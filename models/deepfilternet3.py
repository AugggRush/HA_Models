import sys
sys.path.append("./models")
sys.path.append("./SEtrain/models")
from functools import partial
from typing import Optional, Tuple
from gtcrn_end2end import ERB, DPGRNN
import torch
from torch import Tensor, nn
from putils import torch_asym_stft

from df_modules import (
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    GroupedLinearEinsum,
    Mask,
    DF,
)
from putils.utils import as_complex

PI = 3.1415926535897932384626433


class ModelParams():
    def __init__(self, config):
        super().__init__()
        self.conv_lookahead = config['conv_lookahead']
        self.conv_ch = config['conv_ch']
        self.conv_depthwise = config['conv_depthwise']
        self.convt_depthwise = config['convt_depthwise']
        self.conv_kernel = config['conv_kernel']
        self.convt_kernel = config['convt_kernel']
        self.conv_kernel_inp = config['conv_kernel_inp']
        self.emb_hidden_dim = config['emb_hidden_dim']
        self.emb_num_layers = config['emb_num_layers']
        self.emb_gru_skip_enc = config['emb_gru_skip_enc']
        self.emb_gru_skip = config['emb_gru_skip']
        self.df_hidden_dim = config['df_hidden_dim']
        self.df_gru_skip = config['df_gru_skip']
        self.df_pathway_kernel_size_t = config['df_pathway_kernel_size_t']
        self.enc_concat = config['enc_concat']
        self.df_num_layers = config['df_num_layers']
        self.df_n_iter = config['df_n_iter']
        self.lin_groups = config['lin_groups']
        self.enc_lin_groups = config['enc_lin_groups']
        self.mask_pf = config['mask_pf']
        self.pf_beta = config['pf_beta']
        self.lsnr_dropout = config['lsnr_dropout']

        self.sr = config['sr']
        self.fft_size = config['fft_size']
        self.hop_size = config['hop_size']
        self.nb_erb = config['nb_erb']
        self.nb_df = config['nb_df']
        # Normalization decay factor; used for complex and erb features
        self.norm_tau = config['norm_tau']
        # Local SNR minimum value, ground truth will be truncated
        self.lsnr_max = config['lsnr_max']
        # Local SNR maximum value, ground truth will be truncated
        self.lsnr_min = config['lsnr_min']
        # Minimum number of frequency bins per ERB band
        self.min_nb_freqs = config['min_nb_freqs']
        # Deep Filtering order
        self.df_order = config['df_order']
        # Deep Filtering look-ahead
        self.df_lookahead = config['df_lookahead']
        # Pad mode. By default, padding will be handled on the input side:
        # - `input`, which pads the input features passed to the model
        # - `output`, which pads the output spectrogram corresponding to `df_lookahead`
        self.pad_mode = config['pad_mode']


# def init_model(run_df: bool = True, train_mask: bool = True):
#     p = ModelParams()
#     model = DfNet(erb, erb_inverse, run_df, train_mask)
#     return model.to(device=get_device())


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


class Concat(nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=-1)


class Encoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        # p = ModelParams()
        assert p.nb_erb % 4 == 0, "erb_bins should be divisible by 4"

        self.erb_conv0 = Conv2dNormAct(
            1, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True
        )
        conv_layer = partial(
            Conv2dNormAct,
            in_ch=p.conv_ch,
            out_ch=p.conv_ch,
            kernel_size=p.conv_kernel,
            bias=False,
            separable=True,
        )
        self.erb_conv1 = conv_layer(fstride=2)
        self.erb_conv2 = conv_layer(fstride=2)
        self.erb_conv3 = conv_layer(fstride=1)
        self.df_conv0 = Conv2dNormAct(
            2, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True
        )
        self.df_conv1 = conv_layer(fstride=2)
        self.erb_bins = p.nb_erb
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4
        df_fc_emb = GroupedLinearEinsum(
            p.conv_ch * p.nb_df // 2, self.emb_in_dim, groups=p.enc_lin_groups
        )
        self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))
        if p.enc_concat:
            self.emb_in_dim *= 2
            self.combine = Concat()
        else:
            self.combine = Add()
        self.emb_n_layers = p.emb_num_layers
        self.emb_gru = DPGRNN(p.conv_ch, p.nb_erb // 4, self.emb_dim, p.conv_ch)
        self.lsnr_droput = p.lsnr_dropout
        if p.lsnr_dropout:
            self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def forward(
        self, feat_erb: Tensor, feat_spec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Encodes erb; erb should be in dB scale + normalized; Fe are number of erb bands.
        # erb: [B, 1, T, Fe]
        # spec: [B, 2, T, Fc]
        # b, _, t, _ = feat_erb.shape
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, F]
        e1 = self.erb_conv1(e0)  # [B, C, T, F/2]
        e2 = self.erb_conv2(e1)  # [B, C, T, F/4]
        e3 = self.erb_conv3(e2)  # [B, C, T, F/4]
        c0 = self.df_conv0(feat_spec)  # [B, C, T, Fc]
        c1 = self.df_conv1(c0)  # [B, C, T, Fc/2]
        cemb = c1.permute(0, 2, 3, 1).flatten(2).contiguous()  # [B, T, (C * Fc/2)]
        cemb = self.df_fc_emb(cemb)  # [B, T, C * F/4]
        emb = e3.permute(0, 2, 3, 1).flatten(2).contiguous()  # [B, T, C * F/4]
        emb = self.combine(emb, cemb).reshape(e3.shape[0], -1, e3.shape[2], e3.shape[3])  # [B, C, T, F/4]
        emb = self.emb_gru(emb)  # [B, C, T, F/4]
        if self.lsnr_droput:
            lsnr = self.lsnr_fc(emb.permute(0, 2, 3, 1).flatten(2).contiguous()) * self.lsnr_scale + self.lsnr_offset
        else:
            lsnr = torch.zeros((e3.shape[0], e3.shape[2], 1), device=feat_erb.device)
        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        # p = ModelParams()
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"

        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4
        self.emb_gru = nn.Sequential()
        for i in range(p.emb_num_layers):
            self.emb_gru.add_module(f"df_gru_{i}", DPGRNN(p.conv_ch, p.nb_erb // 4, p.emb_hidden_dim, p.conv_ch),
            )        
        # self.emb_gru = DPGRNN(p.conv_ch, p.nb_erb // 4, p.emb_hidden_dim, p.conv_ch)
        tconv_layer = partial(
            ConvTranspose2dNormAct,
            kernel_size=p.convt_kernel,
            bias=False,
            separable=True,
        )
        conv_layer = partial(
            Conv2dNormAct,
            bias=False,
            separable=True,
        )
        # convt: TransposedConvolution, convp: Pathway (encoder to decoder) convolutions
        self.conv3p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt3 = conv_layer(p.conv_ch, p.conv_ch, kernel_size=p.conv_kernel)
        self.conv2p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt2 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv1p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt1 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv0p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.conv0_out = conv_layer(
            p.conv_ch, 1, kernel_size=p.conv_kernel, activation_layer=nn.Sigmoid
        )

    def forward(self, emb: Tensor, e3: Tensor, e2: Tensor, e1: Tensor, e0: Tensor) -> Tensor:
        # Estimates erb mask
        # b, c, t, f8 = e3.shape
        emb = self.emb_gru(emb)  # [B, C, T, F/4]
        # emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)  # [B, C, T, F/8]
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]
        return m


class DfOutputReshapeMF(nn.Module):
    """Coefficients output reshape for multiframe/MultiFrameModule

    Requires input of shape B, C, T, F, 2.
    """

    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        new_shape = list(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        coefs = coefs.view(new_shape).contiguous()
        coefs = coefs.permute(0, 3, 1, 2, 4).contiguous()
        return coefs


class DfDecoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        # p = ModelParams()
        self.layer_width = p.conv_ch

        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.df_hidden_dim

        self.df_n_hidden = p.conv_ch * p.nb_erb // 4
        self.df_n_layers = p.df_num_layers
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_out_ch = p.df_order * 2

        conv_layer = partial(Conv2dNormAct, separable=True, bias=False)
        kt = p.df_pathway_kernel_size_t
        self.df_convp = conv_layer(self.layer_width, self.df_out_ch, fstride=1, kernel_size=(kt, 1))

        self.df_gru = nn.Sequential()
        for i in range(p.df_num_layers):
            self.df_gru.add_module(f"df_gru_{i}", DPGRNN(p.conv_ch, p.nb_erb // 4, p.df_hidden_dim, p.conv_ch),
            )
        self.df_out: nn.Module
        out_dim = self.df_bins * self.df_out_ch
        df_out = GroupedLinearEinsum(self.df_n_hidden, out_dim, groups=p.lin_groups)
        self.df_out = nn.Sequential(df_out, nn.Tanh())

    def forward(self, emb: Tensor, c0: Tensor) -> Tensor:
        b, c, t, f4 = emb.shape
        
        c = self.df_gru(emb)  # [B, C, T, F/4]
        c = c.permute(0, 2, 3, 1).contiguous().flatten(2)  # [B, T, H], H: df_n_hidden
        c0 = self.df_convp(c0).permute(0, 2, 3, 1).contiguous()  # [B, T, F, O*2], channels_last
        c = self.df_out(c)  # [B, T, F*O*2], O: df_order
        c = c.view(b, t, self.df_bins, self.df_out_ch).contiguous() + c0  # [B, T, F, O*2]
        return c


class DfNet(nn.Module):
    # run_df: Final[bool]
    # run_erb: Final[bool]
    # lsnr_droput: Final[bool]
    # post_filter: Final[bool]
    # post_filter_beta: Final[float]

    def __init__(
        self,
        config,
        run_df: bool = True,
        train_mask: bool = True,
    ):
        super().__init__()
        p = ModelParams(config)
        layer_width = p.conv_ch
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        self.df_lookahead = p.df_lookahead
        self.nb_df = p.nb_df
        self.freq_bins: int = p.fft_size // 2 + 1
        self.emb_dim: int = layer_width * p.nb_erb
        self.erb_bins: int = p.nb_erb
        if p.conv_lookahead > 0:
            assert p.conv_lookahead >= p.df_lookahead
            self.pad_feat = nn.ConstantPad2d((0, 0, -p.conv_lookahead, p.conv_lookahead), 0.0)
        else:
            self.pad_feat = nn.Identity()
        if p.df_lookahead > 0:
            self.pad_spec = nn.ConstantPad3d((0, 0, 0, 0, -p.df_lookahead, p.df_lookahead), 0.0)
        else:
            self.pad_spec = nn.Identity()

        self.erb_c = ERB(12, 20, p.fft_size, p.sr // 2, p.sr)
        self.erb_c.requires_grad_(False)  # 冻结所有参数
        
        self.enc = Encoder(p)
        self.erb_dec = ErbDecoder(p)
        self.mask = Mask()
        # self.erb_inv_fb = erb_inv_fb
        self.post_filter = p.mask_pf
        self.post_filter_beta = p.pf_beta

        self.df_order = p.df_order
        self.nb_df = p.nb_df
        self.df_op = DF(num_freqs=p.nb_df, frame_size=p.df_order, lookahead=self.df_lookahead)
        self.df_dec = DfDecoder(p)
        self.df_out_transform = DfOutputReshapeMF(self.df_order, p.nb_df)

        self.run_erb = p.nb_df + 1 < self.freq_bins
        # if not self.run_erb:
        #     logger.warning("Running without ERB stage")
        self.run_df = run_df
        # if not run_df:
        #     logger.warning("Running without DF stage")
        self.train_mask = train_mask
        self.lsnr_droput = p.lsnr_dropout

        self.stft = torch_asym_stft.STFT_asym(
            filter_length=p.fft_size,
            hop_length=p.hop_size,
            win_length=p.fft_size,
            window="asqrthann",
            M=p.hop_size
        )
        self.stft.requires_grad_(False)  # 冻结所有参数

        assert p.df_n_iter == 1

    def forward(
        self,
        audio: Tensor,
        # spec: Tensor,
        # feat_erb: Tensor,
        # feat_spec: Tensor,  # Not used, take spec modified by mask instead
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward method of DeepFilterNet2.

        Args:
            audio (Tensor): Input audio waveform of shape [B, T]
            spec (Tensor): Spectrum of shape [B, 1, T, F, 2]
            feat_erb (Tensor): ERB features of shape [B, 1, T, E]
            feat_spec (Tensor): Complex spectrogram features of shape [B, 1, T, F', 2]

        Returns:
            spec (Tensor): Enhanced spectrum of shape [B, 1, T, F, 2]
            m (Tensor): ERB mask estimate of shape [B, 1, T, E]
            lsnr (Tensor): Local SNR estimate of shape [B, T, 1]
        """
        # device = audio.device
        n_samples = audio.shape[1]
        audio = audio.unsqueeze(-1) # B, T, C
        spec = self.stft.transform_cpx(audio) # B, C, T, F, 2
        feat_spec = spec[...,:self.nb_df,:].squeeze(1).permute(0, 3, 1, 2).contiguous() # B, C, T, F, 2 -> B, 2, T, F

        spec_real = spec[..., 0].permute(0,1,2,3).contiguous() # B, C, T, F
        spec_imag = spec[..., 1].permute(0,1,2,3).contiguous() # B, C, T, F
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        # feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,F)

        feat = self.erb_c.bm(spec_mag) 
        feat_erb = self.pad_feat(feat)
        feat_spec = self.pad_feat(feat_spec)
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)

        if self.lsnr_droput:
            idcs = lsnr.squeeze() > -10.0
            b, t = (spec.shape[0], spec.shape[2])
            m = torch.zeros((b, 1, t, self.erb_bins), device=spec.device)
            df_coefs = torch.zeros((b, t, self.nb_df, self.df_order * 2))
            spec_m = spec.clone()
            emb = emb[:, idcs]
            e0 = e0[:, :, idcs]
            e1 = e1[:, :, idcs]
            e2 = e2[:, :, idcs]
            e3 = e3[:, :, idcs]
            c0 = c0[:, :, idcs]

        if self.run_erb:
            if self.lsnr_droput:
                m[:, :, idcs] = self.erb_dec(emb, e3, e2, e1, e0)
            else:
                m = self.erb_dec(emb, e3, e2, e1, e0)
            m = self.erb_c.bs(m)
            spec_m = self.mask(spec, m)

        else:
            m = torch.zeros((), device=spec.device)
            spec_m = spec.clone()

        if self.run_df:
            if self.lsnr_droput:
                df_coefs[:, idcs] = self.df_dec(emb, c0)
            else:
                df_coefs = self.df_dec(emb, c0)
            df_coefs = self.df_out_transform(df_coefs)
            spec_e = self.df_op(spec_m.clone(), df_coefs)
            spec_e[..., self.nb_df :, :] = spec_m[..., self.nb_df :, :]
        else:
            df_coefs = torch.zeros((), device=spec.device)
            spec_e = spec_m

        if self.post_filter:
            beta = self.post_filter_beta
            eps = 1e-12
            mask = (as_complex(spec_e).abs() / as_complex(spec).abs().add(eps)).clamp(eps, 1)
            mask_sin = mask * torch.sin(PI * mask / 2).clamp_min(eps)
            pf = (1 + beta) / (1 + beta * mask.div(mask_sin).pow(2))
            spec_e = spec_e * pf.unsqueeze(-1)

        enh = self.stft.inverse_cpx(spec_e)
        m_output = enh.squeeze(-1)
        m_output = torch.nn.functional.pad(m_output, (0, n_samples-m_output.shape[1]))  

        return m_output, spec_e, m, lsnr, df_coefs

def detect_unused_params(model: nn.Module, sample_input: Tensor, device: Optional[torch.device] = None):
    """在一次前向推理中检测未被使用的参数并打印，返回未使用参数名列表。
    通过给包含参数的 module 注册 forward hook，在 module 被调用时标记其 own parameters 为已使用。
    """
    model.eval()
    # map param id -> name
    param_name = {id(p): n for n, p in model.named_parameters()}
    used_ids = set()
    hooks = []

    # 为每个拥有 parameters(recurse=False) 的 module 注册 hook
    for module in model.modules():
        own_params = [p for p in module.parameters(recurse=False)]
        if not own_params:
            continue
        ids = [id(p) for p in own_params]
        def make_hook(ids_local):
            def hook(module, inp, out):
                used_ids.update(ids_local)
            return hook
        hooks.append(module.register_forward_hook(make_hook(ids)))

    # move sample_input to device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    sample_input = sample_input.to(device)

    with torch.no_grad():
        try:
            # 仅执行一次前向推理以触发 hooks；若 forward 需要额外参数，可捕获异常
            model(sample_input)
        except Exception:
            # 忽略前向执行时因输入形状不匹配或额外参数导致的异常，
            # hooks 仍已在被调用的模块上记录到 used_ids。
            pass

    # 移除 hooks
    for h in hooks:
        h.remove()

    # 收集未使用参数名称
    unused = []
    for name, p in model.named_parameters():
        if id(p) not in used_ids:
            unused.append(name)

    # 打印结果
    if unused:
        print(f"Detected {len(unused)} unused parameters:")
        for n in unused:
            print("  ", n)
    else:
        print("No unused parameters detected.")

    return unused

if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load('../configs/df_train_cfg.yaml')
    # 将 OmegaConf 的 DictConfig/ListConfig 等转换为原生的 Python 容器（dict/list）
    # 这样 configs 中的 snd_db: [0, 15] 会成为 Python 列表 [0, 15]
    try:
        config = OmegaConf.to_container(config, resolve=True)
    except Exception:
        # 若转换失败则保持原始 config（兼容性），后续可手动转换字段
        pass
        
    model = DfNet(config['network_config']).eval()

    # 新增：示例运行未使用参数检测（按需调整样本长度/批次）
    try:
        sample = torch.zeros(1, 24000)  # 1 x 时间长度，按你的输入采样长度调整
        detect_unused_params(model, sample)
    except Exception as e:
        print("detect_unused_params failed:", e)

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (24000,), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(flops, params/1e3)