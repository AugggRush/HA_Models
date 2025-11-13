"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params
"""
import sys
sys.path.append(".")
sys.path.append("..")
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import putils.torch_asym_stft as torch_asym_stft
from models.lisennet.generator.dpr_layer import DPR, CustomLayerNorm

class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1].clone()
        x_high = self.erb_fc(x[..., self.erb_subband_1:]).clone()
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1].clone()
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:]).clone()
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1,2))[0]
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)
        
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = TRA(in_channels//2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1 = self.tra(h1)

        x =  self.shuffle(h1, x2)
        
        return x


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h== None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h
    
    
class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, output_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, output_size)
        self.intra_ln = nn.LayerNorm((width, output_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, output_size)
        self.inter_ln = nn.LayerNorm(((width, output_size)), eps=1e-8)
    
    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1).contiguous()  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.output_size) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3).contiguous()  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.output_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3).contiguous()   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3*2, 8, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            ConvBlock(8, 4, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(4, 2, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            GTConvBlock(4, 2, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            GTConvBlock(4, 2, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs

class LearnableTanh2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.tanh(self.slope * x)
    
class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(4, 2, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(4, 2, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(4, 2, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            ConvBlock(4, 8, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(8, 2, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs, de_s = None):
        N_layers = len(self.de_convs)
        de_outs = []
        for i in range(N_layers):
            if de_s != None and i > 0:
                inp = x + en_outs[N_layers-1-i] + de_s[i-1]
            else:
                inp = x + en_outs[N_layers-1-i]
            x = self.de_convs[i](inp)
            de_outs.append(x)
        return x, de_outs
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s
    
def deepfilter_complex_filtering_1x1_conv_einsum(spectrum, mask):
    """
    使用einsum优化的版本，适用于1x1卷积输出的mask
    """
    B, _, T, F = spectrum.shape
    
    # 验证mask通道数
    assert mask.shape[1] == 10, "mask的通道数应该是2 * 5=10"
    
    # 重塑mask
    mask_reshaped = mask.view(B, 2, 5, T, F).permute(0, 1, 3, 4, 2)  # (B, 2, T, F, 5)
    
    # 频谱填充和unfold
    spectrum_padded = torch.nn.functional.pad(spectrum, (2, 2), mode='constant', value=0)
    unfolded_spectrum = spectrum_padded.unfold(3, 5, 1)  # (B, 2, T, F, 5)
    
    # 使用einsum进行高效的复数乘加运算
    # 实部: sum(Re(spec)*Re(mask) - Im(spec)*Im(mask))
    real_real = torch.einsum('btfk,btfk->btf', 
                           unfolded_spectrum[:, 0], mask_reshaped[:, 0])
    imag_imag = torch.einsum('btfk,btfk->btf', 
                           unfolded_spectrum[:, 1], mask_reshaped[:, 1])
    real_part = (real_real + imag_imag)
    
    # 虚部: sum(Re(spec)*Im(mask) + Im(spec)*Re(mask))
    real_imag = torch.einsum('btfk,btfk->btf', 
                           unfolded_spectrum[:, 0], mask_reshaped[:, 1])
    imag_real = torch.einsum('btfk,btfk->btf', 
                           unfolded_spectrum[:, 1], mask_reshaped[:, 0])
    imag_part = (real_imag + imag_real)
    
    # 合并结果
    filtered_spectrum = torch.stack([real_part, imag_part], dim=1)  # (B, 2, T, F)
    
    return filtered_spectrum

class PreEnhNet(nn.Module):
    def __init__(
        self,
        n_fft=256,
        hop_len=48,
        win_len=256,
        in_channels=1,
        emb_dim=8,
        hidden_dim=8 * 2,
        n_freqs=41,
        dropout_p=0.1,        
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim//4, (1, 1), (1, 1)),
            CustomLayerNorm((1, n_freqs), stat_dims=(1, 3)),
            nn.PReLU(emb_dim//4),
        )
        self.conv_2 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(emb_dim//4, emb_dim//2, (2, 3), (1, 2), groups=emb_dim//4), # 32
            CustomLayerNorm((1, n_freqs//2), stat_dims=(1, 3)),
            nn.PReLU(emb_dim//2),
        )
        self.conv_3 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(emb_dim//2, emb_dim, (2, 3), (1, 2), groups=emb_dim//2),  # 16
            CustomLayerNorm((1, n_freqs//4), stat_dims=(1, 3)),
            nn.PReLU(emb_dim),
        )

        self.dpr = DPGRNN(emb_dim, n_freqs//4, hidden_dim, emb_dim)
        self.linear_block = nn.Sequential(
            nn.LayerNorm((emb_dim * (n_freqs//4))),
            nn.Linear((emb_dim * (n_freqs//4)), n_freqs),
            nn.PReLU(),
            nn.Dropout(dropout_p)
        )
        self.lsigmoid = LearnableSigmoid2d(n_freqs, beta=1)

    def forward(self, x):
        # x:(b,d,t,f)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        
        x = self.dpr(x)
        x = x.permute(0, 2, 3, 1).flatten(2).contiguous()  # (b,t,d*f)

        x = self.linear_block(x).unsqueeze(-1)  # (b,t,f,1)
        x = self.lsigmoid(x.permute(0,2,1,3)).permute(0,3,2,1)  # (b,1,t,f)
        return x

class GTCRN(nn.Module):
    def __init__(
        self,
        n_fft=256,
        hop_len=48,
        win_len=192,
        all_stage=False
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        
        self.erb1 = ERB(5,19, nfft=n_fft, high_lim=12000, fs=24000)
        self.erb1.requires_grad_(False)

        self.sfe = SFE(3, 1)
        self.preh = PreEnhNet(
            n_fft=n_fft,
            hop_len=hop_len,
            win_len=win_len,
            in_channels=3,
            emb_dim=8,
            hidden_dim=8*2,
            n_freqs=24,
            dropout_p=0.1
        )
        self.all_stage = all_stage
        if all_stage:

            self.erb2 = ERB(21, 20, nfft=n_fft, high_lim=12000, fs=24000)
            self.erb2.requires_grad_(False)
            self.encoder = Encoder()
            
            self.dpgrnn1 = DPGRNN(4, 11, 8, 4)
            self.dpgrnn2 = DPGRNN(4, 11, 8, 4)
            
            self.decoder = Decoder()

            self.num_features1 = 24
            self.num_features2 = 41

            self.mask = Mask()

        self.stft = torch_asym_stft.STFT_asym(
            filter_length=n_fft, hop_length=hop_len, 
            win_length=win_len, window='asqrthann', M=hop_len)

        # 固定这些模块的参数
        self.sfe.requires_grad_(False)
        self.stft.requires_grad_(False)

    def load_preh_from_checkpoint(self, ckpt_path, map_location=None, strict=False, prefix="preh."):
        """
        从 checkpoint 中只加载 PreEnhNet(self.preh) 的参数到当前模型。
        支持以下几种 checkpoint/state_dict 格式：
         - torch.save(model.state_dict(), path)
         - torch.save({'state_dict': model.state_dict(), ...}, path)
         - DataParallel 保存时带 'module.' 前缀
        参数:
         - ckpt_path: checkpoint 文件路径
         - map_location: 传给 torch.load 的 map_location（默认 cpu）
         - strict: 传给 load_state_dict 的 strict
         - prefix: 在 state_dict 中定位 preh 的键前缀，默认 'preh.'
        返回:
         - loaded_keys_count: 成功加载的键数量
        典型用法（Trainer 中）：
         gtcrn = GTCRN(all_stage=True)
         gtcrn.load_preh_from_checkpoint('pretrained_preh.pth')
        """
        map_location = map_location if map_location is not None else "cpu"
        ckpt = torch.load(ckpt_path, map_location=map_location)

        # 提取 state_dict
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        else:
            # 其他情况直接当作 state_dict 处理
            state_dict = ckpt

        # 处理 DataParallel 的 module. 前缀
        def _strip_module(key):
            if key.startswith("module."):
                return key[len("module."):]
            return key

        # 收集以 prefix 开头的键
        preh_state = {}
        for k, v in state_dict.items():
            k_strip = _strip_module(k)
            if k_strip.startswith(prefix):
                new_k = k_strip[len(prefix):]  # 去掉 'preh.' 前缀后传给 self.preh
                preh_state[new_k] = v

        if len(preh_state) == 0:
            print(f"[load_preh_from_checkpoint] 未在 checkpoint 中找到以 '{prefix}' 为前缀的参数，请确认 checkpoint 内容。")
            return 0

        # 加载到 self.preh
        try:
            missing, unexpected = self.preh.load_state_dict(preh_state, strict=strict)
            # load_state_dict 返回 None 或 NamedTuple（PyTorch 2.x 会抛出异常或返回 None）
            # 统一打印信息
            print(f"[load_preh_from_checkpoint] 从 {ckpt_path} 加载 PreEnhNet 参数，键数量: {len(preh_state)}， strict={strict}")
            return len(preh_state)
        except Exception as e:
            # 在旧版本 PyTorch 上 load_state_dict 直接接受 dict 并返回 None 或抛异常
            # 为兼容性，尝试用 non-strict 方式加载以打印更多信息
            if strict:
                print(f"[load_preh_from_checkpoint] 严格加载失败，错误: {e}。可尝试 strict=False 继续。")
                raise
            else:
                # 最后尝试宽松加载
                self.preh.load_state_dict(preh_state, strict=False)
                print(f"[load_preh_from_checkpoint] 宽松模式加载成功（strict=False），键数量: {len(preh_state)}")
                return len(preh_state)

    def forward(self, x):
        """
        x: (B, L)
        """
        n_samples = x.shape[1]
        x = x.unsqueeze(-1) # B, T, C
        spec = self.stft.transform_cpx(x) # B, C, T, F, 2
        spec = spec.squeeze(1) # B, T, F, 2

        spec_real = spec[..., 0] # B, T, F
        spec_imag = spec[..., 1]
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)

        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,F)
        # Pre-enhancement
        p_feat = self.erb1.bm(feat)  # (B,3,T,24)
        p_feat = self.preh(p_feat)  # (B,1,T,F)
        pm = self.erb1.bs(p_feat)  # (B,1,T,F)
        if not self.all_stage:
            spec_enh = pm * spec.permute(0,3,1,2)  # (B,2,T,F)
        else:
            feat_enh = pm * feat  # (B,3,T,F)

            feat_cat = torch.cat([feat, feat_enh], dim=1)  # (B,6,T,F)
            feat = self.erb2.bm(feat_cat)  # (B,6,T,97)
            feat = self.sfe(feat)     # (B,18,T,97)

            feat, en_outs = self.encoder(feat)
            
            feat1 = self.dpgrnn1(feat) # (B,16,T,25)
            feat2 = self.dpgrnn2(feat1) # (B,16,T,25)
            m_feat, de_s = self.decoder(feat2, en_outs)

            m = self.erb2.bs(m_feat) * pm  # (B,2,T,F)
            spec_enh = self.mask(m, spec.permute(0,3,1,2)) # (B,2,T,F)

        spec_enh = spec_enh.permute(0,2,3,1)  # (B,T,F,2)
        spec_enh = spec_enh.unsqueeze(1) # B, C, T, F, 2
        m_output = self.stft.inverse_cpx(spec_enh)
        m_output = m_output.squeeze(-1)
        m_output = torch.nn.functional.pad(m_output, (0, n_samples-m_output.shape[1]))

        return m_output

# 顶层便捷函数，Trainer 可直接调用
def load_preh_into_gtcrn(gtcrn_model, ckpt_path, map_location=None, strict=False, prefix="preh."):
    """
    Trainer 可调用的便捷函数：将 checkpoint 中的 PreEnhNet 参数载入给定的 gtcrn_model。
    """
    if not isinstance(gtcrn_model, GTCRN):
        raise ValueError("gtcrn_model 必须是 GTCRN 实例")
    return gtcrn_model.load_preh_from_checkpoint(ckpt_path, map_location=map_location, strict=strict, prefix=prefix)

if __name__ == "__main__":
    model = GTCRN(all_stage=True).eval()

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (24000,), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(flops, params/1e3)

    # """causality check"""
    # a = torch.randn(1, 16000)
    # b = torch.randn(1, 16000)
    # c = torch.randn(1, 16000)
    # x1 = torch.cat([a, b], dim=1)
    # x2 = torch.cat([a, c], dim=1)

    # y1 = model(x1)[0]
    # y2 = model(x2)[0]

    # print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())
    # print((y1[16000:] - y2[16000:]).abs().max())