"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from modules.convolution import StreamConv2d, StreamConvTranspose2d
import torch_asym_stft_frame

import soundfile as sf
import librosa
import time
from tqdm import tqdm
from gtcrn import GTCRN
from modules.convert import convert_to_stream

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
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
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


class StreamTRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.Linear(channels, channels*2)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x, tra_cache):
        """
        x: (B,C,T,F)
        """
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1,2))
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At, tra_cache


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class StreamGTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        stream_conv_module = StreamConvTranspose2d if use_deconv else StreamConv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)
        
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = stream_conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = StreamTRA(in_channels//2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])  # (B,2C,T,F)
        return x

    def forward(self, x, conv_cache, tra_cache):
        """
        x: (B, C, T, F)
        conv_cache: (B, C, (kT-1)*dT, F)
        tra_cache: (1, B, C)
        """
        x1, x2 = x[:,:x.shape[1]//2], x[:, x.shape[1]//2:]

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1, conv_cache = self.depth_conv(h1, conv_cache)
        h1 = self.depth_act(self.depth_bn(h1))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1, tra_cache = self.tra(h1, tra_cache)

        x =  self.shuffle(h1, x2)
        
        return x, conv_cache, tra_cache



class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = torch.nn.parameter.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = torch.nn.parameter.Parameter(torch.Tensor(*param_size).to(torch.float32))
        torch.nn.init.ones_(self.gamma)
        torch.nn.init.zeros_(self.beta)
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        assert x.ndim == self.num_dims, print(
            "Expect x to have {} dimensions, but got {}".format(self.num_dims, x.ndim))

        mu_ = x.mean(dim=self.stat_dims, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class StreamConvolutionalGLU(nn.Module):
    """纯流式 Convolutional GLU，forward 需要并返回 conv_cache"""
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.norm = CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        # 只能使用流式 depthwise conv（因果），不再提供离线 fallback
        self.dwconv = StreamConv2d(hidden_dim, hidden_dim, kernel_size=(3,3), stride=(1,1), padding=(0,1), groups=hidden_dim)
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, conv_cache):
        """
        x: (b, d, t, f)
        conv_cache: 流式 conv 的缓存（由 StreamConv2d 定义），必须提供
        返回: (out, conv_cache)
        """
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        # 流式 depthwise conv：返回更新后的 conv_cache
        x, conv_cache = self.dwconv(x, conv_cache)
        x = self.act(x) * v
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + res
        return x, conv_cache


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
        self.ouput_size = output_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, output_size)
        self.intra_ln = nn.LayerNorm((width, output_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, output_size)
        self.inter_ln = nn.LayerNorm(((width, output_size)), eps=1e-8)
    
    def forward(self, x, inter_cache):
        """
        x: (B, C, T, F)
        inter_cache: (1, BF, hidden_size)
        """
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.ouput_size) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x, inter_cache = self.inter_rnn(inter_x, inter_cache)     # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.ouput_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out, inter_cache


class StreamEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, 8, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(8, 4, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            StreamGTConvBlock(4, 2, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            StreamGTConvBlock(4, 2, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            StreamGTConvBlock(4, 2, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x, conv_cache, tra_cache):
        """
        x: (B,C,T,F)
        conv_cache: (B,C, (kT-1)*8, F)
        tra_cache: (3,1,B,C)
        """
        en_outs = []
        for i in range(2):
            x = self.en_convs[i](x)
            en_outs.append(x)
        
        x, conv_cache[:,:, :2, :], tra_cache[0] = self.en_convs[2](x, conv_cache[:,:, :2, :], tra_cache[0]); en_outs.append(x)
        x, conv_cache[:,:, 2:6, :], tra_cache[1] = self.en_convs[3](x, conv_cache[:,:, 2:6, :], tra_cache[1]); en_outs.append(x)
        x, conv_cache[:,:, 6:16, :], tra_cache[2] = self.en_convs[4](x, conv_cache[:,:, 6:16, :], tra_cache[2]); en_outs.append(x)
            
        return x, en_outs, conv_cache, tra_cache


class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


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
    

class StreamGTCRN(nn.Module):
    def __init__(        
            self,
            n_fft=256,
            hop_len=48,
            win_len=256,
            bt_size=1,
            postfilter=False
        ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.post_filter = postfilter

        self.erb2 = ERB(24, 24, nfft=n_fft, high_lim=12000, fs=24000)
        self.sfe = SFE(3, 1)

        self.encoder = StreamEncoder()
        
        self.dpgrnn1 = DPGRNN(4, 12, 16, 4)
        self.dpgrnn2 = DPGRNN(4, 12, 16, 4)
        

        self.glu = StreamConvolutionalGLU(4, n_freqs=12, expansion_factor=2, dropout_p=0.1)
        
        # self.decoder = Decoder()

        self.num_features2 = 48

        self.mask = Mask()

        self.linear_block = nn.Sequential(
            nn.LayerNorm((4 * 12)),
            nn.Linear((4 * 12), self.num_features2),
            # nn.PReLU(),
            # nn.Linear(self.num_features2, self.num_features2),
            nn.Dropout(0.1)
        )
        # self.ltanh = LearnableTanh2d(41, beta=1)
        self.lsigm = LearnableSigmoid2d(self.num_features2, beta=1)

        self.stft = torch_asym_stft_frame.StftAsymFrame(
            filter_length=n_fft, hop_length=hop_len, 
            win_length=win_len, window='asqrthann', M=hop_len)
        self.stft.reset_buffer(bt_size)  
        # 固定这些模块的参数
        self.sfe.requires_grad_(False)
        self.stft.requires_grad_(False)

        self.mask = Mask()

    def forward(self, x, conv_cache, tra_cache, inter_cache, glu_cache):
        """
        x: (B, T, C) = (1, 1, 1)
        conv_cache: [en_cache, de_cache], (2, B, C, 8(kT-1)+2, F) = (2, 1, 16, 16, 33)
        tra_cache: [en_cache, de_cache], (2, B, C, 2, F) = (2, 1, 16, 2, 33)
        inter_cache: [cache1, cache2], (2, 1, BF, C) = (2, 1, 33, 16)
        """
        # n_samples = x.shape[1]
        x = x.unsqueeze(-1) # B, T, C
        spec = self.stft.transform_cpx_frame(x) # B, C, T, F, 2
        spec = spec.squeeze(1) # B, T, F, 2

        spec_real = spec[..., 0]
        spec_imag = spec[..., 1]
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        feat = self.erb2.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)     # (B,9,T,129)

        feat, en_outs, conv_cache[0], tra_cache[0] = self.encoder(feat, conv_cache[0], tra_cache[0])

        feat, inter_cache[0] = self.dpgrnn1(feat, inter_cache[0]) # (B,16,T,33)
        feat, glu_cache = self.glu(feat, glu_cache)  # (B,16,T,33)
        feat, inter_cache[1] = self.dpgrnn2(feat, inter_cache[1]) # (B,16,T,33)

        feat_flat = feat.permute(0, 2, 3, 1).flatten(2).contiguous()  # (B,T,16*25)
        mask_linear = self.linear_block(feat_flat)  # (B,T,256)
        # 假设 mask_tanh.shape == [B, T, nfft]
        # F = mask_linear.shape[-1] // 2
        # mask_real = mask_linear[..., :F]      # [B, T, F]
        # mask_imag = mask_linear[..., F:]      # [B, T, F]
        # mask_c = torch.stack([mask_real, mask_imag], dim=1)  # [B, 2, T, F]
        mask_c = mask_linear.unsqueeze(1)  # (B,1,T,256)   
        mask_tanh = self.lsigm(mask_c.permute(0,3,2,1)).permute(0,3,2,1)  # (B,T,256)

        m = self.erb2.bs(mask_tanh)  # (B,2,T,F)
        # spec_enh = self.mask(m, spec.permute(0,3,1,2)) # (B,2,T,F)
        spec_enh = (m * spec.permute(0,3,1,2)) # (B,2,T,F)
        if self.post_filter:
            beta = 0.02
            eps = 1e-12
            mask = (m * spec_mag.unsqueeze(1) / (spec_mag.unsqueeze(1) + eps)).clamp(eps, 1)
            mask_sin = mask * torch.sin(torch.pi * mask / 2).clamp_min(eps)
            pf = (1 + beta) / (1 + beta * mask.div(mask_sin).pow(2))
            spec_enh = spec_enh * pf
        spec_enh = spec_enh.permute(0,2,3,1)  # (B,T,F,2)
        spec_enh = spec_enh.unsqueeze(1) # B, C, T, F, 2
        m_output = self.stft.inverse_cpx_frame(spec_enh)
        m_output = m_output.squeeze(-1)
        # m_output = torch.nn.functional.pad(m_output, (0, n_samples-m_output.shape[1]))

        # m_feat, conv_cache[1], tra_cache[1] = self.decoder(feat, en_outs, conv_cache[1], tra_cache[1])
        
        # m = self.erb.bs(m_feat)

        # spec_enh = self.mask(m, spec_ref.permute(0,3,2,1)) # (B,2,T,F)
        # spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)

        return m_output, conv_cache, tra_cache, inter_cache, glu_cache


if __name__ == "__main__":

	import glob, os, argparse, math
	from tqdm import tqdm

	# 解析命令行参数：wav_dir、out_dir、batch_size
	parser = argparse.ArgumentParser(description="Batch / streaming inference for GTCRN stream model")
	parser.add_argument("--wav_dir", type=str, default="test_wavs", help="输入 wav 文件夹")
	parser.add_argument("--out_dir", type=str, default="test_wavs/stream_infer", help="输出增强 wav 文件夹")
	parser.add_argument("--batch_size", type=int, default=2, help="每次并行推理的文件数（内存受限时减小）")
	args = parser.parse_args()

	# 如果检测到 GPU 则使用 GPU 推理，否则使用 CPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# 加载并移动到选定设备（map_location 以兼容权重文件）
	model = GTCRN(n_fft=128, hop_len=48, win_len=128).to(device).eval()
	state = torch.load('onnx_models/best_model_371.tar', map_location=device)
	model.load_state_dict(state['model'])
	# 流式模型也移动到 device
	stream_model = StreamGTCRN(n_fft=128, hop_len=48, win_len=128, bt_size=args.batch_size).to(device).eval()

	convert_to_stream(stream_model, model)

	# 输入 / 输出 配置（使用 args）
	wav_dir = args.wav_dir
	out_dir = args.out_dir
	batch_size = max(1, args.batch_size)
	os.makedirs(out_dir, exist_ok=True)

	# 收集输入 wav 文件（过滤掉已有增强文件）
	all_wavs = sorted(glob.glob(os.path.join(wav_dir, '*.wav')))
	all_wavs = [p for p in all_wavs if ('_enh.wav' not in os.path.basename(p)) and ('_enh_stream.wav' not in os.path.basename(p))]
	assert len(all_wavs) > 0, "No wav files found in {}".format(wav_dir)

	print("Found {} wav files for inference, batch_size={}".format(len(all_wavs), batch_size))

	target_sr = 24000
	inf_lines = []

	# 按 batch_size 分批处理文件，避免一次性占用过多内存
	num_batches = math.ceil(len(all_wavs) / batch_size)
	for bi in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
		batch_files = all_wavs[bi*batch_size:(bi+1)*batch_size]
		basenames = []
		orig_srs = []
		signals = []

		# 读取并（必要时）重采样到 model 的采样率（24k）
		for p in batch_files:
			data, sr = sf.read(p, dtype='float32')
			if data.ndim > 1:
				data = data[:, 0]
			if sr != target_sr:
				r_data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
			else:
				r_data = data
			signals.append(torch.from_numpy(r_data).float())
			orig_srs.append(sr)
			basenames.append(os.path.splitext(os.path.basename(p))[0])

		# 流式并行推理（仅对该批次内文件并行）
		hop = stream_model.hop_len if hasattr(stream_model, "hop_len") else 48
		num_frames = [s.shape[0] // hop for s in signals]
		max_frames = max(num_frames) if len(num_frames) > 0 else 0

		# 准备 padded batch_stream（24k）
		batch_stream = []
		for s in signals:
			need = max_frames * hop
			if s.shape[0] < need:
				pad = torch.zeros(need - s.shape[0], dtype=s.dtype)
				s = torch.cat([s, pad], dim=0)
			batch_stream.append(s.unsqueeze(0))
		# 将流式输入也移动到 device（以充分利用 GPU）
		batch_stream = torch.cat(batch_stream, dim=0).to(device) if len(batch_stream)>0 else torch.zeros(0,0).to(device)
		B = batch_stream.shape[0] if batch_stream.ndim==2 else 0

		# 初始化缓存（按 batch 大小）
		if B > 0:
			# 在所选 device 上创建缓存
			conv_cache = torch.zeros(2, B, 2, 18, 12, device=device)
			tra_cache = torch.zeros(2, 3, B, 1, 8, device=device)
			inter_cache = torch.zeros(2, 1, B*12, 16, device=device)
			glu_cache = torch.zeros(B, 8, 2, 12, device=device)
		else:
			conv_cache = tra_cache = inter_cache = glu_cache = None

		out_buffers = [[] for _ in range(B)]
		stream_model.stft.reset_buffer(B)
		for fi in tqdm(range(max_frames), desc=f"Streaming batch {bi+1}/{num_batches}", unit="frame"):
			xi = batch_stream[:, fi*hop:(fi+1)*hop]  # (B, hop)
			with torch.no_grad():
				yi, conv_cache, tra_cache, inter_cache, glu_cache = stream_model(xi, conv_cache, tra_cache, inter_cache, glu_cache)
			yi_cpu = yi.detach().cpu()
			for b in range(B):
				out_buffers[b].append(yi_cpu[b:b+1])

		# 合并并保存该批次流式输出（按各自原始采样率）
		for b in range(B):
			if len(out_buffers[b]) == 0:
				ys = torch.zeros(1, 0)
			else:
				ys = torch.cat(out_buffers[b], dim=1)
			if ys.shape[1] > hop:
				ys = ys[:, hop:]
			y_np = ys.numpy().squeeze()
			if orig_srs[b] != target_sr:
				out_stream = librosa.resample(y_np, orig_sr=target_sr, target_sr=orig_srs[b])
			else:
				out_stream = y_np
			out_stream_path = os.path.join(out_dir, f"{basenames[b]}_enh_stream.wav")
			sf.write(out_stream_path, out_stream, orig_srs[b])

		# 释放本批次临时张量以降低内存占用
		del batch_stream, conv_cache, tra_cache, inter_cache, glu_cache
		# 如果使用 GPU 可以适当清理缓存
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	# 写入 inf.scp（所有批次完成后）
	inf_scp_path = os.path.join(out_dir, "inf.scp")
	with open(inf_scp_path, "w", encoding="utf-8") as f:
		f.writelines(inf_lines)

	print("Inference done. Results saved in:", os.path.abspath(out_dir))
