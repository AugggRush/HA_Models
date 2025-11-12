
import torch
from torch import nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("../../")
sys.path.append("../")
sys.path.append(".")

from models.base.cf_bn import ChannelFreqBatchNorm
from models.base.fft2band import BandConverter
from models.base.torch_asym_stft import STFT_asym
from models.base.torch_stft import STFT
from torch.amp import autocast
from einops import rearrange


EPS = 1e-8

def power_compress(x):
    # x: ...2
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag) + EPS
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = (mag + EPS)**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)

class cLN(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(cLN, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class DenoiseGruNet(nn.Module):
    def __init__(self, mic_num=1, fs=16000,
                 NFFT=128, window_len=128, hop_size=32,
                 eband_num=32, gru1_size=96, gru2_size=64, symmetry_win=False, compression="log",
                 mix_precision = False, tbptt_seg=1, M=32, mask_limit=0.0
                 ):
        super(DenoiseGruNet, self).__init__()
        self.mic_num = mic_num
        if symmetry_win:
            self.trans = STFT(filter_length=NFFT, hop_length=hop_size, win_length=window_len, window='hann')
        else:
            self.trans = STFT_asym(filter_length=NFFT, hop_length=hop_size, win_length=window_len, window='asqrthann', M=M)
            print(f"asqrthann")
        bin_num = NFFT // 2 + 1

        self.fft2band = BandConverter(band_num=eband_num, freq_bins=bin_num, fs=fs, band_method="erb")

        self.conv1 = nn.Conv2d(in_channels=self.mic_num*2, out_channels=8, kernel_size=(1, 4), stride=(1, 4), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=7, kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.bn1 = ChannelFreqBatchNorm(2 * eband_num)
        self.bn2 = ChannelFreqBatchNorm(eband_num//8*7)

        self.grp = 1
        self.gru1 = nn.GRU(input_size=eband_num//self.grp, hidden_size=gru1_size, num_layers=1, batch_first=True)  # 48-48
        self.gru2 = nn.GRU(input_size=gru1_size, hidden_size=gru2_size, num_layers=1, batch_first=True)

        self.fc0 = nn.Linear(in_features=gru1_size, out_features=gru1_size)  # 48-32
        self.fc = nn.Linear(in_features=gru2_size*self.grp, out_features=eband_num)  # 48-32

        self.flatten_parameters()
        self.compression = compression
        self.mix_precision = mix_precision
        self.cln1 = cLN(channel_size = eband_num // self.grp)
        self.cln2 = cLN(channel_size = gru1_size)
        self.tbptt_seg = tbptt_seg
        self.mask_limit = mask_limit

    def flatten_parameters(self):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()

    def forward(self, ipt, tbptt_seg=None, snr=20):
        """
        :param ipt: noisy wav [1, 128, 1]
        :mask_limit: in dB
        :return: enhanced wav [1, 128, 1]
        """
        snr_lin = 10.0 ** (-snr / 20.0)
        ipt = ipt.unsqueeze(-1)  # [B,T,C,1] [1, 128, 1, 1]
        if tbptt_seg is not None:
            self.tbptt_seg = tbptt_seg
        with autocast(enabled=self.mix_precision, device_type='cuda'): # auto cast for mixed precision training
            iptc = self.trans.transform_cpx(ipt)  # [B,C,T,F,2(r+i)]  [1, 1, 5, 65, 2]
            eband_features = self.fft2band.forward_band_mat(iptc.permute(0,1,2,4,3))  # (B,C,T,2 M)
            if self.compression == "log":
                o = torch.log(eband_features + 1.0e-8)  # [1, 1, 5, 32]
            elif self.compression == "sqrt":
                o = power_compress(eband_features.permute(0,1,2,4,3)) # B C T M 2
                o = rearrange(o, "b c t f ri -> b (c ri) t f", ri=2)
            elif self.compression == "pcen":
                o = self.pcen(eband_features.squeeze(1)).unsqueeze(1)

            #print(f"bn1-in {self.bn1.bn.num_batches_tracked} {self.bn1.bn.running_mean} {self.bn1.bn.running_var}")
            o = F.gelu(self.bn1(self.conv1(o)))  # [B,C,T,F] [1, 6, 5, 32]
            o = F.gelu(self.bn2(self.conv2(o)))  # [B,C,T,F] [1, 3, 5, 32]
            condition = torch.ones_like(o[:, :1, :, :]) * snr_lin
            o = torch.cat([o, condition], dim=1)
            #print(f"bn1-out {self.bn1.bn.num_batches_tracked} {self.bn1.bn.running_mean} {self.bn1.bn.running_var}")
            # 分组GRU(layer=2, group=2)
            o = o.permute(0, 2, 1, 3)  # [B,T,C,F] [1, 5, 3, 32]
            o = o.contiguous().view(o.shape[0], o.shape[1], -1)  # [B,T,C*F//2] [1, 5, 48]
            o = rearrange(o, "b t (c g f) ->  (b g) t (c f)", g = self.grp, c=3)
            o = self.cln1(o.permute(0,2,1)).permute(0,2,1)

            if self.tbptt_seg > 1:
                o = rearrange(o, "b (g t) f ->  (b g) t f", g = self.tbptt_seg)
            o, _ = self.gru1(o)  # B T F [1, 5, 48]
            o = F.gelu(self.fc0(o))
            o = rearrange(o, "(b g) t f -> b t (g f)", g = self.grp)
            o = rearrange(o, "b t (f g) -> (b g) t f", g = self.grp)
            o = self.cln2(o.permute(0,2,1)).permute(0,2,1)
            o, _ = self.gru2(o)  # B T F [1, 5, 32]
            if self.tbptt_seg > 1:
                o = rearrange(o, "(b g) t f ->  b (g t) f", g = self.tbptt_seg)
            o = rearrange(o, "(b g) t f -> b t (f g)", g = self.grp)
            eband_mask = torch.sigmoid(self.fc(o))  # B T F [1, 5, 32]
            eband_mask = eband_mask.unsqueeze(1)  # B 1 T F [1, 1, 5, 32]
            band_mask = self.fft2band.inverse_band_mat(eband_mask)  # [1, 1, 5, 65]
            optc = iptc[:, 0:self.mic_num, :, :, :] * band_mask.unsqueeze(-1)  # [B,C,T,F,2] [1, 1, 5, 65, 2]

            enhance_wav = self.trans.inverse_cpx(optc)  # [B,T,C] [1, 128, 1]
            enhance_wav = enhance_wav.float()
            noise_est = ipt - enhance_wav  # 估计噪声
        return enhance_wav, noise_est

def Test_DenoiseGroupGruNet_128():
    from thop import profile
    torch.manual_seed(123)
    device = "cpu"  # if not torch.cuda.is_available() else "cuda:0"
    NFFT, window_len, frame_len = 512, 512, 32
    data = torch.rand((1, 16000, 1)).to(device)
    model = DenoiseGruNet(symmetry_win=False,
                                        NFFT=NFFT,
                                        window_len=window_len,
                                        hop_size=frame_len,
                                        compression="sqrt", gru1_size=96, gru2_size=96, eband_num=96, tbptt_seg=1, M=32).to(device)
    outputs = model(data)
    flops, params = profile(model, inputs=(data,))
    print("Flops: %.3fM, Params: %.3fKB" % (flops / 1e6, params / 1e3))

if __name__ == "__main__":

    Test_DenoiseGroupGruNet_128()
