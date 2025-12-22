import torch
import torch.nn as nn
import sys
sys.path.append(r'../')
sys.path.append(r'.')
import soundfile as sf
from putils.torch_asym_stft import STFT_asym
from putils.fft2band_opt import BandConverter
from putils.torch_stft import STFT
import numpy as np
from percep_loss_core  import PercepLoss
EPS = 1.0e-8


class IRMProc(nn.Module):
    def __init__(self, erb_num=96, hop_size=32, symmetry_win=True, gama=0.1, fs=16000, win_size=512):
        super(IRMProc, self).__init__()
        # print(f"DenoiseGroupGruNet: mic_num:{mic_num}, eband_num:{eband_num}")
        symmetry_win = symmetry_win
        NFFT = win_size
        hop_size = hop_size
        self.trans = STFT_asym(filter_length=NFFT, hop_length=hop_size, win_length=win_size, window='asqrthann')
        bin_num = NFFT // 2 + 1
        self.fft2band = BandConverter(band_num=erb_num, freq_bins=bin_num, fs=fs, band_method="erb")
        self.gama = gama
        self.fs = fs
        self.seg_freq_idx = int(2000 / fs * 512)
        print(f"self.seg_freq_idx {self.seg_freq_idx}")

    def post_procss(self, mask, mag_clean):
        g_b = mask
        g_b_w = g_b * torch.sin(3.1415926/2 * g_b) + 1e-10
        mask = (((1+self.gama)*g_b/g_b_w)/(1+self.gama*(g_b/g_b_w)**2))*g_b_w
        mask[mask < self.gama] = 0
        #print(f"mag_clean {mag_clean.shape}")
        #mag_clean = mag_clean / (torch.mean(mag_clean, dim=-1, keepdim=True)+EPS)
        #mask[mag_clean<0.2] = 0
        return mask

    def forward(self, noisy, clean):
        """
        Args: noisy clean wav [B T 1]
        ret: enhanced  [B T 1]
        """

        ipt_noisy = self.trans.transform_cpx(noisy)  # [B,C,T,F,2(r+i)]  [1, 1, 5, 65, 2]
        mag_noisy = torch.sqrt(ipt_noisy[:, :, :, :, 0] ** 2 + ipt_noisy[:, :, :, :, 1] ** 2 + EPS)  # 幅度谱 (B,C,T,F) [1, 1, 5, 65]

        ipt_clean = self.trans.transform_cpx(clean)  # [B,C,T,F,2(r+i)]  [1, 1, 5, 65, 2]
        mag_clean = torch.sqrt(ipt_clean[:, :, :, :, 0] ** 2 + ipt_clean[:, :, :, :, 1] ** 2 + EPS)  # 幅度谱 (B,C,T,F) [1, 1, 5, 65]

        band_noisy = self.fft2band.forward_band_mat(mag_noisy)  # (B,C,T,M) [1, 1, 5, 32]
        band_clean = self.fft2band.forward_band_mat(mag_clean)  # (B,C,T,M) [1, 1, 5, 32]
        band_mask = band_clean / band_noisy
        band_inv_mask = self.fft2band.inverse_band_mat(band_mask)  # [1, 1, 5, 65]
        optc = ipt_noisy * band_inv_mask.unsqueeze(-1)  # [B,C,T,F,2] [1, 1, 5, 65, 2]
        band_inv_mask = self.post_procss(band_inv_mask, mag_clean)
        amp = mag_noisy * band_inv_mask
        pha_clean = torch.atan2(ipt_clean[:, :, :, :, 1], ipt_clean[:, :, :, :, 0])  # B 1 T F
        pha_noisy = torch.atan2(ipt_noisy[:, :, :, :, 1], ipt_noisy[:, :, :, :, 0])  # B 1 T F
        pha = torch.cat([pha_clean[:,:,:,:self.seg_freq_idx], pha_noisy[:,:,:,self.seg_freq_idx:]], dim=-1)
        pha = pha_noisy
        optc = torch.stack((amp * torch.cos(pha), amp * torch.sin(pha)), dim=-1)  # B 1 T F 2

        enhance_wav = self.trans.inverse_cpx(optc)  # [B,T,C] [1, 128, 1]
        enhance_wav = enhance_wav.float()
        return enhance_wav


class CpxCompressSpecDistWithConsistency(nn.Module):
    def __init__(self, alpha=0.3, beta=0.3, fft_size=512, norm_flag=False, eps=1e-12, loss_type="mse", revise_clean=False, hop_size=128, percep_weight=0.0, mix_noise=0.0):
        super(CpxCompressSpecDistWithConsistency, self).__init__()
        self.percep_weight = percep_weight
        self.percep_obj = PercepLoss(nbarks=49, win_length=fft_size, n_fft=fft_size, hop_length=hop_size)
        self.eps = eps
        self.alpha = alpha  # complex loss ratio
        self.beta = beta  # spectrum compress factor
        self.norm_flag = norm_flag
        self.loss_type = loss_type
        self.trans = STFT(
            filter_length=fft_size,
            hop_length=hop_size,
            win_length=fft_size,
            window='hann'
        )
        self.idx = 0
        self.revise_clean = revise_clean
        if self.revise_clean:
            self.irm_proc = IRMProc(erb_num=48, hop_size=hop_size, symmetry_win=True, win_size=fft_size, fs=24000)

        self.mix_noise = mix_noise
        self.loss_weight_amp_stat = 0.0
        self.loss_weight_cpx_stat = 0.0



    def forward(self, clean, estimation, source_lengths=None, noisy=None, noise=None,improve_snr=None, spk_out=None, aug=False):
        # clean: B T C
        # estimation: B T C
        # print(f"clean.shape {clean.shape} {estimation.shape}")
        self.idx += 1
        if improve_snr is None:
            if self.mix_noise > 0.001:
                noise = self.mix_noise * (noisy - clean)
                clean = clean + noise
        else:
            #pass
            snr_lin = 10.0 ** (-improve_snr / 20.0)
            noise = snr_lin.view(-1, 1, 1) * noise
            clean = clean + noise

        #clean = clean[:, :, :1].repeat(1, 1, estimation.shape[-1])
        # estimation = estimation[:,:,:].repeat(1, 1, clean.shape[-1])
        # clean = clean[:, :, :estimation.shape[-1]]
        clean = clean.unsqueeze(-1)  # B T C
        estimation = estimation.unsqueeze(-1)  # B T C
        if self.norm_flag:
            assert noisy is not None, f"noisy should not be None!"
            noisy = noisy.unsqueeze(-1) if noisy is not None else None  # B T C
            # cat_sig = torch.cat([clean, estimation], dim=1)  # B T1 C -> B T2 C
            max_v = torch.max(torch.abs(noisy).view(noisy.shape[0], -1), dim=-1, keepdim=True).values
            max_v = max_v.unsqueeze(-1)
            gain = 1.0 / (max_v + self.eps)
            gain[gain > 100] = 100
            clean_cpx = self.trans.transform_cpx(clean * gain)  # B C T F 2(real+imag)
            est_cpx = self.trans.transform_cpx(estimation * gain)  # B C T F 2(real+imag)
        else:
            clean_cpx = self.trans.transform_cpx(clean)  # B C T F 2(real+imag)
            est_cpx = self.trans.transform_cpx(estimation)  # B C T F 2(real+imag)

        if self.revise_clean:
            clean = self.irm_proc(noisy, clean)
        assert clean.shape == estimation.shape
        assert clean.shape[-1] < 10
        if self.percep_weight > 1e-4:
            percep_loss = -1 * self.percep_weight * self.percep_obj.mos(clean[:,:,0], estimation[:,:,0])
            percep_loss = torch.mean(percep_loss)
        else:
            percep_loss = 0.0

        clean_cpx = clean_cpx[:, :, :, 1:, :]
        est_cpx = est_cpx[:, :, :, 1:, :]

        if aug:
            fs = 24000
            fft_size = 512
            low, high = 460, 600
            freqs = torch.linspace(0, fs/2, fft_size//2, device=noise.device)
            weight = torch.ones_like(freqs)
            weight[(freqs >= low) & (freqs <= high)] = 2.0   # 权重加倍

        # clean_mag: B C T F 1
        clean_mag = (clean_cpx[:, :, :, :, 0] ** 2 + clean_cpx[:, :, :, :, 1] ** 2 + self.eps).unsqueeze(-1) ** 0.5
        clean_mag_compress = clean_mag ** self.beta
        clean_cpx_compress = clean_cpx * clean_mag_compress / clean_mag
        est_mag = (est_cpx[:, :, :, :, 0] ** 2 + est_cpx[:, :, :, :, 1] ** 2 + self.eps).unsqueeze(-1) ** 0.5
        est_mag_compress = est_mag ** self.beta
        est_cpx_compress = est_cpx * est_mag_compress / est_mag
        if self.loss_type == "mse":
            diff_amp = (clean_mag_compress - est_mag_compress) ** 2.0
            if aug:
                diff_amp = diff_amp * weight[None, None, None, :, None]
            loss_spe = (1 - self.alpha) * torch.mean(diff_amp)
            diff_cpx = (clean_cpx_compress - est_cpx_compress) ** 2.0
            if aug:
                diff_cpx = diff_cpx * weight[None, None, None, :, None]
            loss_spe += self.alpha * torch.mean(diff_cpx)
            diff_amp = torch.mean(diff_amp, dim=[0,1,2,4])
            diff_cpx = torch.mean(diff_cpx, dim=[0,1,2,4])
            if self.idx % 20 == 21:
                self.loss_weight_amp_stat = self.loss_weight_amp_stat * 0.995 + 0.005 * (1 - self.alpha) * diff_amp
                self.loss_weight_cpx_stat = self.loss_weight_cpx_stat * 0.995 + 0.005 * (1 - self.alpha) * diff_cpx
                print(self.loss_weight_amp_stat, self.loss_weight_amp_stat.shape)
                print(f"self.loss_weight_cpx_stat: {self.loss_weight_cpx_stat}")
        elif self.loss_type == "log_cosh":
            loss_spe = (1 - self.alpha) * torch.mean(torch.log(torch.cosh(clean_mag_compress - est_mag_compress)))
            loss_spe += self.alpha * torch.mean(torch.log(torch.cosh(clean_cpx_compress - est_cpx_compress)))

        # if loss_spe > 1:
        #     print(f"idx {self.idx} abnormal loss: {loss_spe} gain {gain}")
        #     clean = clean.detach().transpose(0,1).contiguous().view(-1, clean.shape[0]).cpu().numpy()
        #     noisy = noisy.detach().transpose(0,1).contiguous().view(-1, noisy.shape[0]).cpu().numpy()
        #     est = estimation.detach().transpose(0,1).contiguous().view(-1, estimation.shape[0]).cpu().numpy()
        #     sf.write(f"/data1/weiran/train_version/train_single_channel_G_2ms_4ms_2out_hafb_2.67/workdir/train_denoise_test_ir_greynoise/train_denoise/scripts/{self.idx}_clean.wav", clean, 16000)
        #     sf.write(f"/data1/weiran/train_version/train_single_channel_G_2ms_4ms_2out_hafb_2.67/workdir/train_denoise_test_ir_greynoise/train_denoise/scripts/{self.idx}_noisy.wav", noisy, 16000)
        #     sf.write(f"/data1/weiran/train_version/train_single_channel_G_2ms_4ms_2out_hafb_2.67/workdir/train_denoise_test_ir_greynoise/train_denoise/scripts/{self.idx}_est.wav", est, 16000)
        #     if spk_out is not None:
        #         spk_out = spk_out.detach().transpose(0,1).contiguous().view(-1, spk_out.shape[0]).cpu().numpy()
        #         sf.write(f"/data1/weiran/train_version/train_single_channel_G_2ms_4ms_2out_hafb_2.67/workdir/train_TF_snake_test_ir_5%0916_ss/train_TF_snake/scripts/{self.idx}_spk_out.wav", spk_out, 16000)

        return loss_spe + percep_loss


if __name__ == '__main__':
    ipt = torch.rand(4, 24000, 2, dtype=torch.float32) * 0.06
    tgt = ipt - 1
    noy = tgt + 0.05 * torch.rand(4, 24000, 2, dtype=torch.float32)
    loss = CpxCompressSpecDistWithConsistency(norm_flag=True)(ipt, tgt, noisy=noy)
    print(loss)
