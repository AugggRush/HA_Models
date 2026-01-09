import math
import numpy as np
import torch
import torch.nn as nn
from putils.torch_asym_stft import STFT_asym

class HybridLoss(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        compress_factor=0.3,
        eps=1e-12,
        lamda_ri=30,
        lamda_mag=70):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.window = torch.hann_window(win_len)
        self.c = compress_factor
        self.eps = eps
        self.lamda_ri = lamda_ri
        self.lamda_mag = lamda_mag

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        
        device = y_true.device
        
        pred_stft = torch.stft(y_pred, self.n_fft, self.hop_len, self.win_len, self.window.to(device), return_complex=True)
        true_stft = torch.stft(y_true, self.n_fft, self.hop_len, self.win_len, self.window.to(device), return_complex=True)

        pred_mag = torch.abs(pred_stft).clamp(self.eps)
        true_mag = torch.abs(true_stft).clamp(self.eps)
        
        pred_stft_c = pred_stft / pred_mag**(1 - self.c)
        true_stft_c = true_stft / true_mag**(1 - self.c)

        real_loss = torch.mean((pred_stft_c.real - true_stft_c.real)**2)
        imag_loss = torch.mean((pred_stft_c.imag - true_stft_c.imag)**2)
        mag_loss = torch.mean((pred_mag**self.c - true_mag**self.c)**2)

        # SISNR loss
        y_norm = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)
        sisnr = - 2*torch.log10(
            torch.norm(y_norm, dim=-1, keepdim=True) / 
            torch.norm(y_pred - y_norm, dim=-1, keepdim=True).clamp(self.eps) + 
            self.eps
        ).mean()
        
        return self.lamda_ri*(real_loss + imag_loss) + self.lamda_mag*mag_loss + 0.1 * sisnr


class STFTLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_len=120, win_len=600, window="hann_window", weight_sc=1, weight_mag=1):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.weight_sc = weight_sc
        self.weight_mag = weight_mag
        self.register_buffer("window", getattr(torch, window)(win_len))

    def loss_spectral_convergence(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

    def loss_log_magnitude1(self, x_mag, y_mag):
        return torch.nn.functional.l1_loss((y_mag)**0.3, (x_mag)**0.3)
    def loss_log_magnitude2(self, x_mag, y_mag):
        return torch.nn.functional.mse_loss((y_mag)**0.3, (x_mag)**0.3)
    
    def forward(self, x, y):
        """x, y: (B, T), in time domain"""
        x = torch.stft(x, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        y = torch.stft(y, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        x_mag = torch.abs(x).clamp(1e-8)
        y_mag = torch.abs(y).clamp(1e-8)
        
        sc_loss = self.loss_spectral_convergence(x_mag, y_mag)
        mag_loss = self.loss_log_magnitude2(x_mag, y_mag)
        loss = self.weight_sc * sc_loss + self.weight_mag * mag_loss

        return loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[240, 120, 50],
        win_lengths=[1200, 600, 240],
        window="hann_window",
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, hs, wl, window)]

    def forward(self, x, y):
        loss = 0.0
        for f in self.stft_losses:
            loss += f(x, y)
        loss = loss / len(self.stft_losses)
        return loss
    
class DfLoss(nn.Module):
    def __init__(self,            
                fft_sizes=[2048, 1024, 512],
                hop_sizes=[240, 120, 50],
                win_lengths=[1200, 600, 240],
                window="hann_window",
                compress_factor=0.3,
                eps=1e-12,
                lamda_ri=30,
                lamda_mag=70,                
                lambda_Mr = 1.0,
                lambda_Hyb = 1.0
            ):
        super().__init__()
        self.loss_mr = MultiResolutionSTFTLoss(
            fft_sizes,
            hop_sizes,
            win_lengths,
            window
            )
        self.loss_hyb = HybridLoss(
            fft_sizes[-1], 
            hop_sizes[-1], 
            win_lengths[-1],
            compress_factor,
            eps,
            lamda_ri,
            lamda_mag)
        
        self.lambda_Mr = lambda_Mr
        self.lambda_Hyb = lambda_Hyb

    def forward(self, y_pred, y_true):
        y_pred = y_pred.clone()
        y_true = y_true.clone()
        loss1 = self.loss_mr(y_pred, y_true)
        loss2 = self.loss_hyb(y_pred, y_true)
        return self.lambda_Mr * loss1 + self.lambda_Hyb * loss2, loss1, loss2


class MelSubbandLoss(nn.Module):
	"""
	Mel-subband weighted MSE loss.

	行为：
	- 在 Mel 轴上等距划分得到 K+2 个边界频率 fc。
	- 第 i 个子带包含频率在 fc[i] 到 fc[i+2] 的 FFT 频点。
	- 对时域信号做 STFT -> log-power (dB)，在每个子带上计算 MSE。
	- 通过 Equal-loudness (40 phon) 查表获取 SPL(fc[i])，权重为 w[i]=SPL(1000)/SPL(fc[i])。
	- 返回加权后归一化的标量 loss。
	"""
	def __init__(self, sample_rate=16000, nfft=512, hop_size=256, K=10, phon=40, eps=1e-8, device='cpu'):
		super(MelSubbandLoss, self).__init__()
		self.sr = sample_rate
		self.nfft = nfft
		self.hop = hop_size
		self.K = K
		self.phon = phon
		self.eps = eps
		self.device = device

		# 预计算 FFT 频点（0..sr/2）
		self.freq_bins = np.linspace(0.0, self.sr / 2.0, self.nfft // 2 + 1)

		# 准备 equal-loudness 查表（40 phon）——此表为近似值，可替换为精确 ISO226 数据
		self._ref_freqs = np.array([20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500], dtype=float)
		self._spl_40 = np.array([
			122.0,110.5,100.5,92.5,86.0,80.0,74.2,69.5,65.2,61.0,
			57.5,54.0,50.8,48.0,45.5,43.5,41.8,40.0,39.2,39.8,
			41.2,43.0,45.8,49.0,52.5,58.0,66.0,75.0,85.0
		], dtype=float)
		self._spl_1000 = float(np.interp(1000.0, self._ref_freqs, self._spl_40))
		
		self.stft = STFT_asym(
			filter_length=nfft, hop_length=hop_size, 
			win_length=nfft, window='asqrthann', M=hop_size)

		# -------------------------
		# 预计算 mel 边界、子带掩码和权重 -> 注册为 buffer（随 model.to(device) 移动）
		# -------------------------
		nyq = self.sr / 2.0
		def hz_to_mel(f):
			return 2595.0 * math.log10(1.0 + f / 700.0)
		def mel_to_hz(m):
			return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

		mel_low = hz_to_mel(0.0)
		mel_high = hz_to_mel(nyq)
		mel_points = np.linspace(mel_low, mel_high, self.K + 2)
		fc = mel_to_hz(mel_points)  # (K+2,)

		# mask matrix (K, F) 以及每 band 的频点计数
		F = self.nfft // 2 + 1
		band_mask_np = np.zeros((self.K, F), dtype=np.float32)
		for i in range(self.K):
			low = fc[i]
			high = fc[i + 2]
			inds = np.where((self.freq_bins >= low - 1e-12) & (self.freq_bins <= high + 1e-12))[0]
			if inds.size > 0:
				band_mask_np[i, inds] = 1.0

		# band 权重（使用 mel 网格中间频率 fc[1..K]）
		fc_mid = fc[1:self.K+1]  # (K,)
		spl_fc = np.interp(fc_mid, self._ref_freqs, self._spl_40)
		spl_fc = np.maximum(spl_fc, 1e-6)
		weights_np = (self._spl_1000 / spl_fc).astype(np.float32)  # (K,)
		# 对于没有频点的 band，权重设为 0
		counts = band_mask_np.sum(axis=1)
		weights_np[counts == 0] = 0.0

		# register buffers (float tensors) -> 会随 model.to(device) 自动转移
		self.register_buffer('band_mask', torch.from_numpy(band_mask_np))    # (K, F), float32
		self.register_buffer('band_weight', torch.from_numpy(weights_np))   # (K,)

	def forward(self, enhance, clean):
		"""
		enhance, clean: Tensor (B, T) or (T,)  (single-channel)
		返回: 标量损失
		"""
		# 规范化输入形状到 (B, T)
		if enhance.dim() == 1:
			enhance = enhance.unsqueeze(0)
		if clean.dim() == 1:
			clean = clean.unsqueeze(0)
		assert enhance.shape == clean.shape, "enhance and clean must have same shape"

		# 直接使用 module buffers（会在 model.to(device) 时移动）
		device = enhance.device if enhance.is_cuda else torch.device(self.device)

		# 3) STFT -> log-power (dB)
		def log_power(x):
			real = x[..., 0]
			imag = x[..., 1]
			power = real * real + imag * imag
			lp = 10.0 * torch.log10(power + self.eps)
			return lp  # (B, time, freq_bins)
		
		x = clean.unsqueeze(-1) # B, T, C
		spec_clean = self.stft.transform_cpx(x) # B, C, T, F, 2
		spec_clean = spec_clean.squeeze(1) # B, T, F, 2
		x = enhance.unsqueeze(-1) # B, T, C
		spec_enhance = self.stft.transform_cpx(x) # B, C, T, F, 2
		spec_enhance = spec_enhance.squeeze(1) # B, T, F, 2		
		lp_enh = log_power(spec_enhance)
		lp_clean = log_power(spec_clean)

		# -------------------------
		# 向量化并行计算每个子带的 MSE（避免按 band Python 循环）
		# -------------------------
		B, T, F = lp_enh.shape
		# diff^2 形状 (B, T, F)
		diff_sq = (lp_enh - lp_clean).pow(2)  # (B, T, F)
		# band_mask: (K, F) -> 扩展为 (1,1,K,F)
		mask = self.band_mask.view(1, 1, self.K, F)  # float
		# diff_sq unsqueeze -> (B, T, 1, F) ，相乘后 (B, T, K, F)
		sum_masked = torch.sum(diff_sq.unsqueeze(2) * mask, dim=(0,1,3))  # (K,)
		counts = torch.sum(self.band_mask, dim=1)  # (K,)
		denom = counts * (B * T)
		# 防止除以零
		zero_mask = denom == 0
		denom = denom.clone()
		denom[zero_mask] = 1.0
		mse_tensor = sum_masked / denom
		# 对于无频点的 band，设为 0
		if zero_mask.any():
			mse_tensor = mse_tensor.masked_fill(zero_mask, 0.0)

		# 6) 加权平均并归一化 (使用已注册的 band_weight)
		weights_t = self.band_weight.to(device)
		weight_sum = torch.sum(weights_t)
		if weight_sum.item() <= 0.0:
			loss_spec = torch.mean(mse_tensor)
		else:
			loss_spec = (torch.mean(weights_t * mse_tensor) / weight_sum
)

		return loss_spec

if __name__=='__main__':
    # pass
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('configs/gtcrn_cfg_train.yaml')
    # 将 OmegaConf 的 DictConfig/ListConfig 等转换为原生的 Python 容器（dict/list）
    # 这样 configs 中的 snd_db: [0, 15] 会成为 Python 列表 [0, 15]
    try:
        config = OmegaConf.to_container(config, resolve=True)
    except Exception:
        # 若转换失败则保持原始 config（兼容性），后续可手动转换字段
        pass
        
    a = torch.randn(2, 192000)
    b = torch.randn(2, 192000)

    loss_func = MelSubbandLoss(
		**config['loss']
	)
    loss = loss_func(b, a)
    print(loss)