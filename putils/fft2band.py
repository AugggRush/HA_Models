import numpy as np
import torch
import torch.nn as nn


class BandConverter(nn.Module):
    def __init__(self, band_num=256, freq_bins=257, fs=16000, band_method="erb"):
        super(BandConverter, self).__init__()
        self.band_num = band_num
        self.freq_bins = freq_bins
        self.band_method = band_method
        self.fs = fs
        self.band_segment_idx = self.get_segment_index()
        if self.freq_bins == self.band_num:
            to_band_matrix = torch.eye(self.freq_bins, self.band_num, requires_grad=False)
            inv_to_band_matrix = torch.eye(self.freq_bins, self.band_num, requires_grad=False)
        else:
            to_band_matrix = torch.zeros(self.freq_bins, self.band_num, requires_grad=False)
            inv_to_band_matrix = torch.zeros(self.band_num, self.freq_bins, requires_grad=False)
            for b in range(self.band_num):
                if b < (self.band_num-1):
                    band_size = self.band_segment_idx[b+1] - self.band_segment_idx[b]
                    for f in range(band_size):
                        frac = 1.0 - (float(f) / band_size)
                        to_band_matrix[f + self.band_segment_idx[b], b] = frac  # right of central frequency
                if b > 0.5:
                    band_size = self.band_segment_idx[b] - self.band_segment_idx[b-1]
                    for f in range(band_size):
                        frac = float(f) / band_size
                        to_band_matrix[f + self.band_segment_idx[b - 1], b] = frac  # left of central frequency
            # to_band_matrix[:, 0] *= 2.0
            # to_band_matrix[:, -1] *= 2.0
            for b in range(self.band_num - 1):
                band_size = self.band_segment_idx[b+1] - self.band_segment_idx[b]
                for f in range(band_size):
                    frac = float(f) / band_size
                    inv_to_band_matrix[b, self.band_segment_idx[b]+f] = 1.0 - frac
                    inv_to_band_matrix[b+1, self.band_segment_idx[b]+f] = frac

        self.register_buffer('to_band_matrix', to_band_matrix.float())
        self.register_buffer('inv_to_band_matrix', inv_to_band_matrix.float())
        #self.to_band_matrix = nn.Parameter(torch.FloatTensor(to_band_matrix))
        #self.inv_to_band_matrix = nn.Parameter(torch.FloatTensor(inv_to_band_matrix))
        print("================ parameters in BandConverter =================")
        print(f"self.to_band_matrix {self.to_band_matrix}")
        print(f"self.inv_to_band_matrix {self.inv_to_band_matrix}")
        # np.savetxt("band_trans_mat.txt", self.to_band_matrix.permute(1, 0).detach().cpu().numpy(), fmt='%.1f')
        # np.savetxt("inv_band_trans_mat.txt", self.inv_to_band_matrix.detach().cpu().numpy(), fmt='%.1f')
        print(f"self.band_segment_idx {self.band_segment_idx}")
        print(f"self.band_num {self.band_num}")
        print(f"self.freq_bins {self.freq_bins}")
        print(f"self.fs {self.fs}")
        print(f"self.band_method {self.band_method}")
        print("=================================================================")

    def forward_band_mat(self, stft_mag, method="magnitude"):
        """
        Args:
            stft_mag: [B C T F] or [B T F]
            method: "magnitude" or "energy"
        """
        # np.savetxt("band_trans_mat.txt", self.to_band_matrix.permute(1, 0).detach().cpu().numpy(), fmt='%.1f')
        # np.savetxt("inv_band_trans_mat.txt", self.inv_to_band_matrix.detach().cpu().numpy(), fmt='%.1f')
        if method == "energy":
            stft_mag = stft_mag ** 2
        band_mag = torch.matmul(stft_mag, self.to_band_matrix)
        # print(f"band_mag {torch.max(self.to_band_matrix, dim=-1)} {torch.max(self.inv_to_band_matrix, dim=-1)}")
        return band_mag

    def inverse_band_mat(self, band_mag, method="magnitude"):
        """
        Args:
            band_mag:
            method:
        Returns:
            stft_mag
        """
        # stft_mag: [B C T F] or [B T F]
        # print(f"band_mag {band_mag.shape} {self.inv_to_band_matrix.shape}")
        stft_mag = torch.matmul(band_mag, self.inv_to_band_matrix)
        if method == "energy":
            stft_mag = stft_mag ** 0.5
        return stft_mag

    def forward_band(self, stft_mag, method="magnitude"):
        """
        Args:
            stft_mag: [B C T F] or [B T F]
            method: "magnitude" or "energy"
        """
        #  stft_mag: [B T F]
        with torch.no_grad():
            nframe = stft_mag.shape[1]
            nbatch = stft_mag.shape[0]
            band_mag = torch.zeros(nbatch, nframe, self.band_num, device=stft_mag.device)
            if method == "energy":
                stft_mag = stft_mag**2
            for i in range(self.band_num-1):
                band_size = self.band_segment_idx[i+1] - self.band_segment_idx[i]
                for j in range(band_size):
                    frac = float(j) / band_size
                    tmp = stft_mag[:, :, self.band_segment_idx[i]+j]
                    band_mag[:, :, i] += (1. - frac) * tmp
                    band_mag[:, :, i + 1] += frac * tmp
            #band_mag[:, :, 0] *= 2
            #band_mag[:, :, self.band_num - 1] *= 2
            return band_mag

    def inverse_band(self, band_mag, method="magnitude"):
        #  band_mag: [B T F]
        with torch.no_grad():
            nframe = band_mag.shape[1]
            nbatch = band_mag.shape[0]
            gain = torch.zeros(nbatch, nframe, self.freq_bins, device=band_mag.device)
            for i in range(self.band_num-1):
                band_size = self.band_segment_idx[i+1] - self.band_segment_idx[i]
                for j in range(band_size):
                    frac = float(j) / band_size
                    gain[:, :, self.band_segment_idx[i]+j] = (1 - frac) * band_mag[:, :, i] + frac * band_mag[:, :, i+1]
            if method == "energy":
                gain = gain ** 0.5
            return gain

    def get_segment_index(self):
        """
        This function computes an array of band_num frequencies uniformly spaced on ERB or Bark scale.
        For a definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983). "Suggested formulae for
        calculating auditory-filter bandwidths and excitation patterns," J. Acoust. Soc. Am. 74, 750-753
        """
        if self.band_method == "bark":
            assert (self.fs == 24000) or (self.fs == 16000), 'specific band num is nonsupport under sampling>24000'
            if self.fs == 16000:
                x = 0.9041 * (np.arange(self.band_num) / (self.band_num-1))
            elif self.fs == 24000:
                x = 0.9802 * (np.arange(self.band_num) / (self.band_num - 1))
            cf = [1.552e+05, -3.929e+05, 3.996e+05, -1.936e+05, 4.779e+04, -2855, 128.8]

            fx = x**6*cf[0]+x**5*cf[1] + x**4*cf[2] + x**3*cf[3] + x**2*cf[4] + x*cf[5] + cf[6]
            band_seg_idx = np.around(fx / self.fs * 2.0 * self.freq_bins).astype(int)

        elif self.band_method == "erb":
            # Change the following three parameters if you wish to use a different ERB scale.
            EarQ = 9.26449
            minBW = 24.7
            bin_space = self.fs / (self.freq_bins - 1) / 2
            highFreq = self.fs // 2
            # lowFreq = 31.25
            lowFreq = 0
            # All of the followFreqing expressions are derived in Apple TR #35, "An Efficient Implementation
            # of the Patterson-Holdsworth Cochlear Filter Bank."
            band_seg_idx = -(EarQ * minBW) + np.exp((np.arange(1, self.band_num + 1)) * (-np.log(highFreq + EarQ * minBW) + np.log(lowFreq + EarQ * minBW)) / self.band_num) * (highFreq + EarQ * minBW)
            band_seg_idx = np.round(np.flip(band_seg_idx, axis=0) / bin_space)
            # make each band at least one bin
        else:
            raise ValueError(f"invalid method {self.method}!")

        band_seg_idx[0] = np.array(0).astype(int)
        band_seg_idx[-1] = np.array(self.freq_bins-1).astype(int)

        for i in range(1, self.band_num): # remove repeat band
            if band_seg_idx[i] <= band_seg_idx[i - 1]:
                band_seg_idx[i] = band_seg_idx[i - 1] + 1

        return band_seg_idx.astype(int)


if __name__ == '__main__':
    ipt = torch.rand([4, 100, 256])
    print(f"================================band trans================================")
    converter = BandConverter(band_num=128, freq_bins=256, fs=16000)
    opt1 = converter.forward_band_mat(ipt)
    opt2 = converter.forward_band(ipt)
    print(f"opt1 {opt1}")
    print(f"opt2 {opt2}")
    print(f"trans abs error {torch.abs(opt1 - opt2).mean()}")

    print(f"===============================inv band trans================================")
    ipt = torch.randn(16, 10, 64, device='cuda:0')
    opt1 = converter.inverse_band_mat(opt1)
    opt2 = converter.inverse_band(opt2)
    print(f"inv trans abs error {torch.abs(opt1 - opt2).mean()}")
