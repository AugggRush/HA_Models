import numpy as np
import torch
import torch.nn as nn
import sys
#sys.path.append(r'/data/baron/git/denoise/train//model/third_party/witin_nn/ext_v2.4.5_anke/')
#from witin_nn import LayerConfigFactory, HandleNegInType, HardwareType, WitinConv2d, WitinMatMul


class BandConverter(nn.Module):
    def __init__(self, band_num=256, freq_bins=257, fs=16000, band_method="erb", impl_type="mat", quantize=False, norm=False, use_witin=False, layer_config=None):
        """
        impl_type: "loop" "mat" or "conv"
        """
        super(BandConverter, self).__init__()
        self.band_num = band_num
        self.freq_bins = freq_bins
        self.band_method = band_method
        self.quantize = quantize
        self.fs = fs
        self.impl_type = impl_type
        self.band_segment_idx = self.get_segment_index()
        self.use_witin = use_witin
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
            for b in range(self.band_num - 1):
                band_size = self.band_segment_idx[b+1] - self.band_segment_idx[b]
                for f in range(band_size):
                    frac = float(f) / band_size
                    inv_to_band_matrix[b, self.band_segment_idx[b]+f] = 1.0 - frac
                    inv_to_band_matrix[b+1, self.band_segment_idx[b]+f] = frac

            if norm:
                to_band_matrix = to_band_matrix / torch.sum(to_band_matrix, dim=0, keepdim=True)
                inv_to_band_matrix = inv_to_band_matrix / torch.sum(inv_to_band_matrix, dim=1, keepdim=True)
        self.to_band_matrix = to_band_matrix
        self.inv_to_band_matrix = inv_to_band_matrix
        if self.use_witin:
            from witin_nn import WitinMatMul
            self.forward_mat = WitinMatMul(layer_config=layer_config[0])
            self.inverse_mat = WitinMatMul(layer_config=layer_config[1])

    def forward(self, stft_mag, method="magnitude"):
        """
        :param stft_mag: [B C T F] or [B T F]
        :param method: "magnitude" or "energy"
        :return:
        """
        if not self.use_witin:
            band_mag = torch.matmul(stft_mag, self.to_band_matrix.to(stft_mag.device))
        else:
            band_mag = self.forward_mat(stft_mag, self.to_band_matrix.to(stft_mag.device))
        return band_mag

    def inverse(self, band_mag, method="magnitude"):
        """
        :param band_mag:
        :param method:
        :return:
        """
        if not self.use_witin:
            stft_mag = torch.matmul(band_mag, self.inv_to_band_matrix.to(band_mag.device))
        else:
            stft_mag = self.inverse_mat(band_mag, self.inv_to_band_matrix.to(band_mag.device))
        return stft_mag

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
            band_seg_idx[0] = np.array(0).astype(int)
            band_seg_idx[-1] = np.array(self.freq_bins-1).astype(int)

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

        for i in range(1, self.band_num): # remove repeat band
            if band_seg_idx[i] <= band_seg_idx[i - 1]:
                band_seg_idx[i] = band_seg_idx[i - 1] + 1

        band_seg_idx[0] = np.array(0).astype(int)
        band_seg_idx[-1] = np.array(self.freq_bins-1).astype(int)

        return band_seg_idx.astype(int)


if __name__ == '__main__':
    ipt = torch.rand([4, 100, 256])*2 - 1
    band_num = 192
    print(f"================================band trans================================")
    converter_loop = BandConverter(band_num=band_num, freq_bins=256, fs=16000, impl_type="loop")
    converter_mat = BandConverter(band_num=band_num, freq_bins=256, fs=16000, impl_type="mat", quantize=True)
    converter_conv = BandConverter(band_num=band_num, freq_bins=256, fs=16000, impl_type="conv", quantize=True)
    opt_loop = converter_loop.forward(ipt)
    opt_mat = converter_mat.forward(ipt)
    print(f"max in out {torch.max(torch.abs(ipt))} {torch.max(torch.abs(opt_mat))}")
    opt_conv = converter_conv.forward(ipt)

    print(f"loop vs mat abs error {torch.abs(opt_loop - opt_mat).mean()}")
    print(f"conv vs mat abs error {torch.abs(opt_conv - opt_mat).mean()}")
    print(f"conv vs mat relative error {torch.abs(opt_conv - opt_mat).mean() / torch.abs(opt_mat).mean()}") 

    print(f"===============================inv band trans================================")
    inv_opt_loop = converter_loop.inverse(opt_loop)
    inv_opt_mat = converter_mat.inverse(opt_mat)
    inv_opt_conv = converter_conv.inverse(opt_conv)

    print(f"inv loop vs mat abs error {torch.abs(inv_opt_loop - inv_opt_mat).mean()}")
    print(f"inv conv vs mat abs error {torch.abs(inv_opt_conv - inv_opt_mat).mean()}")
    print(f"inv conv vs mat relative error {torch.abs(inv_opt_conv - inv_opt_mat).mean() / torch.abs(inv_opt_mat).mean()}")
    print(f"================================")
