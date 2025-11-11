import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
from librosa.util import pad_center, tiny
from einops import rearrange
import soundfile as sf
import matplotlib.pyplot as plt


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, size=n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


class STFT_asym(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None,
                 window='hann', N1=None, N2=None, alpha=None, d=None, M=None):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris)
                (default: {'hann'})

            N1 {int} -- orka window argument, normally same as hop_length
            N2 {int} -- orka window argument, N2 = filter_length - hop_length
            alpha {float} -- tukey window argument, 0.0625 in Wang's paper (1/16 for 16ms window length)
            d {int} -- asqrthann window argument, normally same as hop_length or set to 0
        """
        super(STFT_asym, self).__init__()
        self.filter_length = filter_length
        self.M = M
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        # self.pad_amount = int(self.filter_length / 2)  # 正常stft
        self.pad_amount = int(self.filter_length - self.hop_length)  # 非对称window
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        assert (filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        if window == 'hann':
            fft_window = get_window(window, self.win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window
            inverse_basis *= fft_window
        elif window == 'orka':
            # CEC2 E008 Technical Paper
            N1 = N1 if N1 else hop_length
            N2 = N2 if N2 else filter_length - hop_length
            forward_window = self.getOrkaAnalysisWindow(filter_length, N1, N2, hop_length)
            backward_window = self.getOrkaSynthesisWindow(filter_length, N1, N2, hop_length)
            forward_window = torch.from_numpy(forward_window).float()
            backward_window = torch.from_numpy(backward_window).float()
            forward_basis *= forward_window
            inverse_basis *= backward_window
        elif window == 'tukey':
            # STFT-Domain Neural Speech Enhancement with Very Low Algorithmic Latency
            alpha = alpha if alpha else 0.0625
            forward_window = self.getTukeyAnalysisWindow(filter_length, alpha)
            backward_window = self.getTukeySynthesisWindow(filter_length, hop_length * 2, hop_length, alpha)
            forward_window = torch.from_numpy(forward_window).float()
            backward_window = torch.from_numpy(backward_window).float()
            forward_basis *= forward_window
            inverse_basis *= backward_window
        # 最优，16ms窗长
        elif window == 'asqrthann':
            # STFT-Domain Neural Speech Enhancement with Very Low Algorithmic Latency
            d = d if d else 0
            if self.M is None:
                forward_window = self.getAsqrtAnalysisWindow(filter_length, hop_length, d)
                backward_window = self.getAsqrtSynthesisWindow(filter_length, hop_length, d)
            else:
                forward_window = self.getAsqrtAnalysisWindow(filter_length, M, d)
                backward_window = self.getAsqrtSynthesisWindow(filter_length, self.M, d)
            forward_window = torch.from_numpy(forward_window).float()
            backward_window = torch.from_numpy(backward_window).float()
            forward_basis *= forward_window
            inverse_basis *= backward_window

        self.forward_basis = nn.Parameter(torch.FloatTensor(forward_basis), requires_grad=False)
        self.inverse_basis = nn.Parameter(torch.FloatTensor(inverse_basis), requires_grad=False)

    def transform_cpx(self, input_data):
        """Take input data (audio) to STFT domain.

        Arguments:
            input_data: with shape (B T C)
        Returns:
            out: with shape [B C T F 2]
        """
        channels = input_data.shape[-1]
        self.num_samples = input_data.shape[1]
        input_data = rearrange(input_data, 'b t c -> (b c) t').unsqueeze(1).contiguous()

        input_data = F.pad(
            input_data.unsqueeze(1),
            (self.pad_amount, self.pad_amount, 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1).clone()

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        # cutoff = int((self.filter_length / 2) + 1)
        out = rearrange(forward_transform, '(b c) (ri f) t-> b c t f ri', c=channels, ri=2).contiguous()

        return out

    def inverse_cpx(self, cpx_ipt):
        """Call the inverse STFT (iSTFT), given real  and imag tensors produced
        by the ```transform``` function.

        Arguments:
            cpx_ipt: complex out of STFT with shape (B C T F 2)
        Returns:
            inverse_transform: Reconstructed audio (B T C)
        """
        assert len(cpx_ipt.shape) == 5, f"input spectrum shape must be [B C T F 2]"
        channels = cpx_ipt.shape[1]
        cpx_ipt = rearrange(cpx_ipt, 'b c t f ri -> (b c) t (ri f)').contiguous()
        cpx_ipt = cpx_ipt.clone().permute(0, 2, 1).contiguous()

        inverse_transform = F.conv_transpose1d(cpx_ipt,
                                               self.inverse_basis,
                                               stride=self.hop_length,
                                               padding=0)

        if self.window is not None:
            if self.window == 'hann':
                window_sum = window_sumsquare(
                    self.window, cpx_ipt.shape[-1], hop_length=self.hop_length,
                    win_length=self.win_length, n_fft=self.filter_length,
                    dtype=np.float32)
                # remove modulation effects
                approx_nonzero_indices = torch.from_numpy(
                    np.where(window_sum > tiny(window_sum))[0])
                window_sum = torch.from_numpy(window_sum).to(inverse_transform.device)
                inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            if self.M is None:
                inverse_transform *= (float(self.filter_length) / self.hop_length)
            else:
                inverse_transform *= (float(self.filter_length) / self.M)

        inverse_transform = inverse_transform[..., self.pad_amount:]
        inverse_transform = inverse_transform[..., :self.num_samples]
        inverse_transform = inverse_transform.squeeze(1)  # B T
        inverse_transform = rearrange(inverse_transform, '(b c) t -> b t c', c=channels).contiguous()

        return inverse_transform

    def getOrkaAnalysisWindow(self, filter_length, N1, N2, hop_length):
        analysisWindow = np.zeros(filter_length)
        for i in range(filter_length):
            analysisWindow[i] = self.forward_win(i, N1, N2, hop_length)
        return analysisWindow

    def getOrkaSynthesisWindow(self, filter_length, N1, N2, hop_length):
        synthesisWindow = np.zeros(filter_length)
        for i in range(filter_length):
            synthesisWindow[i] = self.backward_win(i, N1, N2, hop_length)
        return synthesisWindow

    def forward_win(self, N, N1, N2, hop_size):
        assert N >= 0
        if N < N1:
            return (np.sin(N * np.pi / (2 * N1))) ** 2
        elif N <= N2:
            return 1
        elif N <= N2 + hop_size and N > N2:
            return (np.sin(np.pi * (N2 + hop_size - N) / (2 * hop_size)))

    def backward_win(self, N, N1, N2, hop_size):
        assert N >= 0
        if N < N2 - hop_size:
            return 0
        elif N <= N2:
            return (np.cos((np.pi * (N - N2)) / (2 * hop_size))) ** 2
        elif N <= N2 + hop_size:
            return (np.sin(np.pi * (N2 + hop_size - N) / (2 * hop_size)))

    def getTukeyAnalysisWindow(self, filter_length, alpha):
        analysisWindow = np.zeros(filter_length)
        for i in range(filter_length):
            analysisWindow[i] = self.TukeyAW(i, filter_length, alpha)
        return analysisWindow

    def TukeyAW(self, n, N, alpha):
        # assert n >= 0
        if n < alpha * N:
            return 0.5 * (1 - np.cos(np.pi * n / (alpha * N)))
        elif n <= N - alpha * N:
            return 1
        elif n <= N:
            return 0.5 * (1 - np.cos(np.pi * (N - n) / (alpha * N)))

    def getTukeySynthesisWindow(self, N, A, B, alpha):
        synthesisWindow = np.zeros(A)
        for i in range(A):
            x = N - A + i
            numerator = self.TukeyAW(x, N, alpha)
            denonminator = 0
            for k in range(int(A / B)):
                y = N - A + i % B + k * B
                denonminator += self.TukeyAW(y, N, alpha) ** 2
            synthesisWindow[i] = numerator / denonminator

        synthesisWindow = np.pad(synthesisWindow, (N - A, 0), 'constant', constant_values=0)
        return synthesisWindow

    def getAsqrtAnalysisWindow(self, N, M, d):
        risingSqrtHann = np.sqrt(np.hanning(2 * (N - M - d) + 1)[:(N - M - d)])     # 上升
        fallingSqrtHann = np.sqrt(np.hanning(2 * M + 1)[:2 * M])                    # 下降

        window = np.zeros(N)
        window[:d] = 0
        window[d:N - M] = risingSqrtHann[:N - M - d]
        window[N - M:] = fallingSqrtHann[-M:]

        return window

    def getAsqrtSynthesisWindow(self, N, M, d):
        risingSqrtHannAnalysis = np.sqrt(np.hanning(2 * (N - M - d) + 1)[:(N - M - d)])
        risingNoramlizedHann = np.hanning(2 * M + 1)[:M]
        risingNoramlizedHann[1:M] = risingNoramlizedHann[1:M] / risingSqrtHannAnalysis[N - 2 * M - d + 1:N - M - d]
        fallingSqrtHann = np.sqrt(np.hanning(2 * M + 1)[:2 * M])

        window = np.zeros(N)
        window[:-2 * M] = 0
        window[-2 * M:-M] = risingNoramlizedHann
        window[-M:] = fallingSqrtHann[-M:]

        return window


def Test_stft():
    audio_np, sr = sf.read('/data/goodman/data/ha_fix_test/wav/sample_0048_snr_0.wav', dtype="float32")
    audio_torch = torch.from_numpy(audio_np).unsqueeze(0).to("cuda:0").detach()
    audio_torch = torch.stack([audio_torch, audio_torch, audio_torch], dim=-1)
    print(f"shape of audio_torch {audio_torch.shape}")

    stft = STFT_asym(filter_length=320, hop_length=160, win_length=320).to('cuda:0')
    audio_stft = stft.transform_cpx(audio_torch)
    print(f"audio_stft {audio_stft.shape}")
    audio_recovered = stft.inverse_cpx(audio_stft)
    audio_recovered_np = audio_recovered.detach().cpu().numpy()
    print(
        f"mean energy {np.abs(audio_np).mean()} {np.abs(audio_recovered_np).mean()} {np.abs(audio_np).mean() / np.abs(audio_recovered_np).mean()}")
    print('stft shape:', audio_stft.shape)
    # sf.write('../test_datas/wavs/stft_out.wav', audio_recovered_np[0, :, 0], sr)

    plt.imshow(torch.log(audio_stft[0, 0, :, :, 0] ** 2 + audio_stft[0, 0, :, :, 1] ** 2).transpose(0,
                                                                                                    1).detach().cpu().numpy())
    plt.show()
    # plt.savefig('../test_datas/figs/torch_stft.png', format='png', dpi=1000)


def Test_asym_stft():
    x = torch.randn(1, 16000, 1)
    NFFT = 512
    hop_length = 32
    win_length = 512
    stft = STFT_asym(filter_length=NFFT, hop_length=hop_length, win_length=win_length, window='asqrthann', M=64)
    x_stft = stft.transform_cpx(x)  # [1, 1, 503, 65, 2]
    x_recover = stft.inverse_cpx(x_stft)
    print(x_recover.shape)
    # 查看误差
    print(torch.mean((x - x_recover) ** 2))
    # orka 5.8949e-14
    # hann 3.3134e-14

    torch_stft = torch.stft(x.squeeze(-1), n_fft=NFFT, hop_length=hop_length, win_length=win_length,
                            window=torch.hann_window(win_length, device='cpu'),return_complex=True,
                            center=True)
    torch_recover = torch.istft(torch_stft, n_fft=NFFT, hop_length=hop_length,
                                win_length=win_length,
                                window=torch.hann_window(win_length, device='cpu'),
                                center=True)
    print(torch.mean((x - torch_recover.unsqueeze(-1)) ** 2))   # 9.9718e-15


if __name__ == "__main__":
    Test_stft()
    Test_asym_stft()
