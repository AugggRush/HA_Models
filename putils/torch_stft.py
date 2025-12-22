import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
import librosa.util as librosa_util
from librosa.util import pad_center, tiny
from einops import rearrange
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
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
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None,
                 window='hann'):
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
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        # self.pad_amount = int(self.filter_length / 2)
        self.pad_amount = filter_length - hop_length
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        assert(filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size = filter_length)
        fft_window = torch.from_numpy(fft_window).float()
        # window the bases
        forward_basis *= fft_window
        inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform_cpx(self, input_data):
        """Take input data (audio) to STFT domain.

        Arguments:
            input_data: with shape (B T C)
        Returns:
            out: with shape [B C T F 2]
        """
        channels = input_data.shape[-1]
        self.num_samples = input_data.shape[1]
        input_data = rearrange(input_data, 'b t c -> (b c) t').unsqueeze(1)

        input_data = F.pad(
            input_data.unsqueeze(1),
            (self.pad_amount, self.pad_amount, 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        #cutoff = int((self.filter_length / 2) + 1)
        out = rearrange(forward_transform, '(b c) (ri f) t-> b c t f ri', c=channels, ri=2)

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
        cpx_ipt = rearrange(cpx_ipt, 'b c t f ri -> (b c) t (ri f)')
        cpx_ipt = cpx_ipt.permute(0,2,1)

        inverse_transform = F.conv_transpose1d(cpx_ipt,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
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
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[..., self.pad_amount:]
        inverse_transform = inverse_transform[..., :self.num_samples]
        inverse_transform = inverse_transform.squeeze(1) # B T
        inverse_transform = rearrange(inverse_transform, '(b c) t -> b t c', c=channels)

        return inverse_transform


def test_stft():
    filter_length = 512
    hop_length = 100

    audio_np, sr = sf.read('../test_data/wavs/mono_16k.wav', dtype="float32")
    audio_torch = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(-1)
    print(f"shape of audio_torch: {audio_torch.shape}")

    stft = STFT(filter_length=filter_length, hop_length=hop_length, win_length=filter_length)
    audio_stft = stft.transform_cpx(audio_torch)
    print(f"audio_stft {audio_stft.shape}")
    audio_recovered = stft.inverse_cpx(audio_stft)
    audio_recovered_np = audio_recovered.cpu().numpy()
    error = audio_np - audio_recovered_np[0,:,0]
    print(f"recovered error: {np.abs(error).mean()}")
    print(f"mean energy {np.abs(audio_np).mean()} {np.abs(audio_recovered_np).mean()} {np.abs(audio_np).mean()/np.abs(audio_recovered_np).mean()}")
    print('stft shape:', audio_stft.shape)
    sf.write('../test_data/wavs/stft_out.wav', audio_recovered_np[0,:,0], sr)
    """
    plt.imshow(torch.log(audio_stft[0,0,:,:,0]**2+audio_stft[0,0,:,:,1]**2).transpose(0,1).cpu().numpy())
    plt.show()
    plt.savefig('../test_datas/figs/torch_stft.png', format='png', dpi=1000)
    """

if __name__ == "__main__":
    test_stft()

