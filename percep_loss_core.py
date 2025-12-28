import torch
import math
import numpy as np

from scipy.signal import butter
from torch.nn.functional import unfold
from torch.nn import Parameter
from torchaudio.functional import lfilter
from torchaudio.transforms import Spectrogram

from typeguard import typechecked
from scipy import interpolate
import profile


#############################################################################################
# fmt: off
nr_of_hz_bands_per_bark_band_16k = [
    1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2,
    1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4,
    3, 4, 5, 4, 5, 6, 6, 7, 8, 9, 9, 12,12,15,16,
    18,21,25,20]

centre_of_band_bark_16k = [
    0.078672,     0.316341,     0.636559,     0.961246,     1.290450,
    1.624217,     1.962597,     2.305636,     2.653383,     3.005889,
    3.363201,     3.725371,     4.092449,     4.464486,     4.841533,
    5.223642,     5.610866,     6.003256,     6.400869,     6.803755,
    7.211971,     7.625571,     8.044611,     8.469146,     8.899232,
    9.334927,     9.776288,     10.223374,     10.676242,     11.134952,
    11.599563,     12.070135,     12.546731,     13.029408,     13.518232,
    14.013264,     14.514566,     15.022202,     15.536238,     16.056736,
    16.583761,     17.117382,     17.657663,     18.204674,     18.758478,
    19.319147,     19.886751,     20.461355,     21.043034]

centre_of_band_hz_16k = [
    7.867213,     31.634144,     63.655895,     96.124611,     129.044968,
    162.421738,     196.259659,     230.563568,     265.338348,     300.588867,
    336.320129,     372.537140,     409.244934,     446.448578,     484.568604,
    526.600586,     570.303833,     619.423340,     672.121643,     728.525696,
    785.675964,     846.835693,     909.691650,     977.063293,     1049.861694,
    1129.635986,     1217.257568,     1312.109497,     1412.501465,     1517.999390,
    1628.894165,     1746.194336,     1871.568848,     2008.776123,     2158.979248,
    2326.743164,     2513.787109,     2722.488770,     2952.586670,     3205.835449,
    3492.679932,     3820.219238,     4193.938477,     4619.846191,     5100.437012,
    5636.199219,     6234.313477,     6946.734863,     7796.473633]

width_of_band_bark_16k = [
    0.157344,     0.317994,     0.322441,     0.326934,     0.331474,
    0.336061,     0.340697,     0.345381,     0.350114,     0.354897,
    0.359729,     0.364611,     0.369544,     0.374529,     0.379565,
    0.384653,     0.389794,     0.394989,     0.400236,     0.405538,
    0.410894,     0.416306,     0.421773,     0.427297,     0.432877,
    0.438514,     0.444209,     0.449962,     0.455774,     0.461645,
    0.467577,     0.473569,     0.479621,     0.485736,     0.491912,
    0.498151,     0.504454,     0.510819,     0.517250,     0.523745,
    0.530308,     0.536934,     0.543629,     0.550390,     0.557220,
    0.564119,     0.571085,     0.578125,     0.585232]

width_of_band_hz_16k = [
    15.734426,     31.799433,     32.244064,     32.693359,     33.147385,
    33.606140,     34.069702,     34.538116,     35.011429,     35.489655,
    35.972870,     36.461121,     36.954407,     37.452911,     40.269653,
    42.311859,     45.992554,     51.348511,     55.040527,     56.775208,
    58.699402,     62.445862,     64.820923,     69.195374,     76.745667,
    84.016235,     90.825684,     97.931152,     103.348877,     107.801880,
    113.552246,     121.490601,     130.420410,     143.431763,     158.486816,
    176.872803,     198.314697,     219.549561,     240.600098,     268.702393,
    306.060059,     349.937012,     398.686279,     454.713867,     506.841797,
    564.863770,     637.261230,     794.717285,     931.068359]

pow_dens_correction_factor_16k = [
    100.000000,     99.999992,     100.000000,     100.000008,     100.000008,
    100.000015,     99.999992,     99.999969,     50.000027,     100.000000,
    99.999969,     100.000015,     99.999947,     100.000061,     53.047077,
    110.000046,     117.991989,     65.000000,     68.760147,     69.999931,
    71.428818,     75.000038,     76.843384,     80.968781,     88.646126,
    63.864388,     68.155350,     72.547775,     75.584831,     58.379192,
    80.950836,     64.135651,     54.384785,     73.821884,     64.437073,
    59.176456,     65.521278,     61.399822,     58.144047,     57.004543,
    64.126297,     54.311001,     61.114979,     55.077751,     56.849335,
    55.628868,     53.137054,     54.985844,     79.546974]
# fmt: on
Sp_16k = 6.910853e-006


def interp(values: list, nelms_new: int):

    nelms = len(values)
    interp = interpolate.interp1d(np.arange(nelms), values)
    return torch.tensor(interp(np.linspace(0, 49.0, nelms_new, endpoint=False)))


class BarkScale(torch.nn.Module):

    def __init__(self, nfreqs: int = 256, nbarks: int = 49, sample_rate: int = 16000):
        super(BarkScale, self).__init__()

        # scale factor relative to the 16kHz base constants
        hz_scale = float(sample_rate) / 16000.0

        # pow_dens_correction uses same base factors, keep Sp scaling as before
        self.pow_dens_correction = Parameter(
            interp(pow_dens_correction_factor_16k, nbarks) * Sp_16k, requires_grad=False
        )
        # width/centre in Hz need to be scaled according to sample_rate
        width_hz_scaled = [w * hz_scale for w in width_of_band_hz_16k]
        centre_hz_scaled = [c * hz_scale for c in centre_of_band_hz_16k]
        self.width_hz = Parameter(interp(width_hz_scaled, nbarks), requires_grad=False)
        self.width_bark = Parameter(interp(width_of_band_bark_16k, nbarks), requires_grad=False)
        self.centre = Parameter(interp(centre_hz_scaled, nbarks), requires_grad=False)

        fbank = torch.zeros(nbarks, nfreqs)

        if nfreqs == 256 and nbarks == 49:
            current = 0
            for i in range(nbarks):
                end = current + nr_of_hz_bands_per_bark_band_16k[i]

                fbank[i, current:end] = 1.0
                current = end
        else:
            prev, bin_width = 0, 8000.0 / nfreqs
            for i in range(nbarks):
                stride = self.width_hz[i] / bin_width
                centre = self.centre[i] / bin_width
                start, end = max(prev, int(math.floor(centre - stride / 2))), min(
                    nfreqs, int(math.ceil(centre + stride / 2))
                )
                fbank[i, start:end] = 1.0
                prev = end

        self.fbank = Parameter(fbank, requires_grad=False)
        self.total_width = self.width_bark[1:].sum()

    @typechecked
    def weighted_norm(
        self, tensor, p: float = 2
    ):
        return self.total_width * (
            self.width_bark * tensor / self.total_width ** (1 / p)
        )[:, :, 1:].norm(p, dim=2)

    @typechecked
    def forward(
        self, tensor
    ):

        bark_powspec = torch.einsum("ij,klj->kli", self.fbank, tensor[:, :, :-1])
        return bark_powspec * self.pow_dens_correction


#############################################################################################
# fmt: off
abs_thresh_power_16k = [
    51286152.000000,     2454709.500000,     70794.593750,     4897.788574,     1174.897705,
    389.045166,     104.712860,     45.708820,     17.782795,     9.772372,
    4.897789,     3.090296,     1.905461,     1.258925,     0.977237,
    0.724436,     0.562341,     0.457088,     0.389045,     0.331131,
    0.295121,     0.269153,     0.257040,     0.251189,     0.251189,
    0.251189,     0.251189,     0.263027,     0.288403,     0.309030,
    0.338844,     0.371535,     0.398107,     0.436516,     0.467735,
    0.489779,     0.501187,     0.501187,     0.512861,     0.524807,
    0.524807,     0.524807,     0.512861,     0.478630,     0.426580,
    0.371535,     0.363078,     0.416869,     0.537032]
# fmt: on

zwicker_power = 0.23
Sl_16k = 1.866055e-001

class Loudness(torch.nn.Module):

    def __init__(self, nbark: int = 49, sample_rate: int = 16000):
        super(Loudness, self).__init__()

        # For now use the same absolute thresholds (interpolated) as base 16k values.
        # If desired, a more precise calibration for other sample rates can be added here.
        self.threshs = Parameter(
            interp(abs_thresh_power_16k, nbark).unsqueeze(0).unsqueeze(0),
            requires_grad=False,
        )

        exp = 6 / (torch.tensor(centre_of_band_bark_16k) + 2.0)
        self.exp = Parameter(
            exp.clamp(min=1.0, max=2.0) ** 0.15 * zwicker_power, requires_grad=False
        )

    @typechecked
    def total_audible(
        self, tensor, factor: float = 1.0
    ):

        mask = tensor > self.threshs * factor

        tmp = (tensor * mask).sum(dim=2)
        return tmp

    @typechecked
    def forward(
        self, pow_dens
    ):

        loudness = (2.0 * self.threshs) ** self.exp * (
            (0.5 + 0.5 * pow_dens / self.threshs) ** self.exp - 1
        )
        loudness[pow_dens <= self.threshs] = 0.0

        return loudness * Sl_16k

#############################################################################################

class PercepLoss(torch.nn.Module):
    """PercepLoss"""

    def __init__(
        self,
        nbarks: int = 49, #
        win_length: int = 512,
        n_fft: int = 512,
        hop_length: int = 256,
        sample_rate: int = 16000,
    ):
        super(PercepLoss, self).__init__()

        self.to_spec = Spectrogram(
            win_length=win_length,
            n_fft=n_fft,
            hop_length=hop_length,
            window_fn=torch.hann_window,
            power=2,
            normalized=False,
            center=False,
        )

        # construct Bark/Loudness with sample_rate awareness
        self.fbank = BarkScale(n_fft // 2, nbarks, sample_rate=sample_rate)
        self.loudness = Loudness(nbarks, sample_rate=sample_rate)

        # redesign the butter bandpass using the provided sample_rate
        out = np.asarray(butter(5, [325, 3250], fs=sample_rate, btype="band"))
        self.power_filter = Parameter(
            torch.as_tensor(out, dtype=torch.float32), requires_grad=False
        )
        self.pre_filter = Parameter(
            torch.tensor([[2.740826, -5.4816519, 2.740826], [1.0, -1.9444777, 0.94597794]],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def align_level(
        self, signal
    ):
        #filtered_signal = lfilter(
        #    signal, self.power_filter[1], self.power_filter[0], clamp=False
        #)
        filtered_signal = signal
        power = (
            (filtered_signal**2).sum(dim=1, keepdim=True)
            / (filtered_signal.shape[1] + 5120)
            / 1.04684
        )
        signal = signal * (10**7 / power).sqrt()

        return signal

    def preemphasize(
        self, signal
    ):

        emp = torch.linspace(0, 15, 16, device=signal.device)[1:] / 16.0
        signal[:, :15] *= emp
        signal[:, -15:] *= torch.flip(emp, dims=(0,))

        signal = lfilter(signal, self.pre_filter[1], self.pre_filter[0], clamp=False)

        return signal

    def raw(
        self, ref, deg
    ):
        """Calculate symmetric and asymmetric distances"""
        deg, ref = torch.atleast_2d(deg), torch.atleast_2d(ref)

        # equalize to [-1, 1] range
        max_val = torch.max(
            torch.amax(deg.abs(), dim=1, keepdim=True),
            torch.amax(ref.abs(), dim=1, keepdim=True),
        )
        deg, ref = deg / max_val, ref / max_val

        ref, deg = self.align_level(ref), self.align_level(deg)
        #ref, deg = self.preemphasize(ref), self.preemphasize(deg)

        deg, ref = self.to_spec(deg).swapaxes(1, 2), self.to_spec(ref).swapaxes(1, 2)

        deg[:, :, 0] = 0.0
        ref[:, :, 0] = 0.0

        deg, ref = self.fbank(deg), self.fbank(ref)

        equ_ref = ref
        equ_deg = deg

        deg_loud, ref_loud = self.loudness(equ_deg), self.loudness(equ_ref)

        deadzone = 0.25 * torch.min(deg_loud, ref_loud)
        disturbance = deg_loud - ref_loud
        disturbance = disturbance.sign() * (disturbance.abs() - deadzone).clamp(min=0)

        symm_distu = self.fbank.weighted_norm(disturbance, p=2)
        symm_distu = symm_distu.clamp(min=1e-20)

        asymm_scaling = ((equ_deg + 50.0) / (equ_ref + 50.0)) ** 1.2
        asymm_scaling[asymm_scaling < 3.0] = 0.0
        asymm_scaling = asymm_scaling.clamp(max=12.0)

        asymm_distu = self.fbank.weighted_norm(disturbance * asymm_scaling, p=1)
        asymm_distu = asymm_distu.clamp(min=1e-20)

        h = ((self.loudness.total_audible(equ_ref, 1) + 1e5) / 1e7) ** 0.04
        symm_distu, asymm_distu = (symm_distu / h).clamp(max=45.0), (asymm_distu / h).clamp(max=45.0)

        d_symm = symm_distu.square().mean(dim=1).sqrt()
        d_asymm = asymm_distu.square().mean(dim=1).sqrt()
        return d_symm, d_asymm

    def mos(
        self, ref, deg
    ):
        """
        """

        d_symm, d_asymm = self.raw(ref, deg)

        mos = 4.5 - 0.1 * d_symm - 0.0309 * d_asymm

        mos = 0.999 + 4 / (1 + torch.exp(-1.3669 * mos + 3.8224))

        return mos


if __name__ == '__main__':
    ipt = torch.rand(32, 24000, dtype=torch.float32).to("cuda:0")
    tgt = ipt - 1
    loss_fun = PercepLoss().to("cuda:0")
    # p = LineProfiler()
    # p.add_function(loss_fun.raw)
    # lp_wrapper = p(loss_fun.raw)
    # for i in range(20):
    #     loss = lp_wrapper(ipt, tgt)

    # p.print_stats()
