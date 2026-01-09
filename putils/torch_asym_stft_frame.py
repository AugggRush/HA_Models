import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
from librosa.util import pad_center, tiny
from einops import rearrange
import soundfile as sf
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from torch.autograd import Function

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


class StftAsymFrame(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None,
                 window='hann', N1=None, N2=None, alpha=None, d=None, M=None,
                 use_precision='float32'):
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
        super(StftAsymFrame, self).__init__()
        self.filter_length = filter_length
        self.M = M
        self.hop_length = hop_length
        self.overlap_length = filter_length - hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length - self.hop_length)  # 非对称window
        self.overlap_amount = self.filter_length - self.hop_length
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        if use_precision == 'float32':
            forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
            inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

            assert (filter_length >= self.win_length)
            # STFT-Domain Neural Speech Enhancement with Very Low Algorithmic Latency
            d = d if d else 0
            forward_window = self.getAsqrtAnalysisWindow(filter_length, M, d)
            backward_window = self.getAsqrtSynthesisWindow(filter_length, self.M, d)
            forward_window = torch.from_numpy(forward_window).float()
            backward_window = torch.from_numpy(backward_window).float()
            forward_basis *= forward_window
            inverse_basis *= backward_window

            self.forward_basis = nn.Parameter(torch.FloatTensor(forward_basis), requires_grad=False)
            self.inverse_basis = nn.Parameter(torch.FloatTensor(inverse_basis), requires_grad=False)
        elif use_precision == 'float64':
            forward_basis = torch.DoubleTensor(fourier_basis[:, None, :])
            inverse_basis = torch.DoubleTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

            assert (filter_length >= self.win_length)
            # STFT-Domain Neural Speech Enhancement with Very Low Algorithmic Latency
            d = d if d else 0
            forward_window = self.getAsqrtAnalysisWindow(filter_length, M, d)
            backward_window = self.getAsqrtSynthesisWindow(filter_length, self.M, d)
            forward_window = torch.from_numpy(forward_window).double()
            backward_window = torch.from_numpy(backward_window).double()
            forward_basis *= forward_window
            inverse_basis *= backward_window

        # print(f"self.forward_basis {self.forward_basis.shape} {self.inverse_basis.shape}")
        
        self.trans_buffer = None
        self.inv_trans_buffer = None
        self.inv_frame_idx = 0
        self.frame_out_buffer = None
        self.forward_window = forward_window.unsqueeze(0).unsqueeze(1)
        self.backward_window = backward_window.unsqueeze(0).unsqueeze(-1)
        
        # print(f"self.forward_window {self.backward_window.shape} {self.backward_window.shape}")

        # if self.trans_buffer is None:
        #     self.trans_buffer = torch.zeros(self.batch_size*self.ch, 1, self.filter_length).to('cuda')
        #     self.forward_window = self.forward_window.to('cuda')
        # if self.frame_out_buffer is None:
        #     self.frame_out_buffer = torch.zeros(self.batch_size*self.ch, self.filter_length, 1)#.to('cuda')
        #     self.backward_window = self.backward_window.to('cuda')

        self.F = self.filter_length        # 全频谱维度
        # 构造逆DFT矩阵
        k = np.arange(self.F).reshape(-1, 1)
        n = np.arange(self.F).reshape(1, -1)
        kernel = np.exp(2j * np.pi * k * n / self.F)
        idft_matrix = (1 / self.F) * kernel.conj().T
        self.IDFT_matrix = torch.from_numpy(idft_matrix).to(torch.complex64)

        

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
            mode='constant', value=0.0).squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        out = rearrange(forward_transform, '(b c) (ri f) t-> b c t f ri', c=channels, ri=2)
        return out
    

    def transform_cpx_frame(self, input_data): #B T C
        channels = input_data.shape[-1]
        input_data = rearrange(input_data, 'b t c -> (b c) t').unsqueeze(1) # B 1 T
        if self.trans_buffer is None:
            self.trans_buffer = torch.zeros(input_data.shape[0], 1, self.filter_length).to(input_data.device)
            self.forward_window = self.forward_window.to(input_data.device)
        # self.trans_buffer = self.trans_buffer.to(input_data.device)
        # self.forward_window = self.forward_window.to(input_data.device)
        # self.trans_buffer[:, :, :self.overlap_amount] = self.trans_buffer[:, :, self.hop_length:].clone()
        # self.trans_buffer[:, :, self.overlap_amount:] = input_data
        new_trans_buffer = torch.zeros_like(self.trans_buffer).to(self.trans_buffer.device)
        new_trans_buffer[:, :, :self.overlap_amount] = self.trans_buffer[:, :, self.hop_length:].detach()
        new_trans_buffer[:, :, self.overlap_amount:] = input_data.detach()
        self.trans_buffer = new_trans_buffer
        use_fft = True
        # use_fft = False
        if not use_fft:
            forward_transform = F.conv1d(self.trans_buffer, self.forward_basis, stride=1, padding=0)
            out = rearrange(forward_transform, '(b c) (ri f) t-> b c t f ri', c=channels, ri=2)
        else:
            input_win = self.trans_buffer * self.forward_window
            forward_transform = torch.fft.fft(input_win, dim=-1)
            forward_transform = torch.stack([torch.real(forward_transform), torch.imag(forward_transform)], dim=-1) # B T F 2
            forward_transform = forward_transform[:, :, :(self.win_length // 2 + 1), :]
            out = rearrange(forward_transform, '(b c) t f ri -> b c t f ri', c=channels, ri=2)
        return out

    
    def reset_buffer(self, batch_channels):
        """根据当前批次的通道数和设备重置缓冲区"""
        self.frame_out_buffer = torch.zeros(
            batch_channels, self.filter_length, 1,
            device=self.inverse_basis.device,
            dtype=self.inverse_basis.dtype,
            requires_grad=False  # 缓冲区不需要梯度
        )

    def reset_stft(self, filter_length, hop_length, M, trans_buffer, frame_out_buffer, use_precision = 'float32', d=None):
        self.filter_length = filter_length
        self.M = M
        self.hop_length = hop_length
        self.overlap_length = filter_length - hop_length
        self.win_length = filter_length
        self.forward_transform = None
        self.pad_amount = int(self.filter_length - self.hop_length)  # 非对称window
        self.overlap_amount = self.filter_length - self.hop_length
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        if use_precision == 'float32':
            forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
            inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

            assert (filter_length >= self.win_length)
            # STFT-Domain Neural Speech Enhancement with Very Low Algorithmic Latency
            d = d if d else 0
            forward_window = self.getAsqrtAnalysisWindow(filter_length, M, d)
            backward_window = self.getAsqrtSynthesisWindow(filter_length, self.M, d)
            forward_window = torch.from_numpy(forward_window).float()
            backward_window = torch.from_numpy(backward_window).float()
            forward_basis *= forward_window
            inverse_basis *= backward_window

            self.forward_basis = nn.Parameter(torch.FloatTensor(forward_basis), requires_grad=False)
            self.inverse_basis = nn.Parameter(torch.FloatTensor(inverse_basis), requires_grad=False)
        elif use_precision == 'float64':
            forward_basis = torch.DoubleTensor(fourier_basis[:, None, :])
            inverse_basis = torch.DoubleTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

            assert (filter_length >= self.win_length)
            # STFT-Domain Neural Speech Enhancement with Very Low Algorithmic Latency
            d = d if d else 0
            forward_window = self.getAsqrtAnalysisWindow(filter_length, M, d)
            backward_window = self.getAsqrtSynthesisWindow(filter_length, self.M, d)
            forward_window = torch.from_numpy(forward_window).double()
            backward_window = torch.from_numpy(backward_window).double()
            forward_basis *= forward_window
            inverse_basis *= backward_window

        # print(f"self.forward_basis {self.forward_basis.shape} {self.inverse_basis.shape}")
        
        self.trans_buffer = trans_buffer
        self.frame_out_buffer = frame_out_buffer


    def inverse_cpx_frame(self, cpx_ipt):
        """Call the inverse STFT (iSTFT), given real  and imag tensors produced
        by the ```transform``` function.

        Arguments:
            cpx_ipt: complex out of STFT with shape (B C T F 2)
        Returns:
            inverse_transform: Reconstructed audio (B T C)
        """
        assert len(cpx_ipt.shape) == 5, f"input spectrum shape must be [B C T F 2]"
        cpx_ipt = torch.complex(cpx_ipt[:, :, :, :, 0], cpx_ipt[:, :, :, :, 1])
        channels = cpx_ipt.shape[1]
        cpx_ipt = rearrange(cpx_ipt, 'b c t f -> (b c) t f')
        cpx_ipt = cpx_ipt.permute(0, 2, 1) # B F T(1)
        
        if self.frame_out_buffer is None:
            self.reset_buffer(cpx_ipt.shape[0]*cpx_ipt.shape[1])
            self.backward_window = self.backward_window.to(cpx_ipt.device)

        self.frame_out_buffer = self.frame_out_buffer.to(cpx_ipt.device)
        self.backward_window = self.backward_window.to(cpx_ipt.device)
        prev_buffer = self.frame_out_buffer  # 当前缓冲区

        cpx_ipt_conj = torch.conj(torch.flip(cpx_ipt[:, 1:-1, :], dims=[1])) #+ 1e-8
        cpx_ipt = torch.cat([cpx_ipt, cpx_ipt_conj], dim=1)

        temp = torch.fft.ifft(cpx_ipt, dim=1).real  # iFFT
        temp = temp * self.backward_window #add win

        new_buffer = torch.zeros_like(prev_buffer)
        new_buffer[:, :self.overlap_length, :] = prev_buffer[:, self.hop_length:, :]
        new_buffer += temp  # 关键：梯度会流经 prev_buffer 和 current_frame
        
        # 更新缓冲区（不使用 detach()）
        self.frame_out_buffer = new_buffer  # 直接赋值（Python-level），计算图已记录
        inverse_transform = new_buffer[:, self.overlap_length-self.hop_length:self.overlap_length, :]     

        inverse_transform = inverse_transform.squeeze(-1)  # B 1 T
        inverse_transform = rearrange(inverse_transform, '(b c) t -> b t c', c=channels)
        return inverse_transform


    def inverse_cpx_frame_idft(self, cpx_ipt):
        """Call the inverse STFT (iSTFT), given real and imag tensors produced
        by the ```transform``` function. Optimized for parallel processing.

        Arguments:
            cpx_ipt: complex out of STFT with shape (B C T F 2)
        Returns:
            inverse_transform: Reconstructed audio (B T C)
        """
        assert len(cpx_ipt.shape) == 5, f"input spectrum shape must be [B C T F 2]"
        cpx_ipt = torch.complex(cpx_ipt[:, :, :, :, 0], cpx_ipt[:, :, :, :, 1])
        channels = cpx_ipt.shape[1]
        cpx_ipt = rearrange(cpx_ipt, 'b c t f -> (b c) t f')
        cpx_ipt = cpx_ipt.permute(0, 2, 1)  # B F T(1)

        if self.frame_out_buffer is None:
            self.frame_out_buffer = torch.zeros((cpx_ipt.shape[0], self.filter_length, 1), device=cpx_ipt.device)
            self.backward_window = self.backward_window.to(cpx_ipt.device)
            
            # 创建IDFT矩阵 - 只需要计算一次
            N = self.filter_length
            self.idft_matrix = self._create_idft_matrix(N, device=cpx_ipt.device)

        cpx_ipt_conj = torch.conj(torch.flip(cpx_ipt[:, 1:-1, :], dims=[1]))
        cpx_ipt = torch.cat([cpx_ipt, cpx_ipt_conj], dim=1)
        
        batch_size = cpx_ipt.shape[0]
        time_frames = cpx_ipt.shape[2]
        
        # 使用批处理矩阵乘法进行并行处理
        # 重塑频域数据以便批处理
        cpx_ipt_reshaped = cpx_ipt.permute(0, 2, 1)  # [B, T, F]
        cpx_ipt_flat = cpx_ipt_reshaped.reshape(-1, cpx_ipt_reshaped.shape[-1])  # [B*T, F]
        
        # 应用IDFT矩阵 - 一次性计算所有批次和时间帧
        time_domain_flat = torch.matmul(cpx_ipt_flat, self.idft_matrix.t())  # [B*T, N]
        
        # 重塑回原始维度
        temp = time_domain_flat.reshape(batch_size, time_frames, self.filter_length)  # [B, T, N]
        temp = temp.permute(0, 2, 1).real  # [B, N, T] 并提取实部
        
        # 应用窗口函数
        temp = temp * self.backward_window
        
        # 执行重叠相加
        new_frame_out_buffer = torch.zeros_like(self.frame_out_buffer)
        new_frame_out_buffer[:, :self.overlap_length, :] = self.frame_out_buffer[:, self.hop_length:, :].clone()
        new_frame_out_buffer = new_frame_out_buffer + temp
        
        # 提取当前帧
        inverse_transform = new_frame_out_buffer[:, self.overlap_length-self.hop_length:self.overlap_length, :]
        self.frame_out_buffer = new_frame_out_buffer.detach().clone()
        
        # 重塑输出
        inverse_transform = inverse_transform.squeeze(-1)
        inverse_transform = rearrange(inverse_transform, '(b c) t -> b t c', c=channels)
        
        return inverse_transform

    def _create_idft_matrix(self, N, device):
        """创建IDFT矩阵
        
        Arguments:
            N: 信号长度
            device: 计算设备
        
        Returns:
            idft_matrix: IDFT矩阵，形状为(N, N)
        """
        # 创建行和列索引
        n = torch.arange(N, device=device).view(-1, 1)  # 时域索引 [0,1,...,N-1]
        k = torch.arange(N, device=device).view(1, -1)  # 频域索引 [0,1,...,N-1]
        
        # 计算IDFT矩阵元素
        # IDFT: x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^{j*2π*k*n/N}
        angle = 2.0 * torch.pi * k * n / N
        cos_terms = torch.cos(angle) / N
        sin_terms = torch.sin(angle) / N
        
        # 创建复数IDFT矩阵
        return torch.complex(cos_terms, sin_terms)

    def reset(self):
        if self.trans_buffer is not None:
            self.trans_buffer = self.trans_buffer *  0.0
        if self.frame_out_buffer is not None:
            self.frame_out_buffer = self.frame_out_buffer * 0.0

  
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
        cpx_ipt = cpx_ipt.permute(0, 2, 1) # B F T

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
        inverse_transform = rearrange(inverse_transform, '(b c) t -> b t c', c=channels)

        return inverse_transform

    
    def inverse_cpx_frame_conv(self, cpx_ipt, prev_buffer):
        """以逐帧方式进行STFT逆变换
        
        Arguments:
            cpx_ipt: 单帧频谱数据，形状为 (B C 1 F 2)
        Returns:
            frame_output: 当前帧对应的时域信号片段，形状为 (B hop_size C)
        """
        assert len(cpx_ipt.shape) == 5, f"input spectrum shape must be [B C 1 F 2]"
        assert cpx_ipt.shape[2] == 1, f"Time dimension must be 1 for frame-by-frame processing"
        
        channels = cpx_ipt.shape[1]
        batch_size = cpx_ipt.shape[0]
        
        # 重组频谱数据
        cpx_ipt = rearrange(cpx_ipt, 'b c t f ri -> (b c) t (ri f)')
        cpx_ipt = cpx_ipt.permute(0, 2, 1)  # (B*C) F 1
        
        # 初始化帧缓冲区（如果需要）
        if not hasattr(self, 'frame_out_buffer') or self.frame_out_buffer is None:
            # 缓冲区长度需要考虑重叠 = 滤波器长度
            self.frame_out_buffer = torch.zeros((batch_size * channels, 1, self.filter_length), 
                                        device=cpx_ipt.device)
                    
        # 进行转置卷积（逆STFT的一部分）
        current_frame = F.conv_transpose1d(cpx_ipt,
                                        self.inverse_basis,
                                        stride=self.hop_length,
                                        padding=0)  # (B*C) 1 filter_length
        
        # 应用窗口函数和缩放
        if self.window is not None:
            if self.window == 'hann':
                window_sum = window_sumsquare(
                    self.window, 1, hop_length=self.hop_length,
                    win_length=self.win_length, n_fft=self.filter_length,
                    dtype=np.float32)
                # 移除调制效应
                approx_nonzero_indices = torch.from_numpy(
                    np.where(window_sum > tiny(window_sum))[0])
                window_sum = torch.from_numpy(window_sum).to(current_frame.device)
                current_frame[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            
            # 按跳跃比例缩放
            if self.M is None:
                current_frame *= (float(self.filter_length) / self.hop_length)
            else:
                current_frame *= (float(self.filter_length) / self.M)
        
        # new_frame_out_buffer = torch.zeros_like(self.frame_out_buffer, device=self.frame_out_buffer.device)
        # new_frame_out_buffer[:, :, :self.overlap_length] = self.frame_out_buffer[:, :, self.hop_length:].clone()
        # new_frame_out_buffer = new_frame_out_buffer + current_frame  # overlap add
        # frame_output = new_frame_out_buffer[:, :, :self.hop_length]
        # self.frame_out_buffer = new_frame_out_buffer.detach().clone()

        # # 移除填充并重组输出
        # frame_output = frame_output.squeeze(1)  # (B*C) hop_size
        # frame_output = rearrange(frame_output, '(b c) t -> b t c', b=batch_size, c=channels)

        # 重叠相加，不使用detach()
        new_frame_out_buffer = torch.zeros_like(prev_buffer)
        new_frame_out_buffer[:, :, :self.overlap_length] = prev_buffer[:, :, self.hop_length:]
        new_frame_out_buffer += current_frame  # Overlap-Add
        
        # 提取当前帧输出并更新缓冲区
        frame_output = new_frame_out_buffer[:, :, :self.hop_length]
        next_buffer = new_frame_out_buffer  # 不断开梯度
        
        # 整理输出形状
        frame_output = frame_output.squeeze(1)
        frame_output = rearrange(frame_output, '(b c) t -> b t c', b=batch_size, c=channels)
        
        return frame_output, next_buffer
    # def inverse_cpx_frame_conv(self, cpx_ipt):
    #     """修正后的逐帧逆变换函数"""
    #     assert len(cpx_ipt.shape) == 5, "输入频谱形状应为[B, C, 1, F, 2]"
    #     assert cpx_ipt.shape[2] == 1, "时间维度必须为1（逐帧处理）"
        
    #     # --- 前向传播处理 ---
    #     channels = cpx_ipt.shape[1]
    #     batch_size = cpx_ipt.shape[0]
        
    #     # 重组频谱数据 (B C 1 F 2) -> (B*C 1 F*2)
    #     cpx_ipt = rearrange(cpx_ipt, 'b c t f ri -> (b c) t (ri f)')
    #     cpx_ipt = cpx_ipt.permute(0, 2, 1)  # (B*C) F 1
        
    #     # 逆变换核心操作（保持与整句变换相同的数学操作）
    #     current_frame = F.conv_transpose1d(
    #         cpx_ipt,
    #         self.inverse_basis,
    #         stride=self.hop_length,
    #         padding=0
    #     )  # (B*C, 1, win_length)
        
    #     # 应用窗函数补偿
    #     if self.window is not None:
    #         if self.window == 'hann':
    #             window_sum = window_sumsquare(
    #                 self.window, 1, hop_length=self.hop_length,
    #                 win_length=self.win_length, n_fft=self.filter_length,
    #                 dtype=np.float32)
    #             # 移除调制效应
    #             approx_nonzero_indices = torch.from_numpy(
    #                 np.where(window_sum > tiny(window_sum))[0])
    #             window_sum = torch.from_numpy(window_sum).to(current_frame.device)
    #             current_frame[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            
    #         # 按跳跃比例缩放
    #         if self.M is None:
    #             current_frame *= (float(self.filter_length) / self.hop_length)
    #         else:
    #             current_frame *= (float(self.filter_length) / self.M)
        
    #     # --- 自定义重叠相加 ---
    #     if not hasattr(self, '_frame_buffer'):
    #         # 初始化缓冲区 (batch*channels, 1, win_length)
    #         self._frame_buffer = torch.zeros(
    #             (batch_size * channels, 1, self.win_length),
    #             dtype=current_frame.dtype,
    #             device=current_frame.device
    #         )
        
    #     # 执行自定义重叠相加操作
    #     combined_output = OverlapAddFunction.apply(
    #         self._frame_buffer,  # 历史缓冲区（自动断开计算图）
    #         current_frame,       # 当前帧（保留完整计算图）
    #         self.hop_length,
    #         self.win_length
    #     )
        
    #     # 更新缓冲区（保留最新win_length长度）
    #     self._frame_buffer = combined_output[..., -self.win_length:]
        
    #     # 输出当前hop_size对应的时域片段
    #     output_frame = combined_output[..., :self.hop_length]  # (B*C, 1, hop_size)
        
    #     # --- 后处理 ---
    #     # 调整形状恢复通道维度 (B*C, 1, hop_size) -> (B, hop_size, C)
    #     output_frame = rearrange(output_frame, '(b c) t s -> b (t s) c', c=channels)
        
    #     return output_frame  # (B, hop_size, C)
    

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
        risingNoramlizedHann = np.hanning(2 * M + 1)[:M] / risingSqrtHannAnalysis[N - 2 * M - d:N - M - d]
        fallingSqrtHann = np.sqrt(np.hanning(2 * M + 1)[:2 * M])

        window = np.zeros(N)
        window[:-2 * M] = 0
        window[-2 * M:-M] = risingNoramlizedHann
        window[-M:] = fallingSqrtHann[-M:]

        return window


class OverlapAddFunction(Function):
    @staticmethod
    def forward(ctx, buffer_history, current_frame, hop_size, win_length):
        # 显式复制张量以延长生命周期
        ctx.hop_size = hop_size
        ctx.win_length = win_length
        
        # 分离历史缓冲区的计算图但保留数据
        if buffer_history is not None:
            buffer_history = buffer_history.detach().clone()
        ctx.save_for_backward(buffer_history, current_frame.clone())
        
        # 重叠相加逻辑保持不变
        if buffer_history is None:
            output = current_frame
        else:
            output = buffer_history.clone()
            output[..., hop_size:] += current_frame[..., :-hop_size]
            output = torch.cat([output, current_frame[..., -hop_size:]], dim=-1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 获取已持久化的张量副本
        buffer_history, current_frame = ctx.saved_tensors
        buffer_grad = current_grad = None
        
        # 计算梯度时重新构建必要计算图
        if ctx.needs_input_grad:  # buffer_history梯度
            buffer_grad = grad_output[..., :ctx.win_length].clone()
            buffer_grad = buffer_grad.detach()  # 阻断进一步反向传播
        
        if ctx.needs_input_grad:  # current_frame梯度
            current_grad = torch.zeros_like(current_frame)
            current_grad[..., :-ctx.hop_size] = grad_output[..., ctx.hop_size:ctx.win_length]
            current_grad[..., -ctx.hop_size:] = grad_output[..., -ctx.hop_size:]
        
        return buffer_grad, current_grad, None, None



def Test_asym_stft():
    x = torch.randn(1, 16000, 1)
    x = sf.read("../stream/test_wavs/mix.wav", always_2d= True, dtype="float32")[0][:32000, :]
    x = torch.from_numpy(x).unsqueeze(0)
    NFFT = 512
    hop_length = 32
    win_length = 512
    stft = StftAsymFrame(filter_length=NFFT, hop_length=hop_length, win_length=win_length, window='asqrthann', M=32)
    stft.reset_buffer(1)    
    chunck_num = x.shape[1] // hop_length
    x_recover = []
    for i in range(chunck_num):
        x_seg = x[:, i * hop_length:(i + 1) * hop_length, :]
        x_stft = stft.transform_cpx_frame(x_seg)
        o = stft.inverse_cpx_frame(x_stft)
        x_recover.append(o.clone())
    x_recover = torch.cat(x_recover, dim=1)
    x_recover = x_recover[:, hop_length:, :]
    x = x[:, :x_recover.shape[1], :]
    print(torch.mean(torch.abs((x - x_recover))), torch.max(torch.abs(x)), torch.max(torch.abs(x_recover)), torch.max(torch.abs(x_recover))/torch.max(torch.abs(x)))
    print(x_recover.shape)
    sf.write('../stream/test_wavs/stft_out1.wav', x_recover[0, :, 0].detach().cpu().numpy(), 16000)
    sf.write('../stream/test_wavs/in2.wav', x[0, :, 0].detach().cpu().numpy(), 16000)


if __name__ == "__main__":
    # test_stft()
    Test_asym_stft()
