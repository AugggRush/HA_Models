import os
import torch
import h5py
import lmdb
import zlib
import random
import pickle
import time
import logging
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils import data
from typing import Tuple, Optional, Dict, Any
from multiprocessing import Pool, cpu_count
import torch.utils
import torch.utils.data
from tqdm import tqdm
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
SPEECH_DATABASE_TRAIN = WORK_DIR+'/prepare_datasets/train_clean_24k.csv'
NOISE_DATABASE_TRAIN =  WORK_DIR+'/prepare_datasets/train_noise_24k.csv'
SPEECH_DATABASE_VALID =  WORK_DIR+'/prepare_datasets/val_clean_24k.csv'
NOISE_DATABASE_VALID =  WORK_DIR+'/prepare_datasets/val_noise_24k.csv'
RIR_DATABASE_TRAIN =  WORK_DIR+'/prepare_datasets/train_rir_24k.csv'
RIR_DATABASE_VALID =  WORK_DIR+'/prepare_datasets/val_rir_24k.csv'

class HaSimuDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        clean_file_path: str,
        noise_file_path: str,
        rir_file_path: str,
        fs=24000,
        length_in_seconds=8,
        num_data_per_epoch=400000,
        train=True,
        snr_db=0.0,               # 新增：目标信噪比（dB）或范围，如 (min,max)
        dr_db=(-40, -10),      # 新增：强制动态范围 (RMS dBFS) 在此范围内
        pure_noise_prob=0.1,  # 新增：纯噪声样本比例
        random_seed=1234,
        max_readers=126):
        if train:
            print("You are using this training data:", clean_file_path, noise_file_path, rir_file_path)
        else:
            print("You are using this validation data:", clean_file_path, noise_file_path, rir_file_path)

        # --- 初始化随机种子，保证调试可重复 ---
        try:
            seed = int(random_seed)
        except Exception:
            seed = 1234
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)            # Python random
        np.random.seed(seed)         # numpy
        torch.manual_seed(seed)      # torch CPU
        try:
            torch.cuda.manual_seed_all(seed)  # torch GPU（若可用）
        except Exception:
            pass
        # 可选：使部分操作确定性（可能影响性能）
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        self.fs = fs
        self.target_rir_length = int(0.2 * fs)
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train
        self.pure_noise_prob = pure_noise_prob  
        # 支持传入单值或(min, max)范围
        if isinstance(snr_db, (list, tuple)) and len(snr_db) == 2:
            self.snr_db_min = float(snr_db[0])
            self.snr_db_max = float(snr_db[1])
        else:
            self.snr_db_min = float(snr_db)
            self.snr_db_max = float(snr_db)
        if isinstance(dr_db, (list, tuple)) and len(dr_db) == 2:
            self.dr_db_min = float(dr_db[0])
            self.dr_db_max = float(dr_db[1])
        else:
            self.dr_db_min = float(dr_db)
            self.dr_db_max = float(dr_db) 
        self.snr_db = -10
        self.in_memory = False
        self.max_readers = max_readers
        self.length_in_seconds = length_in_seconds
        self.L = int(length_in_seconds * fs)

        # --- 新增：加载并筛选 CSV 中的路径（按时长） ---
        def _find_column(cols, names):
            cols_l = [c.lower() for c in cols]
            for n in names:
                if n in cols_l:
                    return cols[cols_l.index(n)]
            return None

        def _load_and_filter(csv_path, min_duration):
            if not os.path.exists(csv_path):
                # 若文件不存在，返回空列表（调用处需处理）
                print(f"Warning: csv not found: {csv_path}")
                return []
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Warning: failed to read csv {csv_path}: {e}")
                return []
            # 尝试找到路径列（常见名称）
            path_col = _find_column(df.columns, ['file_dir', 'file_path', 'filepath', 'path', 'filename', 'file'])
            if path_col is None:
                # 退而求其次，使用第一列作为路径
                path_col = df.columns[0]
            # 尝试找到时长列（常见名称）
            dur_col = _find_column(df.columns, ['duration', 'dur', 'length', 'seconds', 'length_seconds'])
            if dur_col is None:
                # 无时长信息：返回所有路径（字符串化）
                return df[path_col].astype(str).tolist()
            else:
                # 过滤时长 >= min_duration（秒）
                try:
                    df[dur_col] = pd.to_numeric(df[dur_col], errors='coerce')
                    filtered = df[df[dur_col] >= float(min_duration)]
                    return filtered[path_col].astype(str).tolist()
                except Exception as e:
                    print(f"Warning: failed to filter by duration in {csv_path}: {e}")
                    return df[path_col].astype(str).tolist()
        # 加载训练/验证的 speech / noise / rir CSV（若存在），并按 length_in_seconds 筛选
        if self.train:
            self.speech_database_train = _load_and_filter(clean_file_path, self.length_in_seconds - 1)
            self.noise_database_train = _load_and_filter(noise_file_path, self.length_in_seconds - 1)
            # RIR 可能没有持续时间字段，但仍尝试加载（不强制时长）
            self.rir_database_train = _load_and_filter(rir_file_path, 0)
        else:
            self.speech_database_valid = _load_and_filter(clean_file_path, self.length_in_seconds - 1)
            self.noise_database_valid = _load_and_filter(noise_file_path, self.length_in_seconds - 1)
            # RIR 可能没有持续时间字段，但仍尝试加载（不强制时长）
            self.rir_database_valid = _load_and_filter(rir_file_path, 0)
        
    def sample_data_per_epoch(self):
        # 若可用数据少于需要数，允许有放回采样
        available = len(self.speech_database_train)
        if available == 0:
            print("Warning: no training speech files available after filtering.")
            self.speech_database_train = []
            return
        if available >= self.num_data_per_epoch:
            self.speech_database_train = random.sample(self.speech_database_train, self.num_data_per_epoch)
        else:
            # 使用有放回采样补齐
            self.speech_database_train = random.choices(self.speech_database_train, k=self.num_data_per_epoch)

    def __getitem__(self, idx):
        # 选择语音、噪声与 RIR 列表（训练/验证）
        if self.train:
            speech_list = self.speech_database_train
            noise_list = self.noise_database_train
            rir_list = self.rir_database_train
        else:
            speech_list = self.speech_database_valid
            noise_list = self.noise_database_valid
            rir_list = self.rir_database_valid

        # 若索引越界或列表为空，返回零信号
        if idx >= len(speech_list) or len(speech_list) == 0:
            clean = np.zeros(self.L, dtype=np.float32)
            noisy = np.zeros(self.L, dtype=np.float32)
            print("若索引越界或列表为空，返回零信号 {:d}".format(idx))
            return torch.from_numpy(clean).float(), torch.from_numpy(noisy).float()

        def rms(x):
            return np.sqrt(np.mean(x ** 2)) if x.size > 0 else 0.0
        
        def _read_mono(path):
            try:
                sig, sr = sf.read(path, dtype='float32')
            except Exception as e:
                print(f"Warning: failed to read audio {path}: {e}")
                return None, None
            sig = np.asarray(sig, dtype=np.float32)
            if sig.ndim > 1:
                # sig = sig.mean(axis=1)
                sig = sig[:, 1]  # 仅使用第二个通道，假定为右声道
            return sig, sr

        # 读取语音并截取/补零到 self.L
        speech_path = speech_list[idx]
        speech_sig, sr_s = _read_mono(speech_path)
        if speech_sig is None:
            clean = np.zeros(self.L, dtype=np.float32)
        else:
            # 如果采样率与 self.fs 不一致，尝试不做重采样（假定一致），否则可能产生长度误差
            total_len = len(speech_sig)
            if total_len >= self.L:
                # 随机截取一段
                max_begin = total_len - self.L
                begin = np.random.randint(0, max_begin + 1)
                clean = speech_sig[begin:begin + self.L].copy()
            else:
                # 补零到 self.L
                clean = np.zeros(self.L, dtype=np.float32)
                clean[:total_len] = speech_sig

        # 读取并应用 RIR（随机选择一个）
        rir_sig = np.zeros(self.L, dtype=np.float32)
        while(rms(rir_sig) == 0):
            if len(rir_list) > 0:
                rir_path = random.choice(rir_list)
                rir_sig_raw, sr_r = _read_mono(rir_path)
                if rir_sig_raw is not None:
                    rir_sig = rir_sig_raw.copy()
      
        # 如果有 RIR 则卷积，否则保持原始 clean
        if rir_sig is not None and np.any(rir_sig):
            # 卷积并取前 self.L
            conv = np.convolve(clean, rir_sig)[:self.L]
            # 若 conv 长度不足 self.L，则补零
            if len(conv) < self.L:
                tmp = np.zeros(self.L, dtype=np.float32)
                tmp[:len(conv)] = conv
                conv = tmp
            clean_conv = conv.astype(np.float32)
            # 前五十毫秒的 RIR作为target
            rir_trim = rir_sig[:self.target_rir_length]  # 裁剪或补零到目标长度       
            # 卷积并取前 self.L
            conv_trim = np.convolve(clean, rir_trim)[:self.L]
            # 若 conv 长度不足 self.L，则补零
            if len(conv_trim) < self.L:
                tmp = np.zeros(self.L, dtype=np.float32)
                tmp[:len(conv_trim)] = conv_trim
                conv_trim = tmp
            clean_conv_trim = conv_trim.astype(np.float32)                 
        else:
            clean_conv = clean.astype(np.float32)
            clean_conv_trim = clean.astype(np.float32)

        # 读取噪声并截取/补零到 self.L（随机选择一个噪声文件）
        noise_sig = np.zeros(self.L, dtype=np.float32)
        while(rms(noise_sig) == 0):
            if len(noise_list) > 0:
                noise_path = random.choice(noise_list)
                n_sig, sr_n = _read_mono(noise_path)
                if n_sig is not None:
                    n_len = len(n_sig)
                    if n_len >= self.L:
                        # 随机截取一段
                        start_n = np.random.randint(0, n_len - self.L + 1)
                        noise_seg = n_sig[start_n:start_n + self.L].copy()
                    else:
                        noise_seg = np.zeros(self.L, dtype=np.float32)
                        noise_seg[:n_len] = n_sig
                    noise_sig = noise_seg.astype(np.float32)

        # 调整噪声幅度以匹配目标 SNR（dB），每次从范围内随机采样一个 SNR
        rms_clean = rms(clean_conv)
        rms_noise = rms(noise_sig)
        if rms_noise == 0 or rms_clean == 0:
            noisy = clean_conv + noise_sig
            noise_scaled = noise_sig
            print(f"rms zero occurs: {rms_clean}, {rms_noise}")
        else:
            # 随机采样一个 SNR（dB）
            self.snr_db = int(random.uniform(self.snr_db_min, self.snr_db_max))
            # 要使 SNR = 20*log10(rms_clean / rms_noise_scaled)
            # 则 rms_noise_scaled = rms_clean / (10^(SNR/20))
            target_linear = 10.0 ** (-self.snr_db / 20.0)
            scale = (rms_clean / rms_noise) * target_linear
            noise_scaled = noise_sig * scale
            noisy = clean_conv + noise_scaled

        # 防止 NaN / inf
        noisy = np.nan_to_num(noisy).astype(np.float32)
        clean_out = np.nan_to_num(clean_conv_trim).astype(np.float32)
        noise_scaled = np.nan_to_num(noise_scaled).astype(np.float32)
        # --- 新增：强制 noisy 的动态范围 (RMS dBFS) 在 [-40, -10] 之间 ---
        # 计算 RMS 与 dB
        # 计算噪声音频的RMS并转换为分贝
        eps = 1e-10
        rms_noisy = rms(noisy)
        db_noisy = 20.0 * np.log10(np.maximum(rms_noisy, eps))
        # 使用torch.clamp进行裁剪（等效于np.clip）
        db_target = int(random.uniform(self.dr_db_min, self.dr_db_max))  # 每次随机选择一个目标范围上限
        # 若需要调整则缩放音频
        if np.abs(db_target - db_noisy) > 1e-6:
            scale = 10.0 ** ((db_target - db_noisy) / 20.0)
            noisy = noisy * scale
            clean_out = clean_out * scale
            noise_scaled = noise_scaled * scale
        # 防止峰值溢出，进行峰值归一化
        if noisy.size != 0:
            peak = np.max([np.max(np.abs(noisy)), np.max(np.abs(clean_out)), np.max(np.abs(noise_scaled))])
            if peak > 1.0:
                noisy = noisy / peak
                clean_out = clean_out / peak
                noise_scaled = noise_scaled / peak
        # --- 新增：按 pure_noise_prob 概率返回纯噪声样本 ---
        if self.train and self.pure_noise_prob > 1e-6:
            if random.random() < self.pure_noise_prob:
                # 返回纯噪声样本
                clean_out = np.zeros_like(noisy)    
                return torch.from_numpy(noise_scaled).float(), torch.from_numpy(clean_out).float(), self.snr_db
            if random.random() < self.pure_noise_prob:
                # 返回纯噪声样本
                noisy = np.zeros_like(clean_out)    
                return torch.from_numpy(noisy).float(), torch.from_numpy(clean_out).float(), self.snr_db    
        # 实时检查数据质量
        if np.max(np.abs(noisy)) < 1e-5 or np.max(np.abs(clean)) < 1e-5:
            print(f"警告: 样本 {idx} [d峰值过低: noisy={np.max(np.abs(noisy)):.2e}, clean={np.max(np.abs(clean)):.2e}")
                    
        # 返回 torch tensors: (clean, noisy) 按原始代码习惯可调整顺序
        return torch.from_numpy(noisy).float(), torch.from_numpy(clean_out).float(), self.snr_db

    def __len__(self):
        if self.train:
            return self.num_data_per_epoch
        else:
            if self.num_data_per_epoch < len(self.speech_database_valid):
                return self.num_data_per_epoch
            else: return len(self.speech_database_valid)

class HaSimuDataset_HD5(torch.utils.data.Dataset):
    """
    用于加载HDF5音频数据的PyTorch Dataset类
    支持从HDF5文件中读取音频波形数据或频谱特征
    """

    def __init__(
                self, 
                clean_file_path: str,
                noise_file_path: str,
                rir_file_path: str,
                fs=24000,
                length_in_seconds=8,
                num_data_per_epoch=400000,
                train=True,
                snr_db=0.0,               # 新增：目标信噪比（dB）或范围，如 (min,max)
                dr_db=(-40, -10),      # 新增：强制动态范围 (RMS dBFS) 在此范围内
                random_seed=1234,
                max_readers=126):
        """
        初始化HDF5音频数据集
        
        参数:
            h5_file_path: HDF5文件路径
        """
        super(HaSimuDataset_HD5, self).__init__()

        # --- 初始化随机种子，保证调试可重复 ---
        try:
            seed = int(random_seed)
        except Exception:
            seed = 1234
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)            # Python random
        np.random.seed(seed)         # numpy
        torch.manual_seed(seed)      # torch CPU
        try:
            torch.cuda.manual_seed_all(seed)  # torch GPU（若可用）
        except Exception:
            pass
        # 可选：使部分操作确定性（可能影响性能）
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        self.target_rir_length = int(0.5 * fs)
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train
        # 支持传入单值或(min, max)范围
        if isinstance(snr_db, (list, tuple)) and len(snr_db) == 2:
            self.snr_db_min = float(snr_db[0])
            self.snr_db_max = float(snr_db[1])
        else:
            self.snr_db_min = float(snr_db)
            self.snr_db_max = float(snr_db)
        if isinstance(dr_db, (list, tuple)) and len(dr_db) == 2:
            self.dr_db_min = float(dr_db[0])
            self.dr_db_max = float(dr_db[1])
        else:
            self.dr_db_min = float(dr_db)
            self.dr_db_max = float(dr_db)  
        self.clean_h5_file_path = clean_file_path
        self.noise_h5_file_path = noise_file_path
        self.rir_h5_file_path = rir_file_path
        print("You are using this dataset:", clean_file_path,"\t", noise_file_path,"\t", rir_file_path)            
        # 打开HDF5文件并读取 干净语音数据
        with h5py.File(clean_file_path, 'r') as h5f:
            # 获取音频数据集
            if "audio_data" not in h5f:
                raise KeyError(f"Audio key 'audio_data' not found in HDF5 file")
            h5_datas = h5f['audio_data']
            # 获取数据集基本信息
            self.clean_num_samples = h5_datas.attrs['num_files'] # 总样本数量     
            self.clean_audio_duration = h5_datas.attrs['duration'] # 每个音频段的持续时间（秒）
            self.clean_sampling_rate = h5_datas.attrs['sampling_rate']  # 采样率
            self.clean_dtype = h5_datas.attrs['dtype'] # 数据类型
            self.clean_num_channels = h5_datas.attrs['num_channels'] # 通道数
        # 打开HDF5文件并读取 噪声数据
        with h5py.File(noise_file_path, 'r') as h5f:
            # 获取音频数据集
            if "audio_data" not in h5f:
                raise KeyError(f"Audio key 'audio_data' not found in HDF5 file")
            h5_datas = h5f['audio_data']
            # 获取数据集基本信息
            self.noise_num_samples = h5_datas.attrs['num_files'] # 总样本数量    
            self.noise_audio_duration = h5_datas.attrs['duration'] # 每个音频段的持续时间（秒）
            self.noise_sampling_rate = h5_datas.attrs['sampling_rate']  # 采样率
            self.noisen_dtype = h5_datas.attrs['dtype'] # 数据类型
            self.noise_num_channels = h5_datas.attrs['num_channels'] # 通道数
        # 打开HDF5文件并读取 冲击响应数据
        with h5py.File(rir_file_path, 'r') as h5f:
            # 获取音频数据集
            if "audio_data" not in h5f:
                raise KeyError(f"Audio key 'audio_data' not found in HDF5 file")
            h5_datas = h5f['audio_data']
            # 获取数据集基本信息
            self.rir_num_samples = h5_datas.attrs['num_files'] # 总样本数量    
            self.rir_audio_duration = h5_datas.attrs['duration'] # 每个音频段的持续时间（秒）
            self.rir_sampling_rate = h5_datas.attrs['sampling_rate']  # 采样率
            self.rir_dtype = h5_datas.attrs['dtype'] # 数据类型
            self.rir_num_channels = h5_datas.attrs['num_channels'] # 通道数
        # 检查采样率是否一致
        if not (self.clean_sampling_rate == self.noise_sampling_rate == self.rir_sampling_rate == fs):
            raise ValueError("Sampling rates of clean, noise, and RIR data must match the specified fs")
        # 检查音频持续时间是否一致
        if not (self.clean_audio_duration == self.noise_audio_duration == length_in_seconds):
            raise ValueError("Audio durations of clean and noise data must match the specified length_in_seconds")
        self.length_in_seconds = length_in_seconds
        self.L = int(length_in_seconds * fs)


    def __len__(self) -> int:
        """返回数据集中的样本数量"""
        if self.clean_num_samples < self.num_data_per_epoch:
            return self.clean_num_samples
        else: return self.num_data_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        根据索引获取音频样本和标签
        
        参数:
            idx: 样本索引
            
        返回:
            audio_tensor: 音频张量 [channels, length] 或特征张量
            label_tensor: 标签张量（如果有标签）
        """
        # 从HDF5文件中读取音频数据
        with h5py.File(self.clean_h5_file_path, 'r') as h5f:
            audio_data = h5f['audio_data'][idx]
        # 从HDF5文件中随机选择一个噪声和RIR样本
        with h5py.File(self.noise_h5_file_path, 'r') as h5f:
            noise_data = h5f['audio_data'][random.randint(0, self.noise_num_samples - 1)]
        with h5py.File(self.rir_h5_file_path, 'r') as h5f:
            rir_data = h5f['audio_data'][random.randint(0, self.rir_num_samples - 1)]
        
        # 转换为PyTorch张量       
        audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
        noise_tensor = torch.from_numpy(noise_data.astype(np.float32))
        rir_tensor = torch.from_numpy(rir_data.astype(np.float32))
        if self.rir_num_channels > 1:
            rir_tensor = rir_tensor[:,0]  # 仅使用第一个通道

        def _torch_convolve(signal, kernel, mode='full'):
            """
            使用 PyTorch 的 conv1d 函数模拟 NumPy 的 convolve 行为。

            Args:
                signal: 输入信号，一维 PyTorch 张量。
                kernel: 卷积核，一维 PyTorch 张量。
                mode: 卷积模式，'full', 'same', 或 'valid' [2](@ref)。

            Returns:
                一维张量，卷积结果。
            """
            # 确保输入是一维的
            signal = signal.view(1, 1, -1)  # 形状变为 (batch_size=1, in_channels=1, length)
            kernel = kernel.flip(dims=[0])  # 翻转卷积核以匹配np.convolve的互相关操作 [5](@ref)
            kernel = kernel.view(1, 1, -1)  # 权重形状: (out_channels=1, in_channels/groups=1, kernel_size) [9,10](@ref)

            # 根据模式设置填充 [6](@ref)
            signal_length = signal.shape[-1]
            kernel_length = kernel.shape[-1]
            
            if mode == 'full':
                padding = kernel_length - 1
            elif mode == 'same':
                padding = (kernel_length - 1) // 2
            elif mode == 'valid':
                padding = 0
            else:
                raise ValueError("模式必须是 'full', 'same', 或 'valid'")

            # 使用卷积操作，分组数设为1 [9](@ref)
            result = torch.nn.functional.conv1d(signal, kernel, padding=padding, groups=1)
            return result.squeeze()  # 将输出恢复为一维

        conv = _torch_convolve(audio_tensor, rir_tensor)[:self.L]
        # 前五十毫秒的 RIR作为target
        rir_trim = rir_tensor[:self.target_rir_length]  # 裁剪或补零到目标长度       
        # 卷积并取前 self.L
        conv_trim = _torch_convolve(audio_tensor, rir_trim)[:self.L]               

        # 调整噪声幅度以匹配目标 SNR（dB），每次从范围内随机采样一个 SNR
        def _rms(x, dim=None, keepdim=False):
            """
            PyTorch版本的RMS计算函数
            
            参数:
                x: 输入张量
                dim: 沿指定维度计算RMS（默认为全局计算）
                keepdim: 是否保持维度
            """
            if x.numel() == 0:
                return torch.tensor(0.0, device=x.device)
            
            # 计算平方值的均值，然后开方[1](@ref)
            squared = x ** 2
            if dim is not None:
                mean_squared = torch.mean(squared, dim=dim, keepdim=keepdim)
            else:
                mean_squared = torch.mean(squared)
            
            return torch.sqrt(mean_squared)

        rms_cov = _rms(conv)
        rms_noise = _rms(noise_tensor)
        if rms_noise == 0 or rms_cov == 0:
            noisy = conv + noise_tensor
        else:
            # 随机采样一个 SNR（dB）
            snr_db = int(random.uniform(self.snr_db_min, self.snr_db_max))
            # 要使 SNR = 20*log10(rms_clean / rms_noise_scaled)
            # 则 rms_noise_scaled = rms_clean / (10^(SNR/20))
            target_linear = 10.0 ** (-snr_db / 20.0)
            scale = (rms_cov / rms_noise) * target_linear
            noise_scaled = noise_tensor * scale
            noisy = conv + noise_scaled

        # 防止 NaN / inf
        noisy = torch.nan_to_num(noisy).to(torch.float32)
        clean_out = torch.nan_to_num(conv_trim).to(torch.float32)

        # --- 新增：强制 noisy 的动态范围 (RMS dBFS) 在 [-40, -5] 之间 ---
        # 计算 RMS 与 dB
        # 计算噪声音频的RMS并转换为分贝
        eps = 1e-10
        rms_noisy = _rms(noisy)
        db_noisy = 20.0 * torch.log10(torch.max(rms_noisy, torch.tensor(eps, device=noisy.device)))
        # 使用torch.clamp进行裁剪（等效于np.clip）
        db_target = int(random.uniform(-40, -10))  # 每次随机选择一个目标范围上限
        # 若需要调整则缩放音频
        if torch.abs(db_target - db_noisy) > 1e-6:
            scale = 10.0 ** ((db_target - db_noisy) / 20.0)
            noisy = noisy * scale
            clean_out = clean_out * scale
        # 防止峰值溢出，进行峰值归一化
        if noisy.numel() > 0:
            peak = torch.max(torch.abs(noisy))
            if peak > 1.0:
                noisy = noisy / peak
                clean_out = clean_out / peak
        
        return noisy, clean_out
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """返回数据集的详细信息"""
        info_dict = {
            'snrrange in dB': (self.snr_db_min, self.snr_db_max),
            'dynamic range in dB range': (self.dr_db_min, self.dr_db_max),
            'sampling rate': self.clean_sampling_rate,
            'audio segment length in s': self.length_in_seconds,
            'target rir reserve in s': int(self.target_rir_length / self.clean_sampling_rate),
            'num audio file (8s) per epoch': self.num_data_per_epoch,
            'is training now': self.train
        }
        print("Dataset info:\n", info_dict)
        return info_dict


class HaSimulate_LMDB(torch.utils.data.Dataset):
    """
    从LMDB数据库加载WAV音频数据的PyTorch Dataset类
    优化了读取性能并包含完整的错误处理[1,6](@ref)
    """
    
    def __init__(
                self, 
                clean_file_path: str,
                noise_file_path: str,
                rir_file_path: str,
                fs=24000,
                length_in_seconds=8,
                num_data_per_epoch=400000,
                train=True,
                snr_db=0.0,               # 新增：目标信噪比（dB）或范围，如 (min,max)
                dr_db=(-40, -10),      # 新增：强制动态范围 (RMS dBFS) 在此范围内
                random_seed=1234,
                max_readers=126):
        """
        初始化LMDB数据集
        
        参数:

        """

        # --- 初始化随机种子，保证调试可重复 ---
        try:
            seed = int(random_seed)
        except Exception:
            seed = 1234
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)            # Python random
        np.random.seed(seed)         # numpy
        torch.manual_seed(seed)      # torch CPU
        try:
            torch.cuda.manual_seed_all(seed)  # torch GPU（若可用）
        except Exception:
            pass
        # 可选：使部分操作确定性（可能影响性能）
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        self.fs = fs
        self.target_rir_length = int(0.5 * fs)
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train
        # 支持传入单值或(min, max)范围
        if isinstance(snr_db, (list, tuple)) and len(snr_db) == 2:
            self.snr_db_min = float(snr_db[0])
            self.snr_db_max = float(snr_db[1])
        else:
            self.snr_db_min = float(snr_db)
            self.snr_db_max = float(snr_db)
        if isinstance(dr_db, (list, tuple)) and len(dr_db) == 2:
            self.dr_db_min = float(dr_db[0])
            self.dr_db_max = float(dr_db[1])
        else:
            self.dr_db_min = float(dr_db)
            self.dr_db_max = float(dr_db)  

        self.clean_lmdb_path = clean_file_path
        self.noise_lmdb_path = noise_file_path
        self.rir_lmdb_path = rir_file_path

        self.in_memory = False
        self.max_readers = max_readers

        # --- 移除原有的 self._init_db 调用 ---
        # 我们不再在这里初始化 self.clean_env, self.clean_txn 等
        # 改为读取全局元数据（如果需要的话，可以用一个临时环境）
        # 注意：全局元数据通常很小，可以安全地读取并存储为普通数据类型
        try:
            # 临时打开一个环境来获取全局元数据
            clean_env = lmdb.open(self.clean_lmdb_path, readonly=True, lock=False, max_readers=1)
            self.clean_keys = []
            with clean_env.begin(write=False) as temp_txn:
                # 使用cursor遍历所有键，`iternext(keys=True, values=False)`只获取key
                self.clean_keys = [key.decode('utf-8') for key in temp_txn.cursor().iternext(keys=True, values=False)]
                global_meta_bytes = temp_txn.get(b'__global_metadata__')
                if global_meta_bytes:
                    self.clean_global_metadata = pickle.loads(global_meta_bytes)
                else:
                    raise ValueError("Global metadata not found in clean LMDB")
            self.clean_length = len(self.clean_keys)
            clean_env.close()
            noise_env = lmdb.open(self.noise_lmdb_path, readonly=True, lock=False, max_readers=1)
            with noise_env.begin(write=False) as temp_txn:
                # 使用cursor遍历所有键，`iternext(keys=True, values=False)`只获取key
                self.noise_keys = [key.decode('utf-8') for key in temp_txn.cursor().iternext(keys=True, values=False)]                
                global_meta_bytes = temp_txn.get(b'__global_metadata__')
                if global_meta_bytes:
                    self.noise_global_metadata = pickle.loads(global_meta_bytes)
                else:
                    raise ValueError("Global metadata not found in clean LMDB")
            self.noise_length = len(self.noise_keys)
            noise_env.close()
            rir_env = lmdb.open(self.rir_lmdb_path, readonly=True, lock=False, max_readers=1)
            with rir_env.begin(write=False) as temp_txn:
                # 使用cursor遍历所有键，`iternext(keys=True, values=False)`只获取key
                self.rir_keys = [key.decode('utf-8') for key in temp_txn.cursor().iternext(keys=True, values=False)]                
                global_meta_bytes = temp_txn.get(b'__global_metadata__')
                if global_meta_bytes:
                    self.rir_global_metadata = pickle.loads(global_meta_bytes)
                else:
                    raise ValueError("Global metadata not found in clean LMDB")
            self.rir_length = len(self.rir_keys)     
            rir_env.close() 
        except Exception as e:
            logger.error(f"Error reading global metadata: {e}")
            raise

        if not (self.clean_global_metadata['sampling_rate'] == \
                self.noise_global_metadata['sampling_rate'] == \
                self.rir_global_metadata['sampling_rate'] == fs):
            raise ValueError("Sampling rates of clean, noise, and RIR data must match the specified fs")
        if not (self.clean_global_metadata['duration'] == \
                self.noise_global_metadata['duration'] == length_in_seconds):
            raise ValueError("Audio durations of clean and noise data must match the specified length_in_seconds")
        
        self.length_in_seconds = length_in_seconds
        self.L = int(length_in_seconds * fs)

    def __len__(self) -> int:
        """返回数据集中的样本数量"""
        # 假设 clean_global_metadata 里包含了总样本数 'num_files'
        total_files = self.clean_length
        if total_files < self.num_data_per_epoch:
            return total_files
        else:
            return self.num_data_per_epoch

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        根据索引获取数据样本
        
        参数:
            index: 样本索引
            
        返回:
            包含音频数据和元数据的字典
        """
        if index >= self.clean_length:
            raise IndexError(f"Index {index} out of range for dataset of size {self.clean_length}")
        
        try:
            self.clean_env = lmdb.open(self.clean_lmdb_path, readonly=True, lock=False, max_readers=self.max_readers)
            self.noise_env = lmdb.open(self.noise_lmdb_path, readonly=True, lock=False, max_readers=self.max_readers)
            self.rir_env = lmdb.open(self.rir_lmdb_path, readonly=True, lock=False, max_readers=self.max_readers)
            # 构建键（8位数字格式）
            with self.clean_env.begin(write=False) as clean_txn, \
                 self.noise_env.begin(write=False) as noise_txn, \
                 self.rir_env.begin(write=False) as rir_txn:
                key = self.clean_keys[index].encode()  # 使用预存的键列表
                # 从LMDB读取语音数据
                value_bytes = clean_txn.get(key)
                if value_bytes is None:
                    raise KeyError(f"Key {key} not found in LMDB database")
                # 反序列化数据
                data_dict = pickle.loads(value_bytes) 
                # 提取音频数据和元数据
                clean_data = data_dict['audio_data']

                # 构建键（8位数字格式）
                n_idx = random.randint(0, self.noise_length - 1)
                key = self.noise_keys[n_idx].encode()  # 使用预存的键列表
                # 从LMDB读取噪声数据
                noise_value_bytes = noise_txn.get(key)
                if noise_value_bytes is None:
                    raise KeyError(f"Key {key} not found in noise LMDB database")
                noise_dict = pickle.loads(noise_value_bytes)
                noise_data = noise_dict['audio_data']
                
                # 构建键（8位数字格式）
                r_idx = random.randint(0, self.rir_length - 1)
                key = self.rir_keys[r_idx].encode()  # 使用预存的键列表
                # 从LMDB读取RIR数据
                rir_value_bytes = rir_txn.get(key)
                if rir_value_bytes is None:
                    raise KeyError(f"Key {key} not found in RIR LMDB database")
                rir_dict = pickle.loads(rir_value_bytes)
                rir_data = rir_dict['audio_data']   
            
            if self.rir_global_metadata['num_channels'] > 1:
                rir_data = rir_data[:,1]  # 仅使用第一个通道

            clean_conv = np.convolve(clean_data, rir_data)[:self.L]
            # 前五十毫秒的 RIR作为target
            rir_trim = rir_data[:self.target_rir_length]  # 裁剪或补零到目标长度       
            # 卷积并取前 self.L
            conv_trim = np.convolve(clean_data, rir_trim)[:self.L]   

            # 调整噪声幅度以匹配目标 SNR（dB），每次从范围内随机采样一个 SNR
            def _rms(x, dim=None, keepdim=False):
                """
                PyTorch版本的RMS计算函数
                
                参数:
                    x: 输入张量
                    dim: 沿指定维度计算RMS（默认为全局计算）
                    keepdim: 是否保持维度
                """
                if x.size == 0:
                    return np.zeros(x.size, dtype=np.float32)
                
                # 计算平方值的均值，然后开方[1](@ref)
                squared = x ** 2
                if dim is not None:
                    mean_squared = np.mean(squared, dim=dim, keepdim=keepdim)
                else:
                    mean_squared = np.mean(squared)
                
                return np.sqrt(mean_squared)

            rms_cov = _rms(clean_conv)
            rms_noise = _rms(noise_data)
            if rms_noise == 0 or rms_cov == 0:
                noisy = clean_conv + noise_data
            else:
                # 随机采样一个 SNR（dB）
                snr_db = int(random.uniform(self.snr_db_min, self.snr_db_max))
                # 要使 SNR = 20*log10(rms_clean / rms_noise_scaled)
                # 则 rms_noise_scaled = rms_clean / (10^(SNR/20))
                target_linear = 10.0 ** (-snr_db / 20.0)
                scale = (rms_cov / rms_noise) * target_linear
                noise_scaled = noise_data * scale
                noisy = clean_conv + noise_scaled

            # 防止 NaN / inf
            noisy = np.nan_to_num(noisy)
            clean_out = np.nan_to_num(conv_trim)

            # --- 新增：强制 noisy 的动态范围 (RMS dBFS) 在 [-40, -5] 之间 ---
            # 计算 RMS 与 dB
            # 计算噪声音频的RMS并转换为分贝
            eps = 1e-10
            rms_noisy = _rms(noisy)
            db_noisy = 20.0 * np.log10(np.maximum(rms_noisy, eps))
            # 使用torch.clamp进行裁剪（等效于np.clip）
            db_target = int(random.uniform(self.dr_db_min, self.dr_db_max))  # 每次随机选择一个目标范围上限
            # 若需要调整则缩放音频
            if np.abs(db_target - db_noisy) > 1e-6:
                scale = 10.0 ** ((db_target - db_noisy) / 20.0)
                noisy = noisy * scale
                clean_out = clean_out * scale
            # 防止峰值溢出，进行峰值归一化
            if noisy.size != 0:
                peak = np.max(np.abs(noisy))
                if peak > 1.0:
                    noisy = noisy / peak
                    clean_out = clean_out / peak

        # 包含音频数据和元数据的字典    
        except Exception as e:
            logger.error(f"Error loading sample {index}: {e}")
            # 返回空样本或进行错误处理
            return self._get_empty_sample(index)
        return torch.from_numpy(noisy).to(torch.float), torch.from_numpy(clean_out).to(torch.float)
    

    def _get_empty_sample(self, index: int) -> Dict[str, Any]:
        """返回空样本用于错误处理"""
        empty_audio = torch.zeros(self.clean_length, dtype=torch.float)
        return empty_audio, empty_audio
    
    def get_global_metadata(self) -> Dict[str, Any]:
        """获取全局元数据"""
        return self.clean_global_metadata.copy() if self.clean_global_metadata else {}
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'env') and self.env:
            self.env.close()
    
    def close(self):
        """显式关闭数据库连接"""
        if self.clean_env is not None:
            self.clean_env.close()
            self.clean_env = None
        if self.noise_env is not None:
            self.noise_env.close()
            self.noise_env = None
        if self.rir_env is not None:
            self.rir_env.close()
            self.rir_env = None

class HaSimuDatasetToLMDB:
    def __init__(self, original_dataset, lmdb_path, num_workers=None):
        """
        将HaSimuDataset转换为LMDB格式
        
        Args:
            original_dataset: HaSimuDataset实例
            lmdb_path: LMDB数据库保存路径
            num_workers: 进程数，默认为CPU核心数
        """
        self.dataset = original_dataset
        self.lmdb_path = lmdb_path
        self.num_workers = num_workers if num_workers else cpu_count()
        self.map_size = 1024 ** 4  # 1TB LMDB映射大小[6](@ref)
        
    def _calculate_map_size(self):
        """估算LMDB所需映射大小"""
        # 采样几个数据点来估算平均大小
        sample_size = min(100, len(self.dataset))
        total_size = 0
        
        for i in range(sample_size):
            try:
                noisy, clean, snr = self.dataset[i]
                # 估算序列化后的大小（包含压缩）
                sample_data = {
                    'noisy': noisy.numpy(),
                    'clean': clean.numpy()
                }
                serialized = pickle.dumps(sample_data)
                compressed = zlib.compress(serialized)
                total_size += len(compressed)
            except:
                continue
        
        if sample_size > 0:
            avg_size = total_size / sample_size
            total_estimated_size = avg_size * len(self.dataset) * 2  # 2倍安全系数
            self.map_size = max(total_estimated_size, 1024 ** 3)  # 至少1GB
        else:
            self.map_size = 1024 ** 4  # 默认1TB[6](@ref)
    
    def _process_sample(self, idx):
        """处理单个样本（用于多进程）"""
        try:
            # 获取数据
            noisy, clean, snr = self.dataset[idx]
            
            # 准备数据字典
            sample_data = {
                'snr': snr,
                'noisy': noisy.numpy(),
                'clean': clean.numpy(),
                'index': idx
            }
            
            # 序列化并压缩
            serialized_data = pickle.dumps(sample_data)
            compressed_data = zlib.compress(serialized_data)
            
            return idx, compressed_data, None
        except Exception as e:
            print(f"error: {e}")
            return idx, None, str(e)
    
    def _process_batch(self, batch_indices):
        """处理批次数据"""
        results = []
        for idx in batch_indices:
            result = self._process_sample(idx)
            results.append(result)
        return results
    
    def convert_to_lmdb(self):
        """执行LMDB转换"""
        print(f"开始将数据集转换为LMDB格式...")
        print(f"数据量: {len(self.dataset)} 个样本")
        print(f"使用进程数: {self.num_workers}")
        print(f"LMDB保存路径: {self.lmdb_path}")
        
        # 计算映射大小
        # self._calculate_map_size()
        print(f"LMDB映射大小: {self.map_size / (1024**3):.2f} GB")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(self.lmdb_path) if os.path.dirname(self.lmdb_path) else '.', exist_ok=True)
        
        # 删除已存在的数据库[3](@ref)
        if os.path.exists(self.lmdb_path):
            import shutil
            shutil.rmtree(self.lmdb_path)
            print("删除已存在的LMDB数据库")
        
        start_time = time.time()
        
        # 创建LMDB环境[1](@ref)
        env = lmdb.open(self.lmdb_path, map_size=self.map_size, max_readers=126, readonly=False)
        
        try:
            # 使用多进程处理数据
            batch_size = 1000  # 每个批次的样本数
            total_samples = len(self.dataset)
            
            # 创建进度条
            pbar = tqdm(total=total_samples, desc="转换进度")
            
            with Pool(processes=self.num_workers) as pool:
                for start_idx in range(0, total_samples, batch_size):
                    end_idx = min(start_idx + batch_size, total_samples)
                    batch_indices = list(range(start_idx, end_idx))
                    
                    # 为每个批次创建独立的事务[1](@ref)
                    with env.begin(write=True) as txn:  # 每个批次使用新事务
                        batch_results = []
                        for i in range(0, len(batch_indices), 100):
                            sub_batch = batch_indices[i:i+100]
                            batch_results.extend(pool.map(self._process_sample, sub_batch))
                        
                        successful_writes = 0
                        for idx, data, error in batch_results:
                            if data is not None:
                                key = f"{idx:010d}".encode('ascii')
                                txn.put(key, data)
                                successful_writes += 1
                        
                    
                    pbar.update(len(batch_indices))
            
            pbar.close()
            
            # 元数据写入使用独立事务
            with env.begin(write=True) as txn:
                meta_info = {
                    'total_samples': total_samples,
                    'sample_length': self.dataset.L,
                    'sample_rate': self.dataset.fs,
                    'creation_time': time.time(),
                    'dataset_type': 'train' if self.dataset.train else 'val'
                }
                meta_key = b'meta_info'
                meta_value = pickle.dumps(meta_info)
                txn.put(meta_key, zlib.compress(meta_value))
            
            end_time = time.time()
            print(f"转换完成! 耗时: {end_time - start_time:.2f} 秒")
            print(f"LMDB数据库位置: {self.lmdb_path}")
            
        except Exception as e:
            print(f"转换过程中发生错误: {e}")
            raise
        finally:
            env.close()

def get_available_memory():
    """获取系统可用内存"""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        # 如果psutil不可用，返回一个保守估计值
        return 16 * 1024**3  # 假设16GB

class HaDataSetsFromLMDB(torch.utils.data.Dataset):
    """终极解决方案：完全多进程安全的LMDB Dataset"""
    
    def __init__(self, lmdb_path, max_reader=512):
        self.lmdb_path = lmdb_path
        self.max_reader = max_reader
        
        # 在主进程仅获取元数据
        env = lmdb.open(lmdb_path, max_readers=max_reader, readonly=True, lock=False)
        with env.begin() as txn:
            meta_value = txn.get(b'meta_info')
            if meta_value:
                meta_info = pickle.loads(zlib.decompress(meta_value))
                self.total_samples = meta_info['total_samples']
                self.sample_length = meta_info['sample_length']
                self.sample_rate = meta_info['sample_rate']
            else:
                self.total_samples = sum(1 for _ in txn.cursor() if _[0] != b'meta_info')
        env.close()
        
        # 关键：不在__init__中初始化环境，由worker_init_fn处理
        self.env = None

    def _init_env(self):
        """为当前工作进程初始化LMDB环境（惰性初始化）"""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                max_readers=self.max_reader,
                readonly=True,
                lock=False,  # 多读取器必须设置lock=False[2](@ref)
                create=False,
                subdir=True
            )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 确保环境已初始化
        self._init_env()
        
        with self.env.begin() as txn:
            key = f"{idx:010d}".encode('ascii')
            compressed_data = txn.get(key)
            
            if compressed_data is None:
                raise KeyError(f"Key {key} not found in LMDB database")
            
            serialized_data = zlib.decompress(compressed_data)
            sample_data = pickle.loads(serialized_data)
            
            # 添加数据验证
            noisy = torch.from_numpy(sample_data['noisy'].copy())  # 使用copy()确保数据独立性
            clean = torch.from_numpy(sample_data['clean'].copy())
            
            # 实时检查数据质量
            if torch.max(torch.abs(noisy)) < 1e-5 or torch.max(torch.abs(clean)) < 1e-5:
                print(f"警告: 样本 {idx} [d峰值过低: noisy={torch.max(torch.abs(noisy)):.2e}, clean={torch.max(torch.abs(clean)):.2e}")
            
        return noisy, clean

def lmdb_worker_init_fn(worker_id):
    """DataLoader工作进程初始化函数"""
    # 确保每个工作进程有独立的随机种子
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        torch.manual_seed(worker_info.seed % (2**32 - 1))

if __name__=='__main__':
    # pass
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('configs/cfg_train.yaml')
    # 将 OmegaConf 的 DictConfig/ListConfig 等转换为原生的 Python 容器（dict/list）
    # 这样 configs 中的 snd_db: [0, 15] 会成为 Python 列表 [0, 15]
    try:
        config = OmegaConf.to_container(config, resolve=True)
    except Exception:
        # 若转换失败则保持原始 config（兼容性），后续可手动转换字段
        pass

        
    train_dataset = HaSimuDataset(**config['train_dataset'])
    train_dataset.sample_data_per_epoch()
    # 创建转换器并执行转换
    converter = HaSimuDatasetToLMDB(train_dataset, './prepare_datasets/training_audio_24k_pure10_trim200.lmdb', 4)
    converter.convert_to_lmdb()

    valid_dataset = HaSimuDataset(**config['validation_dataset'])
    # 创建转换器并执行转换
    converter = HaSimuDatasetToLMDB(valid_dataset, './prepare_datasets/validation_audio_24k_pure10_trim200.lmdb', 4)
    converter.convert_to_lmdb()

    # output_dir = WORK_DIR + "/prepare_datasets/check_data_samples/lmdb_audios"
    # os.makedirs(output_dir, exist_ok=True)

    # # 保存训练数据的音频
    # datasets = HaDataSetsFromLMDB('./prepare_datasets/validation_audio_24k_pure10_trim200.lmdb', max_reader=512)
    # dataloader = torch.utils.data.DataLoader(
    #     datasets, 
    #     batch_size=16, 
    #     shuffle=False, 
    #     num_workers=4, 
    #     pin_memory=False,
    #     worker_init_fn=lmdb_worker_init_fn  # 添加worker初始化函数
    # )
    # for i, (noisy, clean) in enumerate(tqdm(dataloader, desc="Processing train data")):
    #     peak_1 = np.max(np.abs(noisy.numpy()))
    #     peak_2 = np.max(np.abs(clean.numpy()))
    #     if peak_1 <= 1e-5 or peak_2 <= 1e-5 or peak_1 > 1 or peak_2 > 1:
    #         print(f"Warning: Peak value {peak_1, peak_2} ...")
    #         noisy_path = os.path.join(output_dir, f"amp_noisy_{i}.wav")
    #         clean_path = os.path.join(output_dir, f"amp_clean_{i}.wav")
    #         sf.write(noisy_path, noisy[0].numpy(), config['validation_dataset']['fs'])
    #         sf.write(clean_path, clean[0].numpy(), config['validation_dataset']['fs'])
    #         break
        # if i > 1000:
        #     break
    # # 保存验证数据的音频
    # for i, (noisy, clean) in enumerate(tqdm(validation_dataloader, desc="Processing validation data")):
    #     noisy_path = os.path.join(output_dir, f"val_noisy_{i}.wav")
    #     clean_path = os.path.join(output_dir, f"val_clean_{i}.wav")
    #     sf.write(noisy_path, noisy[0].numpy(), config['validation_dataset']['fs'])
    #     sf.write(clean_path, clean[0].numpy(), config['validation_dataset']['fs'])
    #     if i >= 9:  # 仅保存前10个样本
    #         break
