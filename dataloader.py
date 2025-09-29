import os
import torch
import h5py
import random
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils import data
from typing import Tuple, Optional, Dict, Any

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
        fs=24000,
        length_in_seconds=8,
        num_data_per_epoch=400000,
        random_start_point=False,
        train=True,
        snr_db=0.0,               # 新增：目标信噪比（dB）或范围，如 (min,max)
        random_seed=1234
    ):
        if train:
            print("You are using this training data:", SPEECH_DATABASE_TRAIN, NOISE_DATABASE_TRAIN, RIR_DATABASE_TRAIN)
        else:
            print("You are using this validation data:", SPEECH_DATABASE_VALID, NOISE_DATABASE_VALID, RIR_DATABASE_VALID)
        
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

        self.L = int(length_in_seconds * fs)
        self.random_start_point = random_start_point
        self.fs = fs
        self.target_rir_len = int(0.3 * fs)  # 50ms
        self.length_in_seconds = length_in_seconds
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train
        # 支持传入单值或(min, max)范围
        if isinstance(snr_db, (list, tuple)) and len(snr_db) == 2:
            self.snr_db_min = float(snr_db[0])
            self.snr_db_max = float(snr_db[1])
        else:
            self.snr_db_min = float(snr_db)
            self.snr_db_max = float(snr_db)

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
            self.speech_database_train = _load_and_filter(SPEECH_DATABASE_TRAIN, self.length_in_seconds)
            self.noise_database_train = _load_and_filter(NOISE_DATABASE_TRAIN, self.length_in_seconds)
            # RIR 可能没有持续时间字段，但仍尝试加载（不强制时长）
            self.rir_database_train = _load_and_filter(RIR_DATABASE_TRAIN, 0)
        else:
            self.speech_database_valid = _load_and_filter(SPEECH_DATABASE_VALID, self.length_in_seconds)
            self.noise_database_valid = _load_and_filter(NOISE_DATABASE_VALID, self.length_in_seconds)
            # RIR 可能没有持续时间字段，但仍尝试加载（不强制时长）
            self.rir_database_valid = _load_and_filter(RIR_DATABASE_VALID, 0)
        
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
            return torch.from_numpy(clean).float(), torch.from_numpy(noisy).float()

        speech_path = speech_list[idx]

        def _read_mono(path):
            try:
                sig, sr = sf.read(path, dtype='float32')
            except Exception as e:
                print(f"Warning: failed to read audio {path}: {e}")
                return None, None
            sig = np.asarray(sig, dtype=np.float32)
            if sig.ndim > 1:
                # sig = sig.mean(axis=1)
                sig = sig[:, 0]
            return sig, sr

        # 读取语音并截取/补零到 self.L
        speech_sig, sr_s = _read_mono(speech_path)
        if speech_sig is None:
            clean = np.zeros(self.L, dtype=np.float32)
        else:
            # 如果采样率与 self.fs 不一致，尝试不做重采样（假定一致），否则可能产生长度误差
            total_len = len(speech_sig)
            if total_len >= self.L:
                if self.random_start_point:
                    max_begin = total_len - self.L
                    begin = np.random.randint(0, max_begin + 1)
                else:
                    begin = 0
                clean = speech_sig[begin:begin + self.L].copy()
            else:
                # 补零到 self.L
                clean = np.zeros(self.L, dtype=np.float32)
                clean[:total_len] = speech_sig

        # 读取并应用 RIR（随机选择一个）
        rir_sig = None
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
            rir_trim = rir_sig[:self.target_rir_len]  # 裁剪或补零到目标长度       
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
        def rms(x):
            return np.sqrt(np.mean(x ** 2)) if x.size > 0 else 0.0

        rms_clean = rms(clean_conv)
        rms_noise = rms(noise_sig)
        if rms_noise == 0 or rms_clean == 0:
            noisy = clean_conv + noise_sig
        else:
            # 随机采样一个 SNR（dB）
            snr_db = int(random.uniform(self.snr_db_min, self.snr_db_max))
            # 要使 SNR = 20*log10(rms_clean / rms_noise_scaled)
            # 则 rms_noise_scaled = rms_clean / (10^(SNR/20))
            target_linear = 10.0 ** (-snr_db / 20.0)
            scale = (rms_clean / rms_noise) * target_linear
            noise_scaled = noise_sig * scale
            noisy = clean_conv + noise_scaled

        # 防止 NaN / inf
        noisy = np.nan_to_num(noisy).astype(np.float32)
        clean_out = np.nan_to_num(clean_conv_trim).astype(np.float32)

        # --- 新增：强制 noisy 的动态范围 (RMS dBFS) 在 [-40, -5] 之间 ---
        # 计算 RMS 与 dB
        eps = 1e-10
        def _rms(x):
            return np.sqrt(np.mean(x ** 2)) if x.size > 0 else 0.0
        rms_noisy = _rms(noisy)
        db_noisy = 20.0 * np.log10(max(rms_noisy, eps))
        # 裁剪到目标范围
        db_target = np.clip(db_noisy, -60.0, -25.0)
        # 若需要调整则缩放 noisy
        if abs(db_target - db_noisy) > 1e-6:
            scale = 10.0 ** ((db_target - db_noisy) / 20.0)
            noisy = noisy * scale
            clean_out = clean_out * scale
        # 防止峰值溢出（裁剪），再按峰值归一化（这可能会降低 RMS，但避免失真）
        peak = np.max(np.abs(noisy)) if noisy.size > 0 else 0.0
        if peak > 1.0:
            noisy = noisy / peak
            clean_out = clean_out / peak

        # 返回 torch tensors: (clean, noisy) 按原始代码习惯可调整顺序
        return torch.from_numpy(noisy).float(), torch.from_numpy(clean_out).float()

    def __len__(self):
        if self.train:
            return self.num_data_per_epoch
        else:
            return len(self.speech_database_valid)


class HaSimuDataset_HD5(torch.utils.data.Dataset):
    """
    用于加载HDF5音频数据的PyTorch Dataset类
    支持从HDF5文件中读取音频波形数据或频谱特征
    """

    def __init__(
                self, 
                clean_h5_file_path: str,
                noise_h5_file_path: str,
                rir_h5_file_path: str,
                fs=24000,
                length_in_seconds=8,
                num_data_per_epoch=400000,
                train=True,
                snr_db=0.0,               # 新增：目标信噪比（dB）或范围，如 (min,max)
                dr_db=(-40, -10),      # 新增：强制动态范围 (RMS dBFS) 在此范围内
                random_seed=1234):
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
                  
        print("You are using this dataset:", clean_h5_file_path,"\t", noise_h5_file_path,"\t", rir_h5_file_path)            
        # 打开HDF5文件并读取 干净语音数据
        with h5py.File(clean_h5_file_path, 'r') as h5f:
            # 获取音频数据集
            if "audio_data" not in h5f:
                raise KeyError(f"Audio key 'audio_data' not found in HDF5 file")
            self.audio_data = h5f['audio_data']
            # 获取数据集基本信息
            self.clean_num_samples = h5f['num_files'] # 总样本数量    
            self.clean_audio_duration = h5f['duration'] # 每个音频段的持续时间（秒）
            self.clean_sampling_rate = h5f['sampling_rate']  # 采样率
            self.clean_dtype = h5f['dtype'] # 数据类型
            self.clean_num_channels = h5f['num_channels'] # 通道数
        # 打开HDF5文件并读取 噪声数据
        with h5py.File(noise_h5_file_path, 'r') as h5f:
            # 获取音频数据集
            if "audio_data" not in h5f:
                raise KeyError(f"Audio key 'audio_data' not found in HDF5 file")
            self.noise_data = h5f['audio_data']
            # 获取数据集基本信息
            self.noise_num_samples = h5f['num_files'] # 总样本数量    
            self.noise_audio_duration = h5f['duration'] # 每个音频段的持续时间（秒）
            self.noise_sampling_rate = h5f['sampling_rate']  # 采样率
            self.noisen_dtype = h5f['dtype'] # 数据类型
            self.noise_num_channels = h5f['num_channels'] # 通道数
        # 打开HDF5文件并读取 冲击响应数据
        with h5py.File(rir_h5_file_path, 'r') as h5f:
            # 获取音频数据集
            if "audio_data" not in h5f:
                raise KeyError(f"Audio key 'audio_data' not found in HDF5 file")
            self.rir_data = h5f['audio_data']
            # 获取数据集基本信息
            self.rir_num_samples = h5f['num_files'] # 总样本数量    
            self.rir_audio_duration = h5f['duration'] # 每个音频段的持续时间（秒）
            self.rir_sampling_rate = h5f['sampling_rate']  # 采样率
            self.rir_dtype = h5f['dtype'] # 数据类型
            self.rir_num_channels = h5f['num_channels'] # 通道数
        # 检查采样率是否一致
        if not (self.clean_sampling_rate == self.noise_sampling_rate == self.rir_sampling_rate == fs):
            raise ValueError("Sampling rates of clean, noise, and RIR data must match the specified fs")
        # 检查音频持续时间是否一致
        if not (self.clean_audio_duration == self.noise_audio_duration == length_in_seconds):
            raise ValueError("Audio durations of clean and noise data must match the specified length_in_seconds")
        self.L = int(length_in_seconds * fs)


    def __len__(self) -> int:
        """返回数据集中的样本数量"""
        if self.train:
            return self.num_data_per_epoch
        else:
            return len(self.clean_num_samples)
    
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
        conv_trim = _torch_convolve(clean, rir_trim)[:self.L]               

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
            'sampling rate': self.fs,
            'audio segment length in s': self.length_in_seconds,
            'target rir reserve in s': int(self.target_rir_length / self.fs),
            'num audio file (8s) per epoch': self.num_data_per_epoch,
            'is training now': self.train
        }
        print("Dataset info:\n", info_dict)
        return info_dict


if __name__=='__main__':
    from tqdm import tqdm 
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
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    train_dataloader.dataset.sample_data_per_epoch()

    validation_dataset = HaSimuDataset(**config['validation_dataset'])
    validation_dataloader = data.DataLoader(validation_dataset, **config['validation_dataloader'])

    print(len(train_dataloader), len(validation_dataloader))


    output_dir = WORK_DIR + "/prepare_datasets/train_data_samples"
    os.makedirs(output_dir, exist_ok=True)

    # 保存训练数据的音频
    for i, (noisy, clean) in enumerate(tqdm(train_dataloader, desc="Processing train data")):
        noisy_path = os.path.join(output_dir, f"train_noisy_{i}.wav")
        clean_path = os.path.join(output_dir, f"train_clean_{i}.wav")
        sf.write(noisy_path, noisy[0].numpy(), config['train_dataset']['fs'])
        sf.write(clean_path, clean[0].numpy(), config['train_dataset']['fs'])
        if i >= 15:  # 仅保存前10个样本
            break

    # # 保存验证数据的音频
    # for i, (noisy, clean) in enumerate(tqdm(validation_dataloader, desc="Processing validation data")):
    #     noisy_path = os.path.join(output_dir, f"val_noisy_{i}.wav")
    #     clean_path = os.path.join(output_dir, f"val_clean_{i}.wav")
    #     sf.write(noisy_path, noisy[0].numpy(), config['validation_dataset']['fs'])
    #     sf.write(clean_path, clean[0].numpy(), config['validation_dataset']['fs'])
    #     if i >= 9:  # 仅保存前10个样本
    #         break
