import os
import torch
import lmdb
import zlib
import random
import pickle
import time
import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import torch.utils
import torch.utils.data
import pytorch_lightning as pl
from tqdm import tqdm
import soundfile as sf
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.assigned_snr_db = None  # 新增：允许外部强制指定一个固定的 SNR 值
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
            # # 前五十毫秒的 RIR作为target
            # rir_trim = rir_sig[:self.target_rir_length]  # 裁剪或补零到目标长度       
            # # 卷积并取前 self.L
            # conv_trim = np.convolve(clean, rir_trim)[:self.L]
            # # 若 conv 长度不足 self.L，则补零
            # if len(conv_trim) < self.L:
            #     tmp = np.zeros(self.L, dtype=np.float32)
            #     tmp[:len(conv_trim)] = conv_trim
            #     conv_trim = tmp
            # clean_conv_trim = conv_trim.astype(np.float32)                 
        else:
            clean_conv = clean.astype(np.float32)
            # clean_conv_trim = clean.astype(np.float32)

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
            if self.assigned_snr_db != None:
                self.snr_db = self.assigned_snr_db
            else: self.snr_db = int(random.uniform(self.snr_db_min, self.snr_db_max))
            # 要使 SNR = 20*log10(rms_clean / rms_noise_scaled)
            # 则 rms_noise_scaled = rms_clean / (10^(SNR/20))
            target_linear = 10.0 ** (-self.snr_db / 20.0)
            scale = (rms_clean / rms_noise) * target_linear
            noise_scaled = noise_sig * scale
            noisy = clean_conv + noise_scaled

        # 防止 NaN / inf
        noisy = np.nan_to_num(noisy).astype(np.float32)
        clean_out = np.nan_to_num(clean_conv).astype(np.float32)
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
        return torch.from_numpy(noisy.copy()).float(), torch.from_numpy(clean_out.copy()).float(), self.snr_db

    def __len__(self):
        if self.train:
            return self.num_data_per_epoch
        else:
            if self.num_data_per_epoch < len(self.speech_database_valid):
                return self.num_data_per_epoch
            else: return len(self.speech_database_valid)

    def make_wavs_noisy(self, idx, snr):
        """将验证集中的干净语音转换为带噪语音并保存到指定目录"""
        self.assigned_snr_db = snr  # 强制使用指定的 SNR
        if idx < len(self):
            noisy, clean, snr_db = self.__getitem__(idx)
            noisy_np = noisy.numpy()
            clean_np = clean.numpy()
            noise_np =noisy_np - clean_np
        # 准备数据字典
        sample_data = {
            'snr': snr_db,
            'noisy': noisy_np,
            'clean': clean_np,
            'noise': noise_np,
            'index': idx
        }                
        return sample_data

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
            snr = sample_data.get('snr', -100)
                  
        return noisy, clean, snr

    def make_wavs_noisy(self, idx, snr):
        """将验证集中的干净语音转换为带噪语音并保存到指定目录"""
        self.assigned_snr_db = snr  # 强制使用指定的 SNR
        if idx < len(self):
            noisy, clean, snr_db = self.__getitem__(idx)
            noisy_np = noisy.numpy()
            clean_np = clean.numpy()
            noise_np =noisy_np - clean_np
        # 准备数据字典
        sample_data = {
            'snr': snr_db,
            'noisy': noisy_np,
            'clean': clean_np,
            'noise': noise_np,
            'index': idx
        }                
        return sample_data

class PlDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_src_dir,
        val_src_dir,
        batch_size, 
        num_workers,
        max_reader=512,
    ):
        super().__init__()
        self.train_src_dir = train_src_dir
        self.val_src_dir = val_src_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_reader = max_reader

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = HaDataSetsFromLMDB(self.train_src_dir, self.max_reader)
            self.val_dataset = HaDataSetsFromLMDB(self.val_src_dir, self.max_reader)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
    
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
  
    # train_dataset = HaSimuDataset(**config['train_dataset'])
    # train_dataset.sample_data_per_epoch()
    # # 创建转换器并执行转换
    # converter = HaSimuDatasetToLMDB(train_dataset, './prepare_datasets/training_audio_tau_24k_noDereverb.lmdb', 4)
    # converter.convert_to_lmdb()

    valid_dataset = HaSimuDataset(**config['validation_dataset'])
    # 创建转换器并执行转换
    converter = HaSimuDatasetToLMDB(valid_dataset, './prepare_datasets/validation_audio_tau_24k_noDereverb.lmdb', 4)
    converter.convert_to_lmdb()

    # 输出目录
    # output_dir = "/minioData/goodman/train_data/ha_lmdb/valid_demo_noReverb/"
    # os.makedirs(output_dir, exist_ok=True)

    # # 从 LMDB 数据集读取
    # datasets = HaDataSetsFromLMDB('./prepare_datasets/validation_audio_24k_noDereverb.lmdb', max_reader=512)

    # # 创建子目录
    # audio_types = ['noisy', 'noise', 'clean']
    # for audio_type in audio_types:
    #     os.makedirs(os.path.join(output_dir, audio_type), exist_ok=True)

    # # 随机抽取 100 个样本（不足则放回抽样）
    # num_to_save = 100
    # total = len(datasets)
    # replace = total < num_to_save
    # indices = np.random.choice(total, size=num_to_save, replace=replace)

    # sample_rate = getattr(datasets, 'sample_rate', 24000)
    # saved = 0
    # for idx in indices:
    #     try:
    #         noisy, clean, snr = datasets[int(idx)]

    #         # 转 numpy
    #         if isinstance(noisy, torch.Tensor):
    #             noisy_np = noisy.cpu().numpy()
    #         else:
    #             noisy_np = np.asarray(noisy)
    #         if isinstance(clean, torch.Tensor):
    #             clean_np = clean.cpu().numpy()
    #         else:
    #             clean_np = np.asarray(clean)

    #         # 计算 noise
    #         noise_np = noisy_np - clean_np

    #         # 处理 snr，四舍五入
    #         snr_val = 0.0 if snr is None else float(snr)
    #         snr_round = int(round(snr_val))

    #         prefix = f"sample_{int(idx):05d}_snr_{snr_round}"
    #         sf.write(os.path.join(output_dir, 'noisy', prefix + "_noisy.wav"), noisy_np, sample_rate)
    #         sf.write(os.path.join(output_dir, 'clean', prefix + "_clean.wav"), clean_np, sample_rate)
    #         sf.write(os.path.join(output_dir, 'noise', prefix + "_noise.wav"), noise_np, sample_rate)

    #         saved += 1
    #     except Exception as e:
    #         print(f"保存样本 idx={idx} 失败: {e}")

    # print(f"完成：已保存 {saved}/{num_to_save} 个样本到 {output_dir}")