import lmdb
import pandas as pd
import os
import random
import soundfile as sf
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import logging
import gc
import json
import pickle
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _rms(x):
    return np.sqrt(np.mean(x ** 2)) if x.size > 0 else 0.0

def pad_audio(data, target_length, min_length, mode='constant'):
    """将音频数据填充或截断到目标长度"""
    current_length = len(data)
    if current_length == target_length:
        return data
    elif current_length <= min_length:
        return None
    elif current_length > target_length:
        start = np.random.randint(0, current_length - target_length)
        return data[start:start + target_length]
    else:
        pad_width = target_length - current_length
        if data.ndim == 1:
            padded_data = np.pad(data, (0, pad_width), mode=mode)
        else:
            padded_data = np.pad(data, ((0, pad_width), (0, 0)), mode=mode)
        return padded_data

def read_wav_file(file_path):
    """读取单个WAV文件"""
    try:
        data, samplerate = sf.read(file_path)
        return file_path, data, samplerate
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return file_path, None, None

def get_file_path_from_csv(csv_file):
    """从CSV文件获取文件路径列表"""
    df = pd.read_csv(csv_file)
    return df['file_dir'].tolist(), df['duration'].tolist()

def serialize_audio_data(audio_data):
    """序列化音频数据和元数据"""
    data_dict = {
        'audio_data': audio_data.astype(np.float32)  # 统一数据类型
    }
    return pickle.dumps(data_dict)

def main(args):
    """
    主函数：将WAV文件转换为LMDB数据库
    
    参数:
        csv_path: 包含WAV文件路径的CSV文件
        clip_duration: 每个音频剪辑的持续时间（秒）
        sample_sr: 目标采样率
        output_lmdb_path: 输出的LMDB文件路径
        num_processes: 进程池大小
        num_files_to_save: 要保存的文件数量
    """
    print("Starting WAV to LMDB conversion...")
    print(f"Configuration: \n{args}")
    
    # 解析参数
    csv_path = args["csv_path"]
    clip_duration = args["clip_duration"]
    output_lmdb_path = args["output_lmdb_path"]
    num_processes = args.get("num_processes", os.cpu_count())
    num_files_to_save = args.get("num_files_to_save", 100000)
    
    # 获取文件列表
    wav_files, durations = get_file_path_from_csv(csv_path)
    max_duration = max(durations)
    print(f"Max duration in CSV: {max_duration} seconds")
    
    total_files = len(wav_files)
    if total_files < num_files_to_save:
        num_files_to_save = total_files
    
    if num_files_to_save == 0:
        logger.error("No WAV files found.")
        return
    
    logger.info(f"Found {total_files} WAV files, will process {num_files_to_save} files.")
    wav_files = random.sample(wav_files, num_files_to_save)
    
    # 预读取样本文件获取参数
    _, sample_data, sample_sr = read_wav_file(wav_files[0])
    if sample_data is None:
        logger.error("Failed to read sample file.")
        return
    
    if sample_sr != args["sample_sr"]:
        logger.warning(f"Sample rate mismatch: expected {args['sample_sr']}, got {sample_sr}")
        return
    
    # 计算目标长度和参数
    target_length = sample_sr * clip_duration
    if max_duration < clip_duration:
        logger.warning(f"Max duration {max_duration}s < clip_duration {clip_duration}s")
        target_length = int(max_duration * sample_sr)
    
    num_channels = 1 if sample_data.ndim == 1 else sample_data.shape[1]
    dtype = sample_data.dtype
    
    # 估算LMDB所需空间（平均文件大小 × 文件数量 × 安全系数）
    avg_file_size = target_length * num_channels * 4  # 假设float32
    map_size = avg_file_size * num_files_to_save * 3  # 安全系数为3
    
    # 创建LMDB环境
    # os.makedirs(os.path.dirname(output_lmdb_path), exist_ok=True)
    
    with lmdb.open(output_lmdb_path, map_size=map_size, max_readers=num_processes*2) as env:
        # 存储全局元数据
        global_metadata = {
            'sampling_rate': sample_sr,
            'duration': target_length / sample_sr,
            'num_channels': num_channels,
            'dtype': str(dtype),
            'num_files': num_files_to_save,
            'target_length': target_length
        }
        
        with env.begin(write=True) as txn:
            txn.put(b'__global_metadata__', pickle.dumps(global_metadata))
        
        # 多进程处理
        logger.info("Starting parallel WAV file processing...")
        success_count = 0
        zeros_count = 0
        skipped_count = 0
        
        batch_size = 1000
        total_batches = (num_files_to_save + batch_size - 1) // batch_size
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_files_to_save)
                batch_files = wav_files[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_files)} files)")
                
                batch_results = list(executor.map(read_wav_file, batch_files))
                
                # 批量写入LMDB
                with env.begin(write=True) as txn:
                    for i, (file_path, audio_data, sr) in enumerate(batch_results):
                        if audio_data is not None:
                            file_index = start_idx + i
                            
                            # 音频数据处理
                            padded_audio = pad_audio(audio_data, target_length, 
                                                    min_length=(clip_duration-1)*sample_sr)
                            if padded_audio is None:
                                skipped_count += 1
                                continue
                            
                            # RMS检测
                            rms = _rms(padded_audio)
                            if rms < 1e-5:
                                zeros_count += 1
                                continue
                            
                            # 序列化并存储
                            key = f"{file_index:08d}".encode()
                            value = serialize_audio_data(padded_audio)
                            txn.put(key, value)
                            
                            success_count += 1
                        else:
                            logger.warning(f"Read error: {os.path.basename(file_path)}")
                
                # 清理和进度报告
                del batch_results
                if batch_idx % 10 == 0:
                    gc.collect()
                    logger.info(f"Batch {batch_idx + 1}: Success={success_count}, Zeros={zeros_count}, Skipped={skipped_count}")
                
                if success_count >= 100000:
                    logger.info("Reached 100,000 files, stopping early.")
                    break
        
        logger.info(f"Conversion completed: {success_count} files saved to {output_lmdb_path}")

def read_lmdb_and_save_wav_samples(lmdb_path, output_dir, num_samples=10):
    """
    从LMDB读取数据并保存样本WAV文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with lmdb.open(lmdb_path, readonly=True) as env:
        # 读取全局元数据
        with env.begin() as txn:
            global_meta_bytes = txn.get(b'__global_metadata__')
            if global_meta_bytes:
                global_metadata = pickle.loads(global_meta_bytes)
                sampling_rate = global_metadata['sampling_rate']
        
        zero_count = 0
        with env.begin() as txn:
            cursor = txn.cursor()
            total_samples = sum(1 for _ in cursor) - 1  # 减去元数据键
            
            samples_to_check = min(num_samples, total_samples)
            indices = random.sample(range(total_samples), samples_to_check)
            
            for idx in indices:
                key = f"{idx:08d}".encode()
                value = txn.get(key)
                
                if value:
                    data_dict = pickle.loads(value)
                    audio_data = data_dict['audio_data']
                    
                    # RMS检测
                    rms = _rms(audio_data)
                    if rms < 1e-5:
                        zero_count += 1
                        print(f"Sample {idx}: Low RMS ({rms})")
                    
                    # 保存WAV文件
                    output_path = os.path.join(output_dir, f"sample_{idx:06d}.wav")
                    sf.write(output_path, audio_data, sampling_rate)
                    print(f"Saved sample {idx}: RMS={rms:.6f}")
                else:
                    print(f"Sample {idx}: Not found in LMDB")
        print(f"Found {zero_count} samples with low RMS out of {samples_to_check} checked")

if __name__ == "__main__":
    # config_dict = {
    #     "csv_path": "val_rir_24k.csv",
    #     "clip_duration": 8,
    #     "sample_sr": 24000,
    #     "output_lmdb_path": "val_rir_24k.lmdb",  # LMDB路径
    #     "num_processes": 8,
    #     "num_files_to_save": 2000
    # }
    
    # main(args=config_dict)
    
    # 测试读取
    read_lmdb_and_save_wav_samples("val_clean_24k.lmdb", "extracted_samples", 200)