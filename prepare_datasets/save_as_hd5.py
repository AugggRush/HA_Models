import h5py
import pandas as pd
import os
import tqdm
import random
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import logging
import gc

# 配置日志，方便追踪进度和排查问题
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _rms(x):
    return np.sqrt(np.mean(x ** 2)) if x.size > 0 else 0.0

def pad_audio(data, target_length, min_length, mode='constant'):
    """将音频数据填充或截断到目标长度。"""
    current_length = len(data)
    if current_length == target_length:
        return data  # 已经是目标长度
    elif current_length <= min_length:
        return None  # 跳过短于等于min_length的文件
    elif current_length > target_length:
        # 截断：从中间或随机位置截取目标长度
        start = np.random.randint(0, current_length - target_length) # 随机起点
        # start = np.random.randint(0, current_length - target_length + 1) # 随机截断
        return data[start:start + target_length]
    else:
        # 填充：在末尾添加静音（零值）
        pad_width = target_length - current_length
        # 对于单声道 data.ndim == 1, 对于多声道需要指定axis
        if data.ndim == 1:
            padded_data = np.pad(data, (0, pad_width), mode=mode)
        else:
            padded_data = np.pad(data, ((0, pad_width), (0, 0)), mode=mode)

    return padded_data
    
def read_wav_file(file_path):
    """
    使用soundfile读取单个WAV文件。
    返回文件路径、音频数据和采样率。
    """
    try:
        data, samplerate = sf.read(file_path)
        return file_path, data, samplerate
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return file_path, None, None
# 将包装函数定义为模块级别的全局函数
def read_wav_file_wrapper(file_path):
    """
    用于多进程池的全局包装函数。
    简单地调用 read_wav_file 并返回其结果。
    """
    return read_wav_file(file_path)

def get_file_path_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df['file_dir'].tolist(), df['duration'].tolist()

def main(args):
    """
    主函数：扫描目录下的WAV文件，使用多进程读取，并存储到HDF5文件。
    
    参数:
        wav_dir: 包含WAV文件的目录路径。
        output_h5_path: 输出的HDF5文件路径。
        num_processes: 进程池大小，默认为CPU核心数。
    """
    # 在主函数中，创建数据集前确定目标长度
    print("Starting WAV to HDF5 conversion...")
    print(f"Configuration: \n {args}")
    csv_path = args["csv_path"]
    clip_duration = args["clip_duration"]
    output_h5_path = args["output_h5_path"]
    num_processes = args.get("num_processes", os.cpu_count())
    num_files_to_save = args.get("num_files_to_save", 100000)  # 默认保存100000个文件
    wav_files, durations = get_file_path_from_csv(csv_path)
    max_duration = max(durations)
    print(f"Max duration in CSV: {max_duration} seconds")
    total_files = len(wav_files)
    if total_files < num_files_to_save:
        num_files_to_save = len(wav_files)    
    if num_files_to_save == 0:
        logger.error("No WAV files found in the specified directory.")
        return
    logger.info(f"Found {total_files} WAV files.")
    wav_files = random.sample(wav_files, num_files_to_save)  # 打乱文件顺序，避免顺序偏差
    # 预读取一个文件以获取数据形状和采样率，用于创建HDF5数据集
    _, sample_data, sample_sr = read_wav_file(wav_files[0])
    if sample_data is None:
        logger.error("Failed to read the first WAV file, cannot determine data shape.")
        return
    if sample_sr != args["sample_sr"]:
        logger.warning(f"Sample rate mismatch: expected {args['sample_sr']}, got {sample_sr}")
        return
    # 确定每个文件的样本点数和通道数
    target_length = sample_sr * clip_duration  # 假设目标为8秒音频，采样率24000Hz
    if max_duration < clip_duration:
        logger.warning(f"Warning: max_duration {max_duration}s is less than clip_duration {clip_duration}s. \
                       Setting target_length to max_duration.")
        target_length = int(max_duration * sample_sr)
    samples_per_file = target_length
    num_channels = 1 if sample_data.ndim == 1 else sample_data.shape[1]
    dtype = sample_data.dtype

    # 2. 创建HDF5文件并初始化数据集
    with h5py.File(output_h5_path, 'w') as h5f:
        # 创建一个三维数据集: (文件索引, 样本点, 通道)
        # 如果文件是单声道，通道数为1，数据集为二维可能更高效，但三维通用性更好。
        if num_channels == 1:
            dataset_shape = (num_files_to_save, samples_per_file)  # 使用二维数组存储单声道数据
        else:
            dataset_shape = (num_files_to_save, samples_per_file, num_channels)  # 三维数组存储多声道数据

        dset = h5f.create_dataset(
            'audio_data',  # 数据集名称
            shape=dataset_shape,
            dtype=dtype,
            # *** 关键性能优化：分块存储 ***
            # 分块形状与每个文件的数据形状对齐，实现每个文件存储在一个独立的块中。
            chunks=(1,) + dataset_shape[1:],  # 例如 (1, samples_per_file, num_channels)
            compression='gzip',  # 启用压缩以节省存储空间 [1,2](@ref)
            compression_opts=6   # 压缩级别(0-9)，越高压缩比越大但速度越慢
        )

        # 存储采样率作为全局属性（假设所有文件采样率相同）
        dset.attrs['sampling_rate'] = sample_sr
        dset.attrs['duration'] = target_length / sample_sr  # 以秒为单位的持续时间
        dset.attrs['num_channels'] = num_channels
        dset.attrs['dtype'] = str(dtype)
        dset.attrs['num_files'] = num_files_to_save
        # 3. 使用多进程池读取WAV文件
        logger.info("Starting parallel WAV file reading...")
        success_count = 0
        zeros_count = 0
        skipped_count = 0
        # 关键优化：设置批次大小，避免一次性提交所有任务[4](@ref)
        batch_size = 1000  # 根据内存情况调整此值
        total_batches = (num_files_to_save + batch_size - 1) // batch_size
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # 使用分批处理替代一次性提交所有任务[1,4](@ref)
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_files)
                batch_files = wav_files[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_files)} files)")
                
                # 使用map方法进行批量处理，减少Future对象数量[1](@ref)
                batch_results = list(executor.map(read_wav_file, batch_files))
                
                # 处理当前批次的结果
                for i, (file_path, audio_data, sr) in enumerate(batch_results):
                    if audio_data is not None:
                        file_index = start_idx + i
                        
                        # 对音频数据进行填充处理
                        padded_audio = pad_audio(audio_data, target_length, min_length=(clip_duration-1)*sample_sr)             
                        if padded_audio is None:
                            skipped_count += 1
                            logger.warning(f"Skipped short file: {os.path.basename(file_path)}")
                            continue
                        # 计算RMS并检查有效性
                        rms = _rms(padded_audio)
                        if rms < 1e-5:
                            zeros_count += 1
                            logger.warning(f"Skipped low RMS file: {os.path.basename(file_path)} (RMS: {rms})")
                            continue
                        
                        # 写入HDF5数据集
                        if num_channels == 1:
                            dset[file_index, :] = padded_audio
                        else:
                            dset[file_index, :, :] = padded_audio

                        success_count += 1
                        logger.debug(f"Successfully processed: {os.path.basename(file_path)}")
                    else:
                        logger.warning(f"Skipped due to read error: {os.path.basename(file_path)}")
                
                # 每批处理完成后进行内存清理[4](@ref)
                del batch_results
                if batch_idx % 10 == 0:  # 每10个批次清理一次
                    gc.collect()
                    h5f.flush()  # 刷新HDF5缓存到磁盘
                    logger.info(f"Batch {batch_idx + 1} completed. Successfully processed {success_count} files so far.")
                    logger.info(f"Batch {batch_idx + 1} completed. zero rms processed {zeros_count} files so far.")
                    logger.info(f"Batch {batch_idx + 1} completed. too short processed {skipped_count} files so far.")
                
                # 测试用提前终止条件
                if success_count >= 100000:
                    logger.info("Reached 100,000 successfully processed files, stopping early for testing.")
                    break

        logger.info(f"All files processed. HDF5 file saved to: {output_h5_path}")


def read_hdf5_and_save_wav_samples(hdf5_path, output_dir, num_samples=10):
    """
    从HDF5文件中读取数据，随机选择指定数量的样本保存为WAV文件，并检查数据有效性。
    
    参数:
        hdf5_path: HDF5文件路径
        output_dir: WAV文件输出目录
        num_samples: 要随机选择的样本数量
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取HDF5文件
    with h5py.File(hdf5_path, 'r') as h5f:
        # 获取所有音频数据集的名称
        zero_count = 0
        for i in range(num_samples):
            if 'audio_data' in h5f:
                audio_group = h5f['audio_data']
            # 2. 随机选择指定数量的样本
            if audio_group.shape[0] < num_samples:
                print(f"警告: 文件只有 {audio_group.shape[0]} 个样本，将选择所有样本")
            else:
                # 生成一个随机的行索引
                random_index = np.random.choice(audio_group.shape[0])
                selected_datasets = audio_group[random_index]
            # 3. 处理每个选中的样本
            try:
                # 获取数据集
                dataset = selected_datasets
                rms = np.sqrt(np.mean(dataset ** 2)) if dataset.size > 0 else 0.0
                if rms < 1e-5:
                    zero_count += 1
                    print(f"样本 {random_index+1:03d} 的RMS值过低，可能是静音或无效数据 (RMS: {rms})")
                # 4. 保存为WAV文件
                output_filename = f"sample_{random_index+1:03d}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                sf.write(output_path, dataset, audio_group.attrs['sampling_rate'])
                print(f"已保存: {output_filename} (采样率: {audio_group.attrs['sampling_rate']}Hz, 长度: {len(dataset)}样本)")
            
            except Exception as e:
                print(f"处理样本 {random_index+1:03d} 时出错: {e}")
        print(f"总共发现 {zero_count} 个RMS值过低的样本")
if __name__ == "__main__":
    # 示例用法
    # config_dict = {
    #     "csv_path": "val_clean_24k.csv",  # 包含WAV文件路径的CSV文件
    #     "clip_duration": 8,           # 每个剪辑的目标持续时间（秒）
    #     "sample_sr": 24000,           # 目标采样率
    #     "output_h5_path": "val_clean_24k_checkzero.h5",  # 输出HDF5文件路径
    #     "num_processes": 8,           # 使用的进程数
    #     "num_files_to_save": 10000   # 要保存的文件数量
    # }
    # main(args=config_dict)

    # 配置参数
    hdf5_file_path = "val_clean_24k_checkzero.h5"  # 替换为您的HDF5文件路径
    output_directory = "extracted_wav_samples"  # WAV文件输出目录
    num_samples_to_extract = 200  # 要提取的样本数量
    # 执行主要功能
    read_hdf5_and_save_wav_samples(hdf5_file_path, output_directory, num_samples_to_extract)
    
    # print(f"\n所有操作完成。提取的WAV文件保存在: {output_directory}")    