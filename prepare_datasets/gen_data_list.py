# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
from concurrent.futures import as_completed

# 获取所有WAV文件路径（多线程处理）
from concurrent.futures import ThreadPoolExecutor

def find_wav_to_csv():
    # 输入和输出路径
    input_dir = "/minioData/alice/nn_data/train_data/纯净语音数据_2000h/mtfaa_mono_denoise_wav_aishu"  # 指定输入目录
    output_dir = "/minioData/goodman/train_data/clean_speech_24k"
    output_csv = "/data/goodman/torch_nn_train/SEtrain/prepare_datasets/train_clean_24k.csv"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    def find_wav_files(root_dir):
        wav_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".wav"):
                    wav_files.append(os.path.join(root, file))
        return wav_files

    # 使用多线程遍历目录
    all_wav_files = []
    num_threads = 8  # 根据需求调整线程数量
    sub_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务
        future_to_dir = {executor.submit(find_wav_files, sub_dir): sub_dir for sub_dir in sub_dirs}
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(future_to_dir), total=len(future_to_dir), desc="扫描目录"):
            wav_files = future.result()
            all_wav_files.extend(wav_files)

    print(f"找到 {len(all_wav_files)} 个WAV文件.")

    # 定义处理单个文件的函数
    def process_file(input_path):
        if not os.path.isfile(input_path):
            return None, f"文件未找到: {input_path}"

        try:
            # 读取WAV文件
            data, sample_rate = librosa.load(input_path, sr=None)  # 保留原始采样率
            duration = len(data) / sample_rate  # 计算时长（秒）

            # 如果采样率为48K，则下采样为24K
            if sample_rate == 48000:
                new_sample_rate = 24000
                data = librosa.resample(data, orig_sr=sample_rate, target_sr=new_sample_rate)
            else:
                new_sample_rate = sample_rate

            # 生成输出路径，保留子目录结构
            relative_path = os.path.relpath(input_path, start=input_dir)
            output_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存下采样后的WAV文件
            sf.write(output_path, data, new_sample_rate)

            # 返回结果
            return {"file_dir": os.path.abspath(output_path), "duration": duration}, None

        except Exception as e:
            return None, f"处理文件时出错: {input_path}, 错误: {e}"

    # 使用多线程处理文件
    results = []
    errors = []
    # 设置线程数量
    num_threads = 8  # 根据需求调整线程数量
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务
        future_to_file = {executor.submit(process_file, wav_file): wav_file for wav_file in wav_files}
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="处理进度"):
            result, error = future.result()
            if result:
                results.append(result)
            if error:
                errors.append(error)

    # 保存结果到CSV
    result_df = pd.DataFrame(results, columns=["file_dir", "duration"])  # 指定列名
    result_df.to_csv(output_csv, index=False)  # 保存为CSV文件，不包含索引
    print(f"处理完成，结果已保存到: {output_csv}")

    # 打印错误信息
    if errors:
        print("以下文件处理失败:")
        for error in errors:
            print(error)

def pick_up_validation_csv():
    input_csv = "/data/goodman/torch_nn_train/SEtrain/prepare_datasets/train_rir_24k.csv"
    output_csv = "/data/goodman/torch_nn_train/SEtrain/prepare_datasets/val_rir_24k.csv"

    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 按照5%的比例随机抽取验证集
    val_df = df.sample(frac=0.05, random_state=42)  # 设置随机种子以确保可重复性

    # 将剩余数据保存回训练集CSV
    train_df = df.drop(val_df.index)  # 删除验证集对应的行
    train_df.to_csv(input_csv, index=False)  # 保存更新后的训练集CSV

    # 保存验证集到新的CSV文件
    val_df.to_csv(output_csv, index=False)

    print(f"验证集已保存到: {output_csv}, 包含 {len(val_df)} 条记录.")
    print(f"更新后的训练集已保存到: {input_csv}, 包含 {len(train_df)} 条记录.")


def find_rir_to_csv():
    # 输入和输出路径
    input_dir = "/data/alice/rir/A_record/rir24k"  # 指定输入目录
    output_dir = "/minioData/goodman/train_data/record_rir_24k/"
    output_csv = "/data/goodman/torch_nn_train/SEtrain/prepare_datasets/train_rir_24k.csv"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有WAV文件路径（多线程处理）
    from concurrent.futures import ThreadPoolExecutor

    def find_wav_files(root_dir):
        wav_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".wav"):
                    wav_files.append(os.path.join(root, file))
        return wav_files

    # 使用多线程遍历目录
    all_wav_files = []
    num_threads = 8  # 根据需求调整线程数量
    sub_dirs = [input_dir]
    for d in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, d)):
            sub_dirs.append(os.path.join(input_dir, d))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务
        future_to_dir = {executor.submit(find_wav_files, sub_dir): sub_dir for sub_dir in sub_dirs}
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(future_to_dir), total=len(future_to_dir), desc="扫描目录"):
            wav_files = future.result()
            all_wav_files.extend(wav_files)

    print(f"找到 {len(all_wav_files)} 个WAV文件.")

    # 定义处理单个文件的函数
    def process_file(input_path):
        if not os.path.isfile(input_path):
            return None, f"文件未找到: {input_path}"

        try:
            # 使用 soundfile 读取，保证多通道不被降为单声道
            data, sample_rate = sf.read(input_path, always_2d=True)  # data shape: (frames, channels)
            frames = data.shape[0] # 右耳助听麦2
            duration = frames / sample_rate  # 计算时长（秒）

            # # 如果原采样率为16000，则重采样为24000（保持原逻辑）
            # if sample_rate == 48000:
            #     new_sample_rate = 24000
            #     # 对每个通道单独重采样，然后按列堆叠回多通道数组
            #     if data.shape[1] == 1:
            #         resampled = librosa.resample(data[:, 0], orig_sr=sample_rate, target_sr=new_sample_rate)
            #         data_out = resampled  # 1D
            #     else:
            #         channels = []
            #         for ch in range(data.shape[1]):
            #             ch_res = librosa.resample(data[:, ch], orig_sr=sample_rate, target_sr=new_sample_rate)
            #             channels.append(ch_res)
            #         # 将每个通道堆叠为 (frames, channels)
            #         data_out = np.stack(channels, axis=1)
            # else:
            #     new_sample_rate = sample_rate
            #     # 如果只有一个通道，保持为 1D，否则保持 (frames, channels)
            #     data_out = data if data.shape[1] > 1 else data[:, 0]

            # # 生成输出路径，保留子目录结构
            # relative_path = os.path.relpath(input_path, start=input_dir)
            # output_path = os.path.join(output_dir, relative_path)
            # os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # # 保存（sf.write 支持多通道数组）
            # sf.write(output_path, data_out, new_sample_rate)

            # 返回结果
            return {"file_dir": os.path.abspath(input_path), "duration": duration}, None

        except Exception as e:
            return None, f"处理文件时出错: {input_path}, 错误: {e}"

    # 使用多线程处理文件
    results = []
    errors = []
    # 设置线程数量
    num_threads = 8  # 根据需求调整线程数量
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务（修复变量名，使用 all_wav_files）
        future_to_file = {executor.submit(process_file, wav_file): wav_file for wav_file in all_wav_files}
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="处理进度"):
            result, error = future.result()
            if result:
                results.append(result)
            if error:
                errors.append(error)

    # 保存结果到CSV
    result_df = pd.DataFrame(results, columns=["file_dir", "duration"])  # 指定列名
    result_df.to_csv(output_csv, index=False)  # 保存为CSV文件，不包含索引
    print(f"处理完成，结果已保存到: {output_csv}")

    # 打印错误信息
    if errors:
        print("以下文件处理失败:")
        for error in errors:
            print(error)

def clip_wav_to_csv(input_dir, output_dir, output_csv):
    """
    读取WAV文件，转换采样率到24k，切分为10s片段，并生成CSV文件
    """
    clip_duration = 10  # 切片时长（秒）
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    def find_wav_files(root_dir):
        wav_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".wav"):
                    wav_files.append(os.path.join(root, file))
        return wav_files

    def process_file(input_path):
        if not os.path.isfile(input_path):
            return None, f"文件未找到: {input_path}"

        try:
            # 读取WAV文件
            data, sample_rate = librosa.load(input_path, sr=None)
            
            # 转换采样率到24k
            if sample_rate != 24000:
                data = librosa.resample(data, orig_sr=sample_rate, target_sr=24000)
                sample_rate = 24000

            # 计算每个片段的采样点数
            samples_per_clip = clip_duration * sample_rate
            
            # 切分音频
            clips = []
            for i in range(0, len(data), samples_per_clip):
                clip = data[i:i + samples_per_clip]
                
                # 只保存长度等于目标长度的片段
                if len(clip) == samples_per_clip:
                    # 生成输出文件名
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    clip_name = f"{base_name}_clip_{i//samples_per_clip}.wav"
                    clip_path = os.path.join(output_dir, clip_name)
                    
                    # 保存片段
                    sf.write(clip_path, clip, sample_rate)
                    clips.append({
                        "file_dir": os.path.abspath(clip_path),
                        "duration": clip_duration
                    })

            return clips, None

        except Exception as e:
            return None, f"处理文件时出错: {input_path}, 错误: {e}"

    # 获取所有WAV文件
    print("正在扫描WAV文件...")
    wav_files = find_wav_files(input_dir)
    print(f"找到 {len(wav_files)} 个WAV文件")

    # 使用多线程处理文件
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_file = {executor.submit(process_file, wav_file): wav_file for wav_file in wav_files}
        for future in tqdm(as_completed(future_to_file), total=len(wav_files), desc="处理进度"):
            result, error = future.result()
            if result:
                results.extend(result)
            if error:
                errors.append(error)

    # 保存结果到CSV
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_csv, index=False, mode='a')
        print(f"处理完成，结果已保存到: {output_csv}")
        print(f"总共生成了 {len(results)} 个音频片段")

    # 打印错误信息
    if errors:
        print("\n处理过程中出现以下错误:")
        for error in errors:
            print(error)

def split_train_validation():
    """
    从train_noise_24k_new.csv中随机选取五分之一的数据作为验证集
    """
    # 文件路径
    train_csv = "./train_noise_24k_new.csv"
    val_csv = "./val_noise_24k_new.csv"
    
    # 读取训练集CSV文件
    df = pd.read_csv(train_csv)
    
    # 计算要抽取的样本数量(五分之一)
    val_size = len(df) // 5
    
    # 随机抽取验证集数据
    val_df = df.sample(n=val_size, random_state=42)  # 设置随机种子以确保可重复性
    
    # 从训练集中删除验证集数据
    train_df = df.drop(val_df.index)
    
    # 保存更新后的训练集
    train_df.to_csv(train_csv, index=False)
    
    # 保存验证集
    val_df.to_csv(val_csv, index=False)
    
    print(f"验证集已保存到: {val_csv}, 包含 {len(val_df)} 条记录")
    print(f"更新后的训练集已保存到: {train_csv}, 包含 {len(train_df)} 条记录")


if __name__ == "__main__":
    # pick_up_validation_csv()
    split_train_validation()