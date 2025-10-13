import os
import wave
import contextlib
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm  # 从模块中导入tqdm类
def get_all_wav_files(root_dir):
    """
    获取文件夹及其子文件夹中所有WAV文件的绝对路径
    """
    wav_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                abs_path = os.path.abspath(os.path.join(root, file))
                wav_files.append(abs_path)
    return wav_files

def get_wav_duration(file_path):
    """
    获取WAV文件的时长（秒）- 线程安全版本
    """
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"无法读取文件时长 {file_path}: {e}")
        return 0

def process_wav_file_batch(file_paths, train_files_set, results_lock, validation_data):
    """
    批量处理WAV文件 - 每个线程处理一批文件[6,8](@ref)
    """
    batch_results = []
    
    for file_path in file_paths:
        # 检查是否在训练数据中
        if file_path not in train_files_set:
            duration = get_wav_duration(file_path)
            batch_results.append({
                'file_dir': file_path,
                'duration': duration
            })
    
    # 使用锁安全地更新共享结果列表[8](@ref)
    with results_lock:
        validation_data.extend(batch_results)

def process_single_wav_file(args):
    """
    处理单个WAV文件 - 用于ThreadPoolExecutor.map[6](@ref)
    """
    file_path, train_files_set = args
    if file_path not in train_files_set:
        duration = get_wav_duration(file_path)
        return {'file_dir': file_path, 'duration': duration}
    return None

def main():
    # 配置参数
    root_dir = input("请输入root dir: ").strip().strip('"')
    train_csv_path = "./prepare_datasets/train_noise_24k.csv"
    valid_csv_path = "./prepare_datasets/valid_noise_24k.csv"
    
    # 线程配置
    max_workers = int(input("请输入线程数 (推荐4-16，根据CPU核心数调整): ") or 8)
    
    # 检查根目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误：目录 '{root_dir}' 不存在")
        return
    
    # 获取所有WAV文件
    print("正在扫描WAV文件...")

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
    sub_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务
        future_to_dir = {executor.submit(find_wav_files, sub_dir): sub_dir for sub_dir in sub_dirs}
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(future_to_dir), total=len(future_to_dir), desc="扫描目录"):
            wav_files = future.result()
            all_wav_files.extend(wav_files)

    print(f"找到 {len(all_wav_files)} 个WAV文件.")

    print(f"找到 {len(all_wav_files)} 个WAV文件")
    
    if not all_wav_files:
        print("未找到WAV文件，程序退出")
        return
    
    # 读取训练数据CSV文件（如果存在）
    train_files = set()
    if os.path.exists(train_csv_path):
        try:
            train_df = pd.read_csv(train_csv_path)
            if not train_df.empty and len(train_df.columns) > 0:
                # 获取第一列的所有文件路径并标准化
                train_files = set(train_df.iloc[:, 0].astype(str).str.strip())
                # 统一路径格式，确保比较准确
                train_files = {os.path.abspath(path) for path in train_files if os.path.exists(path)}
                print(f"从 {train_csv_path} 中读取 {len(train_files)} 个有效训练文件路径")
        except Exception as e:
            print(f"读取 {train_csv_path} 时出错: {e}")
            return
    else:
        print(f"警告：{train_csv_path} 不存在，将把所有WAV文件视为验证文件")
    
    # 方法1: 使用ThreadPoolExecutor并行处理（推荐）[6,8](@ref)
    print(f"使用多线程处理（{max_workers}个线程）...")
    
    validation_data = []
    processed_count = 0
    total_files = len(all_wav_files)
    
    # 准备参数列表
    process_args = [(file_path, train_files) for file_path in all_wav_files]
    
    # 使用ThreadPoolExecutor并行处理[6](@ref)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_wav_file, args): args[0] 
            for args in process_args
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    validation_data.append(result)
                
                processed_count += 1
                if processed_count % 100 == 0 or processed_count == total_files:
                    print(f"处理进度: {processed_count}/{total_files} ({processed_count/total_files*100:.1f}%)")
                    
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                processed_count += 1
    
    # 方法2: 批量处理版本（适合极大文件数量，减少线程创建开销）
    def process_with_batching():
        """批量处理版本，减少线程间同步开销[6](@ref)"""
        validation_data_batch = []
        results_lock = threading.Lock()
        
        # 将文件列表分成批次
        batch_size = max(1, len(all_wav_files) // (max_workers * 2))
        batches = [all_wav_files[i:i + batch_size] for i in range(0, len(all_wav_files), batch_size)]
        
        print(f"将{len(all_wav_files)}个文件分成{len(batches)}个批次进行处理")
        
        threads = []
        for i, batch in enumerate(batches):
            thread = threading.Thread(
                target=process_wav_file_batch,
                args=(batch, train_files, results_lock, validation_data_batch)
            )
            threads.append(thread)
            thread.start()
            
            # 限制同时运行的线程数量
            if len(threads) >= max_workers:
                for t in threads:
                    t.join()
                threads = []
                print(f"已完成批次 {i+1}/{len(batches)}")
        
        # 等待剩余线程完成
        for thread in threads:
            thread.join()
            
        return validation_data_batch
    
    # 根据文件数量选择处理方法
    if len(all_wav_files) > 1000:
        print("文件数量较多，使用批量处理方法...")
        validation_data = process_with_batching()
    
    print(f"找到 {len(validation_data)} 个验证文件")
    
    # 写入验证数据到CSV
    if validation_data:
        valid_df = pd.DataFrame(validation_data)
        valid_df.to_csv(valid_csv_path, index=False)
        print(f"验证数据已写入 {valid_csv_path}")
        
        # 显示统计信息
        total_duration = valid_df['duration'].sum()
        avg_duration = valid_df['duration'].mean()
        max_duration = valid_df['duration'].max()
        min_duration = valid_df['duration'].min()
        
        print(f"验证集统计信息:")
        print(f"  总文件数: {len(validation_data)}")
        print(f"  总时长: {total_duration:.2f} 秒 ({total_duration/3600:.2f} 小时)")
        print(f"  平均文件时长: {avg_duration:.2f} 秒")
        print(f"  最长文件: {max_duration:.2f} 秒")
        print(f"  最短文件: {min_duration:.2f} 秒")
        
        # 保存统计信息到文件
        stats_path = "validation_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("验证集统计信息\\n")
            f.write("=" * 50 + "\\n")
            f.write(f"总文件数: {len(validation_data)}\\n")
            f.write(f"总时长: {total_duration:.2f} 秒 ({total_duration/3600:.2f} 小时)\\n")
            f.write(f"平均文件时长: {avg_duration:.2f} 秒\\n")
            f.write(f"最长文件: {max_duration:.2f} 秒\\n")
            f.write(f"最短文件: {min_duration:.2f} 秒\\n")
        print(f"统计信息已保存到 {stats_path}")
    else:
        print("没有找到需要处理的验证文件")

if __name__ == "__main__":
    main()