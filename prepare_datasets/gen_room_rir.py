import os
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

class RIRGenerator:
    def __init__(self, base_save_path="./rir_results", fs=24000):
        self.fs = fs
        self.base_save_path = Path(base_save_path)
        self.base_save_path.mkdir(parents=True, exist_ok=True)
        
        # 定义房间大小
        self.room_sizes = {
            'small': [4.0, 4.0, 3.0],
            'medium': [6.0, 5.0, 3.5],
            'large': [8.0, 7.0, 4.0]
        }
        
        # 混响时间列表
        self.rt60_list = [0.1, 0.3, 0.5, 0.7]
        
        # CSV文件锁
        self.csv_lock = threading.Lock()
        
        # 初始化CSV文件
        self.csv_file = self.base_save_path / "rir_database.csv"
        if not self.csv_file.exists():
            df_header = pd.DataFrame(columns=[
                'rir_path', 'diffuse_rir_path', 'rt60', 'distance'
            ])
            df_header.to_csv(self.csv_file, index=False)

    def generate_mic_position(self, room_dim):
        """生成随机麦克风位置，距离墙面至少1米"""
        L, W, H = room_dim
        x = np.random.uniform(1.0, L-1.0)
        y = np.random.uniform(1.0, W-1.0)
        z = np.random.uniform(1.0, H-1.0)
        return [x, y, z]

    def generate_distance(self):
        """生成声源与麦克风距离，满足1-2米占80%的要求"""
        p = np.random.random()
        if p < 0.8:
            return np.random.uniform(1.0, 2.0)
        else:
            if p < 0.9:
                return np.random.uniform(0.5, 1.0)
            else:
                return np.random.uniform(2.0, 3.0)

    def generate_source_position(self, mic_pos, room_dim, distance):
        """生成满足距离要求的声源位置"""
        L, W, H = room_dim
        mic_pos = np.array(mic_pos)
        
        max_attempts = 1000
        for attempt in range(max_attempts):
            # 随机方向向量
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            src_pos = mic_pos + direction * distance
            
            # 检查是否在有效区域内（距离墙面至少0.5米）
            if (0.5 <= src_pos[0] <= L-0.5 and 
                0.5 <= src_pos[1] <= W-0.5 and 
                0.5 <= src_pos[2] <= H-0.5):
                return src_pos.tolist()
        
        # 失败时使用边界值
        src_pos[0] = np.clip(src_pos[0], 0.5, L-0.5)
        src_pos[1] = np.clip(src_pos[1], 0.5, W-0.5)
        src_pos[2] = np.clip(src_pos[2], 0.5, H-0.5)
        return src_pos.tolist()

    def generate_single_rir(self, room_size_name, rt60, rir_index):
        """生成单条RIR（线程安全）"""
        room_dim = self.room_sizes[room_size_name]
        
        # 计算房间材料参数
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        
        # 创建房间
        room = pra.ShoeBox(
            room_dim, 
            fs=self.fs, 
            materials=pra.Material(e_absorption), 
            max_order=max_order
        )
        
        # 生成位置
        mic_pos = self.generate_mic_position(room_dim)
        distance = self.generate_distance()
        src_pos = self.generate_source_position(mic_pos, room_dim, distance)
        actual_distance = np.linalg.norm(np.array(src_pos) - np.array(mic_pos))
        
        # 添加声源和麦克风
        room.add_source(src_pos)
        room.add_microphone_array(pra.MicrophoneArray(np.array([mic_pos]).T, room.fs))
        
        # 计算RIR
        room.compute_rir()
        rir = room.rir[0][0]
        
        return rir, src_pos, mic_pos, actual_distance

    def process_room_rt60_combination(self, room_size_name, rt60, num_rirs=200, num_threads=8):
        """处理单个房间大小和混响时间组合"""
        print(f"开始处理: {room_size_name}房间, RT60={rt60}秒")
        
        # 创建保存目录
        save_dir = self.base_save_path / f"{room_size_name}_room_rt60_{rt60}"
        save_dir.mkdir(exist_ok=True)
        
        # 存储所有RIR数据
        all_rirs = []
        rir_info_list = []
        
        # 使用线程池生成RIR
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self.generate_single_rir, room_size_name, rt60, i): i 
                for i in range(num_rirs)
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    rir, src_pos, mic_pos, distance = future.result()
                    
                    # 保存单条RIR
                    filename = f"rir_{room_size_name}_rt60_{rt60}_{i:03d}_src_{src_pos[0]:.2f}_{src_pos[1]:.2f}_{src_pos[2]:.2f}_mic_{mic_pos[0]:.2f}_{mic_pos[1]:.2f}_{mic_pos[2]:.2f}_dist_{distance:.2f}.wav"
                    filepath = save_dir / filename
                    sf.write(str(filepath), rir, self.fs)
                    
                    # 存储RIR数据和信息
                    all_rirs.append(rir)
                    rir_info_list.append({
                        'filepath': str(filepath),
                        'distance': distance,
                        'index': i
                    })
                    
                    if (i + 1) % 20 == 0:
                        print(f"  {room_size_name}房间 RT60={rt60}: 已完成 {i+1}/{num_rirs}")
                        
                except Exception as e:
                    print(f"生成RIR时出错 (索引 {i}): {e}")
        
        # 计算散射RIR
        if all_rirs:
            min_length = min(len(rir) for rir in all_rirs)
            rirs_truncated = [rir[:min_length] for rir in all_rirs]
            diffuse_rir = np.mean(rirs_truncated, axis=0)
            
            # 保存散射RIR
            diffuse_filename = f"diffuse_rir_{room_size_name}_rt60_{rt60}.wav"
            diffuse_filepath = save_dir / diffuse_filename
            sf.write(str(diffuse_filepath), diffuse_rir, self.fs)
            
            print(f"散射RIR已保存: {diffuse_filename}")
            
            # 更新CSV文件
            self.update_csv_file(rir_info_list, str(diffuse_filepath), rt60)
        
        print(f"完成: {room_size_name}房间, RT60={rt60}秒")

    def update_csv_file(self, rir_info_list, diffuse_rir_path, rt60):
        """更新CSV文件（线程安全）"""
        with self.csv_lock:
            # 读取现有CSV
            if self.csv_file.exists():
                df = pd.read_csv(self.csv_file)
            else:
                df = pd.DataFrame(columns=['rir_path', 'diffuse_rir_path', 'rt60', 'distance'])
            
            # 添加新数据
            new_data = []
            for info in rir_info_list:
                new_data.append({
                    'rir_path': info['filepath'],
                    'diffuse_rir_path': diffuse_rir_path,
                    'rt60': rt60,
                    'distance': info['distance']
                })
            
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df], ignore_index=True)
            
            # 保存CSV
            df.to_csv(self.csv_file, index=False)

    def generate_all_rirs(self, num_threads_per_combination=8):
        """生成所有组合的RIR"""
        print("开始生成所有房间大小和混响时间的RIR...")
        
        # 遍历所有房间大小和混响时间组合
        combinations = []
        for room_size_name in self.room_sizes.keys():
            for rt60 in self.rt60_list:
                combinations.append((room_size_name, rt60))
        
        # 使用线程池处理不同组合（高级并行）
        with ThreadPoolExecutor(max_workers=min(len(combinations), 4)) as executor:
            futures = []
            for room_size_name, rt60 in combinations:
                future = executor.submit(
                    self.process_room_rt60_combination,
                    room_size_name, rt60, 200, num_threads_per_combination
                )
                futures.append(future)
            
            # 等待所有组合完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"处理组合时出错: {e}")
        
        print("所有RIR生成完成！")
        print(f"CSV文件已保存至: {self.csv_file}")

def main():
    # 初始化生成器
    generator = RIRGenerator(base_save_path="/minioData/goodman/train_data/simulate_rir_24k")
    
    # 生成所有RIR
    generator.generate_all_rirs(num_threads_per_combination=8)
    
    # 验证CSV文件
    df = pd.read_csv(generator.csv_file)
    print(f"\n生成的RIR总数: {len(df)}")
    print(f"CSV文件列名: {df.columns.tolist()}")
    print("\n前5条记录:")
    print(df.head())

if __name__ == "__main__":
    main()