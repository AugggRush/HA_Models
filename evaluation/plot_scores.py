import re
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端以支持无显示器环境
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

def process_scp_file(file_path):
    """处理单个SCP文件，计算各信噪比下的平均PESQ分数"""
    snr_scores = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # 使用正则表达式匹配文件名和分数
            match = re.match(r'^sample_\d+_snr_(\d+)\s+([\d.]+)$', line)
            if match:
                snr = match.group(1)
                score = float(match.group(2))
                snr_scores[snr].append(score)
    
    # 计算每个信噪比的平均分
    avg_scores = {}
    for snr, scores in snr_scores.items():
        avg_scores[int(snr)] = np.mean(scores)
    
    return avg_scores


def plot_multiple_results(results_dict, save_path='pesq_comparison.png'):
    """绘制多个SCP文件的PESQ分数对比图"""
    if not results_dict:
        print("没有数据可绘制")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 定义标准信噪比顺序
    standard_snrs = [5, 10, 15, 19]
    
    # 为每个SCP文件绘制一条线
    for label, data in results_dict.items():
        if data is None:
            continue
        # 按照标准信噪比顺序获取分数
        scores = [data.get(snr, 0) for snr in standard_snrs]
        plt.plot(standard_snrs, scores, 'o-', label=label, linewidth=2, markersize=8)
    
    plt.title('PESQ compare', fontsize=14, pad=20)
    plt.xlabel('snr (dB)', fontsize=12)
    plt.ylabel('average score', fontsize=12)
    plt.xticks(standard_snrs)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n图表已保存为 {save_path}")

def main(scp_files, png_save):
    """主函数：处理多个SCP文件并绘制结果"""
    results = {}
    for file_path in scp_files:
        if os.path.exists(file_path):
            # 使用文件名作为标签（不带扩展名）
            label = file_path.split('/')[-3]
            avg_scores = process_scp_file(file_path)
            results[label] = avg_scores
            print(f"Processed {file_path}:")
            for snr, score in sorted(avg_scores.items()):
                print(f"  SNR {snr}dB: {score:.4f}")
        else:
            print(f"File not found: {file_path}")
    
    if results:
        plot_multiple_results(results, png_save)
    else:
        print("No valid results to plot.")

# 示例使用 - 替换为您的实际SCP文件路径
if __name__ == "__main__":
    # 这里添加您的SCP文件路径列表
    scp_files = [
        '/minioData/goodman/train_data/ha_lmdb/eval_sets/gtcrn_dual_decoder_speechCnoise_2025-10-14-10h18m/' \
        'scoring_dnsmos/P808_MOS.scp',
        '/minioData/goodman/train_data/ha_lmdb/eval_sets/without_process/' \
        'ori_scoring_dnsmos/P808_MOS.scp',
    ]
    png_save_path = '/minioData/goodman/train_data/ha_lmdb/eval_sets/gtcrn_dual_decoder_speechCnoise_2025-10-14-10h18m/' \
    'scoring_intrusive/P808_gtcrn_dual_decoder_speechCnoise.png'
    main(scp_files, png_save_path)