import re
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端以支持无显示器环境
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

def process_scp_file(file_path):
    """处理单个SCP文件，计算各信噪比下的平均ESTOI分数"""
    snr_scores = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            uid, val = line.split()
            if uid and val:
                snr = uid.split('_')[-1]  # 假设 SNR 信息在文件名最后，以 '_' 分隔
                score = float(val)
                snr_scores[snr].append(score)
    
    # 计算每个信噪比的平均分
    avg_scores = {}
    for snr, scores in snr_scores.items():
        avg_scores[int(snr)] = np.mean(scores)
    
    # 如果文件名（不含扩展名）为 IEC_SNR，则每个平均分减去对应的 snr 值
    basename = os.path.splitext(os.path.basename(file_path))[0]
    if basename == 'IEC_SNR':
        for snr_int in list(avg_scores.keys()):
            avg_scores[snr_int] = avg_scores[snr_int] - snr_int
    
    return avg_scores


def plot_multiple_results(results_dict, save_path='PESQ_IF.png'):
    """绘制多个SCP文件的PESQ分数对比图"""
    if not results_dict:
        print("没有数据可绘制")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 定义标准信噪比顺序
    standard_snrs = [-10, -5, 0, 5, 10, 15, 20, 25]
    
    # 为每个SCP文件绘制一条线
    for label, data in results_dict.items():
        if data is None:
            continue
        # 按照标准信噪比顺序获取分数
        scores = [data.get(snr, 0) for snr in standard_snrs]
        plt.plot(standard_snrs, scores, 'o-', label=label, linewidth=2, markersize=8)
        
        # 计算一个小的垂直偏移，避免数值标签与点重叠
        try:
            y_min = min(scores)
            y_max = max(scores)
            y_range = max(1e-6, y_max - y_min)
            offset = y_range * 0.02  # 偏移为值域的2%
        except Exception:
            offset = 0.01
        
        # 在每个点上方添加数值标签，保留两位小数
        for x, y in zip(standard_snrs, scores):
            # 如果需要可以在此处过滤掉无效值，如 y is None 或 np.isnan(y)
            plt.text(x, y + offset, f"{y:.2f}", fontsize=9, ha='center', va='bottom')
    
    png_name = save_path.split('/')[-1].split('_')[0]
    plt.title(png_name, fontsize=14, pad=20)
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
        # '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/dp02_v92_9G/' \
        # 'scoring_dnsmos/OVRL.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_29M_Hyber_crm_tau/' \
        'scoring_dnsmos/OVRL.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/Ha_base_69M/' \
        'scoring_dnsmos/OVRL.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/v92_pipeline_nr1_test/' \
        'scoring_dnsmos/OVRL.scp',        
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/BYPASS/' \
        'scoring_dnsmos/OVRL.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_30M_fro-mse_irm_tau/' \
        'scoring_dnsmos/OVRL.scp', 
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau_25dB/' \
        'scoring_dnsmos/OVRL.scp',  
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau/' \
        'scoring_dnsmos/OVRL.scp',                      
    ]
    png_save_path = '/data/goodman/torch_nn_train/SEtrain/Ha_denoise/plot_results/OVRL_IF.png'
    main(scp_files, png_save_path)
    scp_files = [
        # '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/dp02_v92_9G/' \
        # 'scoring_dnsmos/BAK.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_29M_Hyber_crm_tau/' \
        'scoring_dnsmos/BAK.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/Ha_base_69M/' \
        'scoring_dnsmos/BAK.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/v92_pipeline_nr1_test/' \
        'scoring_dnsmos/BAK.scp',  
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/BYPASS/' \
        'scoring_dnsmos/BAK.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_30M_fro-mse_irm_tau/' \
        'scoring_dnsmos/BAK.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau_25dB/' \
        'scoring_dnsmos/BAK.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau/' \
        'scoring_dnsmos/BAK.scp',             
    ]
    png_save_path = '/data/goodman/torch_nn_train/SEtrain/Ha_denoise/plot_results/BAK_IF.png'
    main(scp_files, png_save_path)
    scp_files = [
        # '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/dp02_v92_9G/' \
        # 'scoring_dnsmos/SIG.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_29M_Hyber_crm_tau/' \
        'scoring_dnsmos/SIG.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/Ha_base_69M/' \
        'scoring_dnsmos/SIG.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/v92_pipeline_nr1_test/' \
        'scoring_dnsmos/SIG.scp',          
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/BYPASS/' \
        'scoring_dnsmos/SIG.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_30M_fro-mse_irm_tau/' \
        'scoring_dnsmos/SIG.scp',      
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau_25dB/' \
        'scoring_dnsmos/SIG.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau/' \
        'scoring_dnsmos/SIG.scp',        
    ]
    png_save_path = '/data/goodman/torch_nn_train/SEtrain/Ha_denoise/plot_results/SIG_IF.png'
    main(scp_files, png_save_path)
    scp_files = [
        # '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/dp02_v92_9G/' \
        # 'scoring_dnsmos/P808_MOS.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_29M_Hyber_crm_tau/' \
        'scoring_dnsmos/P808_MOS.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/Ha_base_69M/' \
        'scoring_dnsmos/P808_MOS.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/v92_pipeline_nr1_test/' \
        'scoring_dnsmos/P808_MOS.scp',          
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/BYPASS/' \
        'scoring_dnsmos/P808_MOS.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_30M_fro-mse_irm_tau/' \
        'scoring_dnsmos/P808_MOS.scp', 
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau_25dB/' \
        'scoring_dnsmos/P808_MOS.scp',  
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau/' \
        'scoring_dnsmos/P808_MOS.scp',                     
    ]
    png_save_path = '/data/goodman/torch_nn_train/SEtrain/Ha_denoise/plot_results/P808_MOS_IF.png'
    main(scp_files, png_save_path)

    scp_files = [
        # '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/dp02_v92_9G/' \
        # 'scoring_intrusive/PESQ.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_29M_Hyber_crm_tau/' \
        'scoring_intrusive/PESQ.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/Ha_base_69M/' \
        'scoring_intrusive/PESQ.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/v92_pipeline_nr1_test/' \
        'scoring_intrusive/PESQ.scp',        
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/BYPASS/' \
        'scoring_intrusive/PESQ.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_30M_fro-mse_irm_tau/' \
        'scoring_intrusive/PESQ.scp',   
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau_25dB/' \
        'scoring_intrusive/PESQ.scp',  
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau/' \
        'scoring_intrusive/PESQ.scp',          
    ]
    png_save_path = '/data/goodman/torch_nn_train/SEtrain/Ha_denoise/plot_results/PESQ_IF.png'
    main(scp_files, png_save_path)
    scp_files = [
        # '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/dp02_v92_9G/' \
        # 'scoring_intrusive/ESTOI.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_29M_Hyber_crm_tau/' \
        'scoring_intrusive/ESTOI.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/Ha_base_69M/' \
        'scoring_intrusive/ESTOI.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/v92_pipeline_nr1_test/' \
        'scoring_intrusive/ESTOI.scp',  
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/BYPASS/' \
        'scoring_intrusive/ESTOI.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_30M_fro-mse_irm_tau/' \
        'scoring_intrusive/ESTOI.scp', 
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau_25dB/' \
        'scoring_intrusive/ESTOI.scp',  
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau/' \
        'scoring_intrusive/ESTOI.scp',                      
    ]
    png_save_path = '/data/goodman/torch_nn_train/SEtrain/Ha_denoise/plot_results/ESTOI_IF.png'
    main(scp_files, png_save_path)
    scp_files = [
        # '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/dp02_v92_9G/' \
        # 'scoring_intrusive/SISNR.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_29M_Hyber_crm_tau/' \
        'scoring_intrusive/SISNR.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/Ha_base_69M/' \
        'scoring_intrusive/SISNR.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/v92_pipeline_nr1_test/' \
        'scoring_intrusive/SISNR.scp',          
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/BYPASS/' \
        'scoring_intrusive/SISNR.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_30M_fro-mse_irm_tau/' \
        'scoring_intrusive/SISNR.scp', 
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau_25dB/' \
        'scoring_intrusive/SISNR.scp',      
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau/' \
        'scoring_intrusive/SISNR.scp',         
    ]
    png_save_path = '/data/goodman/torch_nn_train/SEtrain/Ha_denoise/plot_results/SISNR_IF.png'
    main(scp_files, png_save_path)
    scp_files = [
        # '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/dp02_v92_9G/' \
        # 'scoring_intrusive/SDR.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_29M_Hyber_crm_tau/' \
        'scoring_intrusive/SDR.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/Ha_base_69M/' \
        'scoring_intrusive/SDR.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/v92_pipeline_nr1_test/' \
        'scoring_intrusive/SDR.scp',          
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/BYPASS/' \
        'scoring_intrusive/SDR.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_30M_fro-mse_irm_tau/' \
        'scoring_intrusive/SDR.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau_25dB/' \
        'scoring_intrusive/SDR.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau/' \
        'scoring_intrusive/SDR.scp',           
    ]
    png_save_path = '/data/goodman/torch_nn_train/SEtrain/Ha_denoise/plot_results/SDR_IF.png'
    main(scp_files, png_save_path)
    scp_files = [
        # '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/dp02_v92_9G/' \
        # 'scoring_intrusive/IEC_SNR.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_29M_Hyber_crm_tau/' \
        'scoring_intrusive/IEC_SNR.scp',    
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/Ha_base_69M/' \
        'scoring_intrusive/IEC_SNR.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/v92_pipeline_nr1_test/' \
        'scoring_intrusive/IEC_SNR.scp',          
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/BYPASS/' \
        'scoring_intrusive/IEC_SNR.scp',
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_30M_fro-mse_irm_tau/' \
        'scoring_intrusive/IEC_SNR.scp', 
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau_25dB/' \
        'scoring_intrusive/IEC_SNR.scp',  
        '/minioData/goodman/train_data/ha_lmdb/evalsets_noReverb_tau/gtcrn_32M_hyber_irm_tau/' \
        'scoring_intrusive/IEC_SNR.scp',              
    ]
    png_save_path = '/data/goodman/torch_nn_train/SEtrain/Ha_denoise/plot_results/IEC_SNR_IF.png'
    main(scp_files, png_save_path)