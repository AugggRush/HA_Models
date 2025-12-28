import os
import shutil
import torch
import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf
from models.v92_pipeline_model import pipeline_test as Model

def main(args):
    cfg_infer = OmegaConf.load(args.config)
    # cfg_network = OmegaConf.load(cfg_infer.network_config)
    
    noisy_folder = cfg_infer.test_dataset.noisy_dir
    clean_folder = cfg_infer.test_dataset.clean_dir
    rir_folder = cfg_infer.test_dataset.rir_dir
    noise_floder = cfg_infer.test_dataset.noise_dir
    if cfg_infer.network.neg_infer:
        enh_folder = cfg_infer.network.neg_folder
    else:
        enh_folder = cfg_infer.network.enh_folder
    os.makedirs(enh_folder, exist_ok=True)

    model = Model(**cfg_infer['network_config'])
    model.eval()
    
    noisy_wavs = sorted(list(filter(lambda x: x.endswith("wav"), os.listdir(noisy_folder))))
    rir_wavs = sorted(list(filter(lambda x: x.endswith("wav"), os.listdir(rir_folder))))
    
    inf_scp_list = []
    ref_scp_list = []
    # 依序从 rir_wavs 中选取 rir 文件；若 rir 数量少于 noisy 数量则循环使用
    if len(rir_wavs) == 0:
        raise RuntimeError(f"No RIR files found in rir_folder: {rir_folder}")
    for idx, wav_name in enumerate(tqdm(noisy_wavs)):
        if cfg_infer.network.neg_infer:
            noise, fs = sf.read(os.path.join(noise_floder, wav_name), dtype='float32')
            clean, fs = sf.read(os.path.join(clean_folder, wav_name), dtype='float32')
            noisy = -1 * clean +noise
            sf.write('neg_enh.wav', noisy, fs)
            in_file_path = 'neg_enh.wav'
        else:
            in_file_path = os.path.join(noisy_folder, wav_name)
        uid = wav_name.split(".wav")[0]
        # 循环选择 rir 文件
        rir_file = rir_wavs[idx % len(rir_wavs)]
        ir_file_path = os.path.join(rir_folder, rir_file)
        enh_path = os.path.join(enh_folder, uid + f"_enh.wav")
        with torch.inference_mode():
            model(in_file_path, ir_file_path, enh_path)
        
        ref_path = os.path.join(clean_folder, wav_name)
        
        inf_scp_list.append([uid, enh_path])
        ref_scp_list.append([uid, ref_path])
    
    # Save paths into scp file for evaluation
    with open(os.path.join(enh_folder, "inf.scp"), "w") as f:
        for uid, audio_path in inf_scp_list:
            f.write(f"{uid} {audio_path}\n")

    with open(os.path.join(enh_folder, "ref.scp"), "w") as f:
        for uid, audio_path in ref_scp_list:
            f.write(f"{uid} {audio_path}\n")
            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='configs/v92_pipeline_cfg.yaml')

    args = parser.parse_args()
    main(args)
