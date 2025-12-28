import sys
import os
sys.path.append(".")
sys.path.append("..")
import numpy as np
import librosa
import soundfile as sf
import torch
import subprocess
from typing import Iterable, List, Tuple, Union, Dict

class pipeline_test(torch.nn.Module):
    def __init__(self,
        # pipeline bin path
        bin_path = 'aitd_v92_pipeline_test',
        # frame length in ms, default 2ms                 
        frame_len = 2,
        # fft length, default 256
        fft_len = 128,
        # dc mode, 0-off, 1-on, default 1
        dc_mode = 1,
        # afc mode, 0-off, 1, 2, 3, 4, 5, default 1    
        afc_mode = 1,             
        # afc delay in us, default 6000us
        afc_delay = 6000,   
        # wdrc mode, 0-off, 1, 2, 3, default 1     
        wdrc_mode = 1,
        # wdrc noise gate mode, 0-disable, 1-enable, default 1
        wdrc_ng_mode = 1,
        # wdrc noise gate threshold in dB, default -40dB
        wdrc_ng_th = 40,
        # wdrc noise gate ratio, default 0.8
        wdrc_ng_ratio = 0.5,
        # noise reduction mode, 0-disable, 1-enable, default 1
        nr1_mode = 1,
        # noise reduction mode 2, 0-disable, 1-enable, default 0
        nr2_mode = 0,
        # bf mode, 0-disable, 1-enable, default 0
        bf_mode = 0,
        # limiter mode, 0-disable, 1-enable, default 1            
        lim_mode = 1,
        # 0-disable（open loop）, 1-enable（closed loop）
        feedback_enable = 1,         
        # 0-轻度，使用gain1, 1-重度，使用gain2, default 0 
        hear_loss_level = 0,  
        # volume gain value in dB, default 0dB    
        vol_value = 0,             
        # speech gain value in dB, default 0dB
        speech_gain = 0,           
        # bgm gain value in dB, default 0dB
        bgm_gain = -10,        
        # dB 输入前向增益, defaule 0dB      
        g1 = 0,                    
        # dB 输出后向增益, defaule 0dB
        g2 = 0,             
        # dB 闭环路径增益, defaule 0dB       
        g3 = 0,                   
        # freq lowering
        fl_mode = 0,    
        # 是否做帧长的切换
        frame_len_switch = 0,
    ) -> None:
        super().__init__()
        self.bin_path = bin_path
        self.frame_len = frame_len
        self.fft_len = fft_len
        self.dc_mode = dc_mode
        self.afc_mode = afc_mode
        self.afc_delay = afc_delay
        self.wdrc_mode = wdrc_mode
        self.wdrc_ng_mode = wdrc_ng_mode
        self.wdrc_ng_th = wdrc_ng_th
        self.wdrc_ng_ratio = wdrc_ng_ratio
        self.nr1_mode = nr1_mode
        self.nr2_mode = nr2_mode
        self.bf_mode = bf_mode
        self.lim_mode = lim_mode
        self.feedback_enable = feedback_enable
        self.hear_loss_level = hear_loss_level
        self.vol_value = vol_value
        self.speech_gain = speech_gain
        self.bgm_gain = bgm_gain
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.fl_mode = fl_mode
        self.frame_len_switch = frame_len_switch

    def _process_one(self, in_file_path, ir_file_path, out_file_path):
        """
        单文件处理：运行可执行文件并做后处理（裁剪/补零等）。
        返回 out_file_path（字符串）。
        """
        # 构建命令（保持原参数顺序与行为）
        cmd = (f'{self.bin_path} '
            f'{in_file_path} '
            f'{out_file_path} '
            f'{ir_file_path} '
            f'{self.frame_len} '
            f'{self.fft_len} '            
            f'{self.feedback_enable} '
            f'--dc_mode={self.dc_mode} '
            f'--afc_mode={self.afc_mode} '
            f'--afc_delay={self.afc_delay} '
            f'--wdrc_mode={self.wdrc_mode} '
            f'--wdrc_ng_mode={self.wdrc_ng_mode} '
            f'--wdrc_ng_th={self.wdrc_ng_th} '
            f'--wdrc_ng_ratio={self.wdrc_ng_ratio} '
            f'--nr1_mode={self.nr1_mode} '
            f'--nr2_mode={self.nr2_mode} '
            f'--bf_mode={self.bf_mode} '
            f'--lim_mode={self.lim_mode} '
            f'--hear_loss_level={self.hear_loss_level} '
            f'--vol_value={self.vol_value} '
            f'--speech_gain={self.speech_gain} '
            f'--bgm_gain={self.bgm_gain} '
            f'--g1={self.g1} '
            f'--g2={self.g2} '
            f'--g3={self.g3} '
            f'--fl_mode={self.fl_mode} '
            f'--frame_len_switch={self.frame_len_switch} '
        )
        # 使用 subprocess 更可控
        try:
            subprocess.run(cmd, shell=True, check=False)
        except Exception as e:
            print(f"[WARN] 运行 pipeline 可执行文件失败: {e}", file=sys.stderr)
        # 后处理：读取 out_path 与 in_path，裁剪前 n_trim，补零/截断至目标长度
        try:
            out_arr, out_sr = sf.read(str(out_file_path), always_2d=True)
            in_arr, in_sr = sf.read(str(in_file_path), always_2d=True)
            if in_sr != out_sr:
                target_len = int(round(in_arr.shape[0] * (out_sr / float(in_sr))))
            else:
                target_len = in_arr.shape[0]
            n_trim = int(self.frame_len / 1000. * out_sr)
            if out_arr.shape[0] > (n_trim + 1):
                trimmed = out_arr[n_trim:-1, :]
                cur_len = trimmed.shape[0]
                if cur_len < target_len:
                    pad_len = target_len - cur_len
                    pad = np.zeros((pad_len, trimmed.shape[1]), dtype=trimmed.dtype) if trimmed.ndim == 2 else np.zeros(pad_len, dtype=trimmed.dtype)
                    new_data = np.vstack([trimmed, pad]) if trimmed.ndim == 2 else np.concatenate([trimmed, pad])
                else:
                    new_data = trimmed[:target_len, :] if trimmed.ndim == 2 else trimmed[:target_len]
                if isinstance(new_data, np.ndarray) and new_data.ndim == 2 and new_data.shape[1] == 1:
                    new_data = new_data[:, 0]
                sf.write(str(out_file_path), new_data, out_sr)
            else:
                print(f"[WARN] 输出文件过短，无法按要求裁剪: {out_file_path}", file=sys.stderr)
        except Exception as _e:
            print(f"[WARN] 处理并覆盖输出 wav 失败（{out_file_path}): {_e}", file=sys.stderr)
        return out_file_path

    def forward(self, in_file_path, ir_file_path, out_file_path):
        # 支持 batch 输入：如果第一个参数是 list/tuple，视为多样本批次
        if isinstance(in_file_path, (list, tuple)):
            results = []
            for item in in_file_path:
                # item 可能是 (in, ir, out) 或 dict
                if isinstance(item, dict):
                    i = item.get("in")
                    ir = item.get("ir")
                    o = item.get("out")
                else:
                    i, ir, o = item
                results.append(self._process_one(i, ir, o))
            return results
        else:
            # 单样本行为向后兼容（仍返回 out_file_path）
            return self._process_one(in_file_path, ir_file_path, out_file_path)

    def predict_batch(self, data_iter: Iterable[Union[Tuple[str,str,str], Dict]], return_dict: bool=False) -> List:
        """
        方便与 DataLoader 结合的推理接口。
        data_iter: 可迭代对象，每项为 (in_path, ir_path, out_path) 或 {'in':..., 'ir':..., 'out':...}
        return_dict: 若为 True，返回 list of {"out": path, "success": True/False, "error": msg}
        """
        results = []
        # 不需要梯度
        with torch.no_grad():
            for item in data_iter:
                if isinstance(item, dict):
                    in_p = item.get("in")
                    ir_p = item.get("ir")
                    out_p = item.get("out")
                else:
                    in_p, ir_p, out_p = item
                try:
                    out = self._process_one(in_p, ir_p, out_p)
                    res = {"out": out, "success": True, "error": None}
                except Exception as e:
                    res = {"out": out_p, "success": False, "error": str(e)}
                results.append(res if return_dict else res["out"])
        return results
