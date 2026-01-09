#!/usr/bin/env python3
import os
import sys
import wave
import contextlib

def get_wav_duration(path):
	"""返回 WAV 文件时长（秒），失败返回 None。优先使用 wave，其次尝试 soundfile。"""
	try:
		with contextlib.closing(wave.open(path, 'rb')) as wf:
			frames = wf.getnframes()
			rate = wf.getframerate()
			if rate == 0:
				return None
			return frames / float(rate)
	except Exception:
		# 尝试 soundfile（如果可用）
		try:
			import soundfile as sf
			info = sf.info(path)
			if info.samplerate == 0:
				return None
			return info.frames / float(info.samplerate)
		except Exception:
			return None

def format_duration(d):
	"""格式化时长，保留小数但避免冗余 0，保证至少有 '.0'"""
	s = f"{d:.12f}".rstrip('0').rstrip('.')
	if '.' not in s:
		s += '.0'
	return s

def process_scp(input_path, output_csv):
	"""读取 scp，每行取最后一个 token 作为文件路径，写入 csv (path,duration)。"""
	if not os.path.isfile(input_path):
		print(f"[skip] not found: {input_path}")
		return
	out_lines = []
	with open(input_path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			tokens = line.split()
			audio_path = tokens[-1]
			# 若为相对路径，尝试基于 scp 文件目录解析
			if not os.path.isabs(audio_path):
				base = os.path.dirname(os.path.abspath(input_path))
				audio_path = os.path.normpath(os.path.join(base, audio_path))
			if not os.path.isfile(audio_path):
				# 不存在则仍写入原路径但提示（可根据需要改为跳过）
				print(f"[warn] audio not found, skipping: {audio_path}")
				continue
			dur = get_wav_duration(audio_path)
			if dur is None:
				print(f"[warn] cannot get duration, skipping: {audio_path}")
				continue
			out_lines.append(f"{audio_path},{format_duration(dur)}")
	# 写 CSV
	if out_lines:
		with open(output_csv, 'w', encoding='utf-8', newline='\n') as out:
			out.write("\n".join(out_lines))
		print(f"[done] wrote {len(out_lines)} lines to {output_csv}")
	else:
		print(f"[done] no valid entries for {input_path}")

def append_noise_from_scp(scp_path, csv_path, repeats=50):
	"""读取 scp，将每个有效 wav 的 path,duration 写入 csv（追加），重复 repeats 次。"""
	if not os.path.isfile(scp_path):
		print(f"[skip] noise scp not found: {scp_path}")
		return 0
	lines = []
	with open(scp_path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			tokens = line.split()
			audio_path = tokens[-1]
			# 若为相对路径，基于 scp 文件目录解析
			if not os.path.isabs(audio_path):
				base = os.path.dirname(os.path.abspath(scp_path))
				audio_path = os.path.normpath(os.path.join(base, audio_path))
			if not os.path.isfile(audio_path):
				print(f"[warn] audio not found, skipping: {audio_path}")
				continue
			dur = get_wav_duration(audio_path)
			if dur is None:
				print(f"[warn] cannot get duration, skipping: {audio_path}")
				continue
			lines.append(f"{audio_path},{format_duration(dur)}")
	if not lines:
		print(f"[done] no valid entries in {scp_path} to append")
		return 0
	# 将这些行重复 repeats 次并追加到 csv_path
	try:
		# 确保目录存在
		out_dir = os.path.dirname(os.path.abspath(csv_path))
		if out_dir and not os.path.isdir(out_dir):
			os.makedirs(out_dir, exist_ok=True)
		with open(csv_path, 'a', encoding='utf-8', newline='\n') as out:
			out.write("\n".join(lines * repeats) + "\n")
		print(f"[done] appended {len(lines) * repeats} lines to {csv_path} (from {scp_path}, repeated {repeats}x)")
		return len(lines) * repeats
	except Exception as e:
		print(f"[error] failed to append to {csv_path}: {e}")
		return 0

def main():
	# 默认的文件名映射（位于同一目录）
	script_dir = os.path.dirname(os.path.abspath(__file__))
	mapping = {
		os.path.join(script_dir, 'train_picked_dns.scp'): os.path.join(script_dir, 'train_noise_picked_dns.csv'),
		os.path.join(script_dir, 'test_picked_dns.scp'):  os.path.join(script_dir, 'test_noise_picked_dns.csv'),
	}
	# 可通过命令行指定单个 scp -> csv
	if len(sys.argv) == 3:
		process_scp(sys.argv[1], sys.argv[2])
	else:
		for inp, out in mapping.items():
			process_scp(inp, out)

	# 额外行为：若存在 noise.scp，则将其内容按 CSV 格式追加到 train_noise_picked_dns.csv，重复 50 次
	script_dir = os.path.dirname(os.path.abspath(__file__))
	noise_scp = os.path.join(script_dir, 'noise.scp')
	train_noise_csv = os.path.join(script_dir, 'train_noise_picked_dns.csv')
	if os.path.isfile(noise_scp):
		append_noise_from_scp(noise_scp, train_noise_csv, repeats=50)
	else:
		print(f"[skip] noise scp not found: {noise_scp}")

if __name__ == '__main__':
	main()
