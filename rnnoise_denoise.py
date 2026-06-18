#!/usr/bin/env python3
"""RNNoise 降噪工具：自动预处理 + RNNoise 降噪，输出 mp3

核心思路:
  1. 分析音频特征（峰值、RMS、振幅分布百分位）
  2. 自适应预处理：增益放大低音量 → tanh软限幅压制突发尖峰
     - 让 RNNoise 在"正常音量"下工作，避免过度压制低音量信号
  3. RNNoise 降噪（48kHz逐帧处理）
  4. 输出归一化 + 保存 mp3

用法:
    python3 rnnoise_denoise.py input.mp3                    # 全自动
    python3 rnnoise_denoise.py input.mp3 -o output.mp3      # 指定输出
    python3 rnnoise_denoise.py input.mp3 -g 10              # 手动增益10x
    python3 rnnoise_denoise.py input.mp3 --no-preprocess    # 跳过预处理
    python3 rnnoise_denoise.py input.mp3 --target-rms 0.10  # 目标RMS=0.10
"""
import argparse
import os
import sys
import numpy as np
from silero_vad import read_audio
from rnnoise_wrapper import RNNoise


# ============================================================
# Step 1: 音频分析
# ============================================================
def analyze_audio(wav_np, sr=16000):
    """分析音频特征，返回诊断信息"""
    abs_wav = np.abs(wav_np)
    peak = abs_wav.max()
    rms = np.sqrt(np.mean(wav_np**2))
    mean_amp = abs_wav.mean()

    # 振幅分布百分位
    percentiles = {}
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        percentiles[p] = np.percentile(abs_wav, p)

    # 分帧分析（10ms帧），找突发大振幅
    frame_size = int(sr * 0.01)  # 160 samples @ 16kHz
    n_frames = len(wav_np) // frame_size
    frame_energies = np.array([
        np.sqrt(np.mean(wav_np[i*frame_size:(i+1)*frame_size]**2))
        for i in range(n_frames)
    ])

    # 峰值因子：峰值 / 中位数能量 — 越大说明突发越强
    median_energy = np.median(frame_energies[frame_energies > 0]) if np.any(frame_energies > 0) else 0
    crest_factor = peak / max(median_energy, 1e-10)

    # 突发帧比例：能量 > 中位数10倍的帧占比
    if median_energy > 0:
        burst_ratio = (frame_energies > median_energy * 10).sum() / n_frames
    else:
        burst_ratio = 0

    info = {
        'peak': peak,
        'rms': rms,
        'mean_amp': mean_amp,
        'percentiles': percentiles,
        'crest_factor': crest_factor,
        'burst_ratio': burst_ratio,
        'median_energy': median_energy,
        'frame_energies': frame_energies,
        'n_frames': n_frames,
        'frame_size': frame_size,
    }
    return info


def print_analysis(info):
    """打印分析结果"""
    print(f"  峰值: {info['peak']:.6f}, RMS: {info['rms']:.6f}, 均值: {info['mean_amp']:.6f}")
    print(f"  峰值因子(crest): {info['crest_factor']:.1f}x, 突发帧比例: {info['burst_ratio']*100:.2f}%")
    p = info['percentiles']
    print(f"  百分位: p50={p[50]:.6f} p90={p[90]:.6f} p95={p[95]:.6f} "
          f"p99={p[99]:.6f} p99.9={p[99.9]:.6f}")


# ============================================================
# Step 2: 自适应预处理
# ============================================================
def auto_preprocess(wav_np, sr=16000, target_rms=0.15):
    """
    自适应预处理：增益放大 → tanh 软限幅
    
    核心思路:
    1. 计算增益，将 RMS 提升到 target_rms（让 RNNoise 能有效识别语音）
    2. tanh 软限幅：压制增益后的突发尖峰（关门声等）
       - 对正常信号几乎无影响（tanh 在低振幅接近线性）
       - 对突发尖峰有效压缩
    
    统一处理，无需判断"是否有突发"：
    - 低音量无突发：增益放大即可，tanh 几乎不起作用
    - 低音量有突发：增益放大后尖峰也放大了，tanh 压制尖峰
    
    返回: (处理后的音频, 使用的参数dict)
    """
    info = analyze_audio(wav_np, sr)
    p = info['percentiles']

    # ---- Step 1: 计算增益 ----
    # 目标: 将 RMS 提升到 target_rms
    if info['rms'] > 1e-10:
        gain = target_rms / info['rms']
    else:
        gain = 1.0
    # 限制增益范围 [1, 200]
    gain = max(1.0, min(gain, 200.0))

    # ---- Step 2: 软限幅阈值 ----
    # 使用增益后的 p95 作为阈值
    # 这样 95% 的信号在阈值以下（线性区），5% 的尖峰被压缩
    clip_threshold = float(p[95]) * gain
    # 限制阈值范围 [0.15, 0.95]
    # 太低: 压掉正常信号; 太高: 软限幅几乎不起作用
    clip_threshold = max(0.15, min(clip_threshold, 0.95))

    params = {
        'gain': gain,
        'clip_threshold': clip_threshold,
    }

    print(f"  增益: {gain:.1f}x (目标 RMS={target_rms})")
    print(f"  软限幅阈值: {clip_threshold:.4f}")

    # ---- Step 3: 执行: 增益 → tanh 软限幅 ----
    processed = wav_np.copy() * gain

    # tanh 软限幅: 阈值以下接近线性(x≈y)，以上平滑压缩
    # tanh(x/threshold) * threshold
    processed = clip_threshold * np.tanh(processed / clip_threshold)

    # 打印处理效果
    new_peak = np.abs(processed).max()
    new_rms = np.sqrt(np.mean(processed**2))
    print(f"  处理后: 峰值={new_peak:.4f}, RMS={new_rms:.6f}")

    return processed, params


# ============================================================
# Step 3: RNNoise 降噪
# ============================================================
def rnnoise_denoise(wav_16k_float, normalize_output=True):
    """
    对预处理后的 16kHz float 音频做 RNNoise 降噪
    输入范围应在 [-1, 1]，会自动上采样到 48kHz
    """
    wav_np = wav_16k_float.astype(np.float32)

    # 上采样 16kHz -> 48kHz
    wav_48k = np.interp(
        np.arange(len(wav_np) * 3) / 3.0,
        np.arange(len(wav_np)),
        wav_np
    )

    # 转 int16
    wav_48k_int16 = (np.clip(wav_48k, -1.0, 1.0) * 32767).astype(np.int16)

    # RNNoise 逐帧处理
    denoiser = RNNoise()
    frame_size = 480
    n_frames = len(wav_48k_int16) // frame_size
    denoised_frames = []

    print(f'  RNNoise 降噪中... {n_frames} 帧 ({n_frames*10/1000:.1f}s)')
    for i in range(n_frames):
        frame = wav_48k_int16[i*frame_size:(i+1)*frame_size].tobytes()
        vad_prob, denoised = denoiser.process_frame(frame)
        denoised_frames.append(np.frombuffer(denoised, dtype=np.int16))

    denoiser.destroy()
    denoised_48k = np.concatenate(denoised_frames)

    # 下采样回 16kHz
    denoised_16k = denoised_48k[::3].astype(np.float32) / 32767.0

    # 归一化
    if normalize_output:
        max_val = np.abs(denoised_16k).max()
        if max_val > 0:
            denoised_16k = denoised_16k / max_val
        print(f'  降噪后归一化: {max_val:.6f} -> 1.0')
    else:
        print(f'  降噪后保留原量级: 峰值={np.abs(denoised_16k).max():.6f}')

    # 对齐长度
    if len(denoised_16k) > len(wav_np):
        denoised_16k = denoised_16k[:len(wav_np)]
    elif len(denoised_16k) < len(wav_np):
        denoised_16k = np.pad(denoised_16k, (0, len(wav_np) - len(denoised_16k)))

    return denoised_16k


# ============================================================
# Step 4: 保存 mp3
# ============================================================
def save_as_mp3(audio_np, output_path, sr=16000):
    """通过 soundfile + ffmpeg 保存为 mp3"""
    import soundfile as sf
    import subprocess

    tmp_wav = output_path.rsplit('.', 1)[0] + '_tmp.wav'
    sf.write(tmp_wav, audio_np, sr, subtype='PCM_16')

    cmd = ['ffmpeg', '-y', '-i', tmp_wav, '-codec:a', 'libmp3lame', '-b:a', '128k', output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ffmpeg 错误: {result.stderr}', file=sys.stderr)
        print(f'  保留临时 wav: {tmp_wav}')
    else:
        os.remove(tmp_wav)
        print(f'  mp3 已保存: {output_path}')


# ============================================================
# 主流程
# ============================================================
def process_file(input_path, output_path=None, gain=None, auto_preprocess_flag=True,
                 target_rms=0.15, normalize_output=True, sr=16000):
    """完整处理流程"""

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f'{base}_denoised.mp3'

    print(f'=== RNNoise 降噪工具 ===')
    print(f'输入: {input_path}')
    print(f'输出: {output_path}')
    print()

    # 1. 读取音频
    print('1. 读取音频')
    wav = read_audio(input_path)
    wav_np = wav.numpy()
    duration = len(wav_np) / sr
    print(f'  时长: {duration:.1f}s')

    # 2. 分析
    print('\n2. 音频分析')
    info = analyze_audio(wav_np, sr)
    print_analysis(info)

    # 3. 预处理
    print('\n3. 预处理')
    if auto_preprocess_flag:
        processed, params = auto_preprocess(wav_np, sr, target_rms)
    else:
        # 手动增益模式
        if gain is None:
            # 默认增益：将 RMS 放大到 target_rms
            if info['rms'] > 1e-10:
                gain = target_rms / info['rms']
            else:
                gain = 1.0
            gain = max(1.0, min(gain, 200.0))
        print(f'  跳过自动预处理，手动增益: {gain:.1f}x')
        processed = np.clip(wav_np * gain, -1.0, 1.0)
        new_peak = np.abs(processed).max()
        print(f'  处理后: 峰值={new_peak:.4f}')
        params = {'strategy': 'manual', 'gain': gain}

    # 4. RNNoise 降噪
    print('\n4. RNNoise 降噪')
    denoised = rnnoise_denoise(processed, normalize_output=normalize_output)

    # 5. 保存
    print('\n5. 保存')
    save_as_mp3(denoised, output_path, sr)

    print('\n完成!')
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='RNNoise 降噪工具 - 自动预处理 + 降噪',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 rnnoise_denoise.py input.mp3                    # 全自动
  python3 rnnoise_denoise.py input.mp3 -g 10              # 手动增益10x
  python3 rnnoise_denoise.py input.mp3 --no-preprocess    # 跳过预处理
  python3 rnnoise_denoise.py input.mp3 --target-rms 0.10  # 目标RMS=0.10
        """)
    parser.add_argument('input', help='输入音频文件')
    parser.add_argument('-o', '--output', default=None, help='输出 mp3 路径')
    parser.add_argument('-g', '--gain', type=float, default=None,
                        help='手动输入增益(覆盖自动计算), 例如 -g 10')
    parser.add_argument('--auto', dest='auto_preprocess', action='store_true', default=True,
                        help='自动预处理(默认)')
    parser.add_argument('--no-preprocess', dest='auto_preprocess', action='store_false',
                        help='跳过自动预处理，仅做手动增益')
    parser.add_argument('--target-rms', type=float, default=0.15,
                        help='预处理后目标 RMS (默认: 0.15)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='降噪后不归一化')
    parser.add_argument('--sr', type=int, default=16000, help='输出采样率 (默认: 16000)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'错误: 文件不存在: {args.input}', file=sys.stderr)
        sys.exit(1)

    # 如果指定了手动增益，关闭自动预处理
    if args.gain is not None:
        args.auto_preprocess = False

    process_file(
        input_path=args.input,
        output_path=args.output,
        gain=args.gain,
        auto_preprocess_flag=args.auto_preprocess,
        target_rms=args.target_rms,
        normalize_output=not args.no_normalize,
        sr=args.sr,
    )


if __name__ == '__main__':
    main()