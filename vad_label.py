#!/usr/bin/env python3
"""
VAD Label Tool - 基于 Silero VAD 的 MP3 语音活动检测打标工具

对输入的音频文件进行 VAD 检测，输出语音片段的起止时间点。

用法:
    python vad_label.py <audio_file> [选项]

示例:
    python vad_label.py input.mp3
    python vad_label.py input.mp3 --threshold 0.5 --output result.json
    python vad_label.py input.mp3 --format text
"""

import argparse
import json
import sys
from pathlib import Path

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


def format_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS.mmm 格式"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def detect_vad(
    audio_path: str,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 512,
    normalize: bool = True,
) -> list[dict]:
    """
    对音频文件执行 VAD 检测，返回语音片段的时间戳列表。

    Args:
        audio_path: 音频文件路径 (支持 mp3, wav 等格式)
        threshold: VAD 检测阈值 (0.0-1.0)，越高越严格
        min_speech_duration_ms: 最小语音时长 (毫秒)
        min_silence_duration_ms: 最小静音时长 (毫秒)
        window_size_samples: 窗口大小 (采样点数)
        normalize: 是否对音频进行归一化 (音量放大到合理范围)

    Returns:
        语音片段列表，每项包含 start 和 end 时间 (秒)
    """
    model = load_silero_vad()
    wav = read_audio(audio_path)

    if normalize:
        max_val = wav.abs().max()
        if max_val > 0 and max_val < 0.5:
            wav = wav / max_val

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        window_size_samples=window_size_samples,
        return_seconds=True,
    )

    return speech_timestamps


def output_json(speech_timestamps: list[dict], audio_path: str) -> str:
    """以 JSON 格式输出结果"""
    result = {
        "file": str(audio_path),
        "total_segments": len(speech_timestamps),
        "segments": speech_timestamps,
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


def output_text(speech_timestamps: list[dict], audio_path: str) -> str:
    """以文本格式输出结果"""
    lines = [f"文件: {audio_path}", f"检测到 {len(speech_timestamps)} 个语音片段:", ""]

    for i, seg in enumerate(speech_timestamps, 1):
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        lines.append(
            f"  [{i:03d}] {format_time(start)} -> {format_time(end)}  "
            f"(时长: {duration:.3f}s)"
        )

    # 统计总语音时长
    total_speech = sum(seg["end"] - seg["start"] for seg in speech_timestamps)
    lines.append("")
    lines.append(f"总语音时长: {total_speech:.3f}s")

    return "\n".join(lines)


def output_csv(speech_timestamps: list[dict], audio_path: str) -> str:
    """以 CSV 格式输出结果"""
    lines = ["segment_id,start_sec,end_sec,duration_sec,start_fmt,end_fmt"]
    for i, seg in enumerate(speech_timestamps, 1):
        duration = seg["end"] - seg["start"]
        lines.append(
            f"{i},{seg['start']:.3f},{seg['end']:.3f},{duration:.3f},"
            f"{format_time(seg['start'])},{format_time(seg['end'])}"
        )
    return "\n".join(lines)


def output_srt(speech_timestamps: list[dict]) -> str:
    """以 SRT 字幕格式输出结果 (便于在视频播放器中查看)"""
    lines = []
    for i, seg in enumerate(speech_timestamps, 1):
        start_h = int(seg["start"] // 3600)
        start_m = int((seg["start"] % 3600) // 60)
        start_s = seg["start"] % 60
        start_ms = int((start_s % 1) * 1000)
        start_s_int = int(start_s)

        end_h = int(seg["end"] // 3600)
        end_m = int((seg["end"] % 3600) // 60)
        end_s = seg["end"] % 60
        end_ms = int((end_s % 1) * 1000)
        end_s_int = int(end_s)

        lines.append(str(i))
        lines.append(
            f"{start_h:02d}:{start_m:02d}:{start_s_int:02d},{start_ms:03d} --> "
            f"{end_h:02d}:{end_m:02d}:{end_s_int:02d},{end_ms:03d}"
        )
        lines.append(f"[VAD] {seg['start']:.3f}s - {seg['end']:.3f}s")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="基于 Silero VAD 的音频语音活动检测打标工具"
    )
    parser.add_argument("audio", help="输入音频文件路径 (mp3/wav 等)")
    parser.add_argument(
        "--threshold", "-t",
        type=float, default=0.5,
        help="VAD 阈值 (0.0-1.0)，越高越严格 (默认: 0.5)",
    )
    parser.add_argument(
        "--min-speech", "-s",
        type=int, default=250,
        help="最小语音时长，毫秒 (默认: 250)",
    )
    parser.add_argument(
        "--min-silence", "-S",
        type=int, default=100,
        help="最小静音时长，毫秒 (默认: 100)",
    )
    parser.add_argument(
        "--window", "-w",
        type=int, default=512,
        help="窗口大小，采样点数 (默认: 512)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "csv", "srt"],
        default="text",
        help="输出格式 (默认: text)",
    )
    parser.add_argument(
        "--output", "-o",
        help="输出文件路径 (默认输出到终端)",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="禁用音频归一化 (默认自动归一化低音量音频)",
    )

    args = parser.parse_args()

    # 检查文件是否存在
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"错误: 文件不存在: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"正在加载模型并分析: {audio_path}", file=sys.stderr)

    try:
        speech_timestamps = detect_vad(
            str(audio_path),
            threshold=args.threshold,
            min_speech_duration_ms=args.min_speech,
            min_silence_duration_ms=args.min_silence,
            window_size_samples=args.window,
            normalize=not args.no_normalize,
        )
    except Exception as e:
        print(f"错误: VAD 检测失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 格式化输出
    formatters = {
        "json": lambda: output_json(speech_timestamps, str(audio_path)),
        "text": lambda: output_text(speech_timestamps, str(audio_path)),
        "csv": lambda: output_csv(speech_timestamps, str(audio_path)),
        "srt": lambda: output_srt(speech_timestamps),
    }
    result = formatters[args.format]()

    # 写入输出
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"结果已保存到: {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()