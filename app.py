#!/usr/bin/env python3
"""
VAD Label Web API - 基于 Silero VAD 的语音活动检测 Web 服务

提供 REST API 接口，上传音频文件即可获取 VAD 检测结果。

启动:
    uvicorn app:app --host 0.0.0.0 --port 8000

API:
    POST /vad       - 上传音频文件进行 VAD 检测
    GET  /health    - 健康检查
"""

import os
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

# ---------- 全局加载模型（启动时加载一次） ----------

print("正在加载 Silero VAD 模型...")
_model = load_silero_vad()
print("模型加载完成")

app = FastAPI(
    title="VAD Label API",
    description="基于 Silero VAD 的语音活动检测打标服务",
    version="1.0.0",
)

# 支持的音频扩展名
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac"}


# ---------- 核心检测函数 ----------

def do_detect(
    audio_path: str,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 512,
    normalize: bool = True,
) -> list[dict]:
    """执行 VAD 检测，返回语音片段列表"""
    wav = read_audio(audio_path)

    if normalize:
        max_val = wav.abs().max()
        if max_val > 0 and max_val < 0.5:
            wav = wav / max_val

    speech_timestamps = get_speech_timestamps(
        wav,
        _model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        window_size_samples=window_size_samples,
        return_seconds=True,
    )
    return speech_timestamps


def format_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


# ---------- API 路由 ----------

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "model": "silero-vad"}


@app.post("/vad")
async def detect_vad(
    file: UploadFile = File(..., description="音频文件 (mp3/wav/flac/ogg/m4a 等)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="VAD 阈值 (0.0-1.0)，越高越严格"),
    min_speech: int = Query(250, ge=0, description="最小语音时长 (毫秒)"),
    min_silence: int = Query(100, ge=0, description="最小静音时长 (毫秒)"),
    window: int = Query(512, description="窗口大小 (采样点数)"),
    normalize: bool = Query(True, description="是否自动归一化低音量音频"),
    format: str = Query("json", description="输出格式: json / text / csv / srt"),
):
    """
    上传音频文件进行 VAD 语音活动检测。

    - **file**: 音频文件
    - **threshold**: VAD 阈值，默认 0.5
    - **min_speech**: 最小语音片段时长(ms)，默认 250
    - **min_silence**: 最小静音时长(ms)，默认 100
    - **window**: 窗口采样点数，默认 512
    - **normalize**: 自动归一化低音量音频，默认开启
    - **format**: 返回格式 json/text/csv/srt，默认 json
    """
    # 校验文件扩展名
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 保存到临时文件
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"vad_{uuid.uuid4().hex}{ext}")
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        # 执行检测
        speech_timestamps = do_detect(
            tmp_path,
            threshold=threshold,
            min_speech_duration_ms=min_speech,
            min_silence_duration_ms=min_silence,
            window_size_samples=window,
            normalize=normalize,
        )

        # 按格式返回
        if format == "json":
            total_speech = sum(s["end"] - s["start"] for s in speech_timestamps)
            result = {
                "file": file.filename,
                "total_segments": len(speech_timestamps),
                "total_speech_sec": round(total_speech, 3),
                "segments": speech_timestamps,
            }
            return JSONResponse(content=result)

        elif format == "text":
            lines = [f"文件: {file.filename}", f"检测到 {len(speech_timestamps)} 个语音片段:", ""]
            for i, seg in enumerate(speech_timestamps, 1):
                dur = seg["end"] - seg["start"]
                lines.append(
                    f"  [{i:03d}] {format_time(seg['start'])} -> {format_time(seg['end'])}  "
                    f"(时长: {dur:.3f}s)"
                )
            total_speech = sum(s["end"] - s["start"] for s in speech_timestamps)
            lines.append("")
            lines.append(f"总语音时长: {total_speech:.3f}s")
            return PlainTextResponse(content="\n".join(lines))

        elif format == "csv":
            lines = ["segment_id,start_sec,end_sec,duration_sec,start_fmt,end_fmt"]
            for i, seg in enumerate(speech_timestamps, 1):
                dur = seg["end"] - seg["start"]
                lines.append(
                    f"{i},{seg['start']:.3f},{seg['end']:.3f},{dur:.3f},"
                    f"{format_time(seg['start'])},{format_time(seg['end'])}"
                )
            return PlainTextResponse(content="\n".join(lines), media_type="text/csv")

        elif format == "srt":
            lines = []
            for i, seg in enumerate(speech_timestamps, 1):
                sh = int(seg["start"] // 3600)
                sm = int((seg["start"] % 3600) // 60)
                ss = int(seg["start"] % 60)
                sms = int((seg["start"] % 1) * 1000)
                eh = int(seg["end"] // 3600)
                em = int((seg["end"] % 3600) // 60)
                es = int(seg["end"] % 60)
                ems = int((seg["end"] % 1) * 1000)
                lines.append(str(i))
                lines.append(f"{sh:02d}:{sm:02d}:{ss:02d},{sms:03d} --> {eh:02d}:{em:02d}:{es:02d},{ems:03d}")
                lines.append(f"[VAD] {seg['start']:.3f}s - {seg['end']:.3f}s")
                lines.append("")
            return PlainTextResponse(content="\n".join(lines), media_type="text/srt")

        else:
            raise HTTPException(status_code=400, detail=f"不支持的格式: {format}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VAD 检测失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)