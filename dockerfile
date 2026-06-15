FROM mirror.gcr.io/python:3.9-slim

# 安装系统依赖 (ffmpeg 用于 mp3 解码)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存层加速构建
COPY requirements.txt .

# 安装 Python 依赖 (使用 CPU 版 PyTorch 减小镜像体积)
RUN pip3 install --no-cache-dir \
    torch>=1.12.0 torchaudio>=0.12.0 \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    silero-vad fastapi>=0.100.0 uvicorn>=0.20.0 python-multipart>=0.0.6 numpy soundfile

# 复制应用代码
COPY vad_label.py .
COPY app.py .

# 暴露端口
EXPOSE 8000

# 启动 Web API 服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]