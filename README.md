# VAD Label Tool

基于 [Silero VAD](https://github.com/snakers4/silero-vad) 的语音活动检测（Voice Activity Detection）打标工具，提供**命令行工具**和 **Web API** 两种使用方式。

上传一段音频文件，即可自动检测其中包含语音的片段，输出每个语音片段的起止时间点。支持 mp3、wav、flac、ogg、m4a 等常见音频格式。

## 功能特性

- 基于企业级预训练模型 Silero VAD，检测精度高
- 自动归一化低音量音频，无需手动调整
- 支持 4 种输出格式：JSON / Text / CSV / SRT
- 提供 REST API，便于集成到其他系统
- 支持 Docker 一键部署
- CPU 即可运行，无需 GPU

## 项目结构

```
vad/
├── app.py              # FastAPI Web API 服务
├── vad_label.py        # 命令行工具
├── dockerfile          # Docker 部署配置
├── requirements.txt    # Python 依赖
└── README.md
```

---

## 方式一：命令行工具

### 安装依赖

```bash
# 安装 CPU 版 PyTorch（体积小，无需 GPU）
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装 silero-vad 及其他依赖
pip3 install silero-vad

# 系统需要安装 ffmpeg（用于 mp3 解码）
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg
```

或使用 requirements.txt：

```bash
pip3 install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

> **注意**：`pip install torch` 默认安装 CUDA 版本（含 NVIDIA 依赖，体积约 2GB+）。如只需 CPU 推理，务必加上 `--index-url https://download.pytorch.org/whl/cpu`（约 200MB）。

### 基本用法

```bash
python3 vad_label.py <音频文件> [选项]
```

### 参数说明

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `audio` | — | 必填 | 输入音频文件路径 |
| `--threshold` | `-t` | 0.5 | VAD 检测阈值 (0.0-1.0)，越高越严格 |
| `--min-speech` | `-s` | 250 | 最小语音时长（毫秒） |
| `--min-silence` | `-S` | 100 | 最小静音时长（毫秒） |
| `--window` | `-w` | 512 | 窗口大小（采样点数） |
| `--format` | `-f` | text | 输出格式：text / json / csv / srt |
| `--output` | `-o` | — | 输出文件路径（默认输出到终端） |
| `--no-normalize` | — | false | 禁用自动归一化 |

### 使用示例

```bash
# 基本用法 - 文本格式输出
python3 vad_label.py input.mp3

# 降低阈值，检测更多语音片段
python3 vad_label.py input.mp3 -t 0.2

# JSON 格式输出到文件
python3 vad_label.py input.mp3 -f json -o result.json

# CSV 格式输出
python3 vad_label.py input.mp3 -f csv

# SRT 字幕格式（可在播放器中可视化查看 VAD 区域）
python3 vad_label.py input.mp3 -f srt -o vad.srt

# 调整检测参数
python3 vad_label.py input.mp3 -t 0.6 -s 300 -S 150

# 禁用自动归一化
python3 vad_label.py input.mp3 --no-normalize
```

### 输出格式示例

**text 格式**（默认）：

```
文件: input.mp3
检测到 5 个语音片段:

  [001] 00:01.100 -> 00:02.500  (时长: 1.400s)
  [002] 00:04.800 -> 00:05.800  (时长: 1.000s)
  [003] 00:07.000 -> 00:07.600  (时长: 0.600s)
  [004] 00:08.700 -> 00:09.200  (时长: 0.500s)
  [005] 07:14.700 -> 07:15.000  (时长: 0.300s)

总语音时长: 3.800s
```

**json 格式**：

```json
{
  "file": "input.mp3",
  "total_segments": 5,
  "total_speech_sec": 3.8,
  "segments": [
    {"start": 1.1, "end": 2.5},
    {"start": 4.8, "end": 5.8},
    {"start": 7.0, "end": 7.6},
    {"start": 8.7, "end": 9.2},
    {"start": 434.7, "end": 435.0}
  ]
}
```

**csv 格式**：

```csv
segment_id,start_sec,end_sec,duration_sec,start_fmt,end_fmt
1,1.100,2.500,1.400,00:01.100,00:02.500
2,4.800,5.800,1.000,00:04.800,00:05.800
3,7.000,7.600,0.600,00:07.000,00:07.600
4,8.700,9.200,0.500,00:08.700,00:09.200
5,434.700,435.000,0.300,07:14.700,07:15.000
```

**srt 格式**：

```
1
00:00:01,100 --> 00:00:02,500
[VAD] 1.100s - 2.500s

2
00:00:04,800 --> 00:00:05,800
[VAD] 4.800s - 5.800s
```

---

## 方式二：Web API

### Docker 部署（推荐）

```bash
# 构建镜像
podman build -t vad-api .
# 或使用 docker
docker build -t vad-api .

# 运行容器
podman run -d -p 8000:8000 --name vad-api vad-api
# 或使用 docker
docker run -d -p 8000:8000 --name vad-api vad-api
```

### 本地直接启动

```bash
pip3 install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
uvicorn app:app --host 0.0.0.0 --port 8000
```

启动后访问交互式 API 文档：`http://localhost:8000/docs`

### API 接口

#### 健康检查

```
GET /health
```

**响应示例**：

```json
{"status": "ok", "model": "silero-vad"}
```

#### VAD 检测

```
POST /vad
```

**请求参数**：

| 参数 | 位置 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | form-data | file | 必填 | 音频文件 |
| `threshold` | query | float | 0.5 | VAD 阈值 (0.0-1.0) |
| `min_speech` | query | int | 250 | 最小语音时长（毫秒） |
| `min_silence` | query | int | 100 | 最小静音时长（毫秒） |
| `window` | query | int | 512 | 窗口采样点数 |
| `normalize` | query | bool | true | 自动归一化低音量音频 |
| `format` | query | string | json | 输出格式：json / text / csv / srt |

**调用示例**：

```bash
# JSON 格式（默认）
curl -X POST 'http://localhost:8000/vad?threshold=0.2' \
  -F 'file=@input.mp3'

# Text 格式
curl -X POST 'http://localhost:8000/vad?threshold=0.2&format=text' \
  -F 'file=@input.mp3'

# CSV 格式
curl -X POST 'http://localhost:8000/vad?format=csv' \
  -F 'file=@input.mp3'

# SRT 格式
curl -X POST 'http://localhost:8000/vad?format=srt' \
  -F 'file=@input.mp3' -o vad.srt

# 禁用归一化
curl -X POST 'http://localhost:8000/vad?normalize=false' \
  -F 'file=@input.mp3'
```

**JSON 响应示例**：

```json
{
  "file": "input.mp3",
  "total_segments": 5,
  "total_speech_sec": 3.8,
  "segments": [
    {"start": 1.1, "end": 2.5},
    {"start": 4.8, "end": 5.8},
    {"start": 7.0, "end": 7.6},
    {"start": 8.7, "end": 9.2},
    {"start": 434.7, "end": 435.0}
  ]
}
```

**错误响应**：

```json
// 400 - 不支持的文件格式
{"detail": "不支持的文件格式: .avi，支持: .aac, .flac, .m4a, .mp3, .ogg, .wav, .wma"}

// 500 - 检测失败
{"detail": "VAD 检测失败: ..."}
```

---

## 参数调优指南

### threshold（阈值）

最关键的参数，直接影响检测灵敏度：

| 场景 | 建议值 | 说明 |
|------|--------|------|
| 安静环境录音 | 0.5（默认） | 标准灵敏度 |
| 有背景噪声 | 0.6 - 0.7 | 提高阈值过滤噪声 |
| 低音量录音 | 0.2 - 0.3 | 降低阈值捕捉更多语音 |
| 精确检测 | 0.5 - 0.6 | 平衡误检和漏检 |

> 对于低音量音频，工具会自动归一化（将振幅放大到 1.0），因此通常只需降低阈值即可。

### min_speech（最小语音时长）

- 默认 250ms，过滤掉太短的误检
- 如果需要捕捉单字/短词，可降低至 100-150ms
- 如果只关注完整语句，可提高至 500ms

### min_silence（最小静音时长）

- 默认 100ms，相邻语音片段间隔低于此值会合并
- 如果语音中有短暂停顿不应被切分，提高至 200-300ms
- 如果需要精确捕捉每个语音起止点，保持默认

### 自动归一化

当音频最大振幅低于 0.5 时，自动将振幅缩放到 1.0。适用于：
- 录音音量过低的文件
- 远场麦克风录制的音频

如确认音频音量正常且不希望修改原始数据，可通过 `--no-normalize`（命令行）或 `normalize=false`（API）关闭。

---

## 技术细节

- **VAD 模型**：Silero VAD v6，基于 PyTorch JIT，模型大小约 2MB
- **采样率**：Silero VAD 支持 8000Hz 和 16000Hz，`read_audio` 会自动重采样
- **处理速度**：30ms 音频片段处理耗时 < 1ms（CPU），600 秒音频约 3 秒完成检测
- **音频后端**：优先使用 ffmpeg 解码（支持 mp3 等格式），也支持 soundfile（仅 wav/flac）

---

## 依赖说明

| 依赖 | 用途 |
|------|------|
| `silero-vad` | VAD 模型加载与推理 |
| `torch` | PyTorch 运行时（CPU 版即可） |
| `torchaudio` | 音频文件读取与重采样 |
| `fastapi` | Web API 框架 |
| `uvicorn` | ASGI 服务器 |
| `python-multipart` | 文件上传支持 |
| `ffmpeg`（系统） | MP3 等格式解码 |

---

## License

MIT