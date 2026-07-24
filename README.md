# RKLLM Quickstart — 瑞芯微 LLM 部署一站式工程

[![Platform](https://img.shields.io/badge/platform-RK3588%20%7C%20RK3576-blue)](https://www.rock-chips.com/)
[![SDK](https://img.shields.io/badge/RKLLM%20SDK-v1.3.0-green)]()
[![Model](https://img.shields.io/badge/model-DeepSeek--R1--Qwen--1.5B-orange)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

**一句话说明**：帮你把 HuggingFace 大模型 → 量化压缩 → 编译运行 → 部署到 Rockchip 板子上，全流程跑通。

---

## 目录结构

```
rkllm-quickstart/
├── scripts/                ← 所有脚本集中于此
│   ├── export.py           # 模型转换 (HuggingFace → .rkllm)
│   ├── quant_data.py       # 生成量化校准数据
│   ├── build-android.sh    # 编译 Android 可执行文件
│   ├── build-linux.sh      # 编译 Linux 可执行文件
│   └── data_quant.json     # 量化校准样本
├── model/                  ← 模型文件 (需自行下载)
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/  # 下载后放这里
│   │   └── output/                     # 转换产物 .rkllm
│   └── custom/             # 自定义模型配置文件
├── deploy/                 ← 编译后自动生成（可推板）
├── GUIDE.md                ← 详细使用手册（图文）
└── README.md               ← 本文件
```

> **注意**: `sdk/` 目录需从 [Rockchip 官方](https://github.com/airockchip/rknn-llm) 克隆到此，未包含在本仓库中。

---

## 快速开始 (5 步)

### 第 1 步：环境准备

```bash
# 安装 RKLLM Toolkit
pip install rkllm_toolkit-*-cp3*-linux_x86_64.whl

# 安装 Android NDK r21e（仅编译需要）
# 下载: https://dl.google.com/android/repository/android-ndk-r21e-linux-x86_64.zip
unzip android-ndk-r21e-linux-x86_64.zip -d ~/opts/
```

### 第 2 步：下载模型

```bash
cd model/
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

### 第 3 步：模型转换（量化）

```bash
cd scripts/
# 可修改 export.py 中的 target_platform / quantized_dtype
conda activate rkllm
python export.py
# 输出: ../model/DeepSeek-R1-Distill-Qwen-1.5B/output/*.rkllm
```

### 第 4 步：编译板端程序

```bash
cd scripts/
./build-android.sh     # 输出: ../deploy/android/
# 或
./build-linux.sh       # 输出: ../deploy/linux/
```

### 第 5 步：部署运行

```bash
adb push ../deploy/android /data/
adb push ../model/DeepSeek-R1-Distill-Qwen-1.5B/output/*.rkllm /data/android/
adb shell

cd /data/android
export LD_LIBRARY_PATH=./lib
export RKLLM_LOG_LEVEL=1
./llm_demo <model>.rkllm 2048 4096
```

---

## 支持的模型

本项目基于 RKLLM SDK v1.3.0，支持以下模型架构：

| 文本模型 | 多模态模型 |
|----------|-----------|
| LLaMA / TinyLLAMA | Qwen2-VL / Qwen3-VL |
| Qwen2 / 2.5 / 3 / 3.5 | MiniCPM-V-2_6 |
| Phi2 / Phi3 | InternVL2-1B / InternVL3-1B |
| ChatGLM3-6B | Janus-Pro-1B |
| Gemma2 / 3 / 3n / 4 | DeepSeekOCR |
| InternLM2 | SmolVLM |
| MiniCPM3 / 4 | |
| TeleChat2 | |
| DeepSeek-R1-Distill | |
| SmolLM3 | |
| RWKV7 | |

---

## 支持的芯片

| 芯片 | NPU 核数 | 推荐量化 | 备注 |
|------|---------|---------|------|
| **RK3588** | 3 核, 6 TOPS | W8A8 | 支持 W8A8 和 W4A16_GX |
| **RK3576** | 2 核 | W4A16 | 支持 W8A8 和 W4A16 |
| RK3562 | 1 核 | W8A8 | |
| RV1126B | 1 核 | W8A8 | |

---

## 切换模型 / 芯片

### 换芯片 — 改 3 个参数（`scripts/export.py`）

```python
target_platform = "RK3588"   # RK3588 | RK3576 | RK3562 | RV1126B
quantized_dtype = "W8A8"     # RK3588→W8A8, RK3576→W4A16 更省内存
num_npu_core = 3             # RK3588=3, RK3576=2, RK3562=1
```

### 换模型 — 改 1 行 + 创建新脚本

```bash
# 复制模板
cp scripts/export.py scripts/export_qwen3.py

# 编辑第 11 行:
#   modelpath = '.../model/Qwen3-4B-Instruct'
# 第 20 行:
#   dtype="float32"          # CPU 必须 float32，CUDA 可用 float16
# 第 41 行:
#   max_context=8192         # 4B 模型加大上下文
```

### 量化兼容表（选错直接报错）

| 芯片 | W8A8 | W4A16 | 推荐 |
|------|:----:|:-----:|------|
| RK3588 | ✅ | ❌ | `W8A8 normal` |
| RK3576 | ✅ | ✅ | `W4A16 grq` |
| RK3562 | ✅ | ❌ | `W8A8 normal` |
| RV1126B | ✅ | ❌ | `W8A8 normal` |

### 模型参数对照

| 模型 | modelpath | max_context | 板载内存建议 |
|------|-----------|-------------|------------|
| DeepSeek-R1-1.5B | `.../model/DeepSeek-R1-Distill-Qwen-1.5B` | 4096 | 4GB+ |
| Qwen3-4B-Instruct | `.../model/Qwen3-4B-Instruct` | 8192 | 8GB+ |

> 更多详见 **[GUIDE.md 第 3 章](./GUIDE.md)**
>
> 📖 **性能优化实战教程**: **[docs/PERFORMANCE.md](./docs/PERFORMANCE.md)** — 从 0 到生产级的完整优化历程

---

## 实测性能

| 模型 | 芯片 | 量化 | 速度 | 内存 |
|------|------|------|------|------|
| DeepSeek-R1-1.5B | RK3588 | W8A8 | ~20 tok/s | ~2.5GB |

---

详见 **[GUIDE.md](./GUIDE.md)** 获取完整图文教程。
