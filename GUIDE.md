# RKLLM Quickstart 完全指南

> 从零开始在 RK3588/RK3576 上部署大语言模型

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [下载模型](#2-下载模型)
3. [导入新模型（通用流程）](#3-导入新模型通用流程)
4. [模型转换（量化）](#4-模型转换量化)
5. [编译板端程序](#5-编译板端程序)
6. [板端部署与运行](#6-板端部署与运行)
7. [跨平台迁移指南](#7-跨平台迁移指南)
8. [支持的模型与芯片](#8-支持的模型与芯片)
9. [项目结构说明](#9-项目结构说明)
10. [常见问题](#10-常见问题)
11. [Qwen2-VL 多模态部署](#11-qwen2-vl-多模态部署)

---

## 1. 环境搭建

### 1.1 硬件要求

| 阶段 | 环境 | 要求 |
|------|------|------|
| 模型转换 | PC / Linux 虚拟机 | 16GB+ 内存，CPU 或 NVIDIA GPU |
| 模型编译 | PC / Linux 虚拟机 | Android NDK r21e |
| 模型运行 | RK3588 开发板 | Android 或 Linux 系统 |

### 1.2 安装 RKLLM Toolkit

RKLLM Toolkit 是 Rockchip 官方提供的模型转换工具包，用于将 HuggingFace 格式的模型转换为板端可用的 `.rkllm` 格式。

```bash
# 下载 RKLLM SDK (含 toolkit wheel)
git clone https://github.com/airockchip/rknn-llm.git

# 安装对应 Python 版本的 wheel
cd rknn-llm/rkllm-toolkit/packages/
pip install rkllm_toolkit-1.3.0-cp312-cp312-linux_x86_64.whl

# 验证安装
python -c "from rkllm.api import RKLLM; print('RKLLM OK')"
```

支持的 Python 版本: **3.9 / 3.10 / 3.11 / 3.12**

> **注意**: wheel 文件是 Linux x86_64 的，Windows 上无法安装。请使用 Linux 物理机、WSL 或虚拟机。

### 1.3 安装 Android NDK（仅编译需要）

如果你使用的是 Linux 系统（不是 Android），可跳过此步，改用 `build-linux.sh`。

```bash
# 下载 NDK r21e
wget https://dl.google.com/android/repository/android-ndk-r21e-linux-x86_64.zip

# 解压到 ~/opts/
mkdir -p ~/opts
unzip android-ndk-r21e-linux-x86_64.zip -d ~/opts/

# 验证
ls ~/opts/android-ndk-r21e/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++
```

### 1.4 克隆本工程

```bash
git clone <this-repo-url>
cd rkllm-quickstart

# 克隆 RKLLM SDK 到 sdk/ 目录
git clone https://github.com/airockchip/rknn-llm.git sdk
```

---

## 2. 下载模型

从 HuggingFace 下载原始模型权重：

```bash
cd model/

# DeepSeek-R1-Distill-Qwen-1.5B (3.4GB)
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# 或者其他支持的模型，例如：
# git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
```

> **模型目录结构**：
> ```
> model/DeepSeek-R1-Distill-Qwen-1.5B/
> ├── model.safetensors    # 3.4GB, 模型权重
> ├── config.json          # 模型配置
> ├── tokenizer.json       # 分词器
> ├── tokenizer_config.json
> └── generation_config.json
> ```

---

## 3. 导入新模型（通用流程）

> 本节以 Qwen3-4B 为例，完整演示从零接入一个新模型的全过程。

### 3.1 第一步：确定模型名称

在 HuggingFace 上搜索模型时，精确名称容易被截断或混淆。可靠做法：

```bash
# 用 hf 命令列出所有相关模型（显示完整 ID）
hf models ls --search "Qwen3 4B" --author Qwen --no-truncate
```

输出中重点关注 `LIKES` 排行靠前的 `text-generation` 标签模型。例如：

| ID | Likes | 说明 |
|----|-------|------|
| `Qwen/Qwen3-4B-Instruct-2507` | 902 | ✅ 要用这个 |
| `Qwen/Qwen3-4B` | 662 | Base 模型，未指令微调 |

> **教训**: 不要凭猜测写模型名。先 `hf models ls --search` 确认精确 ID。

### 3.2 第二步：下载模型

```bash
cd model/

# 方式1: hf 命令行（推荐，支持断点续传）
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir ./Qwen3-4B-Instruct

# 方式2: Python API（国内网络更稳）
python3 << PYEOF
from huggingface_hub import snapshot_download
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
snapshot_download("Qwen/Qwen3-4B-Instruct-2507", local_dir="./Qwen3-4B-Instruct")
PYEOF
```

> **网络问题**: 国内直连 HF 大概率失败。务必设置 `HF_ENDPOINT=https://hf-mirror.com` 镜像，或使用 ModelScope。

### 3.3 第三步：创建转换脚本

复制模板并修改 4 个关键参数：

```bash
cp scripts/export.py scripts/export_YOUR_MODEL.py
```

需要修改的位置及含义：

```python
# ====== 第 11 行: 模型路径 ======
modelpath = '/home/.../model/Qwen3-4B-Instruct'   # 改成你的模型目录

# ====== 第 20 行: 加载参数 ======
ret = llm.load_huggingface(
    model=modelpath,
    device='cpu',       # CPU 稳定但慢 | CUDA 快但需显卡
    dtype="float32"     # CPU 必须用 float32 | CUDA 可用 float16
)

# ====== 第 34-38 行: 量化参数 ======
target_platform = "RK3588"    # 你的芯片型号
quantized_dtype = "W8A8"      # 量化方式（查下表）
num_npu_core = 3              # NPU 核数
max_context = 8192            # 根据板子内存调整
```

### 3.4 第四步：选择量化方案（关键！）

**不同芯片支持的量化类型不同，选错直接报错**：

| 芯片 | 支持 W8A8 | 支持 W4A16 | 推荐方案 | 4B 模型占用 |
|------|----------|-----------|---------|------------|
| RK3588 | ✅ | ❌ (需 W4A16_GX) | `W8A8 + normal` | ~4GB |
| RK3576 | ✅ | ✅ | `W4A16 + grq` | ~2GB |
| RK3562 | ✅ | ❌ | `W8A8 + normal` | ~4GB |

对应代码：
```python
# RK3588 / RK3562
quantized_dtype = "W8A8"
quantized_algorithm = "normal"

# RK3576（更省内存）
quantized_dtype = "W4A16"
quantized_algorithm = "grq"
```

### 3.5 内存预算速查

| 模型参数量 | FP32 大小 | W8A8 大小 | 推荐板载内存 |
|-----------|----------|----------|-------------|
| 0.5B | 2GB | 0.5GB | 2GB+ |
| 1.5B | 6GB | 1.5GB | 4GB+ |
| 3B-4B | 12-16GB | 3-4GB | 8GB+ |
| 7B | 28GB | 7GB | 16GB+ |

> 加上运行时开销（上下文缓存 + 系统），建议板载内存 ≥ 模型大小 × 2。

### 3.6 支持的模型架构速查

| 架构 | 模型名示例 | 备注 |
|------|---------|------|
| Qwen3 | `Qwen/Qwen3-4B-Instruct-2507` | 需用完整名 |
| Qwen2/2.5 | `Qwen/Qwen2.5-1.5B-Instruct` | |
| DeepSeek-R1 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Distill 版 |
| LLaMA | `meta-llama/Llama-3.2-1B` | |
| Phi | `microsoft/Phi-3-mini-4k-instruct` | |
| Gemma | `google/gemma-2-2b-it` | |
| ChatGLM3 | `THUDM/chatglm3-6b` | 仅 6B 版 |
| InternLM2 | `internlm/internlm2-1.8b` | |
| MiniCPM | `openbmb/MiniCPM3-4B` | |
| RWKV7 | `fla-hub/rwkv-7` | 需 Python 3.12 |

---

## 4. 模型转换（量化）

### 4.1 转换流程图

```
HuggingFace 模型 (FP16/FP32)
    │
    ▼ load_huggingface()
加载到内存
    │
    ▼ build(do_quantization=True)
量化 + 优化 (W8A8 / W4A16)
    │
    ▼ export_rkllm()
.rkllm 格式输出
    │
    ▼
推送到 RK3588 板端推理
```

### 4.2 运行转换

```bash
cd scripts/

# 激活 conda 环境（如果用了 conda）
conda activate rkllm

# 开始转换（CPU 模式，约 20-30 分钟）
python export.py
```

### 4.3 转换参数说明

`scripts/export.py` 中的关键参数：

```python
modelpath = '.../model/DeepSeek-R1-Distill-Qwen-1.5B'  # 模型路径

# 加载参数
device = 'cpu'      # 'cpu' 或 'cuda'（GPU 更快）
dtype = 'float16'   # 'float32' / 'float16' / 'bfloat16'

# 量化参数
target_platform = "RK3588"   # 目标芯片
quantized_dtype = "W8A8"     # W8A8(8bit) / W4A16(4bit)
num_npu_core = 3             # NPU 核数
max_context = 4096           # 最大上下文长度
```

### 4.4 预期输出

```
INFO: rkllm-toolkit version: 1.3.0
Load model ...
Optimizing model: 100%|████████| 28/28 [20:31<00:00]
Converting model: 100%|████████| 339/339 [00:00<00:00]
Exporting the model ...
Model saved to: ../model/DeepSeek-R1-Distill-Qwen-1.5B/output/
                DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm
```

---

## 5. 编译板端程序

### 5.1 Android 编译

```bash
cd scripts/
./build-android.sh
```

输出产物在 `../deploy/android/`:

```
deploy/android/
├── llm_demo          # 可执行文件 (4.7MB)
└── lib/
    ├── librkllmrt.so # RKLLM 运行时 (4.5MB)
    └── libomp.so     # OpenMP 运行时
```

### 5.2 Linux 编译

如果你的板子跑的是 Linux 系统：

```bash
cd scripts/

# 先修改交叉编译工具链路径
# 编辑 build-linux.sh 中的 GCC_COMPILER_PATH

./build-linux.sh
```

---

## 6. 板端部署与运行

### 6.1 推送文件

```bash
# 推 Android 二进制
adb push ../deploy/android /data/

# 推模型
adb push ../model/DeepSeek-R1-Distill-Qwen-1.5B/output/*.rkllm /data/android/

# 推定频脚本（优化 NPU 性能）
adb push ../sdk/scripts/fix_freq_rk3588.sh /data/android/
```

### 6.2 板端运行

```bash
adb shell
cd /data/android

# 设置库路径
export LD_LIBRARY_PATH=./lib

# 定频（提升 NPU 推理稳定性）
sh fix_freq_rk3588.sh

# 开启性能日志
export RKLLM_LOG_LEVEL=1

# 运行推理
./llm_demo DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm 2048 4096
```

用法: `./llm_demo <模型文件> <max_new_tokens> <max_context_len>`

### 6.3 预期输出

```
rkllm init start
rkllm init success

******************可输入以下问题对应序号获取回答/或自定义输入********************

[0] 现有一笼子，里面有鸡和兔子若干只，数一数，共有头14个，腿38条...
[1] 有28位小朋友排成一行,从左边开始数第10位是学豆...

*************************************************************************

user: 0
robot: <think>
首先，设鸡的数量为x，兔子的数量为y...
</think>

鸡有 9 只，兔子有 5 只。
```

---

## 7. 跨平台迁移指南

### 7.1 更换芯片

当你要从 RK3588 切换到其他芯片时，只需修改 `scripts/export.py` 中的 3 个参数：

```python
# ============ 第 27-31 行 ============
target_platform = "RK3576"    # RK3588 → RK3576
quantized_dtype = "W4A16"     # W8A8 → W4A16 (RK3576 推荐)
num_npu_core = 2              # 3 → 2
```

| 芯片 | target_platform | 推荐量��� | num_npu_core |
|------|----------------|----------|-------------|
| RK3588 | `"RK3588"` | `"W8A8"` | `3` |
| RK3576 | `"RK3576"` | `"W4A16"` 或 `"W8A8"` | `2` |
| RK3562 | `"RK3562"` | `"W8A8"` | `1` |
| RV1126B | `"RV1126B"` | `"W8A8"` | `1` |

### 7.2 更换模型

替换模型只需改 `scripts/export.py` 第 11 行的 `modelpath`：

```python
# DeepSeek-R1-1.5B
modelpath = '/path/to/DeepSeek-R1-Distill-Qwen-1.5B'

# 换成 Qwen2.5
modelpath = '/path/to/Qwen2.5-1.5B-Instruct'

# 换成 LLaMA
modelpath = '/path/to/Llama-3.2-3B'
```

### 7.3 编译目标切换

`scripts/build-android.sh` 中的关键变量：

| 变量 | Android | Linux |
|------|---------|-------|
| `ANDROID_NDK_PATH` | `~/opts/android-ndk-r21e` | 不需要 |
| `CMAKE_SYSTEM_NAME` | `Android` | `Linux` |
| `TARGET_ARCH` | `arm64-v8a` | `aarch64` |

### 7.4 修改编译目标架构

在 `build-android.sh` 第 8 行：
```bash
TARGET_ARCH=arm64-v8a     # 64位 ARM
# TARGET_ARCH=armeabi-v7a # 32位 ARM（较少用）
```

在 `build-linux.sh` 中修改交叉编译器路径（第 8 行）：
```bash
GCC_COMPILER_PATH=/your/toolchain/path/bin/aarch64-none-linux-gnu
```

---

## 8. 支持的模型与芯片

### 8.1 模型架构支持表

基于 **RKLLM SDK v1.3.0**：

#### 纯文本 LLM

| 模型家族 | HuggingFace 示例 |
|----------|-----------------|
| LLaMA | `meta-llama/Llama-3.2-1B`, `TinyLlama/TinyLlama-1.1B` |
| Qwen2/2.5/3/3.5 | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen3-0.6B` |
| Phi2/Phi3 | `microsoft/phi-2`, `microsoft/Phi-3-mini-4k-instruct` |
| ChatGLM3 | `THUDM/chatglm3-6b` |
| Gemma2/3/3n/4 | `google/gemma-2-2b-it` |
| InternLM2 | `internlm/internlm2-1.8b` |
| MiniCPM3/4 | `openbmb/MiniCPM3-4B` |
| TeleChat2 | `Tele-AI/TeleChat2-3B` |
| DeepSeek-R1-Distill | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| SmolLM3 | `HuggingFaceTB/SmolLM3-1.7B-Instruct` |
| RWKV7 | RNN 架构，需要 Python 3.12 |

#### 多模态 VL 模型

| 模型家族 | 支持视觉 |
|----------|---------|
| Qwen2-VL / Qwen3-VL | ✅ 图像理解 |
| MiniCPM-V-2_6 | ✅ 图像理解 |
| InternVL2-1B / InternVL3-1B | ✅ 图像理解 |
| Janus-Pro-1B | ✅ 多模态 |
| DeepSeekOCR | ✅ OCR 识别 |
| SmolVLM | ✅ 图像理解 |

Qwen2-VL 需要分别生成 Vision `.rknn` 和 LLM `.rkllm`，不能套用纯文本 `rkchat` 流程。完整实测步骤见 **[Qwen2-VL 多模态部署手册](./docs/QWEN2_VL_DEPLOYMENT.md)**。

### 8.2 添加自定义模型

对于不在官方支持列表中的模型，可以通过 `model/custom/` 目录提供自定义定义：

```
model/custom/
├── configuration_custom.py   # 模型配置类
├── modeling_custom.py        # 模型结构定义
├── tokenizer.json            # 分词器
├── config.json               # 模型参数
└── generation_config.json    # 生成参数
```

然后在 `export.py` 中使用 `custom_config` 参数：
```python
ret = llm.load_huggingface(
    model=modelpath,
    custom_config='/path/to/model/custom',
    device='cpu',
    dtype='float32'
)
```

---

## 9. 项目结构说明

```
rkllm-quickstart/
│
├── scripts/                    # 所有用户脚本
│   ├── export.py              # ★ 核心: 模型转换脚本
│   │   # 修改: modelpath, target_platform, quantized_dtype, num_npu_core
│   │
│   ├── convert_qwen2vl.sh     # Qwen2-VL Vision + LLM 完整流水线
│   ├── export_qwen2vl_llm.py  # Qwen2-VL LLM 量化与 mRoPE 兼容修复
│   │
│   ├── quant_data.py          # 量化校准数据生成
│   │   # 用法: python quant_data.py -m /path/to/model
│   │
│   ├── build-android.sh       # Android 编译
│   │   # 修改: ANDROID_NDK_PATH
│   │
│   ├── build-linux.sh         # Linux 编译
│   │   # 修改: GCC_COMPILER_PATH
│   │
│   └── data_quant.json        # 量化校准样本（20 条中英文混合）
│
├── model/                      # 模型文件
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/  # 从 HF 下载的原始模型
│   │   └── output/            # 转换后的 .rkllm 文件
│   └── custom/                # 自定义模型定义
│
├── deploy/                     # 编译产物（自动生成）
│   ├── android/               # Android 部署包
│   │   ├── llm_demo
│   │   └── lib/
│   └── linux/                 # Linux 部署包
│
├── sdk/                        # RKLLM SDK（需自行克隆）
│   │ git clone https://github.com/airockchip/rknn-llm.git sdk
│   ├── rkllm-toolkit/         # 模型转换工具包
│   ├── rkllm-runtime/         # 运行时库
│   └── scripts/               # 定频/性能评估脚本
│
├── README.md                   # 快速上手
├── GUIDE.md                    # 本详细指南
└── .gitignore                  # Git 忽略规则
```

---

## 10. 常见问题

### Q1: 模型转换报 "target_platform not support quantized_dtype"

```python
# RK3588 不支持 W4A16，改用 W8A8
quantized_dtype = "W8A8"

# RK3576 支持 W4A16 和 W8A8
quantized_dtype = "W4A16"  # 或 "W8A8"
```

### Q2: 推板后报 "No such file or directory"

说明编译产物与板子系统不匹配：
- Android 板子 → 必须用 `build-android.sh` 编译
- Linux 板子 → 必须用 `build-linux.sh` 编译

### Q3: 报 "libomp.so not found"

```bash
# 从 SDK 复制
cp sdk/rkllm-runtime/Android/librkllm_api/arm64-v8a/libomp.so deploy/android/lib/

# 重新推送
adb push deploy/android/lib/libomp.so /data/android/lib/
```

### Q4: 转换速度太慢

- CPU 模式：1.5B 模型约 20-30 分钟
- CUDA 模式：5-10 分钟（需 `device='cuda'` + `dtype='float16'`）
- 大型模型（7B+）建议用 CUDA，否则可能需要数小时

### Q5: 如何加速推理

1. 运行定频脚本：`sh fix_freq_rk3588.sh`
2. 减少 `max_context` 参数（默认 4096）
3. 使用更高压缩比的量化：`W4A16`（如果芯片支持）

### Q6: 内存不足怎么办

- RK3588：DeepSeek-R1-1.5B W8A8 约需 2.5GB RAM
- 减少上下文长度：`max_context = 2048`
- 使用更小模型：Qwen2.5-0.5B / TinyLlama-1.1B

---

## 11. Qwen2-VL 多模态部署

Qwen2-VL 使用两套工具链和两个板端模型：

| 阶段 | 工具链 | 输出 |
|------|--------|------|
| Vision + Projector | RKNN Toolkit2 2.3.2 | `.rknn` |
| Language Model | RKLLM Toolkit 1.3.0 | `.rkllm` |

RK3588 的实测配置为 W8A8、3 NPU 核。服务器端可直接运行：

```bash
cd /path/to/RK_LLM
bash scripts/convert_qwen2vl.sh
```

RKLLM 1.3.0 会把 Qwen2-VL 的语言部分重建为 `Qwen2ForCausalLM`。旧实现会在此过程中丢失 `mrope_section`，并在优化第 1 层时报错。当前 `scripts/export_qwen2vl_llm.py` 已从原始 `text_config.rope_parameters` 恢复该字段，并完成 20 条多模态样本、28 层优化的正式验证。

Windows 侧通过 `X:\RK3576\rknn\05_llm` 获取服务器产物，再使用本机 `adb` 推送到 RK3588 板。具体目录、命令、校验方法和 special token 参数统一维护在 **[Qwen2-VL 多模态部署实测手册](./docs/QWEN2_VL_DEPLOYMENT.md)**。

---

## 参考资源

- [RKLLM GitHub](https://github.com/airockchip/rknn-llm)
- [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [Rockchip 官方文档](https://opensource.rock-chips.com/)
- [HuggingFace DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [Qwen2-VL 多模态部署实测手册](./docs/QWEN2_VL_DEPLOYMENT.md)

---

*最后更新: 2026-07*
