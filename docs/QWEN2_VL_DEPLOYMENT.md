# Qwen2-VL 多模态部署实测手册

本文记录 `Qwen/Qwen2-VL-2B-Instruct` 在 RK3588 上的模型转换、产品整理和 Windows ADB 部署流程。详细步骤以本文件为准。

## 1. 验证范围

验证日期：2026-07-27。

| 项目 | 实测值 |
|------|--------|
| 模型 | `Qwen/Qwen2-VL-2B-Instruct` |
| 目标芯片 | RK3588，3 NPU 核 |
| LLM 量化 | W8A8 normal |
| RKLLM Toolkit | 1.3.0 |
| Transformers / PyTorch | 5.8.0 / 2.6.0 |
| RKNN Toolkit2 | 2.3.2 |
| 校准集 | 20 条图文 `inputs_embeds` 样本 |

## 2. 部署结构

Qwen2-VL 需要两个板端模型：

```text
图片 -> Vision + Projector -> image embeddings
          RKNN (.rknn)              |
                                     +-> LLM (.rkllm) -> 文本回答
文本 prompt -------------------------|
```

- Vision + Projector 使用 RKNN Toolkit2 转换为 `.rknn`。
- Language Model 使用 RKLLM Toolkit 转换为 `.rkllm`。
- 板端使用 `multimodal_model_demo`，不能使用纯文本 `rkchat`。

## 3. 三端路径映射

| 位置 | 路径 |
|------|------|
| Linux 项目目录 | `/path/to/RK_LLM` |
| Windows Samba 示例 | `X:\RK3576\rknn\05_llm` |
| Android 模型目录 | `/data/models` |
| Android demo 目录 | `/data/demo_multimodal` |

服务器负责模型转换和 demo 交叉编译。Windows 从 `X:` 读取产品并使用本机 `adb` 推送。服务会话可能看不到桌面用户的 `X:` 映射，此时先用 SCP 暂存到本地磁盘，不能通过 PowerShell 文本管道传输二进制模型。

## 4. 资源要求

- Linux x86_64 编译环境，建议至少 16 GB RAM。
- 建议至少预留 10 GB 编译磁盘空间。
- 本次量化峰值约 9 GB RSS。
- 本次 `output/` 约 5.8 GB，其中包含 ONNX external data。
- Android demo 使用 NDK r21e。
- 两个板端模型合计约 3.5 GB。

`qwen2_vl_2b_vision.onnx` 主文件会引用同目录下的大量外部权重，生成 RKNN 前不能只保留主 ONNX。

## 5. 环境检查

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh

conda run -n rkllm python -c \
  "import torch, transformers; from rkllm.api import RKLLM; print(transformers.__version__, torch.__version__)"

conda run -n rknn-toolkit2 python -m pip show rknn-toolkit2
```

RKLLM Toolkit 1.3.0 固定依赖 `transformers==5.8.0` 和 `torch==2.6.0`。不要通过降级 Transformers 规避量化错误。

## 6. 完整转换

```bash
cd /path/to/RK_LLM
bash scripts/convert_qwen2vl.sh
```

流水线包含：

1. Vision + Projector 转 ONNX。
2. ONNX 转 RKNN。
3. 生成多模态校准数据并将 LLM 量化为 RKLLM。
4. NDK 存在时编译 Android 多模态 demo。

脚本会跳过已经存在的最终产物。Vision 导出和 LLM 转换对 Transformers 的要求不同，脚本会处理版本切换；不要在流水线运行时并行修改同一 Conda 环境。

## 7. 仅重建语言模型

生成校准数据：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate rkllm
cd /path/to/RK_LLM/sdk/examples/multimodal_model_demo

python data/make_input_embeds_for_quantize.py \
  --path /path/to/RK_LLM/model/Qwen/Qwen2-VL-2B-Instruct \
  --model_type qwen2vl
```

输出为 `data/llm_inputs.json` 和 `data/llm_inputs/sample_*`。

正式量化：

```bash
cd /path/to/RK_LLM
python scripts/export_qwen2vl_llm.py \
  --model model/Qwen/Qwen2-VL-2B-Instruct \
  --target rk3588 \
  --device cpu
```

## 8. `mrope_section` 错误

原错误：

```text
Optimizing model: 0%| | 0/28
ERROR: layer running Error: 'mrope_section'!
```

RKLLM Toolkit 1.3.0 会把 Qwen2-VL 的语言部分重建为 `Qwen2ForCausalLM`。重建出的 `Qwen2Config.rope_parameters` 丢失 `mrope_section`，但优化器仍执行 `Qwen2VLAttention.forward()`，因此触发 `KeyError`。

当前 `scripts/export_qwen2vl_llm.py` 已在 `load_huggingface()` 之后：

1. 使用 `AutoConfig.from_pretrained()` 读取原模型。
2. 获取 `text_config.rope_parameters`。
3. 将完整字典恢复到 `llm.base.model.config.rope_parameters`。
4. 确认 `mrope_section` 存在后再调用 `llm.build()`。

正常日志应包含：

```text
加载成功
mRoPE 配置已恢复: [16, 24, 24]
Optimizing model: 100%|...| 28/28
构建成功
```

以下 warning 是预期行为，表示 RKLLM 只承载语言部分：

```text
rkllm-toolkit only exports Qwen2ForCausalLM of Qwen2VLForConditionalGeneration
```

不要手工修改模型 `config.json`，也不要降级 RKLLM 依赖版本。

## 9. 模型产物与校验

| 文件 | 本次实测大小 |
|------|--------------|
| `qwen2_vl_2b_vision_rk3588.rknn` | 1,401,423,226 bytes |
| `Qwen2-VL-2B-Instruct_w8a8_RK3588.rkllm` | 2,050,725,076 bytes |

```bash
cd /path/to/RK_LLM
OUT=model/Qwen/Qwen2-VL-2B-Instruct/output
stat "$OUT/qwen2_vl_2b_vision_rk3588.rknn"
stat "$OUT/Qwen2-VL-2B-Instruct_w8a8_RK3588.rkllm"
sha256sum "$OUT"/*.rknn "$OUT"/*.rkllm
```

本次 RKLLM SHA-256：

```text
ddf75ba89a6f71d9dd41dfb28ae66299dc210fb0d1c97a2f13ae3ac1bce9a261
```

重新构建后哈希可能变化，应以当前构建结果为准。

## 10. 编译 Android demo

```bash
cd /path/to/RK_LLM/sdk/examples/multimodal_model_demo/deploy
./build-android.sh
```

Android 安装目录为：

```text
install/demo_Android_arm64-v8a/
|-- demo
|-- imgenc
|-- demo.jpg
`-- lib/
```

`scripts/convert_qwen2vl.sh` 会把它整理到仓库的 `deploy/multimodal/`。
流水线还会从 `sdk/rkllm-runtime/Android/librkllm_api/arm64-v8a/libomp.so` 补齐 RKLLM runtime 的 OpenMP 依赖，最终产品的 `lib/` 必须同时包含 `librkllmrt.so`、`librknnrt.so` 和 `libomp.so`。

## 11. Windows ADB 部署

```powershell
adb devices -l
adb shell "df -h /data"
adb shell "mkdir -p /data/models /data/demo_multimodal"

$root = 'X:\RK3576\rknn\05_llm'
$out = "$root\model\Qwen\Qwen2-VL-2B-Instruct\output"

adb push "$out\qwen2_vl_2b_vision_rk3588.rknn" /data/models/
adb push "$out\Qwen2-VL-2B-Instruct_w8a8_RK3588.rkllm" /data/models/
adb push "$root\deploy\multimodal\." /data/demo_multimodal/
```

推送后比较字节数：

```powershell
adb shell "ls -l /data/models /data/demo_multimodal"
```

## 12. 板端运行

```bash
adb shell
cd /data/demo_multimodal
chmod 755 demo imgenc
export LD_LIBRARY_PATH=./lib
ln -sfn /data/models models

./demo demo.jpg \
  models/qwen2_vl_2b_vision_rk3588.rknn \
  models/Qwen2-VL-2B-Instruct_w8a8_RK3588.rkllm \
  256 4096 3 rk3588 \
  "<|vision_start|>" \
  "<|vision_end|>" \
  "<|image_pad|>"
```

`max_context_len` 必须大于文本 token 数、图像 token 数与 `max_new_tokens` 之和。
建议交互产品先将 `max_new_tokens` 设为 128-256；`2048` 会放大异常提示词或未命中 EOS 时的等待时间。项目 demo 已增加 token 回调 `fflush` 和输入 EOF 检查，确保生成内容及时显示且非交互测试不会空循环。

## 13. 常见问题

### 只推 `.rkllm`，无法识别图片

必须同时部署 Vision `.rknn`、LLM `.rkllm` 和 multimodal demo。

### `libomp.so`、`librknnrt.so` 或 `librkllmrt.so` 缺失

推送 `deploy/multimodal/lib/` 中的完整运行库，设置 `LD_LIBRARY_PATH=./lib`，并确保 runtime 与模型工具链属于同一 SDK 版本。`build-android.sh` 的原始 install 目录不会自动安装 `libomp.so`，项目流水线会在整理产品时补入。

### demo 编译成功但产品整理失败

Android 输出目录是 `install/demo_Android_arm64-v8a`，不是 `install/demo_Linux_aarch64`。项目脚本已按 Android 目录整理产品。

### `rknn_init` 返回 `-6`，内核出现 `failed to allocate IOVA`

先检查板上是否已有其他 RKLLM/RKNN 进程占用 NPU 和大块地址空间：

```bash
ps -A -o PID,RSS,NAME,ARGS | grep -E 'rkchat|demo|rknn|rkllm'
dmesg | grep -i -E 'rknpu|IOVA' | tail
```

本次故障由另一个 Qwen3-4B `rkchat` 占用约 5.2 GB RAM 和 NPU/IOVA 资源引起。正常停止旧模型进程后，Vision RKNN 加载恢复。不要在资源有限的板上同时常驻两个大模型服务。

## 14. 本次产品状态

- Vision RKNN：已生成并校验。
- LLM RKLLM：20 样本 W8A8 量化完成，28/28 层优化成功，日志无 ERROR。
- Android demo：已编译；已修复产品包缺少 `libomp.so`、token 输出不 flush 和输入 EOF 空循环。
- Windows ADB 推板：已完成，7 个初始产品文件及补充的 `libomp.so` 均通过服务器、本地、板端 SHA-256 对比。
- 板端图文推理：已完成。128 最大新 token 测试中，LLM 热加载约 2.42 秒，Vision 加载约 1.30 秒，图像编码约 3.70 秒，整条命令约 13.9 秒完成。
- 示例输出：`The image shows an astronaut sitting on the moon with a green beer bottle in his hand, enjoying the view of Earth and the stars.`
