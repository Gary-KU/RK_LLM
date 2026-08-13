---
name: rkllm-multimodal
description: >
  Build and deploy vision-language models on Rockchip NPUs. Covers Qwen2-VL,
  Qwen3-VL, InternVL, MiniCPM-V, DeepSeekOCR, and SmolVLM using a Vision RKNN
  plus an LLM RKLLM. Use for multimodal conversion, quantization, demo builds,
  Windows-to-board ADB deployment, or mRoPE failures on RK3588/RK3576.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# RKLLM Multimodal Deployment Skill

## Canonical Guide

Read `docs/QWEN2_VL_DEPLOYMENT.md` before changing Qwen2-VL scripts or running a deployment. Keep detailed commands there instead of duplicating divergent procedures in this skill.

## Architecture

```text
Image -> Vision + Projector (.rknn) -> image embeddings
                                             |
Text prompt ---------------------------------+-> LLM (.rkllm) -> response
```

Do not use the text-only `rkchat` binary for multimodal inference. Build and deploy the SDK `multimodal_model_demo`.

## Workspace Mapping

- Linux build host: `gary@192.168.180.128`
- Linux project: `/home/gary/RK3576/rknn/05_llm`
- Windows product view: `X:\RK3576\rknn\05_llm`
- Android models: `/data/models`
- Android demo: `/data/demo_multimodal`

Build on Linux, then use Windows `adb` to deploy. If an automation session cannot see `X:`, copy artifacts from Linux to a local staging directory with SCP. Do not assume the drive mapping exists in every Windows session.

## Validated Qwen2-VL Configuration

| Item | Value |
|------|-------|
| Model | `Qwen/Qwen2-VL-2B-Instruct` |
| Target | RK3588, 3 NPU cores |
| Quantization | W8A8 normal |
| RKLLM Toolkit | 1.3.0 |
| Transformers / PyTorch | 5.8.0 / 2.6.0 |
| RKNN Toolkit2 | 2.3.2 |

Use separate `rkllm` and `rknn-toolkit2` Conda environments. Initialize Conda in non-interactive SSH sessions with:

```bash
source /home/gary/miniconda3/etc/profile.d/conda.sh
```

## Conversion

Run the complete resumable pipeline:

```bash
cd /home/gary/RK3576/rknn/05_llm
bash scripts/convert_qwen2vl.sh
```

The stages are Vision to ONNX, ONNX to RKNN, multimodal calibration plus LLM quantization, and Android demo compilation when the NDK is available.

For an LLM-only rebuild:

```bash
source /home/gary/miniconda3/etc/profile.d/conda.sh
conda activate rkllm
cd /home/gary/RK3576/rknn/05_llm
python scripts/export_qwen2vl_llm.py \
  --model model/Qwen/Qwen2-VL-2B-Instruct \
  --target rk3588 \
  --device cpu
```

## RKLLM 1.3.0 mRoPE Compatibility

RKLLM rebuilds `Qwen2VLForConditionalGeneration` as `Qwen2ForCausalLM`, but its optimizer still executes `Qwen2VLAttention`. The rebuilt `Qwen2Config` can lose `mrope_section`, causing:

```text
ERROR: layer running Error: 'mrope_section'!
```

`scripts/export_qwen2vl_llm.py` restores the complete dictionary from `AutoConfig.from_pretrained(model_path).text_config.rope_parameters` after `load_huggingface()` and before `build()`. Do not downgrade the toolkit-pinned Transformers version or patch the source model's `config.json` as a workaround.

This warning is expected because Vision is exported separately through RKNN:

```text
rkllm-toolkit only exports Qwen2ForCausalLM of Qwen2VLForConditionalGeneration
```

## Expected Artifacts

```text
model/Qwen/Qwen2-VL-2B-Instruct/output/
|-- qwen2_vl_2b_vision_rk3588.rknn
`-- Qwen2-VL-2B-Instruct_w8a8_RK3588.rkllm
```

Verify exact byte counts, build-log errors, and SHA-256 before deployment.

## Android Demo

```bash
cd /home/gary/RK3576/rknn/05_llm/sdk/examples/multimodal_model_demo/deploy
./build-android.sh
```

The Android install directory is `install/demo_Android_arm64-v8a`. It must contain `demo`, `imgenc`, `demo.jpg`, and `lib/`.
When packaging `deploy/multimodal`, also copy `sdk/rkllm-runtime/Android/librkllm_api/arm64-v8a/libomp.so` into `lib/`; `librkllmrt.so` requires it at runtime.

## Windows ADB Deployment

```powershell
adb devices -l
adb shell "mkdir -p /data/models /data/demo_multimodal"
adb push X:\RK3576\rknn\05_llm\model\Qwen\Qwen2-VL-2B-Instruct\output\qwen2_vl_2b_vision_rk3588.rknn /data/models/
adb push X:\RK3576\rknn\05_llm\model\Qwen\Qwen2-VL-2B-Instruct\output\Qwen2-VL-2B-Instruct_w8a8_RK3588.rkllm /data/models/
adb push X:\RK3576\rknn\05_llm\deploy\multimodal\. /data/demo_multimodal/
```

Do not delete or overwrite unrelated board files. Compare local and remote sizes after each push.

## Run

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
  "<|vision_start|>" "<|vision_end|>" "<|image_pad|>"
```

## Resource Checks

- Reserve at least 10 GB of build-host disk space. The validated output directory, including ONNX external data, uses about 5.8 GB.
- Expect roughly 9 GB RAM usage for this 2B conversion.
- Check `/data` before pushing approximately 3.5 GB of models.
- Use matching RKLLM and RKNN runtime libraries from the same SDK release family.
- Before a large Vision model returns `rknn_init=-6`, check `dmesg` for IOVA allocation failures and stop unrelated resident RKLLM/RKNN processes. A Qwen3-4B `rkchat` used about 5.2 GB and blocked the validated Qwen2-VL Vision model until it exited.
