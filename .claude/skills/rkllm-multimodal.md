---
name: rkllm-multimodal
description: >
  Deploy VL (Vision-Language) multimodal models on Rockchip NPU.
  Covers Qwen2-VL, Qwen3-VL, InternVL, MiniCPM-V, DeepSeekOCR, SmolVLM.
  Two-step pipeline: Vision Encoder (RKNN) + LLM (RKLLM).
  Use when user asks about multimodal, VL, vision-language, image understanding on RK3588/RK3576.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# RKLLM Multimodal Deployment Skill

## Architecture

```
Image (OpenCV) → Vision Encoder (RKNN) → Image Embedding
                                               │
                                               ▼
                                        Concatenated to prompt
                                               │
Text Prompt ───────────────────────────→ LLM (RKLLM) → Response
```

Two models, two toolkits:

| Component | Format | Toolkit | Compile Script |
|-----------|--------|---------|---------------|
| Vision Encoder | `.rknn` | rknn-toolkit2 (needs `rknn_env`) | `export_vision_qwen2.py` |
| LLM | `.rkllm` | rkllm-toolkit (needs `rkllm` env) | `export.py` |

## Supported VL Models

| Model | Vision Export Script | Notes |
|-------|---------------------|-------|
| Qwen2-VL-2B/7B | `export_vision_qwen2.py` | Best choice for RK3588 (2B) |
| Qwen2.5-VL, Qwen3-VL | `export_vision.py` | `--model_name qwen3-vl` |
| InternVL2-1B, InternVL3-1B | `export_vision.py` | `--model_name internvl` |
| MiniCPM-V-2_6 | `export_vision.py` | `--model_name minicpm` |
| SmolVLM | `export_vision.py` | `--model_name smolvlm` |
| DeepSeekOCR | `export_vision.py` | Needs antialias=False fix |

## Model Download

```bash
# modelscope (China mirror, most reliable)
echo 'from modelscope import snapshot_download' > /tmp/dl.py
echo 'snapshot_download("Qwen/Qwen2-VL-2B-Instruct", cache_dir="/home/gary/RK3576/rknn/05_llm/model")' >> /tmp/dl.py
python /tmp/dl.py

# Or huggingface-cli
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct \
    --local-dir ~/RK3576/rknn/05_llm/model/Qwen2-VL-2B-Instruct \
    --resume-download
```

## Compilation (two steps, two envs)

### Step 1: Vision Encoder → RKNN (rknn-toolkit2 env)

```bash
conda activate <your_rknn_env>

# First pass: generate cu_seqlens and rotary_pos_emb
python sdk/examples/multimodal_model_demo/export/export_vision_qwen2.py \
    --step 1 --path model/Qwen2-VL-2B-Instruct --batch 1 --height 392 --width 392

# Second pass: export ONNX
python sdk/examples/multimodal_model_demo/export/export_vision_qwen2.py \
    --step 0 --path model/Qwen2-VL-2B-Instruct \
    --savepath ./qwen2-vl-vision.onnx --batch 1 --height 392 --width 392

# Convert ONNX to RKNN
python sdk/examples/multimodal_model_demo/export/export_vision_rknn.py \
    --path ./qwen2-vl-vision.onnx --model_name qwen2-vl --height 392 --width 392
```

### Step 2: LLM → RKLLM (rkllm env)

```bash
conda activate rkllm
cp scripts/export.py scripts/export_qwen2vl.py
# Edit: modelpath = 'model/Qwen2-VL-2B-Instruct', target_platform = 'RK3588'
python scripts/export_qwen2vl.py
```

## Build Multimodal Demo

```bash
cd sdk/examples/multimodal_model_demo/deploy
./build-android.sh
```

Output: `llm_vl_demo` binary

## Deploy & Run

Push to device:
```bash
adb push llm_vl_demo /data/local/tmp/mm/
adb push *.rknn /data/local/tmp/mm/
adb push *.rkllm /data/local/tmp/mm/
adb push test_image.jpg /data/local/tmp/mm/
adb push 3rdparty/librknnrt/Android/arm64-v8a/librknnrt.so /data/local/tmp/mm/
adb push 3rdparty/opencv/Android/arm64-v8a/ /data/local/tmp/mm/
```

Run:
```bash
adb shell
cd /data/local/tmp/mm
export LD_LIBRARY_PATH=.:./opencv
./llm_vl_demo vision.rknn llm.rkllm test_image.jpg "Describe this image"
```

## Key Differences from Text-Only rkchat

| | rkchat | Multimodal Demo |
|---|--------|----------------|
| Vision model | None | RKNN (.rknn) |
| Image processing | None | OpenCV |
| Input | Text | Text + Image |
| Dependencies | librkllmrt.so | librkllmrt.so + librknnrt.so + opencv |
| Binary size | ~4.7MB | ~5MB + opencv libs |

## Memory Requirements

| Model | Vision | LLM | Total |
|-------|--------|-----|-------|
| Qwen2-VL-2B | ~1.5 GB | ~2.5 GB | ~4 GB |
| Qwen2-VL-7B | ~2.5 GB | ~7 GB | ~10 GB (too big for RK3588) |

RK3588 with 16GB: Qwen2-VL-2B is the sweet spot.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `rknn-toolkit2` conflicts with `rkllm-toolkit` | Use separate conda envs |
| torch version conflict | rkllm needs torch 2.6, rknn needs torch 2.4 — separate envs mandatory |
| `cu_seqlens` dimension mismatch | Must match batch/height/width in both step 1 and 2 |
| `use_flash_attn` error | Set `"use_flash_attn": false` in config.json before export |
| ONNX export fails | `pip install onnx==1.18.0` |
| Image preprocessing wrong | Image is automatically expanded to square with padding |
