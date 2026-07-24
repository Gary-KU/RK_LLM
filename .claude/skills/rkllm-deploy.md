---
name: rkllm-deploy
description: >
  Deploy, optimize, and debug LLM models on Rockchip NPU (RK3588/RK3576).
  Covers: model compilation (HuggingFace→.rkllm), rkchat building, ADB deployment,
  performance tuning, crash recovery, and API reference. Use when the user asks about
  Rockchip, RKLLM, rkchat, NPU deployment, on-device LLM, or references this repo.
  Triggers on: RK3588, RK3576, rkllm, rkchat, Rockchip, NPU, ADB deployment,
  HuggingFace→rkllm, model quantization (W8A8/W4A16), on-device inference.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# RKLLM Deployment Skill

You are an expert in deploying LLMs on Rockchip NPU boards (RK3588/RK3576) using the RKLLM SDK.

## Project Overview

This repo (`RK_LLM`) is a complete toolkit for:
1. Converting HuggingFace models to `.rkllm` (quantized NPU format)
2. Building a production-grade interactive chat CLI (`rkchat`)
3. Deploying via ADB to Android devices
4. Performance profiling and optimization

## Key Files

| File | Purpose |
|------|---------|
| `src/rkchat.cpp` | Main chat program (~530 lines, single file) |
| `scripts/export.py` | Model conversion (HuggingFace → .rkllm) |
| `scripts/build-android.sh` | Cross-compile for Android via NDK |
| `scripts/rkchat.sh` | Host-side quick-launch script |
| `scripts/rkchat-device.sh` | Device-side launcher |
| `docs/PERFORMANCE.md` | 12-chapter optimization tutorial |
| `docs/RKLLM_API.md` | Complete SDK API reference |
| `README.md` | Project overview with features + customization guide |

## Source Code Architecture (rkchat.cpp)

All in one file, key sections:

| Section | Function/Area | What it does |
|---------|--------------|--------------|
| ANSI Colors | `#define CLR_*` | Terminal color macros |
| Memory Guard | `mem_guard_check()` | Pre-flight /proc/meminfo check |
| Model Detection | `detect_model_template()` | Auto-match Qwen3/LLaMA/ChatGLM templates |
| Math Preprocessor | `preprocess_input()` | Intercept & compute math expressions |
| Eval | `eval_expr/term/factor()` | Recursive-descent math parser |
| Crash Recovery | `on_sigabrt_handler()` + `sigsetjmp` | C-library abort protection |
| Signal Handler | `on_sig()` | Double-tap Ctrl+C |
| Callback | `on_res()` | Streaming output + perf tracking |
| Stats | `print_stats()` | Performance panel |
| Banner | `print_banner()` | Startup ASCII art |
| Commands | `cmd()` | /help /clear /stats /preset /zh /en ... |
| Warmup | `do_warmup()` | Silent cache warmup |
| Main | `main()` | Init → loop → cleanup |

## Common Tasks

### Add a new command
Search for `if (s == "/help")` in `cmd()` and add a new block.

### Change default system prompt
Search for `g_sys_prompt =` (around line 55).

### Add a new model template
Edit `detect_model_template()` — add a new `if` block with the model family name and its `g_chat_prefix`/`g_chat_postfix`.

### Change sampling parameters
Edit the `PRESETS[]` array — each entry has `{name, desc, top_k, top_p, temp, repeat_penalty, ...}`.

### Enable/disable math preprocessor
Search for `preprocess_input()` in main loop — comment out the `if (processed.empty()) continue;` line.

### Adjust memory guard threshold
Search for `safe_margin = 2048` in `mem_guard_check()`.

## Model Compilation

To compile a HuggingFace model to .rkllm:

```bash
cd scripts
conda activate rkllm
# Edit export.py: change modelpath, max_context, quantized_dtype
python export.py
```

Key parameters in export.py:
- `modelpath`: Path to HuggingFace model
- `target_platform`: "RK3588" | "RK3576" | "RK3562" | "RV1126B"
- `quantized_dtype`: "W8A8" (RK3588) | "W4A16" (RK3576)
- `num_npu_core`: 3 (RK3588) | 2 (RK3576) | 1 (RK3562)
- `max_context`: 4096 (1.5B) | 8192 (4B+)

## Build & Deploy

```bash
# Build
cd scripts && ./build-android.sh

# Deploy binary
adb push ../deploy/android/rkchat /data/local/tmp/android/
adb push ../deploy/android/lib/ /data/local/tmp/android/lib/

# Deploy model
adb push ../model/<model>.rkllm /data/local/tmp/android/

# Launch
adb shell
cd /data/local/tmp/android
export LD_LIBRARY_PATH=./lib
./rkchat <model>.rkllm 4096 <context_len>
```

## Performance Baseline

| Model | Size | Memory | Decode Speed |
|-------|------|--------|-------------|
| DeepSeek-R1-1.5B | 2.0 GB | 1.8 GB | **10.2 tok/s** |
| Qwen3-4B-Instruct | 4.6 GB | 5.2 GB | **5.0 tok/s** |

## Known Limitations

- `enabled_cpus_mask` — conflicts with `enabled_cpus_num`, don't set both
- `n_batch` > 1 — causes segfault in SDK 1.3.0
- `rkllm_set_chat_template` — empty prefix/postfix causes tokenizer crash; always use model-specific markers
- RK3588 does NOT support W4A16 quantization (hardware limitation)
- C++ exceptions from `rkllm_run` (extern "C") cannot be caught with try-catch; use sigsetjmp instead

## Chat Template Markers

| Model Family | prefix | postfix |
|-------------|--------|---------|
| Qwen3 | `<\|im_start\|>user\n` | `<\|im_end\|>\n<\|im_start\|>assistant\n` |
| DeepSeek-Distill-Qwen | Same as Qwen3 | Same as Qwen3 |
| LLaMA | `<s>[INST] ` | ` [/INST]` |
| ChatGLM | `[Round 1]\n\n问：` | `\n\n答：` |

## Troubleshooting Quick Reference

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ret=-1` on init | `enabled_cpus_mask` conflict | Remove cpu mask, use default |
| `invalid character` crash | Empty chat template markers | Use model-specific prefix/postfix |
| `Aborted` / SIGABRT | C++ exception across C boundary | sigsetjmp is already in place |
| Math wrong | Model ignores precomputed result | Preprocessor intercepts before model |
| `<think>` tags in output | DeepSeek-R1 reasoning mode | Normal behavior, can filter in callback |
| `Text file busy` | Old process still running | `pkill rkchat; sleep 1` before push |

## When User Asks About Optimization

Always reference `docs/PERFORMANCE.md` for the full optimization journey. Key points:
- Decode speed is hardware-limited (memory bandwidth, not NPU FLOPS)
- Speed = model_size / bandwidth; cannot beat physics
- To go faster: use smaller model or switch to W4A16 (RK3576 only)
- TTFT can be optimized; decode speed fundamentally cannot
