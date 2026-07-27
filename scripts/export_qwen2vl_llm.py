"""
Qwen2-VL-2B-Instruct LLM → RKLLM
完全对齐 SDK 官方 export_rkllm.py 参数
"""

import argparse
import os
import sys

from rkllm.api import RKLLM
from transformers import AutoConfig

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str,
                    default=os.path.join(PROJ, "model/Qwen/Qwen2-VL-2B-Instruct"))
parser.add_argument("--target", type=str, default="rk3588")
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

modelpath = args.model
target = args.target.lower()

# 芯片参数
CHIP = {"rk3588": (3, "w8a8"), "rk3576": (2, "w4a16"),
        "rk3562": (1, "w8a8"), "rv1126b": (1, "w8a8")}
npu_cores, quant_dtype = CHIP[target]

# 量化校准数据
dataset = os.path.join(PROJ, "sdk/examples/multimodal_model_demo/data/llm_inputs.json")
if not os.path.exists(dataset):
    print(f"[WARN] 校准数据不存在: {dataset}, 跳过")
    dataset = None

# 输出路径
out_dir = os.path.join(modelpath, "output")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, f"{os.path.basename(modelpath)}_{quant_dtype}_{target.upper()}.rkllm")


def restore_multimodal_rope_config(rkllm, model_path):
    """Restore mRoPE fields dropped by rkllm-toolkit 1.3.0's VL loader."""
    source_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    source_rope = getattr(source_config.text_config, "rope_parameters", None)
    converted_config = getattr(getattr(rkllm.base, "model", None), "config", None)
    converted_rope = getattr(converted_config, "rope_parameters", None)

    if not isinstance(source_rope, dict) or "mrope_section" not in source_rope:
        raise RuntimeError("Qwen2-VL text_config is missing rope_parameters.mrope_section")
    if not isinstance(converted_rope, dict):
        raise RuntimeError("RKLLM converted model is missing config.rope_parameters")

    converted_rope.update(source_rope)
    return converted_rope["mrope_section"]

print("=" * 60)
print(f"  模型:    {modelpath}")
print(f"  芯片:    {target.upper()}  |  NPU: {npu_cores}核  |  量化: {quant_dtype}")
print(f"  输出:    {out}")
print("=" * 60)

# 加载 (完全对齐 SDK: 不传额外参数)
print("\n[1/3] 加载模型...")
llm = RKLLM()
ret = llm.load_huggingface(model=modelpath, device=args.device)
if ret != 0:
    print("❌ 加载失败!")
    sys.exit(ret)
print("✅ 加载成功")

# rkllm-toolkit 1.3.0 rebuilds Qwen2-VL as Qwen2ForCausalLM and drops
# mrope_section. Its optimizer still executes Qwen2VLAttention, which needs it.
try:
    mrope_section = restore_multimodal_rope_config(llm, modelpath)
except (AttributeError, RuntimeError, TypeError) as exc:
    print(f"❌ mRoPE 配置恢复失败: {exc}")
    sys.exit(1)
print(f"✅ mRoPE 配置已恢复: {mrope_section}")

# 构建 (完全对齐 SDK 参数)
print("\n[2/3] 构建 RKLLM (optimization_level=1)...")
ret = llm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype=quant_dtype,
    quantized_algorithm="normal",
    target_platform=target,
    num_npu_core=npu_cores,
    extra_qparams=None,
    dataset=dataset,
)
if ret != 0:
    print("❌ 构建失败!")
    sys.exit(ret)
print("✅ 构建成功")

# 导出
print("\n[3/3] 导出...")
ret = llm.export_rkllm(out)
if ret != 0:
    print("❌ 导出失败!")
    sys.exit(ret)
print(f"\n✅ Done: {out}")
