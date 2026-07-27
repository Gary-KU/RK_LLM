"""
Qwen2-VL-2B Vision ONNX → RKNN 转换
区别于 Qwen2.5-VL-3B，Qwen2-VL-2B 的 ONNX 只有单个图像输入
"""

from rknn.api import RKNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="ONNX 模型路径")
parser.add_argument("--target", type=str, default="rk3588", help="目标芯片")
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--height", type=int, default=392)
parser.add_argument("--width", type=int, default=392)
parser.add_argument("--output", type=str, default=None, help="输出路径")
args = parser.parse_args()

# Qwen2-VL 的 image_mean/image_std
mean_value = [[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]]
std_value  = [[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]]

# 输出路径
if args.output:
    savepath = args.output
else:
    import os
    base = os.path.splitext(os.path.basename(args.path))[0]
    os.makedirs("./rknn", exist_ok=True)
    savepath = f"./rknn/{base}_{args.target}.rknn"

print("Configuring RKNN...")
rknn = RKNN(verbose=True)
rknn.config(
    target_platform=args.target,
    mean_values=mean_value,
    std_values=std_value,
)

print("Loading ONNX...")
# Qwen2-VL ONNX 只有一个输入，不需要手动指定 input name
ret = rknn.load_onnx(args.path)
if ret != 0:
    print(f"load_onnx failed! ret={ret}")
    exit(ret)

print("Building RKNN (this may take several minutes)...")
ret = rknn.build(do_quantization=False, dataset=None)
if ret != 0:
    print(f"build failed! ret={ret}")
    exit(ret)

print("Exporting RKNN...")
ret = rknn.export_rknn(savepath)
if ret != 0:
    print(f"export_rknn failed! ret={ret}")
    exit(ret)

print(f"\nDone: {savepath}")
