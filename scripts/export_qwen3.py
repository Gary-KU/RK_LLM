from rkllm.api import RKLLM
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

'''
Qwen3-4B-Instruct
https://huggingface.co/Qwen/Qwen3-4B-Instruct
'''

modelpath = '/home/gary/RK3576/rknn/05_llm/model/Qwen3-4B-Instruct'
llm = RKLLM()

# Load model (CPU mode, 4B model needs ~16GB RAM)
# If you have GPU, change device='cuda' and dtype='float16' for faster conversion
ret = llm.load_huggingface(model=modelpath, model_lora=None, device='cpu', dtype="float32", custom_config=None, load_weight=True)
if ret != 0:
    print('Load model failed!')
    exit(ret)

# Build model
# W8A8 for RK3588: ~4GB model size, fits 16GB board
dataset = "./data_quant.json"
target_platform = "RK3588"
optimization_level = 1
quantized_dtype = "W8A8"
quantized_algorithm = "normal"
num_npu_core = 3

ret = llm.build(do_quantization=True, optimization_level=optimization_level, quantized_dtype=quantized_dtype,
                quantized_algorithm=quantized_algorithm, target_platform=target_platform, num_npu_core=num_npu_core, extra_qparams=None, dataset=dataset, hybrid_rate=0, max_context=8192)
if ret != 0:
    print('Build model failed!')
    exit(ret)

# Export rkllm model
ret = llm.export_rkllm(f"../model/Qwen3-4B-Instruct/output/{os.path.basename(modelpath)}_{quantized_dtype}_{target_platform}.rkllm")
if ret != 0:
    print('Export model failed!')
    exit(ret)
