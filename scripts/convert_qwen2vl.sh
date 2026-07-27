#!/bin/bash
# ============================================================================
# Qwen2-VL-2B-Instruct 完整转换流水线 (RK3588)
#
# 自动处理:
#   - rkllm/rknn 环境切换
#   - transformers 版本切换 (A阶段用4.45, C阶段用5.8)
#   - 已完成步骤自动跳过
#
# 用法:
#   bash scripts/convert_qwen2vl.sh
#   MODEL_PATH=/path/to/model ANDROID_NDK_HOME=/path/to/ndk \
#     bash scripts/convert_qwen2vl.sh
# ============================================================================
set -euo pipefail

# === 配置 =====================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/model/Qwen/Qwen2-VL-2B-Instruct}"

CONDA_RKNN="${CONDA_RKNN:-rknn-toolkit2}"
CONDA_RKLLM="${CONDA_RKLLM:-rkllm}"
SDK_DIR="${PROJECT_ROOT}/sdk"
MM_DEMO_DIR="${SDK_DIR}/examples/multimodal_model_demo"

TARGET="${TARGET:-rk3588}"
IMAGE_H="${IMAGE_H:-392}"
IMAGE_W="${IMAGE_W:-392}"
BATCH="${BATCH:-1}"

NDK_PATH="${NDK_PATH:-${ANDROID_NDK_HOME:-${HOME}/opts/android-ndk-r21e}}"
DEMO_INSTALL_DIR="${MM_DEMO_DIR}/deploy/install/demo_Android_arm64-v8a"
LIBOMP_PATH="${SDK_DIR}/rkllm-runtime/Android/librkllm_api/arm64-v8a/libomp.so"

OUT_DIR="${MODEL_PATH}/output"
VISION_ONNX="${OUT_DIR}/qwen2_vl_2b_vision.onnx"
VISION_RKNN="${OUT_DIR}/qwen2_vl_2b_vision_${TARGET}.rknn"
LLM_RKLLM="${OUT_DIR}/Qwen2-VL-2B-Instruct_w8a8_${TARGET^^}.rkllm"

# === 工具函数 =================================================================
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; N='\033[0m'
title()  { echo -e "\n${B}==== [$1] $2 ====${N}"; }
ok()    { echo -e "${G}✅ $1${N}"; }
warn()  { echo -e "${Y}⚠️  $1${N}"; }
die()   { echo -e "${R}❌ $1${N}"; exit 1; }
skip()  { echo -e "${Y}⏭️  跳过 ($1 已存在)${N}"; }

mkdir -p "${OUT_DIR}"
eval "$(conda shell.bash hook)"

# ============================================================================
# Stage A: Vision → ONNX (transformers==4.45.0)
# ============================================================================
title "A" "Vision → ONNX"

if [ -f "${VISION_ONNX}" ]; then
    skip "${VISION_ONNX}"
else
    conda activate "${CONDA_RKLLM}"

    # A阶段需要 transformers 4.x (有 model.visual / model.vision_model)
    CURRENT_TF=$(python -c "import transformers; print(transformers.__version__)")
    if [[ "$CURRENT_TF" == 5.* ]]; then
        echo "  切换 transformers: ${CURRENT_TF} → 4.45.0"
        pip install -q transformers==4.45.0
    fi

    python -c "from transformers import Qwen2VLForConditionalGeneration; print('  transformers OK')"

    cd "${MM_DEMO_DIR}/export"

    echo "  A1) 生成位置编码..."
    python export_vision_qwen2.py \
        --step 1 --path "${MODEL_PATH}" \
        --batch ${BATCH} --height ${IMAGE_H} --width ${IMAGE_W}

    echo "  A2) 导出 ONNX..."
    python export_vision_qwen2.py \
        --step 0 --path "${MODEL_PATH}" \
        --savepath "${VISION_ONNX}" \
        --batch ${BATCH} --height ${IMAGE_H} --width ${IMAGE_W}
    ok "${VISION_ONNX}"
fi

# ============================================================================
# Stage B: ONNX → RKNN (rknn-toolkit2 环境, 不关 transformers 事)
# ============================================================================
title "B" "ONNX → RKNN"

if [ -f "${VISION_RKNN}" ]; then
    skip "${VISION_RKNN}"
else
    conda activate "${CONDA_RKNN}"
    python -c "from rknn.api import RKNN; print('  rknn-toolkit2 OK')"

    python "${PROJECT_ROOT}/scripts/convert_vision_rknn.py" \
        --path "${VISION_ONNX}" \
        --target ${TARGET} \
        --batch ${BATCH} \
        --height ${IMAGE_H} --width ${IMAGE_W} \
        --output "${VISION_RKNN}"
    ok "${VISION_RKNN}"
fi

# ============================================================================
# Stage C: LLM → RKLLM (transformers==5.8.0, rkllm-toolkit 要求)
# ============================================================================
title "C" "LLM: 量化数据 + RKLLM"

if [ -f "${LLM_RKLLM}" ]; then
    skip "${LLM_RKLLM}"
else
    conda activate "${CONDA_RKLLM}"

    # transformers 5.8.0 (rkllm-toolkit 1.3.0 硬要求)
    CURRENT_TF=$(python -c "import transformers; print(transformers.__version__)")
    if [[ "$CURRENT_TF" != 5.8.0 ]]; then
        echo "  切换 transformers: ${CURRENT_TF} → 5.8.0"
        pip install -q transformers==5.8.0
    fi

    # 恢复原始 config.json (之前可能被 patch 过)
    CFG_BAK="${MODEL_PATH}/config.json.bak"
    if [ -f "${CFG_BAK}" ]; then
        cp "${CFG_BAK}" "${MODEL_PATH}/config.json"
        echo "  已恢复原始 config.json"
    fi

    python -c "from rkllm.api import RKLLM; print('  rkllm-toolkit OK')"

    echo "  C1) 生成量化校准数据..."
    cd "${MM_DEMO_DIR}"
    python data/make_input_embeds_for_quantize.py \
        --path "${MODEL_PATH}" \
        --model_type qwen2vl
    ok "校准数据已生成"

    echo "  C2) LLM → RKLLM (optimization_level=1, 约 20-40 分钟)..."
    cd "${PROJECT_ROOT}/scripts"
    python export_qwen2vl_llm.py \
        --model "${MODEL_PATH}" \
        --target ${TARGET} \
        --device cpu
    ok "${LLM_RKLLM}"
fi

# ============================================================================
# Stage D: 编译 demo
# ============================================================================
title "D" "编译板端多模态 demo"

DEPLOY_DIR="${PROJECT_ROOT}/deploy/multimodal"
if [ -f "${DEPLOY_DIR}/demo" ]; then
    skip "${DEPLOY_DIR}/demo"
elif [ -d "${NDK_PATH}" ]; then
    cd "${MM_DEMO_DIR}/deploy"
    sed -i "s|ANDROID_NDK_PATH=.*|ANDROID_NDK_PATH=${NDK_PATH}|g" build-android.sh
    ./build-android.sh
    [ -f "${DEMO_INSTALL_DIR}/demo" ] || die "Android demo 产物不存在: ${DEMO_INSTALL_DIR}"
    [ -f "${LIBOMP_PATH}" ] || die "Android OpenMP 运行库不存在: ${LIBOMP_PATH}"
    mkdir -p "${DEPLOY_DIR}"
    cp -a "${DEMO_INSTALL_DIR}/." "${DEPLOY_DIR}/"
    cp -a "${LIBOMP_PATH}" "${DEPLOY_DIR}/lib/libomp.so"
    ok "编译完成: ${DEPLOY_DIR}"
else
    warn "跳过编译 (NDK 不存在: ${NDK_PATH})"
fi

# ============================================================================
# 完成
# ============================================================================
echo ""
echo -e "${G}============================================================${N}"
echo -e "${G}  🎉 全部完成!${N}"
echo -e "${G}============================================================${N}"
echo ""
echo "产物:"
echo "  ${VISION_RKNN}"
echo "  ${LLM_RKLLM}"
echo ""
echo "推送:"
echo "  adb push ${VISION_RKNN} /data/models/"
echo "  adb push ${LLM_RKLLM} /data/models/"
echo ""
echo "板端运行:"
echo "  adb shell"
echo "  cd /data/demo_multimodal && export LD_LIBRARY_PATH=./lib"
echo "  ln -sf /data/models ."
echo "  ./demo demo.jpg \\"
echo "    models/qwen2_vl_2b_vision_${TARGET}.rknn \\"
echo "    models/Qwen2-VL-2B-Instruct_w8a8_${TARGET^^}.rkllm \\"
echo "    256 4096 3 ${TARGET} \\"
echo '    "<|vision_start|>" "<|vision_end|>" "<|image_pad|>"'
