#!/bin/bash
# Build RKLLM demo for Android
# Usage: cd scripts && ./build-android.sh
set -e

BUILD_TYPE=${BUILD_TYPE:-Release}
ANDROID_NDK_PATH=~/opts/android-ndk-r21e
TARGET_ARCH=arm64-v8a

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
SOURCE_DIR=${PROJECT_DIR}/sdk/examples/rkllm_api_demo/deploy
BUILD_DIR=${SCRIPT_DIR}/build/android_${TARGET_ARCH}_${BUILD_TYPE}
INSTALL_DIR=${PROJECT_DIR}/deploy/android

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SOURCE_DIR}" \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=23 \
    -DCMAKE_ANDROID_ARCH_ABI=${TARGET_ARCH} \
    -DCMAKE_ANDROID_STL_TYPE=c++_static \
    -DCMAKE_ANDROID_NDK=${ANDROID_NDK_PATH} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j4
make install

# CMakeLists.txt 硬编码了安装路径，手动复制到项目 deploy/
OUT_DIR=${SOURCE_DIR}/install/demo_Android_${TARGET_ARCH}
mkdir -p "${INSTALL_DIR}"/lib
cp -v "${OUT_DIR}"/rkchat "${INSTALL_DIR}"/
cp -v "${OUT_DIR}"/lib/librkllmrt.so "${INSTALL_DIR}"/lib/
# 补充 libomp.so（NDK 中的 OpenMP 运行时）
cp -v "${PROJECT_DIR}"/sdk/rkllm-runtime/Android/librkllm_api/arm64-v8a/libomp.so "${INSTALL_DIR}"/lib/ 2>/dev/null || true
# 设备端快捷启动脚本
cp -v "${SCRIPT_DIR}"/rkchat-device.sh "${INSTALL_DIR}"/rkchat.sh 2>/dev/null || true

echo "Done! Output: ${INSTALL_DIR}"
ls -la "${INSTALL_DIR}"