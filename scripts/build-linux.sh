#!/bin/bash
# Build RKLLM demo for Linux aarch64
# Usage: cd scripts && ./build-linux.sh
set -e

BUILD_TYPE=${BUILD_TYPE:-Release}
GCC_COMPILER_PATH=/opt/tool_chain/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu

C_COMPILER=${GCC_COMPILER_PATH}-gcc
CXX_COMPILER=${GCC_COMPILER_PATH}-g++
STRIP_COMPILER=${GCC_COMPILER_PATH}-strip

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
SOURCE_DIR=${PROJECT_DIR}/sdk/examples/rkllm_api_demo/deploy
BUILD_DIR=${SCRIPT_DIR}/build/linux_aarch64_${BUILD_TYPE}
INSTALL_DIR=${PROJECT_DIR}/deploy/linux

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SOURCE_DIR}" \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_C_COMPILER=${C_COMPILER} \
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j4
make install

echo "Done! Output: ${INSTALL_DIR}"
