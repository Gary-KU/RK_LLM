#!/bin/bash
# rkchat.sh — one-command launch rkchat on ADB device
# Usage:
#   ./rkchat.sh              print the launch command
#   ./rkchat.sh --push       push latest build, then print command
#   ./rkchat.sh --go         push + launch directly
#   ./rkchat.sh -m deepseek  use DeepSeek-R1-1.5B model

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---- config ----
MODEL_NAME="${RKCHAT_MODEL:-Qwen3-4B-Instruct}"
MODEL_FILE="${MODEL_NAME}_W8A8_RK3588.rkllm"
DEVICE_DIR="/data/local/tmp/android"
MAX_TOK=4096
CTX_LEN=8192

# check for alternative model
if [ "$1" = "-m" ]; then
    case "$2" in
        deepseek|ds|1.5b) MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
                         MODEL_FILE="${MODEL_NAME}_W8A8_RK3588.rkllm" ;;
        qwen|qwen3|4b)    MODEL_NAME="Qwen3-4B-Instruct"
                         MODEL_FILE="${MODEL_NAME}_W8A8_RK3588.rkllm" ;;
        *) echo "Unknown model: $2"; echo "Known: deepseek, qwen3"; exit 1 ;;
    esac
    shift 2
fi

MODEL_SRC="$PROJECT_DIR/model/$MODEL_NAME/output/$MODEL_FILE"
BIN_SRC="$PROJECT_DIR/deploy/android/rkchat"
LIB_SRC="$PROJECT_DIR/deploy/android/lib"

# ---- helpers ----
push_build() {
    echo "==> Pushing rkchat..."
    adb push "$BIN_SRC" "$DEVICE_DIR/" >/dev/null
    adb push "$LIB_SRC/librkllmrt.so" "$DEVICE_DIR/lib/" >/dev/null
    adb push "$LIB_SRC/libomp.so" "$DEVICE_DIR/lib/" >/dev/null
    adb shell "chmod 755 $DEVICE_DIR/rkchat"
    echo "     done."
}

push_model() {
    if adb shell "[ -f $DEVICE_DIR/$MODEL_FILE ]" 2>/dev/null; then
        echo "==> Model already on device: $MODEL_FILE"
    else
        echo "==> Pushing model ($(du -sh "$MODEL_SRC" | cut -f1))..."
        adb push "$MODEL_SRC" "$DEVICE_DIR/"
        echo "     done."
    fi
}

launch_cmd() {
    echo ""
    echo "  ──────────────────────────────────────────"
    echo "  Copy & paste to start:"
    echo ""
    echo "  adb shell"
    echo "  export LD_LIBRARY_PATH=$DEVICE_DIR/lib"
    echo "  $DEVICE_DIR/rkchat $DEVICE_DIR/$MODEL_FILE $MAX_TOK $CTX_LEN"
    echo "  ──────────────────────────────────────────"
    echo ""
}

# ---- main ----
case "${1:-}" in
    --push)
        push_build
        launch_cmd
        ;;
    --go)
        push_build
        push_model
        echo "==> Launching rkchat..."
        echo "     (Ctrl+C once to stop gen, twice to quit, /exit to leave)"
        exec adb shell "cd $DEVICE_DIR && export LD_LIBRARY_PATH=./lib && ./rkchat $MODEL_FILE $MAX_TOK $CTX_LEN"
        ;;
    --push-model)
        push_model
        launch_cmd
        ;;
    *)
        launch_cmd
        ;;
esac
