#!/system/bin/sh
# rkchat launcher — 启动前自动清理 NPU 占用，避免僵尸进程卡死
DIR=/data/local/tmp/android
cd $DIR
export LD_LIBRARY_PATH=./lib

# ============================================
# 清理旧进程 + 复位 NPU
# ============================================
cleanup_npu() {
    # 杀掉所有旧的 rkchat / llm_demo 进程
    for pid in $(ps -A | grep -E 'rkchat|llm_demo' | awk '{print $2}'); do
        kill -9 $pid 2>/dev/null
    done
    # 复位 NPU
    if [ -f /sys/kernel/debug/rknpu/reset ]; then
        echo 1 > /sys/kernel/debug/rknpu/reset 2>/dev/null
    fi
    sleep 0.5
}

cleanup_npu

# ============================================
# 选模型
# ============================================
case "${1:-fast}" in
    fast|deepseek|ds)
        MODEL=DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm
        CTX=4096 ;;
    qwen|qwen3|4b)
        MODEL=Qwen3-4B-Instruct_W8A8_RK3588.rkllm
        CTX=8192 ;;
    *)
        MODEL="$1"
        CTX=4096 ;;
esac

exec ./rkchat $DIR/$MODEL 4096 $CTX
