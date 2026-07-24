#!/system/bin/sh
# rkchat launcher — 根据参数切换模型
DIR=/data/local/tmp/android
cd $DIR
export LD_LIBRARY_PATH=./lib

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
