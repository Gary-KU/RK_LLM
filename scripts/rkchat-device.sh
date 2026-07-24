#!/system/bin/sh
# rkchat launcher — drop this on device, run from adb shell
DIR=/data/local/tmp/android
MODEL=Qwen3-4B-Instruct_W8A8_RK3588.rkllm
cd $DIR
export LD_LIBRARY_PATH=./lib
exec ./rkchat $MODEL 4096 8192
