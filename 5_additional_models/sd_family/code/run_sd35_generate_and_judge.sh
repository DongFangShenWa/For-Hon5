#!/usr/bin/env bash
# ============================================================
# CollageAttack x SD3.5-Large 一键: 双卡生成图片 -> GPT-4.1 image score 评测
#
# 用法(在服务器上, 任意目录均可):
#   bash run_sd35_generate_and_judge.sh              # 全量 200 条, GPU2 + GPU3 各跑 100
#   LIMIT=5 bash run_sd35_generate_and_judge.sh      # 冒烟: 只用 GPU_A 生成前 5 条并评测
#
# 可用环境变量覆盖默认值:
#   MODEL_PATH  模型路径  (默认 /hub/huggingface/models/stabilityai/stable-diffusion-3.5-large)
#   GPU_A/GPU_B 两张卡的编号 (默认 2 / 3)
#   PYTHON      python 解释器 (默认 python)
#   OFFLOAD     1=开 cpu_offload(默认; 24GB 卡装不下全量 fp16 管线约 27GB, 必须开)
#               0=关(显存 >=40GB 的卡可关, 推理更快)
# ============================================================
set -uo pipefail

cd "$(dirname "$0")"   # 保证在 T2I_CollageAttack/ 下运行, 相对路径才有效

MODEL_PATH="${MODEL_PATH:-/hub/huggingface/models/stabilityai/stable-diffusion-3.5-large}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
PYTHON="${PYTHON:-python}"
LIMIT="${LIMIT:-0}"
TOTAL=200
HALF=$((TOTAL / 2))

# 24GB 卡: 开 cpu_offload, 组件按需上卡(推理时 transformer 独占整卡, 不影响出图)
GEN_ARGS=""
if [ "${OFFLOAD:-1}" = "1" ]; then
    GEN_ARGS="--cpu_offload"
    echo "(cpu_offload 已开启: 单卡 24GB 装不下全量 fp16 管线, offload 下推理时 transformer 独占 GPU)"
fi
# 降低显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# ---------- 阶段 1: 生成图片 ----------
if [ "$LIMIT" -gt 0 ]; then
    echo "[1/2] 冒烟模式: GPU${GPU_A} 生成前 ${LIMIT} 条 ..."
    if ! CUDA_VISIBLE_DEVICES=$GPU_A $PYTHON attack_model_sd35_large.py \
            --model_path "$MODEL_PATH" --start_index 1 --limit "$LIMIT" $GEN_ARGS \
            2>&1 | tee logs/sd35_smoke.log; then
        echo "!! 冒烟生成失败, 查看 logs/sd35_smoke.log, 中止评测"; exit 1
    fi
else
    echo "[1/2] 双卡并行生成: GPU${GPU_A} 跑 1-${HALF}, GPU${GPU_B} 跑 $((HALF+1))-${TOTAL} ..."
    CUDA_VISIBLE_DEVICES=$GPU_A $PYTHON attack_model_sd35_large.py \
        --model_path "$MODEL_PATH" --start_index 1 --limit "$HALF" $GEN_ARGS \
        > "logs/sd35_gpu${GPU_A}.log" 2>&1 &
    PID_A=$!
    CUDA_VISIBLE_DEVICES=$GPU_B $PYTHON attack_model_sd35_large.py \
        --model_path "$MODEL_PATH" --start_index $((HALF + 1)) --limit "$HALF" $GEN_ARGS \
        > "logs/sd35_gpu${GPU_B}.log" 2>&1 &
    PID_B=$!

    # 心跳: 每分钟在控制台报一次进度, 避免双进程日志静默、误以为没在跑
    ( while true; do
        sleep 60
        n=$(ls attack_results/sd3.5-large/combined_hate_*.png 2>/dev/null | wc -l | tr -d ' ')
        g2=$(tail -1 "logs/sd35_gpu${GPU_A}.log" 2>/dev/null | cut -c1-60)
        g3=$(tail -1 "logs/sd35_gpu${GPU_B}.log" 2>/dev/null | cut -c1-60)
        echo "[$(date +%H:%M:%S)] 进度 ${n}/${TOTAL} | GPU${GPU_A}: ${g2} | GPU${GPU_B}: ${g3}"
      done ) &
    MONITOR=$!

    FAIL=0
    wait $PID_A || { echo "!! GPU${GPU_A} 进程异常退出(exit=$?), 查看 logs/sd35_gpu${GPU_A}.log"; FAIL=1; }
    wait $PID_B || { echo "!! GPU${GPU_B} 进程异常退出(exit=$?), 查看 logs/sd35_gpu${GPU_B}.log"; FAIL=1; }
    kill $MONITOR 2>/dev/null
    if [ $FAIL -ne 0 ]; then
        echo "!! 生成阶段有进程异常退出(模型加载失败/CSV 缺失等), 中止评测"
        echo "   已生成的图片仍保留在 attack_results/sd3.5-large/, 修复后重跑本脚本会自动续跑"
        exit 1
    fi
    echo "生成阶段完成 ✅"
fi

GEN_COUNT=$(ls attack_results/sd3.5-large/combined_hate_*.png 2>/dev/null | wc -l | tr -d ' ')
echo "当前图片总数: ${GEN_COUNT}/${TOTAL}"

# ---------- 阶段 2: GPT-4.1 image score 评测 ----------
export MODEL_NAME="sd3.5-large"
echo "[2/2] 运行 judge_image_score.py (GPT-4.1, attack_only, ${MODEL_NAME}) ..."
if ! $PYTHON judge_image_score.py 2>&1 | tee logs/judge_image_score.log; then
    echo "!! 评测脚本报错, 查看 logs/judge_image_score.log"; exit 1
fi

echo "============================================================"
echo "全部完成 ✅"
echo "  图片:   attack_results/sd3.5-large/"
echo "  评分:   judge_image_score/sd3.5-large_attack_score.csv"
echo "  失败记录(若有): attack_results/sd3.5-large/failed_indices.txt"
echo "============================================================"
