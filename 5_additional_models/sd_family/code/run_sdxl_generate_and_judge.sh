#!/usr/bin/env bash
# ============================================================
# CollageAttack x SDXL-base-1.0 一键: 双卡生成图片 -> GPT-4.1 image score 评测
# (GPU0 + GPU1, 与 SD3.5 的 GPU2/GPU3 互不影响, 可同时跑)
#
# 用法(在服务器上, 任意目录均可):
#   bash run_sdxl_generate_and_judge.sh              # 全量 200 条, GPU0 跑 1-100, GPU1 跑 101-200
#   LIMIT=5 bash run_sdxl_generate_and_judge.sh      # 冒烟: 只用 GPU0 生成前 5 条并评测
#
# 可用环境变量覆盖默认值:
#   MODEL_PATH  模型路径 (默认 /hub/huggingface/models/stabilityai/stable-diffusion-xl-base-1.0)
#   GPU_A/GPU_B 两张卡的编号 (默认 0 / 1)
#   PYTHON      python 解释器 (默认 python)
#   OFFLOAD     1=开 cpu_offload / 0=关(默认; SDXL fp16 约 7GB, 24GB 卡绰绰有余)
#
# 注意: SDXL 无 T5, 双 CLIP 上限 77 token, 长攻击提示词会被截断(架构限制,
#       属实验设定的一部分, 详见 attack_model_sdxl.py 头部说明)
# ============================================================
set -uo pipefail

cd "$(dirname "$0")"   # 保证在 T2I_CollageAttack/ 下运行, 相对路径才有效

MODEL_PATH="${MODEL_PATH:-/hub/huggingface/models/stabilityai/stable-diffusion-xl-base-1.0}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
PYTHON="${PYTHON:-python}"
LIMIT="${LIMIT:-0}"
RESULTS_DIR="attack_results/sdxl-base-1.0"
TOTAL=200
HALF=$((TOTAL / 2))

GEN_ARGS=""
if [ "${OFFLOAD:-0}" = "1" ]; then
    GEN_ARGS="--cpu_offload"
    echo "(cpu_offload 已开启: 一般 SDXL 用不到, 除非卡被其他进程占用)"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

# ---------- 阶段 1: 生成图片 ----------
if [ "$LIMIT" -gt 0 ]; then
    echo "[1/2] 冒烟模式: GPU${GPU_A} 生成前 ${LIMIT} 条 ..."
    if ! CUDA_VISIBLE_DEVICES=$GPU_A $PYTHON attack_model_sdxl.py \
            --model_path "$MODEL_PATH" --start_index 1 --limit "$LIMIT" $GEN_ARGS \
            2>&1 | tee logs/sdxl_smoke.log; then
        echo "!! 冒烟生成失败, 查看 logs/sdxl_smoke.log, 中止评测"; exit 1
    fi
else
    echo "[1/2] 双卡并行生成: GPU${GPU_A} 跑 1-${HALF}, GPU${GPU_B} 跑 $((HALF+1))-${TOTAL} ..."
    CUDA_VISIBLE_DEVICES=$GPU_A $PYTHON attack_model_sdxl.py \
        --model_path "$MODEL_PATH" --start_index 1 --limit "$HALF" $GEN_ARGS \
        > "logs/sdxl_gpu${GPU_A}.log" 2>&1 &
    PID_A=$!
    CUDA_VISIBLE_DEVICES=$GPU_B $PYTHON attack_model_sdxl.py \
        --model_path "$MODEL_PATH" --start_index $((HALF + 1)) --limit "$HALF" $GEN_ARGS \
        > "logs/sdxl_gpu${GPU_B}.log" 2>&1 &
    PID_B=$!

    # 心跳: 每分钟在控制台报一次进度, 避免双进程日志静默、误以为没在跑
    ( while true; do
        sleep 60
        n=$(ls ${RESULTS_DIR}/combined_hate_*.png 2>/dev/null | wc -l | tr -d ' ')
        g0=$(tail -1 "logs/sdxl_gpu${GPU_A}.log" 2>/dev/null | cut -c1-60)
        g1=$(tail -1 "logs/sdxl_gpu${GPU_B}.log" 2>/dev/null | cut -c1-60)
        echo "[$(date +%H:%M:%S)] SDXL 进度 ${n}/${TOTAL} | GPU${GPU_A}: ${g0} | GPU${GPU_B}: ${g1}"
      done ) &
    MONITOR=$!

    FAIL=0
    wait $PID_A || { echo "!! GPU${GPU_A} 进程异常退出(exit=$?), 查看 logs/sdxl_gpu${GPU_A}.log"; FAIL=1; }
    wait $PID_B || { echo "!! GPU${GPU_B} 进程异常退出(exit=$?), 查看 logs/sdxl_gpu${GPU_B}.log"; FAIL=1; }
    kill $MONITOR 2>/dev/null
    if [ $FAIL -ne 0 ]; then
        echo "!! 生成阶段有进程异常退出, 中止评测"
        echo "   已生成的图片仍保留在 ${RESULTS_DIR}/, 修复后重跑本脚本会自动续跑"
        exit 1
    fi
    echo "生成阶段完成 ✅"
fi

GEN_COUNT=$(ls ${RESULTS_DIR}/combined_hate_*.png 2>/dev/null | wc -l | tr -d ' ')
echo "当前图片总数: ${GEN_COUNT}/${TOTAL}"

# ---------- 阶段 2: GPT-4.1 image score 评测 ----------
export MODEL_NAME="sdxl-base-1.0"
echo "[2/2] 运行 judge_image_score.py (GPT-4.1, attack_only, ${MODEL_NAME}) ..."
if ! $PYTHON judge_image_score.py 2>&1 | tee logs/judge_image_score_sdxl.log; then
    echo "!! 评测脚本报错, 查看 logs/judge_image_score_sdxl.log"; exit 1
fi

echo "============================================================"
echo "全部完成 ✅"
echo "  图片:   ${RESULTS_DIR}/"
echo "  评分:   judge_image_score/sdxl-base-1.0_attack_score.csv"
echo "  失败记录(若有): ${RESULTS_DIR}/failed_indices.txt"
echo "============================================================"
