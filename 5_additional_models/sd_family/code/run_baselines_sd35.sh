#!/usr/bin/env bash
# ============================================================
# Baseline 实验(Original / PGJ / TYPO-Standard) x SD3.5-Large
# 单卡独占(默认 GPU2)流水线: 逐方法生成, 方法 N 生成完立即
#   ① 后台启动其 GPT-4.1 评测(不占 GPU)  ② GPU 马上生成方法 N+1
#
# 用法:
#   bash run_baselines_sd35.sh                       # 三方法全量
#   METHODS="typo" bash run_baselines_sd35.sh        # 只跑/补跑指定方法
#   LIMIT=5 bash run_baselines_sd35.sh               # 冒烟: 每方法前 5 条
#   GPU=3 bash run_baselines_sd35.sh                 # 换卡
#
# 依赖: baseline_prompts/{original,pgj,typo}_prompts.csv
# 输出: attack_results/sd3.5-large__<method>/ 图片
#       judge_image_score/sd3.5-large__<method>_attack_score.csv 评分
# ============================================================
set -uo pipefail
cd "$(dirname "$0")"

MODEL_SCRIPT="attack_model_sd35_large.py"
MODEL_PATH="${MODEL_PATH:-/hub/huggingface/models/stabilityai/stable-diffusion-3.5-large}"
GPU="${GPU:-2}"                      # 本模型专用卡(0/1/4-7 已被其他任务占用)
PYTHON="${PYTHON:-python}"
# nohup/ssh 非交互环境无 conda 时回退到绝对路径
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=/data/home/Zhiyi/miniconda3/envs/qwen3vl4b/bin/python
TAG="sd3.5-large"
EXTRA_ARGS="--cpu_offload"           # 24GB 卡必须 offload
METHODS="${METHODS:-original pgj typo}"
LIMIT="${LIMIT:-0}"
TOTAL=200

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

JUDGE_PIDS=()
FAILED_METHODS=""

gen_method () {
    local METHOD=$1
    local RES_DIR="attack_results/${TAG}__${METHOD}"
    local CSV="baseline_prompts/${METHOD}_prompts.csv"
    local LOG="logs/${TAG}__${METHOD}_gpu${GPU}.log"

    if [ ! -f "$CSV" ]; then
        echo "!! 缺少 $CSV (先跑 python prep_baseline_prompts.py)"
        FAILED_METHODS="$FAILED_METHODS $METHOD"; return 1
    fi

    echo "=================================================================="
    echo "[${METHOD}] 生成开始 (GPU${GPU}) -> ${RES_DIR}"
    echo "=================================================================="

    local LIMIT_ARG=""
    [ "$LIMIT" -gt 0 ] && LIMIT_ARG="--limit $LIMIT"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $MODEL_SCRIPT \
        --model_path "$MODEL_PATH" --start_index 1 $LIMIT_ARG $EXTRA_ARGS \
        --input_csv "$CSV" --prompt_column prompt --results_dir "$RES_DIR" \
        > "$LOG" 2>&1 &
    local PID=$!

    # 心跳: 每分钟报一次该方法进度
    ( while kill -0 $PID 2>/dev/null; do
          sleep 60
          local n=$(ls ${RES_DIR}/combined_hate_*.png 2>/dev/null | wc -l | tr -d ' ')
          local last=$(tail -1 "$LOG" 2>/dev/null | cut -c1-60)
          echo "[$(date +%H:%M:%S)] ${TAG}__${METHOD} 进度 ${n}/${TOTAL} | ${last}"
      done ) &
    local MON=$!

    wait $PID
    local CODE=$?
    kill $MON 2>/dev/null
    if [ "$CODE" -eq 0 ]; then
        local N=$(ls ${RES_DIR}/combined_hate_*.png 2>/dev/null | wc -l | tr -d ' ')
        echo "[${METHOD}] 生成完成 ${N}/${TOTAL} -> 后台启动评测, GPU 继续下一方法"
        MODEL_NAME="${TAG}__${METHOD}" $PYTHON judge_image_score.py \
            > "logs/judge_${TAG}__${METHOD}.log" 2>&1 &
        JUDGE_PIDS+=($!)
        return 0
    else
        echo "!! [${METHOD}] 生成进程异常退出(exit=${CODE}), 查看 $LOG; 跳过其评测(图片保留可续跑)"
        FAILED_METHODS="$FAILED_METHODS $METHOD"
        return 1
    fi
}

for M in $METHODS; do
    gen_method "$M"
done

echo "============================================================"
echo "生成阶段全部结束${FAILED_METHODS:+(失败:${FAILED_METHODS} )}, 等待剩余后台评测完成..."
echo "============================================================"
JUDGE_FAIL=""
for p in ${JUDGE_PIDS[@]+"${JUDGE_PIDS[@]}"}; do
    wait $p || JUDGE_FAIL="$JUDGE_FAIL $p"
done

echo "============================================================"
echo "全部结束 ✅  生成失败方法:${FAILED_METHODS:- 无}  评测失败:${JUDGE_FAIL:- 无}"
echo "  图片:   attack_results/${TAG}__<method>/"
echo "  评分:   judge_image_score/${TAG}__<method>_attack_score.csv"
echo "  评测日志: logs/judge_${TAG}__<method>.log (429 退避已内置, 失败可单独补:"
echo "           MODEL_NAME=${TAG}__<method> python judge_image_score.py)"
echo "============================================================"
[ -z "$FAILED_METHODS" ] && [ -z "$JUDGE_FAIL" ]
