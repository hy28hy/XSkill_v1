#!/bin/bash
# ============================================================================
# Baseline 评测：只有 Reasoning + 工具调用，关闭 XSkill 框架
# 用法：bash eval/run_baseline.sh [DATA_PATH] [MAX_SAMPLES] [TAG]
#   - DATA_PATH:    默认 benchmark/VisualToolBench/val_50.json
#   - MAX_SAMPLES:  默认 -1（即全量）
#   - TAG:          输出目录后缀，默认 "vtb50"
# ============================================================================

# 加载共享配置
source "$(dirname "$0")/local_env.sh"

# ── 参数 ────────────────────────────────────────────────────────────────────
DATA_PATH="${1:-benchmark/VisualToolBench/val_50.json}"
MAX_SAMPLES="${2:--1}"
TAG="${3:-vtb50}"

OUTPUT_DIR="output/baseline_${TAG}"
LOG_FILE="logs/baseline_${TAG}.log"
TOOL_CONFIG_PATH="eval/configs/tool_configs.yaml"

# ── 推理参数 ────────────────────────────────────────────────────────────────
MAX_TOTAL_TOKENS=32768
MAX_TURNS=20
MAX_IMAGES=100
TEMPERATURE=0.6
TOP_P=1.0
NUM_WORKERS=2
ROLLOUTS_PER_SAMPLE=2
IMAGE_SEARCH_MAX_CALLS=5
WEB_SEARCH_MAX_CALLS=7
SYSTEM_PROMPT_TYPE="multi_tool_agent"
IMAGE_DIR="benchmark"

mkdir -p "$OUTPUT_DIR" logs

echo "==================================================================="
echo "[BASELINE]  data=$DATA_PATH  max_samples=$MAX_SAMPLES  tag=$TAG"
echo "[BASELINE]  output=$OUTPUT_DIR  log=$LOG_FILE"
echo "==================================================================="

# ── 跑！(注意：没有 --skill-enable / --experience-enable，纯裸 backbone) ─────
python3 -u eval/infer_api.py \
    --input-file "$DATA_PATH" \
    --image-folder "$IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-turns $MAX_TURNS \
    --max-images $MAX_IMAGES \
    --max-total-tokens $MAX_TOTAL_TOKENS \
    --system-prompt-key $SYSTEM_PROMPT_TYPE \
    --num-workers $NUM_WORKERS \
    --tool-config-path $TOOL_CONFIG_PATH \
    --max-samples $MAX_SAMPLES \
    --rollouts-per-sample $ROLLOUTS_PER_SAMPLE \
    --image-search-max-calls $IMAGE_SEARCH_MAX_CALLS \
    --web-search-max-calls $WEB_SEARCH_MAX_CALLS \
    2>&1 | tee "$LOG_FILE"
