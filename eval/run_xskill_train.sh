#!/bin/bash
# ============================================================================
# XSkill Phase I：训练（积累 memory bank）
# - 在 single-turn 子集上跑，让 agent 自动 distill 出 SKILL.md / experiences.json
# - 用法：bash eval/run_xskill_train.sh [DATA_PATH] [MAX_SAMPLES] [TAG]
# ============================================================================

source "$(dirname "$0")/local_env.sh"

DATA_PATH="${1:-benchmark/VisualToolBench/val_single.json}"
MAX_SAMPLES="${2:--1}"
TAG="${3:-vtb_train}"

OUTPUT_DIR="output/train_${TAG}"
LOG_FILE="logs/train_${TAG}.log"
TOOL_CONFIG_PATH="eval/configs/tool_configs.yaml"

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

# ── Phase I: 边跑边写入 memory bank ─────────────────────────────────────────
SKILL_LIBRARY="memory_bank/${TAG}/SKILL.md"
EXPERIENCE_LIBRARY="memory_bank/${TAG}/experiences.json"
EXPERIENCE_MAX_OPS=3
EXPERIENCE_LARGE_BATCH=8
EXPERIENCE_MAX_ITEMS=120
SKILL_MAX_LENGTH=1000

mkdir -p "$OUTPUT_DIR" logs "$(dirname "$EXPERIENCE_LIBRARY")"

echo "==================================================================="
echo "[XSKILL TRAIN] data=$DATA_PATH  max=$MAX_SAMPLES  tag=$TAG"
echo "[XSKILL TRAIN] memory_bank/${TAG}/"
echo "==================================================================="

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
    --skill-enable \
    --skill-library "$SKILL_LIBRARY" \
    --skill-inference \
    --skill-refine \
    --skill-max-length $SKILL_MAX_LENGTH \
    --experience-enable \
    --experience-library "$EXPERIENCE_LIBRARY" \
    --experience-retrieval \
    --experience-retrieval-top-k 3 \
    --experience-retrieval-decomposition \
    --experience-retrieval-rewrite \
    --experience-online-generate \
    --experience-library-update \
    --experience-max-ops $EXPERIENCE_MAX_OPS \
    --experience-large-batch $EXPERIENCE_LARGE_BATCH \
    --experience-refine \
    --experience-max-items $EXPERIENCE_MAX_ITEMS \
    2>&1 | tee "$LOG_FILE"
