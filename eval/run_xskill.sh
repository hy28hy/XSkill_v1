#!/bin/bash
# ============================================================================
# XSkill 完整评测：Reasoning + 工具调用 + Skill 库 + Experience 检索
# 用法：bash eval/run_xskill.sh [DATA_PATH] [MAX_SAMPLES] [TAG]
# ============================================================================

source "$(dirname "$0")/local_env.sh"

DATA_PATH="${1:-benchmark/VisualToolBench/val_50.json}"
MAX_SAMPLES="${2:--1}"
TAG="${3:-vtb50}"

OUTPUT_DIR="output/xskill_${TAG}"
LOG_FILE="logs/xskill_${TAG}.log"
TOOL_CONFIG_PATH="eval/configs/tool_configs.yaml"

# ── 推理参数（与 baseline 完全一致，便于对比）─────────────────────────────
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

# ── XSkill memory bank（默认指向训练阶段产出的目录）────────────────────────
SKILL_LIBRARY="${SKILL_LIBRARY:-memory_bank/${TAG}/SKILL.md}"
EXPERIENCE_LIBRARY="${EXPERIENCE_LIBRARY:-memory_bank/${TAG}/experiences.json}"
EXPERIENCE_RETRIEVAL_TOP_K=3

mkdir -p "$OUTPUT_DIR" logs "$(dirname "$EXPERIENCE_LIBRARY")"

echo "==================================================================="
echo "[XSKILL]    data=$DATA_PATH  max_samples=$MAX_SAMPLES  tag=$TAG"
echo "[XSKILL]    output=$OUTPUT_DIR"
echo "[XSKILL]    skill_lib=$SKILL_LIBRARY"
echo "[XSKILL]    experience_lib=$EXPERIENCE_LIBRARY"
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
    --experience-enable \
    --experience-library "$EXPERIENCE_LIBRARY" \
    --experience-retrieval \
    --experience-retrieval-top-k $EXPERIENCE_RETRIEVAL_TOP_K \
    --experience-retrieval-decomposition \
    --experience-retrieval-rewrite \
    2>&1 | tee "$LOG_FILE"
