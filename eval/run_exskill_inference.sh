#!/bin/bash
# ============================================================================
# XSkill 本地化运行脚本（Inference 模式）
# - Reasoning / Verifier / Experience 都用本地 Qwen3.6-27B（vLLM 8080）
# - Embedding 用本地 BGE-M3（vLLM 8081）
# - Web search 用 DuckDuckGo（ddgs 库，无需 key）
# - Image search 用 Serper.dev（含反向图搜）
# - Visit 用 Jina Reader API
# - Image hosting 用 ImgBB（用于反向图搜上传本地图片）
# ============================================================================

# ─── 占位（XSkill 历史脚本里这两个变量被复用，本地化场景不再使用） ───
API_KEY_1="EMPTY"
API_KEY_2="EMPTY"

# ============================================================================
# 重要：不同变量的 endpoint 格式不一样！
#   - REASONING_END_POINT (api_caller.py 直接 requests.post)  → 必须是完整 .../chat/completions
#   - VERIFIER_END_POINT  (llm_judger.py 用 openai.OpenAI 客户端) → 必须是 base url .../v1
#   - EXPERIENCE_END_POINT (llm_client.py 内部会自动 normalize) → base url .../v1 即可
#   - EXPERIENCE_EMBEDDING_ENDPOINT (experience_retriever.py 自动 strip /v1) → .../v1 即可
# ============================================================================

# ============================================================================
# Reasoning Model — 本地 Qwen3.6-27B  (完整 chat/completions URL)
# ============================================================================
export REASONING_MODEL_NAME="Qwen3.6-27B"
export REASONING_API_KEY="EMPTY"
export REASONING_END_POINT="http://localhost:8080/v1/chat/completions"

export REASONING_API_KEY_2="EMPTY"
export REASONING_END_POINT_2="http://localhost:8080/v1/chat/completions"

# ============================================================================
# Verifier Model — 复用 Qwen3.6-27B  (base url，给 openai client)
# ============================================================================
export VERIFIER_MODEL_NAME="Qwen3.6-27B"
export VERIFIER_API_KEY="EMPTY"
export VERIFIER_END_POINT="http://localhost:8080/v1"

# ============================================================================
# Experience Model — 复用 Qwen3.6-27B  (base url 即可，内部自动 normalize)
# ============================================================================
export EXPERIENCE_MODEL_NAME="Qwen3.6-27B"
export EXPERIENCE_API_KEY="EMPTY"
export EXPERIENCE_END_POINT="http://localhost:8080/v1"

export EXPERIENCE_API_KEY_2="EMPTY"
export EXPERIENCE_END_POINT_2="http://localhost:8080/v1"

# ── Embedding：本地 BGE-M3 (vLLM 8081) ──────────────────────────────────────
export EXPERIENCE_EMBEDDING_MODEL="bge-m3"
export EXPERIENCE_EMBEDDING_API_KEY="EMPTY"
export EXPERIENCE_EMBEDDING_ENDPOINT="http://localhost:8081/v1"

# ============================================================================
# 外部工具 API
# ============================================================================
# Serper.dev: 用于 image_search（含反向图搜）。免费 2500/月
export SERPAPI_KEY="8329e96a497ca0683363c8d43bb8e548bd556f9c"

# Jina Reader: 用于 visit（网页内容抓取）
export JINA_API_KEY="jina_eb304caa412f4e8c9d5c95d651aab446-rjZTgq0b_K0DZqTqFIMSPk1vPUn"

# ImgBB: 用于反向图搜时把本地图片传到公网拿到 URL
export IMGBB_API_KEY="e6f022306606fb35f56631aa99171d7a"

export ENABLE_FUNCTION_CALLING="true"

# 启用的工具列表（注意：web_search 已切到 DuckDuckGo，无需 key）
export ENABLED_TOOLS="web_search, image_search, visit, code_interpreter"

TOOL_CONFIG_PATH="eval/configs/tool_configs.yaml"

IMAGE_SEARCH_MAX_CALLS=5
WEB_SEARCH_MAX_CALLS=7

# ============================================================================
# Inference Parameters
# ============================================================================
MAX_TOTAL_TOKENS=32768
MAX_TURNS=20
MAX_IMAGES=100
TEMPERATURE=0.6
TOP_P=1.0

# ============================================================================
# Experience Parameters
# ============================================================================
EXPERIENCE_MAX_OPS=3
EXPERIENCE_MAX_ITEMS=120

EXPERIENCE_RETRIEVAL_TOP_K=3
EXPERIENCE_LIBRARY="memory_bank/test/experiences.json"

# ============================================================================
# Skill Parameters
# ============================================================================
SKILL_LIBRARY="memory_bank/test/SKILL.md"
SKILL_MAX_LENGTH=1000

# ============================================================================
# Running Settings
# ============================================================================
SYSTEM_PROMPT_TYPE="multi_tool_agent"

IMAGE_DIR="benchmark"
DATA_PATH="benchmark/Merged_Test/val.json"

OUTPUT_DIR="output/test_exskill_merged"
LOG_OUTPUT_DIR="logs/test_exskill_merged"

# 单机推理时把并发降低，避免本地 Qwen vLLM 排队
NUM_WORKERS=2
EXPERIENCE_LARGE_BATCH=4
ROLLOUTS_PER_SAMPLE=2

# 9 个合并样本（VisualProbe 4 + Tool_Test 5）
MAX_SAMPLES="9"

# 确保输出目录存在
mkdir -p $OUTPUT_DIR
mkdir -p logs
mkdir -p memory_bank/test

# ============================================================================
# Run Inference
# ============================================================================
python3 -u eval/infer_api.py \
    --input-file $DATA_PATH \
    --image-folder $IMAGE_DIR \
    --output-dir $OUTPUT_DIR \
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
    --skill-library $SKILL_LIBRARY \
    --skill-inference \
    --experience-enable \
    --experience-library $EXPERIENCE_LIBRARY \
    --experience-retrieval \
    --experience-retrieval-top-k $EXPERIENCE_RETRIEVAL_TOP_K \
    --experience-retrieval-decomposition \
    --experience-retrieval-rewrite \
    2>&1 | tee $LOG_OUTPUT_DIR.log
