#!/bin/bash
# ============================================================================
# 共享配置：所有 endpoint / API key / 工具配置都在这里
# 由 run_baseline.sh 和 run_xskill.sh 共同 source 进来
# ============================================================================

# ── Reasoning Model（完整 chat/completions URL）─────────────────────────────
export REASONING_MODEL_NAME="Qwen3.6-27B"
export REASONING_API_KEY="EMPTY"
export REASONING_END_POINT="http://localhost:8080/v1/chat/completions"
export REASONING_API_KEY_2="EMPTY"
export REASONING_END_POINT_2="http://localhost:8080/v1/chat/completions"

# ── Verifier（base url，给 openai client）─────────────────────────────────
export VERIFIER_MODEL_NAME="Qwen3.6-27B"
export VERIFIER_API_KEY="EMPTY"
export VERIFIER_END_POINT="http://localhost:8080/v1"

# ── Experience Model（base url）────────────────────────────────────────────
export EXPERIENCE_MODEL_NAME="Qwen3.6-27B"
export EXPERIENCE_API_KEY="EMPTY"
export EXPERIENCE_END_POINT="http://localhost:8080/v1"
export EXPERIENCE_API_KEY_2="EMPTY"
export EXPERIENCE_END_POINT_2="http://localhost:8080/v1"

# ── Embedding（BGE-M3 vLLM 8081）────────────────────────────────────────────
export EXPERIENCE_EMBEDDING_MODEL="bge-m3"
export EXPERIENCE_EMBEDDING_API_KEY="EMPTY"
export EXPERIENCE_EMBEDDING_ENDPOINT="http://localhost:8081/v1"

# ── 外部工具 API ────────────────────────────────────────────────────────────
export SERPAPI_KEY="8329e96a497ca0683363c8d43bb8e548bd556f9c"
export JINA_API_KEY="jina_eb304caa412f4e8c9d5c95d651aab446-rjZTgq0b_K0DZqTqFIMSPk1vPUn"
export IMGBB_API_KEY="e6f022306606fb35f56631aa99171d7a"

# ── 工具开关 ────────────────────────────────────────────────────────────────
export ENABLE_FUNCTION_CALLING="true"
export ENABLED_TOOLS="web_search, image_search, visit, code_interpreter"
