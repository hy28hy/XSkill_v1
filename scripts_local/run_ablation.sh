#!/bin/bash
# ============================================================================
# run_ablation.sh — XSkill Ablation Study: 3 modes × N datasets × R rollouts
#
# DEFAULT JOB (no env overrides needed):
#   Dataset : VisualToolBench full (1204 questions)
#   Modes   : baseline + xskill_retrieve + xskill_full
#   Rollouts: 4 per sample  (computes pass@1 / pass@4 / acc, paper setup)
#   GPUs    : 2× (default 0,4) tp=2 for Qwen3.6-27B + BGE-M3 sharing GPU 4
#   Backend : Local vLLM (auto-started by this script)
#   ETA     : ~3 days for baseline, ~9-10 days for xskill_retrieve,
#             ~24 days for xskill_full → total ~5-6 weeks of GPU time.
#             Run modes in parallel on separate machines if you can.
#
# RECOMMENDED: Run a smoke test first (~5-10 min) to confirm everything works:
#   DATASETS=visualtoolbench_50 MAX_SAMPLES=4 ROLLOUTS_PER_SAMPLE=1 \
#       MODES=baseline bash scripts_local/run_ablation.sh
#
# ----------------------------------------------------------------------------
# Modes (validating XSkill framework gain on a fixed local backbone):
#   1. baseline        — Bare backbone + tool-calling. Skill/Experience disabled.
#   2. xskill_retrieve — Only enable skill+experience retrieval (no online gen).
#                        Equivalent to using a frozen pre-built memory bank.
#   3. xskill_full     — Full XSkill: retrieval + online generate + library
#                        update + refine (paper setup).
#
# Datasets:
#   visualtoolbench_full   — full 1204 questions  ← DEFAULT
#   visualtoolbench_50     — 50-question sanity subset
#   visualtoolbench_single — single-turn subset   (~603)
#   visualtoolbench_multi  — multi-turn subset    (~601)
#   visualprobe            — 4-question demo (tiny smoke)
#   tool_test              — 5-question demo
#   merged_test            — 9-question merged demo
#
# LLM Backend:
#   - Local mode (default): Local models served by vLLM (this script auto-starts).
#       * Reasoning/Verifier/Experience: Qwen3.6-27B  (port 8080)
#       * Embedding:                     BGE-M3       (port 8081)
#     vLLM runs from a dedicated conda env (default: vllm_qwen36) — eval code
#     runs from the XSkill conda env (default: Xskill).
#   - API mode: Use external OpenAI-compatible API (set REASONING_END_POINT etc.
#               in the environment before launch). XSkill natively supports it.
#
# Hardware: 2× GPUs (default 0,4) on a single host.
#   - Single-model mode (default): 1 vLLM instance for Qwen with tp=2 across
#     QWEN_GPUS, plus a tiny BGE-M3 instance sharing one of the GPUs (~3GB).
#   - To run on more cards: e.g. QWEN_GPUS=0,1,2,3 QWEN_TP=4 BGE_GPUS=4.
#
# Usage:
#   # Default — VisualToolBench full × 3 modes × 4 rollouts (paper-grade):
#   bash scripts_local/run_ablation.sh
#
#   # Smoke test before committing to the full run:
#   DATASETS=visualtoolbench_50 MAX_SAMPLES=4 ROLLOUTS_PER_SAMPLE=1 \
#     bash scripts_local/run_ablation.sh
#
#   # Single mode at a time (recommend running modes on different days/hosts):
#   MODES=baseline         bash scripts_local/run_ablation.sh   # ~3 days
#   MODES=xskill_retrieve  bash scripts_local/run_ablation.sh   # ~9 days
#   MODES=xskill_full      bash scripts_local/run_ablation.sh   # ~24 days
#
#   # Custom GPU set (4 cards for Qwen, BGE on GPU 4):
#   QWEN_GPUS=0,1,2,3 QWEN_TP=4 BGE_GPUS=4 \
#     bash scripts_local/run_ablation.sh
#
#   # Different backbone or dataset:
#   DATASETS=visualtoolbench_multi bash scripts_local/run_ablation.sh
#
#   # Reuse externally started vLLM servers (skip auto-start):
#   AUTO_VLLM=false bash scripts_local/run_ablation.sh
#
#   # API mode (e.g. plug an external OpenAI-compatible service):
#   LLM_BACKEND=api \
#     REASONING_END_POINT=https://your-llm/v1/chat/completions \
#     REASONING_API_KEY=sk-xxx REASONING_MODEL_NAME=gpt-4o \
#     EXPERIENCE_EMBEDDING_ENDPOINT=https://your-embed/v1 \
#     EXPERIENCE_EMBEDDING_API_KEY=sk-xxx \
#     EXPERIENCE_EMBEDDING_MODEL=text-embedding-3-large \
#     bash scripts_local/run_ablation.sh
#
# Output structure (under ${RESULTS_DIR}):
#   summary.csv           — one row per (dataset, mode) with pass@1/@4 metrics
#   configs/<exp>.yaml    — per-experiment config snapshot
#   logs/<exp>.log        — full inference stdout/stderr
#   memory_banks/<exp>/   — XSkill-learned SKILL.md and experiences.json
#   output_<exp>/         — per-sample results.jsonl + metrics_summary.json
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."   # back to XSkill project root
echo "Working directory: $(pwd)"
SRC_DIR="$(pwd)/eval"

# ============ Infra ============
LLM_BACKEND="${LLM_BACKEND:-local}"   # "local" or "api"
AUTO_VLLM="${AUTO_VLLM:-true}"        # auto-start vLLM (local mode only)

# ============ Experiment grid ============
# Default: VisualToolBench full (1204 questions) × all 3 modes.
# Override with: DATASETS="..." MODES="..."  (space-separated lists)
read -ra DATASETS_ARR <<< "${DATASETS:-visualtoolbench_full}"
read -ra MODES_ARR    <<< "${MODES:-baseline xskill_retrieve xskill_full}"

# ============ Paths ============
BASE_DIR="${BASE_DIR:-/data2/chenxuwu/zihaowan_workplace}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/home/chenxuwu/.cache/modelscope/hub/models/Qwen/Qwen3___6-27B}"
BGE_MODEL_PATH="${BGE_MODEL_PATH:-${BASE_DIR}/bge-m3/BAAI/bge-m3}"

# ============ Results directory (includes model flag) ============
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_FLAG="$(basename "${QWEN_MODEL_PATH}")"
RESULTS_DIR="${RESULTS_DIR:-${BASE_DIR}/XSkill/results_ablation_${MODEL_FLAG}_${TIMESTAMP}}"
mkdir -p "${RESULTS_DIR}/logs" "${RESULTS_DIR}/configs" "${RESULTS_DIR}/memory_banks"

# ============ vLLM serving config ============
QWEN_PORT="${QWEN_PORT:-8080}"
BGE_PORT="${BGE_PORT:-8081}"
QWEN_GPUS="${QWEN_GPUS:-0,4}"           # default 0,4 for Qwen3.6-27B (tp=2)
BGE_GPUS="${BGE_GPUS:-4}"               # BGE-M3 sharing GPU 4 (~3GB only)
QWEN_TP="${QWEN_TP:-2}"
QWEN_GPU_MEM="${QWEN_GPU_MEM:-0.80}"    # leave 20% for BGE / safety margin
BGE_GPU_MEM="${BGE_GPU_MEM:-0.05}"      # BGE needs ~3GB; 5% (~5GB) is plenty
QWEN_MAX_LEN="${QWEN_MAX_LEN:-32768}"
QWEN_TOOL_PARSER="${QWEN_TOOL_PARSER:-hermes}"  # qwen3_xml / hermes / qwen3_coder

# ============ External tool API keys ============
# Used by tool implementations under eval/tools/. If already exported in the
# parent shell these defaults will not override.
export SERPAPI_KEY="${SERPAPI_KEY:-8329e96a497ca0683363c8d43bb8e548bd556f9c}"
export JINA_API_KEY="${JINA_API_KEY:-jina_eb304caa412f4e8c9d5c95d651aab446-rjZTgq0b_K0DZqTqFIMSPk1vPUn}"
export IMGBB_API_KEY="${IMGBB_API_KEY:-e6f022306606fb35f56631aa99171d7a}"

# ============ Inference hyperparameters ============
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
MAX_TURNS="${MAX_TURNS:-20}"
MAX_IMAGES="${MAX_IMAGES:-100}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ROLLOUTS_PER_SAMPLE="${ROLLOUTS_PER_SAMPLE:-4}"   # paper setup: pass@4
IMAGE_SEARCH_MAX_CALLS="${IMAGE_SEARCH_MAX_CALLS:-5}"
WEB_SEARCH_MAX_CALLS="${WEB_SEARCH_MAX_CALLS:-7}"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-multi_tool_agent}"
ENABLED_TOOLS="${ENABLED_TOOLS:-web_search, image_search, visit, code_interpreter}"
EXPERIENCE_RETRIEVAL_TOP_K="${EXPERIENCE_RETRIEVAL_TOP_K:-3}"
EXPERIENCE_MAX_OPS="${EXPERIENCE_MAX_OPS:-3}"
EXPERIENCE_MAX_ITEMS="${EXPERIENCE_MAX_ITEMS:-120}"
EXPERIENCE_LARGE_BATCH="${EXPERIENCE_LARGE_BATCH:-8}"
SKILL_MAX_LENGTH="${SKILL_MAX_LENGTH:-1000}"

# ============ Backend validation ============
if [ "${LLM_BACKEND}" = "api" ]; then
    : "${REASONING_END_POINT:?REASONING_END_POINT must be set when LLM_BACKEND=api}"
    : "${REASONING_API_KEY:?REASONING_API_KEY must be set when LLM_BACKEND=api}"
    : "${REASONING_MODEL_NAME:?REASONING_MODEL_NAME must be set when LLM_BACKEND=api}"
    echo "Backend: API mode (${REASONING_MODEL_NAME} @ ${REASONING_END_POINT})"
elif [ "${LLM_BACKEND}" = "local" ]; then
    echo "Backend: Local inference (vLLM)"
    if [ ! -d "${QWEN_MODEL_PATH}" ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║  [ERROR] Local model not found: ${QWEN_MODEL_PATH}"
        echo "╚══════════════════════════════════════════════════════════╝"
        echo ""
        echo "Required models for local mode:"
        echo "  Default: Qwen/Qwen3.6-27B at ${QWEN_MODEL_PATH}"
        echo "    Download via:"
        echo "      modelscope download --model Qwen/Qwen3.6-27B"
        echo "    Or override path:"
        echo "      QWEN_MODEL_PATH=/path/to/your/model bash scripts_local/run_ablation.sh"
        echo ""
        echo "  Alternative — use API backend:"
        echo "    LLM_BACKEND=api REASONING_END_POINT=... REASONING_API_KEY=... \\"
        echo "      REASONING_MODEL_NAME=... bash scripts_local/run_ablation.sh"
        echo ""
        exit 1
    fi
    if [ ! -d "${BGE_MODEL_PATH}" ]; then
        echo ""
        echo "[ERROR] BGE-M3 embedding model not found: ${BGE_MODEL_PATH}"
        echo ""
        echo "Download via:"
        echo "  huggingface-cli download BAAI/bge-m3 --local-dir ${BGE_MODEL_PATH}"
        echo "Or override:"
        echo "  BGE_MODEL_PATH=/path/to/bge-m3 bash scripts_local/run_ablation.sh"
        echo ""
        exit 1
    fi
    echo "  Model: ${QWEN_MODEL_PATH}  (tp=${QWEN_TP}, GPUs=${QWEN_GPUS}, port=${QWEN_PORT})"
    echo "  Embed: ${BGE_MODEL_PATH}  (GPUs=${BGE_GPUS}, port=${BGE_PORT})"
else
    echo "[ERROR] Unknown LLM_BACKEND: ${LLM_BACKEND}. Use 'api' or 'local'."
    exit 1
fi

# ============ Python environment ============
# Strategy:
#   1. If a conda env named ${CONDA_ENV_NAME} exists with XSkill deps, activate it.
#   2. Otherwise fall back to whatever python is on PATH (must have deps).
CONDA_ENV_NAME="${CONDA_ENV_NAME:-Xskill}"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
[ -d "/data2/chenxuwu/miniconda3" ] && CONDA_BASE="${CONDA_BASE_OVERRIDE:-/data2/chenxuwu/miniconda3}"

check_eval_deps() {
    "$1" -c "
import argparse, asyncio, json, yaml, openai, requests
from PIL import Image
" 2>/dev/null
}

if [ -f "${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python" ] \
        && check_eval_deps "${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python"; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    PYTHON="$(command -v python)"
    echo "[py-env] Using conda env: ${CONDA_ENV_NAME} -> ${PYTHON} ($(${PYTHON} --version 2>&1))"
else
    SYSTEM_PYTHON="$(command -v python3 || command -v python || echo /usr/bin/python3)"
    if check_eval_deps "${SYSTEM_PYTHON}"; then
        PYTHON="${SYSTEM_PYTHON}"
        echo "[py-env] Using image-provided python: ${PYTHON} ($(${PYTHON} --version 2>&1))"
    else
        echo ""
        echo "[ERROR] No suitable Python environment found."
        echo "  Tried conda env: ${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
        echo "  Tried system   : ${SYSTEM_PYTHON}"
        echo ""
        echo "Either:"
        echo "  (a) Create the conda env from XSkill's instructions, or"
        echo "  (b) Install eval deps into system python (openai, pyyaml, Pillow, requests)."
        exit 1
    fi
fi
echo "Python (eval): ${PYTHON} ($(${PYTHON} --version 2>&1))"

# ============ vLLM python (for launching vLLM server) ============
# vLLM is typically pinned to a dedicated env (e.g. vllm_qwen36 with cu126 wheels).
# We discover it independently from the eval env.
if [ "${LLM_BACKEND}" = "local" ]; then
    VLLM_CONDA_ENV_NAME="${VLLM_CONDA_ENV_NAME:-vllm_qwen36}"
    VLLM_ENV_DIR="${CONDA_BASE}/envs/${VLLM_CONDA_ENV_NAME}"
    if [ -x "${VLLM_ENV_DIR}/bin/vllm" ]; then
        VLLM_BIN="${VLLM_ENV_DIR}/bin/vllm"
        VLLM_PYTHON="${VLLM_ENV_DIR}/bin/python"
        echo "[vllm-env] Using conda env: ${VLLM_CONDA_ENV_NAME} -> ${VLLM_BIN}"
        echo "           vllm: $(${VLLM_PYTHON} -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'unknown')"
    elif command -v vllm &>/dev/null; then
        VLLM_BIN="$(command -v vllm)"
        VLLM_PYTHON="$(command -v python)"
        VLLM_ENV_DIR="$(dirname "$(dirname "${VLLM_BIN}")")"
        echo "[vllm-env] Using PATH vllm: ${VLLM_BIN}"
    else
        echo "[ERROR] vLLM binary not found."
        echo "  Tried: ${VLLM_ENV_DIR}/bin/vllm and PATH."
        echo "  Install via: pip install vllm  (in a Python 3.10/3.11 env)"
        echo "  Or set: VLLM_CONDA_ENV_NAME=<your_vllm_env_name>"
        exit 1
    fi
fi

# ============ Dataset → val.json mapping ============
resolve_dataset() {
    case "$1" in
        visualtoolbench_50)     echo "benchmark/VisualToolBench/val_50.json" ;;
        visualtoolbench_single) echo "benchmark/VisualToolBench/val_single.json" ;;
        visualtoolbench_multi)  echo "benchmark/VisualToolBench/val_multi.json" ;;
        visualtoolbench_full)   echo "benchmark/VisualToolBench/val_full.json" ;;
        visualprobe)            echo "benchmark/VisualProbe_Test/val.json" ;;
        tool_test)              echo "benchmark/Tool_Test/val.json" ;;
        merged_test)            echo "benchmark/Merged_Test/val.json" ;;
        *) echo "[ERROR] Unknown dataset: $1" >&2; return 1 ;;
    esac
}

# ============ Auto-check VisualToolBench data ================================
# We assume benchmark data has been shipped alongside this repo (see
# benchmark/VisualToolBench/val_*.json). If you keep the data elsewhere,
# override the location with:
#   VTB_DATA_DIR=/your/path/to/VisualToolBench bash scripts_local/run_ablation.sh
# The directory must contain: val_50.json, val_single.json, val_multi.json,
# val_full.json, and an images/ subfolder.
VTB_DATA_DIR="${VTB_DATA_DIR:-$(pwd)/benchmark/VisualToolBench}"

ensure_visualtoolbench_data() {
    # Skip if no visualtoolbench_* dataset is requested
    local need_vtb=false
    for ds in "${DATASETS_ARR[@]}"; do
        case "${ds}" in visualtoolbench_*) need_vtb=true ;; esac
    done
    [ "${need_vtb}" = false ] && return 0

    # Required split for each requested dataset
    local missing=()
    for ds in "${DATASETS_ARR[@]}"; do
        local split
        case "${ds}" in
            visualtoolbench_50)     split="val_50.json" ;;
            visualtoolbench_single) split="val_single.json" ;;
            visualtoolbench_multi)  split="val_multi.json" ;;
            visualtoolbench_full)   split="val_full.json" ;;
        esac
        [ ! -f "${VTB_DATA_DIR}/${split}" ] && missing+=("${split}")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║  [ERROR] VisualToolBench split(s) not found"
        echo "╚══════════════════════════════════════════════════════════╝"
        echo "  Looked under: ${VTB_DATA_DIR}"
        echo "  Missing:      ${missing[*]}"
        echo ""
        echo "  This script expects pre-built JSON splits + images. If you have"
        echo "  the dataset at a different location, point VTB_DATA_DIR to it:"
        echo "    VTB_DATA_DIR=/path/to/VisualToolBench bash scripts_local/run_ablation.sh"
        echo ""
        echo "  The directory must contain:"
        echo "    val_50.json   val_single.json   val_multi.json   val_full.json"
        echo "    images/<doc_id>/img_*.png   (referenced by the JSON files)"
        echo ""
        exit 1
    fi

    # If user pointed VTB_DATA_DIR somewhere outside benchmark/, symlink it in
    # so XSkill's --image-folder=benchmark resolution still works.
    local DEFAULT_DIR="$(pwd)/benchmark/VisualToolBench"
    if [ "${VTB_DATA_DIR}" != "${DEFAULT_DIR}" ] && [ ! -e "${DEFAULT_DIR}" ]; then
        echo "[data] Linking ${VTB_DATA_DIR} -> ${DEFAULT_DIR}"
        mkdir -p "$(dirname "${DEFAULT_DIR}")"
        ln -s "${VTB_DATA_DIR}" "${DEFAULT_DIR}"
    fi
    echo "[data] VisualToolBench OK at ${VTB_DATA_DIR}"
}

# ============ Generate per-experiment YAML ============
# Writes ${RESULTS_DIR}/configs/<dataset>_<mode>.yaml that records every setting
# applied to this run. The yaml is then read back to build CLI args for
# eval/infer_api.py (which is argparse-based, not config-file-driven).
generate_config() {
    local DATASET="$1" MODE="$2"
    local EXP_NAME="${DATASET}_${MODE}"
    local CFG_OUT="${RESULTS_DIR}/configs/${EXP_NAME}.yaml"
    local DATA_PATH; DATA_PATH=$(resolve_dataset "${DATASET}")
    local MEMORY_BANK_DIR="${RESULTS_DIR}/memory_banks/${EXP_NAME}"
    local OUTPUT_DIR="${RESULTS_DIR}/output_${EXP_NAME}"

    "${PYTHON}" - <<PYEOF
import yaml, sys, os

dataset = "${DATASET}"
mode = "${MODE}"
exp_name = "${EXP_NAME}"

cfg = {
    "experiment": {
        "name": exp_name,
        "dataset": dataset,
        "mode": mode,
        "input_file": "${DATA_PATH}",
        "output_dir": "${OUTPUT_DIR}",
        "memory_bank_dir": "${MEMORY_BANK_DIR}",
        "log_file": "${RESULTS_DIR}/logs/${EXP_NAME}.log",
    },
    "backend": {
        "type": "${LLM_BACKEND}",
        "reasoning_model": os.environ.get("REASONING_MODEL_NAME", "Qwen3.6-27B"),
        "reasoning_endpoint": os.environ.get(
            "REASONING_END_POINT", "http://localhost:${QWEN_PORT}/v1/chat/completions"),
        "verifier_endpoint": os.environ.get(
            "VERIFIER_END_POINT", "http://localhost:${QWEN_PORT}/v1"),
        "experience_endpoint": os.environ.get(
            "EXPERIENCE_END_POINT", "http://localhost:${QWEN_PORT}/v1"),
        "embedding_model": os.environ.get("EXPERIENCE_EMBEDDING_MODEL", "bge-m3"),
        "embedding_endpoint": os.environ.get(
            "EXPERIENCE_EMBEDDING_ENDPOINT", "http://localhost:${BGE_PORT}/v1"),
    },
    "vllm": {
        "qwen_model_path": "${QWEN_MODEL_PATH}",
        "qwen_gpus": "${QWEN_GPUS}",
        "qwen_tp": ${QWEN_TP},
        "qwen_port": ${QWEN_PORT},
        "qwen_max_len": ${QWEN_MAX_LEN},
        "qwen_gpu_mem": ${QWEN_GPU_MEM},
        "qwen_tool_parser": "${QWEN_TOOL_PARSER}",
        "bge_model_path": "${BGE_MODEL_PATH}",
        "bge_gpus": "${BGE_GPUS}",
        "bge_port": ${BGE_PORT},
        "bge_gpu_mem": ${BGE_GPU_MEM},
    },
    "inference": {
        "temperature": ${TEMPERATURE},
        "top_p": ${TOP_P},
        "max_turns": ${MAX_TURNS},
        "max_images": ${MAX_IMAGES},
        "max_total_tokens": ${MAX_TOTAL_TOKENS},
        "num_workers": ${NUM_WORKERS},
        "rollouts_per_sample": ${ROLLOUTS_PER_SAMPLE},
        "image_search_max_calls": ${IMAGE_SEARCH_MAX_CALLS},
        "web_search_max_calls": ${WEB_SEARCH_MAX_CALLS},
        "system_prompt_key": "${SYSTEM_PROMPT_TYPE}",
        "enabled_tools": "${ENABLED_TOOLS}",
        "max_samples": int(os.environ.get("MAX_SAMPLES", "-1")),
    },
    # Mode-specific switches consumed by eval/infer_api.py via CLI flags below.
    "skill": {"enable": False, "inference": False, "refine": False,
              "library": None, "max_length": ${SKILL_MAX_LENGTH}},
    "experience": {"enable": False, "library": None,
                   "retrieval": False, "retrieval_top_k": ${EXPERIENCE_RETRIEVAL_TOP_K},
                   "retrieval_decomposition": False, "retrieval_rewrite": False,
                   "online_generate": False, "library_update": False, "refine": False,
                   "max_ops": ${EXPERIENCE_MAX_OPS}, "max_items": ${EXPERIENCE_MAX_ITEMS},
                   "large_batch": ${EXPERIENCE_LARGE_BATCH}},
}

if mode == "baseline":
    pass  # everything stays disabled
elif mode == "xskill_retrieve":
    cfg["skill"].update({
        "enable": True, "inference": True,
        "library": "${MEMORY_BANK_DIR}/SKILL.md",
    })
    cfg["experience"].update({
        "enable": True, "retrieval": True,
        "retrieval_decomposition": True, "retrieval_rewrite": True,
        "library": "${MEMORY_BANK_DIR}/experiences.json",
    })
elif mode == "xskill_full":
    cfg["skill"].update({
        "enable": True, "inference": True, "refine": True,
        "library": "${MEMORY_BANK_DIR}/SKILL.md",
    })
    cfg["experience"].update({
        "enable": True, "retrieval": True,
        "retrieval_decomposition": True, "retrieval_rewrite": True,
        "online_generate": True, "library_update": True, "refine": True,
        "library": "${MEMORY_BANK_DIR}/experiences.json",
    })
else:
    print(f"[ERROR] Unknown mode: {mode}", file=sys.stderr)
    sys.exit(1)

with open("${CFG_OUT}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
print("${CFG_OUT}")
PYEOF
}

# ============ Translate yaml → CLI args for eval/infer_api.py ============
build_cli_args() {
    local CFG="$1"
    "${PYTHON}" - "${CFG}" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
exp = cfg["experiment"]; inf = cfg["inference"]
sk = cfg["skill"]; ex = cfg["experience"]

args = [
    "--input-file", exp["input_file"],
    "--image-folder", "benchmark",
    "--output-dir", exp["output_dir"],
    "--temperature", str(inf["temperature"]),
    "--top-p", str(inf["top_p"]),
    "--max-turns", str(inf["max_turns"]),
    "--max-images", str(inf["max_images"]),
    "--max-total-tokens", str(inf["max_total_tokens"]),
    "--system-prompt-key", inf["system_prompt_key"],
    "--num-workers", str(inf["num_workers"]),
    "--tool-config-path", "eval/configs/tool_configs.yaml",
    "--max-samples", str(inf["max_samples"]),
    "--rollouts-per-sample", str(inf["rollouts_per_sample"]),
    "--image-search-max-calls", str(inf["image_search_max_calls"]),
    "--web-search-max-calls", str(inf["web_search_max_calls"]),
]
if sk["enable"]:
    args += ["--skill-enable"]
    if sk.get("library"):  args += ["--skill-library", sk["library"]]
    if sk.get("inference"): args += ["--skill-inference"]
    if sk.get("refine"):
        args += ["--skill-refine", "--skill-max-length", str(sk["max_length"])]
if ex["enable"]:
    args += ["--experience-enable"]
    if ex.get("library"): args += ["--experience-library", ex["library"]]
    if ex.get("retrieval"):
        args += ["--experience-retrieval",
                 "--experience-retrieval-top-k", str(ex["retrieval_top_k"])]
        if ex.get("retrieval_decomposition"): args += ["--experience-retrieval-decomposition"]
        if ex.get("retrieval_rewrite"):       args += ["--experience-retrieval-rewrite"]
    if ex.get("online_generate"):  args += ["--experience-online-generate"]
    if ex.get("library_update"):
        args += ["--experience-library-update",
                 "--experience-max-ops", str(ex["max_ops"]),
                 "--experience-large-batch", str(ex["large_batch"])]
    if ex.get("refine"):
        args += ["--experience-refine", "--experience-max-items", str(ex["max_items"])]

# Print one arg per line (so callers can read into bash arrays safely)
print("\n".join(args))
PYEOF
}

# ============ Common env exports for evaluation jobs ============
export_common_env() {
    if [ "${LLM_BACKEND}" = "local" ]; then
        export REASONING_MODEL_NAME="${REASONING_MODEL_NAME:-Qwen3.6-27B}"
        export REASONING_API_KEY="${REASONING_API_KEY:-EMPTY}"
        export REASONING_END_POINT="${REASONING_END_POINT:-http://localhost:${QWEN_PORT}/v1/chat/completions}"
        export REASONING_API_KEY_2="${REASONING_API_KEY_2:-EMPTY}"
        export REASONING_END_POINT_2="${REASONING_END_POINT_2:-http://localhost:${QWEN_PORT}/v1/chat/completions}"

        export VERIFIER_MODEL_NAME="${VERIFIER_MODEL_NAME:-Qwen3.6-27B}"
        export VERIFIER_API_KEY="${VERIFIER_API_KEY:-EMPTY}"
        export VERIFIER_END_POINT="${VERIFIER_END_POINT:-http://localhost:${QWEN_PORT}/v1}"

        export EXPERIENCE_MODEL_NAME="${EXPERIENCE_MODEL_NAME:-Qwen3.6-27B}"
        export EXPERIENCE_API_KEY="${EXPERIENCE_API_KEY:-EMPTY}"
        export EXPERIENCE_END_POINT="${EXPERIENCE_END_POINT:-http://localhost:${QWEN_PORT}/v1}"
        export EXPERIENCE_API_KEY_2="${EXPERIENCE_API_KEY_2:-EMPTY}"
        export EXPERIENCE_END_POINT_2="${EXPERIENCE_END_POINT_2:-http://localhost:${QWEN_PORT}/v1}"

        export EXPERIENCE_EMBEDDING_MODEL="${EXPERIENCE_EMBEDDING_MODEL:-bge-m3}"
        export EXPERIENCE_EMBEDDING_API_KEY="${EXPERIENCE_EMBEDDING_API_KEY:-EMPTY}"
        export EXPERIENCE_EMBEDDING_ENDPOINT="${EXPERIENCE_EMBEDDING_ENDPOINT:-http://localhost:${BGE_PORT}/v1}"
    fi
    export ENABLE_FUNCTION_CALLING="true"
    export ENABLED_TOOLS
}

# ============ vLLM Server Management (local mode) ============
QWEN_PID=""
BGE_PID=""

start_vllm_servers() {
    if [ "${LLM_BACKEND}" != "local" ]; then return; fi

    if [ "${AUTO_VLLM}" != "true" ]; then
        echo "[vLLM] AUTO_VLLM=false: skip auto-start. Verifying existing services..."
        if ! curl -s --max-time 5 "http://localhost:${QWEN_PORT}/v1/models" > /dev/null; then
            echo "[ERROR] Port ${QWEN_PORT} (Qwen) not reachable. Start vLLM or set AUTO_VLLM=true." >&2
            exit 1
        fi
        if ! curl -s --max-time 5 "http://localhost:${BGE_PORT}/v1/models" > /dev/null; then
            echo "[ERROR] Port ${BGE_PORT} (BGE-M3) not reachable. Start vLLM or set AUTO_VLLM=true." >&2
            exit 1
        fi
        echo "[vLLM] Existing services on ${QWEN_PORT}/${BGE_PORT} are healthy."
        return
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Starting vLLM servers (env: ${VLLM_CONDA_ENV_NAME})"
    echo "╚══════════════════════════════════════════════════════════╝"

    # vLLM workers spawn subprocesses (ninja, cicc, gcc...) that look up tools
    # via PATH. We must run vllm with its own env's PATH, NOT the eval env's.
    local VLLM_PATH="${VLLM_ENV_DIR}/bin:${PATH}"

    # 1) Qwen3.6-27B (Reasoning/Verifier/Experience)
    if curl -s --max-time 5 "http://localhost:${QWEN_PORT}/v1/models" > /dev/null; then
        echo "  [Qwen] Port ${QWEN_PORT} already serving — reuse."
    else
        echo "  [Qwen] Launching: ${QWEN_MODEL_PATH}"
        echo "         GPUs=${QWEN_GPUS}, tp=${QWEN_TP}, port=${QWEN_PORT}"
        CUDA_VISIBLE_DEVICES=${QWEN_GPUS} \
            PATH="${VLLM_PATH}" \
            CONDA_PREFIX="${VLLM_ENV_DIR}" \
            "${VLLM_BIN}" serve "${QWEN_MODEL_PATH}" \
            --tensor-parallel-size ${QWEN_TP} \
            --port ${QWEN_PORT} \
            --host 0.0.0.0 \
            --served-model-name Qwen3.6-27B \
            --trust-remote-code \
            --dtype bfloat16 \
            --max-model-len ${QWEN_MAX_LEN} \
            --gpu-memory-utilization ${QWEN_GPU_MEM} \
            --enable-auto-tool-choice \
            --tool-call-parser ${QWEN_TOOL_PARSER} \
            ${VLLM_EXTRA_ARGS:-} \
            > "${RESULTS_DIR}/logs/vllm_qwen.log" 2>&1 &
        QWEN_PID=$!
        echo "  [Qwen] PID=${QWEN_PID}"
    fi

    # 2) BGE-M3 (Embedding)
    if curl -s --max-time 5 "http://localhost:${BGE_PORT}/v1/models" > /dev/null; then
        echo "  [BGE]  Port ${BGE_PORT} already serving — reuse."
    else
        echo "  [BGE]  Launching: ${BGE_MODEL_PATH}"
        echo "         GPUs=${BGE_GPUS}, port=${BGE_PORT}"
        CUDA_VISIBLE_DEVICES=${BGE_GPUS} \
            PATH="${VLLM_PATH}" \
            CONDA_PREFIX="${VLLM_ENV_DIR}" \
            "${VLLM_BIN}" serve "${BGE_MODEL_PATH}" \
            --runner pooling \
            --port ${BGE_PORT} \
            --host 0.0.0.0 \
            --served-model-name bge-m3 \
            --trust-remote-code \
            --gpu-memory-utilization ${BGE_GPU_MEM} \
            > "${RESULTS_DIR}/logs/vllm_bge.log" 2>&1 &
        BGE_PID=$!
        echo "  [BGE]  PID=${BGE_PID}"
    fi

    # Wait for both servers to be ready
    echo "  Waiting for vLLM servers (max ${VLLM_START_TIMEOUT:-1800}s)..."
    local MAX_WAIT=${VLLM_START_TIMEOUT:-1800} WAIT_COUNT=0
    while true; do
        local QWEN_OK=false BGE_OK=false
        curl -s --max-time 3 "http://localhost:${QWEN_PORT}/v1/models" > /dev/null 2>&1 && QWEN_OK=true
        curl -s --max-time 3 "http://localhost:${BGE_PORT}/v1/models" > /dev/null 2>&1 && BGE_OK=true
        if [ "$QWEN_OK" = true ] && [ "$BGE_OK" = true ]; then
            echo "  Both vLLM servers ready!"
            break
        fi
        sleep 15
        WAIT_COUNT=$((WAIT_COUNT + 15))
        if [ ${WAIT_COUNT} -ge ${MAX_WAIT} ]; then
            echo "[ERROR] vLLM didn't come up in ${MAX_WAIT}s." >&2
            tail -50 "${RESULTS_DIR}/logs/vllm_qwen.log" 2>/dev/null
            tail -50 "${RESULTS_DIR}/logs/vllm_bge.log" 2>/dev/null
            exit 1
        fi
        if [ -n "${QWEN_PID}" ] && ! kill -0 ${QWEN_PID} 2>/dev/null; then
            echo "[ERROR] Qwen vLLM died." >&2
            tail -80 "${RESULTS_DIR}/logs/vllm_qwen.log" 2>/dev/null
            exit 1
        fi
        if [ -n "${BGE_PID}" ] && ! kill -0 ${BGE_PID} 2>/dev/null; then
            echo "[ERROR] BGE vLLM died." >&2
            tail -80 "${RESULTS_DIR}/logs/vllm_bge.log" 2>/dev/null
            exit 1
        fi
        echo "    ...${WAIT_COUNT}s (qwen=${QWEN_OK}, bge=${BGE_OK})"
    done
    echo ""
}

stop_vllm_servers() {
    # Only stop PIDs spawned by this script (do not touch external services).
    if [ -n "${QWEN_PID}" ] && kill -0 ${QWEN_PID} 2>/dev/null; then
        echo "Stopping Qwen vLLM (pid=${QWEN_PID})..."
        kill -TERM ${QWEN_PID} 2>/dev/null || true
    fi
    if [ -n "${BGE_PID}" ] && kill -0 ${BGE_PID} 2>/dev/null; then
        echo "Stopping BGE vLLM (pid=${BGE_PID})..."
        kill -TERM ${BGE_PID} 2>/dev/null || true
    fi
    sleep 3
    [ -n "${QWEN_PID}" ] && kill -9 ${QWEN_PID} 2>/dev/null || true
    [ -n "${BGE_PID}" ] && kill -9 ${BGE_PID} 2>/dev/null || true
}
trap stop_vllm_servers EXIT

start_vllm_servers
export_common_env
ensure_visualtoolbench_data

# ============ Build job list ============
JOBS=()
for DS in "${DATASETS_ARR[@]}"; do
    for MODE in "${MODES_ARR[@]}"; do
        JOBS+=("${DS}|${MODE}")
    done
done
TOTAL_JOBS=${#JOBS[@]}

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  XSkill Ablation: backbone vs +retrieve vs +full           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "  Backend  : ${LLM_BACKEND}  (auto_vllm=${AUTO_VLLM})"
echo "  Datasets : ${DATASETS_ARR[*]}"
echo "  Modes    : ${MODES_ARR[*]}"
echo "  Total    : ${TOTAL_JOBS} jobs"
echo "  Output   : ${RESULTS_DIR}"
echo ""

# Summary CSV
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
echo "dataset,mode,n_samples,rollouts,pass_at_1,pass_at_2,avg_at_2,acc,duration_sec,status,output_dir" \
    > "${SUMMARY_CSV}"

# ============ Per-job runner ============
run_one() {
    local DS="$1" MODE="$2"
    local TAG="${DS}_${MODE}"
    local LOG_FILE="${RESULTS_DIR}/logs/${TAG}.log"
    local OUTPUT_DIR="${RESULTS_DIR}/output_${TAG}"
    local MEMORY_BANK_DIR="${RESULTS_DIR}/memory_banks/${TAG}"
    mkdir -p "${OUTPUT_DIR}" "${MEMORY_BANK_DIR}"

    # Generate per-experiment YAML config
    local CFG_FILE
    CFG_FILE=$(generate_config "${DS}" "${MODE}")
    [ -z "${CFG_FILE}" ] && { echo "[ERROR] Failed to generate config for ${TAG}" >&2; return 1; }

    # Read CLI args from the YAML (one per line)
    local CLI_ARGS=()
    while IFS= read -r line; do
        [ -n "${line}" ] && CLI_ARGS+=("${line}")
    done < <(build_cli_args "${CFG_FILE}")

    local START_TIME; START_TIME=$(date +%s)
    echo "  [run] ${TAG}  $(date '+%H:%M:%S')  cfg=$(basename "${CFG_FILE}")"

    "${PYTHON}" -u eval/infer_api.py "${CLI_ARGS[@]}" \
        > "${LOG_FILE}" 2>&1 \
        && STATUS="ok" || STATUS="fail"

    local END_TIME; END_TIME=$(date +%s)
    local DURATION=$(( END_TIME - START_TIME ))

    # Parse metrics from log
    local N_SAMPLES ROLLOUTS PASS1 PASS2 AVG2 ACC
    N_SAMPLES=$(grep -oP 'Total samples processed: \K[0-9]+'  "${LOG_FILE}" | tail -1 || echo "?")
    ROLLOUTS=${ROLLOUTS_PER_SAMPLE}
    PASS1=$(grep -oP 'Pass@1:\s*\K[0-9.]+' "${LOG_FILE}" | tail -1 || echo "?")
    PASS2=$(grep -oP 'Pass@2:\s*\K[0-9.]+' "${LOG_FILE}" | tail -1 || echo "?")
    AVG2=$(grep -oP 'Average@2:\s*\K[0-9.]+' "${LOG_FILE}" | tail -1 || echo "?")
    ACC=$(grep -oP 'Overall Accuracy Score:\s*\K[0-9.]+' "${LOG_FILE}" | tail -1 || echo "?")

    (
        flock -x 200
        echo "${DS},${MODE},${N_SAMPLES},${ROLLOUTS},${PASS1},${PASS2},${AVG2},${ACC},${DURATION},${STATUS},${OUTPUT_DIR}" \
            >> "${SUMMARY_CSV}"
    ) 200>"${RESULTS_DIR}/.summary_lock"

    echo "  [done] ${TAG}  ${STATUS}  ${DURATION}s  pass@1=${PASS1} pass@2=${PASS2} acc=${ACC}"
    if [ "${STATUS}" = "fail" ]; then
        echo "  [error] last 30 lines of log:" >&2
        tail -30 "${LOG_FILE}" >&2 || true
    fi
}

# ============ Dispatch (sequential — share same vLLM, avoid contention) ============
echo ""
JOB_NUM=0
for job_str in "${JOBS[@]}"; do
    JOB_NUM=$(( JOB_NUM + 1 ))
    IFS='|' read -r DS MODE <<< "${job_str}"
    echo "  [dispatch ${JOB_NUM}/${TOTAL_JOBS}] ${DS}/${MODE}"
    run_one "${DS}" "${MODE}"
done

echo ""
echo "============================================================"
echo "  All ${TOTAL_JOBS} jobs done."
echo "  Results: ${RESULTS_DIR}"
echo "  Summary: ${SUMMARY_CSV}"
echo "============================================================"
echo ""
if command -v column &>/dev/null; then
    column -t -s, "${SUMMARY_CSV}"
else
    cat "${SUMMARY_CSV}"
fi
echo ""
echo "Detailed logs: ${RESULTS_DIR}/logs/"
echo "Outputs:       ${RESULTS_DIR}/output_<dataset>_<mode>/"
echo "Memory banks:  ${RESULTS_DIR}/memory_banks/<dataset>_<mode>/"
echo "Configs used:  ${RESULTS_DIR}/configs/"
