#!/bin/bash
# ============================================================================
# 一键复现：XSkill 框架在 Qwen3.6-27B 上的增益验证
#
# 三阶段流水线（按顺序执行）：
#   Phase I   train:    在 single-turn 子集 (≈603 条) 上跑训练，积累 memory_bank
#   Phase II  baseline: 在 multi-turn 子集 (≈601 条) 上跑裸 backbone（关 XSkill）
#   Phase II  xskill:   在 multi-turn 子集上跑 +XSkill（用 Phase I 的 memory_bank）
#
# 用法：bash eval/run_full_compare.sh [TAG] [TRAIN_DATA] [EVAL_DATA] [TRAIN_MAX] [EVAL_MAX]
#
# 例：
#   # 50/50 小规模快速验证
#   bash eval/run_full_compare.sh smoke benchmark/VisualToolBench/val_single.json benchmark/VisualToolBench/val_multi.json 50 50
#
#   # 全量评测
#   bash eval/run_full_compare.sh full benchmark/VisualToolBench/val_single.json benchmark/VisualToolBench/val_multi.json -1 -1
# ============================================================================

set -e  # 任一步失败立即退出

TAG="${1:-vtb}"
TRAIN_DATA="${2:-benchmark/VisualToolBench/val_single.json}"
EVAL_DATA="${3:-benchmark/VisualToolBench/val_multi.json}"
TRAIN_MAX="${4:--1}"
EVAL_MAX="${5:--1}"

# 用同一个 TAG 串联三阶段（baseline 和 xskill 共享 memory bank）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."  # 回到 XSkill 项目根

LOG_DIR="logs/full_compare_${TAG}"
mkdir -p "$LOG_DIR"

echo ""
echo "##############################################################"
echo "#  Full Compare Run — TAG=$TAG"
echo "#  TRAIN: $TRAIN_DATA  (max=$TRAIN_MAX)"
echo "#  EVAL:  $EVAL_DATA   (max=$EVAL_MAX)"
echo "#  Log dir: $LOG_DIR"
echo "##############################################################"
echo ""

# ── Stage 1: 训练 ──────────────────────────────────────────────────────────
echo ""
echo "━━━ Stage 1/3: XSkill TRAIN (Phase I) ━━━"
START_T1=$(date +%s)
bash eval/run_xskill_train.sh "$TRAIN_DATA" "$TRAIN_MAX" "$TAG" 2>&1 | tee "$LOG_DIR/01_train.log"
T1=$(($(date +%s) - START_T1))
echo "✓ Stage 1 done in ${T1}s"

# ── Stage 2: baseline 评测 ─────────────────────────────────────────────────
echo ""
echo "━━━ Stage 2/3: BASELINE 评测（裸 backbone）━━━"
START_T2=$(date +%s)
bash eval/run_baseline.sh "$EVAL_DATA" "$EVAL_MAX" "$TAG" 2>&1 | tee "$LOG_DIR/02_baseline.log"
T2=$(($(date +%s) - START_T2))
echo "✓ Stage 2 done in ${T2}s"

# ── Stage 3: xskill 评测（复用 Stage 1 的 memory bank）────────────────────
echo ""
echo "━━━ Stage 3/3: XSKILL 评测（含 memory bank）━━━"
START_T3=$(date +%s)
# memory bank 路径由 run_xskill.sh 默认指向 memory_bank/${TAG}/，正好和 train 阶段一致
bash eval/run_xskill.sh "$EVAL_DATA" "$EVAL_MAX" "$TAG" 2>&1 | tee "$LOG_DIR/03_xskill.log"
T3=$(($(date +%s) - START_T3))
echo "✓ Stage 3 done in ${T3}s"

# ── 汇总 ────────────────────────────────────────────────────────────────────
echo ""
echo "##############################################################"
echo "#  Full Compare 完成"
echo "##############################################################"
echo "Train:    ${T1}s"
echo "Baseline: ${T2}s"
echo "XSkill:   ${T3}s"
echo ""
echo "结果文件:"
echo "  Memory bank:        memory_bank/${TAG}/"
echo "  Baseline output:    output/baseline_${TAG}/"
echo "  XSkill output:      output/xskill_${TAG}/"
echo "  Logs:               $LOG_DIR/"
echo ""
echo "对比 baseline vs xskill 的指标，看 XSkill 的增益:"
echo "  python3 - <<EOF"
echo "  import json"
echo "  for tag in ['baseline_${TAG}', 'xskill_${TAG}']:"
echo "      try:"
echo "          d = json.load(open(f'output/{tag}/metrics_at_k.json'))"
echo "          print(f'{tag:>20s}:', d)"
echo "      except FileNotFoundError:"
echo "          print(f'{tag} not found')"
echo "  EOF"
