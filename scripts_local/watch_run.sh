#!/bin/bash
# ============================================================================
# XSkill 全量评测后台监督脚本
# - 每 5 分钟检查一次关键指标
# - 写入单行结构化日志到 /tmp/xskill_watch.log
# - 检测异常：vLLM 挂掉 / 错误激增 / 进程死亡 / 磁盘不足
# - 异常时额外写一行 [ALERT] 到 /tmp/xskill_alert.log
# ============================================================================

WATCH_LOG="/tmp/xskill_watch.log"
ALERT_LOG="/tmp/xskill_alert.log"
INTERVAL=300   # 5 分钟

XSKILL_DIR="/data2/chenxuwu/zihaowan_workplace/XSkill"
TRAIN_LOG="$XSKILL_DIR/logs/train_vtb_full.log"
BASELINE_LOG="$XSKILL_DIR/logs/baseline_vtb_full.log"
XSKILL_RUN_LOG="$XSKILL_DIR/logs/xskill_vtb_full.log"

# header
{
    echo "# XSkill watch log started at $(date '+%F %T')"
    echo "# columns: time | stage | rollouts_done/total | recent_pass@2 | err_count | vllm_8080 | vllm_8081 | gpu0_used | gpu4_used | disk_free | infer_pid"
} >> "$WATCH_LOG"

while true; do
    TS=$(date '+%F %T')

    # ───── 1. 当前活跃 stage（看哪份 log 最新被写入）─────
    stage="unknown"
    active_log=""
    if   [ -f "$XSKILL_RUN_LOG" ]   && [ "$(stat -c %Y "$XSKILL_RUN_LOG"   2>/dev/null)" -ge "$(stat -c %Y "$BASELINE_LOG" 2>/dev/null || echo 0)" ] \
                                    && [ "$(stat -c %Y "$XSKILL_RUN_LOG"   2>/dev/null)" -ge "$(stat -c %Y "$TRAIN_LOG"    2>/dev/null || echo 0)" ]; then
        stage="xskill"
        active_log="$XSKILL_RUN_LOG"
    elif [ -f "$BASELINE_LOG" ]     && [ "$(stat -c %Y "$BASELINE_LOG"     2>/dev/null)" -ge "$(stat -c %Y "$TRAIN_LOG"    2>/dev/null || echo 0)" ]; then
        stage="baseline"
        active_log="$BASELINE_LOG"
    elif [ -f "$TRAIN_LOG" ]; then
        stage="train"
        active_log="$TRAIN_LOG"
    fi

    # ───── 2. infer_api.py 进程是否还活 ─────
    infer_pid=$(pgrep -f "eval/infer_api.py" 2>/dev/null | head -1)
    [ -z "$infer_pid" ] && infer_pid="DEAD"

    # ───── 3. 已完成 rollouts（从 active log 找最后一行进度条）─────
    rollouts_done="?"
    rollouts_total="?"
    if [ -n "$active_log" ] && [ -f "$active_log" ]; then
        # 形如 "Processing rollouts:  37%|... | 42/112 [..." 或 "Batch 9: 100%|██| 2/2"
        last_progress=$(grep -oE "[0-9]+/[0-9]+ \[" "$active_log" 2>/dev/null | tail -1 | tr -d '[')
        if [ -n "$last_progress" ]; then
            rollouts_done="${last_progress%/*}"
            rollouts_total="${last_progress#*/}"
        fi
    fi

    # ───── 4. 最近 1000 行里 pass@2 平均（最近样本表现）─────
    recent_pass="?"
    if [ -n "$active_log" ] && [ -f "$active_log" ]; then
        recent_pass=$(grep -oE "pass@2=[0-9.]+" "$active_log" 2>/dev/null | tail -20 | awk -F= '{s+=$2; n+=1} END{ if(n>0) printf "%.2f", s/n; else printf "?" }')
    fi

    # ───── 5. 错误计数（HTTP 4xx / Traceback / Failed）─────
    err_count=0
    if [ -n "$active_log" ] && [ -f "$active_log" ]; then
        err_count=$(grep -cE "Traceback|HTTP 4[0-9][0-9]|All API attempts failed|Experience API error" "$active_log" 2>/dev/null || echo 0)
    fi

    # ───── 6. vLLM 健康检查 ─────
    vllm_8080="OK"
    vllm_8081="OK"
    if ! curl -s --max-time 5 -o /dev/null -w "%{http_code}" http://localhost:8080/v1/models 2>/dev/null | grep -q 200; then
        vllm_8080="DOWN"
    fi
    if ! curl -s --max-time 5 -o /dev/null -w "%{http_code}" http://localhost:8081/v1/models 2>/dev/null | grep -q 200; then
        vllm_8081="DOWN"
    fi

    # ───── 7. GPU 使用情况 ─────
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
    gpu0_used=$(echo "$gpu_info" | awk -F', ' '$1==0 {print $2"M"}')
    gpu4_used=$(echo "$gpu_info" | awk -F', ' '$1==4 {print $2"M"}')

    # ───── 8. 磁盘剩余 ─────
    disk_free=$(df -h /data2 | tail -1 | awk '{print $4}')

    # ───── 写入主日志 ─────
    line="$TS | $stage | $rollouts_done/$rollouts_total | recent_pass@2=$recent_pass | err=$err_count | vllm8080=$vllm_8080 | vllm8081=$vllm_8081 | gpu0=$gpu0_used | gpu4=$gpu4_used | disk=$disk_free | pid=$infer_pid"
    echo "$line" >> "$WATCH_LOG"

    # ───── 异常检测 → ALERT ─────
    alert_reason=""
    [ "$infer_pid" = "DEAD" ] && alert_reason="${alert_reason}[INFER_DEAD]"
    [ "$vllm_8080" = "DOWN" ] && alert_reason="${alert_reason}[VLLM_8080_DOWN]"
    [ "$vllm_8081" = "DOWN" ] && alert_reason="${alert_reason}[VLLM_8081_DOWN]"
    # 磁盘剩余少于 30G
    disk_free_g=$(df -BG /data2 | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ -n "$disk_free_g" ] && [ "$disk_free_g" -lt 30 ]; then
        alert_reason="${alert_reason}[DISK_LOW=$disk_free]"
    fi
    # 错误突增（这次比上次 +50）
    last_err=$(grep -oE "err=[0-9]+" "$WATCH_LOG" 2>/dev/null | tail -2 | head -1 | cut -d= -f2)
    if [ -n "$last_err" ] && [ -n "$err_count" ] && [ "$err_count" -gt $((last_err + 50)) ]; then
        alert_reason="${alert_reason}[ERR_SURGE:+$((err_count - last_err))]"
    fi

    if [ -n "$alert_reason" ]; then
        echo "$TS | $alert_reason | $line" >> "$ALERT_LOG"
    fi

    sleep "$INTERVAL"
done
