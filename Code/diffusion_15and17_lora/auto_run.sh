#!/bin/bash

##############################################################################
# 基础配置（可按需修改）
##############################################################################
# 重试间隔时间（秒），3分钟 = 180秒
RETRY_INTERVAL=180
# 训练脚本路径（若不在同一目录，需写绝对路径，如 /home/user/project/train.py）
TRAIN_SCRIPT="train.py"
# 日志文件路径（记录每次尝试结果，避免丢失信息）
LOG_FILE="/home/jiangxinhai/GMABDA/Logs/fine_tuning/auto_train_retry.log"
# 单次训练详细日志目录
DETAIL_LOG_DIR="/home/jiangxinhai/GMABDA/Logs/fine_tuning/train_logs"
# 传递给 train.py 的额外参数（如指定设备、批次大小等，无则留空）
TRAIN_ARGS="--device cuda"

##############################################################################
# 工具函数
##############################################################################
log() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

##############################################################################
# 主执行逻辑
##############################################################################
mkdir -p "$DETAIL_LOG_DIR"  # 创建详细日志目录

log "=== 自动训练重试脚本启动 ==="
log "重试间隔：$RETRY_INTERVAL 秒（3分钟）"
log "训练脚本：$TRAIN_SCRIPT"
log "训练参数：$TRAIN_ARGS"
log "总日志文件：$LOG_FILE"
log "详细日志目录：$DETAIL_LOG_DIR"

# 循环重试
while true; do
    # 生成本次训练的详细日志文件名
    detail_log_file="${DETAIL_LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
    
    log "---------------- 开始新一次训练尝试 ----------------"
    log "本次详细日志：$detail_log_file"
    
    # 执行训练脚本
    python "$TRAIN_SCRIPT" $TRAIN_ARGS > "$detail_log_file" 2>&1
    
    # 检查训练退出码
    if [ $? -eq 0 ]; then
        log "训练成功！脚本将退出。"
        exit 0
    else
        log "本次训练失败！详细信息请查看: $detail_log_file"
        log "将在 $RETRY_INTERVAL 秒后重试..."
        sleep "$RETRY_INTERVAL"
    fi
done