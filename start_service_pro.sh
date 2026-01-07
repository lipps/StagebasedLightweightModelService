#!/bin/bash

# BGE-M3 Pro 版服务启动脚本
# 用法: ./start_service_pro.sh [cpu|gpu]

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== BGE-M3 Pro 版服务启动 ===${NC}\n"

# 检查参数
MODE=${1:-auto}

# 配置环境变量
export MODEL_PATH="/opt/bge-m3/models/bge-m3"
export LOG_LEVEL="INFO"
export ENABLE_WARMUP="true"
export MAX_BATCH_SIZE="128"
export MAX_LENGTH="8192"

# 根据模式配置
case $MODE in
  cpu)
    echo -e "${YELLOW}[模式] CPU 多线程${NC}"
    export DEVICE="cpu"
    export CPU_THREADS="8"
    ;;
  gpu)
    echo -e "${GREEN}[模式] GPU 加速${NC}"
    export DEVICE="cuda"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export CPU_THREADS="1"
    ;;
  auto)
    echo -e "${GREEN}[模式] 自动选择设备${NC}"
    export DEVICE="auto"
    export CPU_THREADS="8"
    ;;
  *)
    echo -e "${RED}错误: 无效模式 '$MODE'${NC}"
    echo "用法: $0 [cpu|gpu|auto]"
    exit 1
    ;;
esac

# 显示配置
echo -e "\n${GREEN}[配置信息]${NC}"
echo "  模型路径: $MODEL_PATH"
echo "  设备模式: $DEVICE"
echo "  CPU 线程: $CPU_THREADS"
echo "  最大批次: $MAX_BATCH_SIZE"
echo "  最大长度: $MAX_LENGTH"
echo "  日志级别: $LOG_LEVEL"
echo "  模型预热: $ENABLE_WARMUP"

# 激活虚拟环境
echo -e "\n${GREEN}[激活环境]${NC}"
if [ -f "/opt/bge-m3/.venv/bin/activate" ]; then
    source /opt/bge-m3/.venv/bin/activate
    echo "  虚拟环境已激活"
else
    echo -e "${RED}错误: 虚拟环境不存在${NC}"
    exit 1
fi

# 检查模型文件
echo -e "\n${GREEN}[检查模型]${NC}"
if [ -f "$MODEL_PATH/model.safetensors" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH/model.safetensors" | cut -f1)
    echo "  ✅ SafeTensors 模型: $MODEL_SIZE"
elif [ -f "$MODEL_PATH/pytorch_model.bin" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH/pytorch_model.bin" | cut -f1)
    echo "  ⚠️  PyTorch 模型: $MODEL_SIZE (建议转换为 SafeTensors)"
else
    echo -e "${RED}错误: 模型文件不存在${NC}"
    exit 1
fi

# 检查 GPU（如果配置为 GPU 模式）
if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "auto" ]; then
    echo -e "\n${GREEN}[GPU 检查]${NC}"
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
        echo "  GPU: $GPU_INFO"
    else
        echo "  ⚠️  nvidia-smi 未找到"
    fi
fi

# 启动服务
echo -e "\n${GREEN}[启动服务]${NC}"
echo "  访问地址: http://localhost:8001"
echo "  API 文档: http://localhost:8001/docs"
echo "  健康检查: http://localhost:8001/health"
echo "  统计信息: http://localhost:8001/stats"
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止服务${NC}\n"

# 启动 uvicorn
exec uvicorn serve_bge_m3_pro:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 1 \
    --log-level info
