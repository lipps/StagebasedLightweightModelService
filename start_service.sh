#!/bin/bash

# BGE-M3 Embedding Service 启动脚本
# 用法: ./start_service.sh

# 指定使用 GPU 0
export CUDA_VISIBLE_DEVICES=0

# 激活虚拟环境
source /opt/bge-m3/.venv/bin/activate

# 启动服务
uvicorn serve_bge_m3:app --host 0.0.0.0 --port 8001 --workers 1
