#!/bin/bash

# PyTorch 升级后验证和测试脚本
# 用法: ./post_upgrade.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  PyTorch 2.9.1 升级后验证流程${NC}"
echo -e "${GREEN}=======================================${NC}\n"

# 激活环境
source .venv/bin/activate

# 步骤 1: 验证 PyTorch 安装
echo -e "${YELLOW}[步骤 1/5] 验证 PyTorch 安装${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo -e "${RED}❌ PyTorch 导入失败！${NC}"
    exit 1
}
echo -e "${GREEN}✅ PyTorch 安装成功${NC}\n"

# 步骤 2: GPU 功能验证
echo -e "${YELLOW}[步骤 2/5] GPU 功能验证${NC}"
python verify_gpu.py || {
    echo -e "${RED}❌ GPU 验证失败！${NC}"
    exit 1
}
echo

# 步骤 3: 模型加载测试
echo -e "${YELLOW}[步骤 3/5] 模型加载测试${NC}"
python -c "
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_ID = '/opt/bge-m3/models/bge-m3'
print('加载分词器...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
print('✅ 分词器加载成功')

print('加载模型（GPU FP16）...')
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True)
model.to('cuda')
model.eval()
print('✅ 模型加载到 GPU 成功')

print(f'显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB')
" || {
    echo -e "${RED}❌ 模型加载测试失败！${NC}"
    exit 1
}
echo -e "${GREEN}✅ 模型加载测试通过${NC}\n"

# 步骤 4: 快速功能测试
echo -e "${YELLOW}[步骤 4/5] 快速功能测试${NC}"
python -c "
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_ID = '/opt/bge-m3/models/bge-m3'
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True)
model.to('cuda')
model.eval()

# 测试编码
with torch.inference_mode():
    inputs = tokenizer(['测试文本'], padding=True, truncation=True, return_tensors='pt')
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)

print(f'输出形状: {outputs.last_hidden_state.shape}')
print('✅ 功能测试通过')
" || {
    echo -e "${RED}❌ 功能测试失败！${NC}"
    exit 1
}
echo -e "${GREEN}✅ 快速功能测试通过${NC}\n"

# 步骤 5: 性能基准测试（可选）
echo -e "${YELLOW}[步骤 5/5] 性能基准测试（可选）${NC}"
read -p "是否运行完整的性能基准测试？（需要 5-10 分钟）[y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${YELLOW}运行性能基准测试...${NC}\n"
    python benchmark_gpu.py
else
    echo -e "${YELLOW}跳过性能基准测试${NC}"
fi

# 总结
echo
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  升级验证完成！${NC}"
echo -e "${GREEN}=======================================${NC}\n"

echo -e "${GREEN}✅ 所有验证通过！${NC}"
echo
echo "PyTorch 2.9.1 + RTX 5090 已成功配置"
echo
echo "下一步操作:"
echo "  1. 启动 Pro 版服务（GPU 模式）:"
echo "     ${YELLOW}./start_service_pro.sh gpu${NC}"
echo
echo "  2. 或启动原版服务（需先修改配置）:"
echo "     修改 serve_bge_m3.py 中的 DEVICE = \"cuda\""
echo "     ${YELLOW}./start_service.sh${NC}"
echo
echo "  3. 运行性能基准测试:"
echo "     ${YELLOW}python benchmark_gpu.py${NC}"
echo
