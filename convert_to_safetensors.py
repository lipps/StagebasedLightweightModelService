#!/usr/bin/env python3
"""
将 pytorch_model.bin 转换为 model.safetensors 格式
解决 PyTorch 2.3.0 的安全漏洞问题
"""

import os
import torch
from safetensors.torch import save_file

MODEL_DIR = "/opt/bge-m3/models/bge-m3"
BIN_FILE = os.path.join(MODEL_DIR, "pytorch_model.bin")
SAFETENSORS_FILE = os.path.join(MODEL_DIR, "model.safetensors")

print("🔄 开始转换模型格式...")
print(f"📂 源文件: {BIN_FILE}")
print(f"📂 目标文件: {SAFETENSORS_FILE}")

# 检查源文件
if not os.path.exists(BIN_FILE):
    raise FileNotFoundError(f"源文件不存在: {BIN_FILE}")

# 使用旧版 torch.load 加载（2.3.0 支持）
print("\n📥 加载 PyTorch 权重...")
# 禁用安全检查以便在旧版本中加载
import warnings
warnings.filterwarnings('ignore')

# 临时设置环境变量跳过安全检查
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

try:
    # 直接使用 torch.load（在 transformers 检查之前）
    state_dict = torch.load(BIN_FILE, map_location='cpu', weights_only=False)
    print(f"✅ 成功加载 {len(state_dict)} 个权重张量")

    # 转换为 safetensors
    print("\n💾 保存为 safetensors 格式...")
    save_file(state_dict, SAFETENSORS_FILE)

    # 验证文件大小
    bin_size = os.path.getsize(BIN_FILE) / (1024**3)
    safe_size = os.path.getsize(SAFETENSORS_FILE) / (1024**3)

    print(f"\n✅ 转换完成!")
    print(f"📊 原始大小: {bin_size:.2f} GB")
    print(f"📊 新文件大小: {safe_size:.2f} GB")
    print(f"📊 差异: {((safe_size - bin_size) / bin_size * 100):+.1f}%")

    print("\n⚠️  建议操作:")
    print("1. 验证模型可以正常加载")
    print("2. 测试通过后可删除原始 pytorch_model.bin 文件")
    print(f"   rm {BIN_FILE}")

except Exception as e:
    print(f"\n❌ 转换失败: {e}")
    raise
