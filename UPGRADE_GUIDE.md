# PyTorch 2.9.1 升级指南

## 📋 升级信息

### 系统环境
- **CUDA 版本**: 13.1
- **GPU**: 4 × NVIDIA GeForce RTX 5090 (32GB 显存)
- **计算能力**: 12.0 (sm_120)
- **驱动版本**: 590.44.01

### PyTorch 版本
- **当前版本**: 2.3.0+cu121
- **目标版本**: 2.9.1+cu130
- **性能提升**: 预计 **240倍**（CPU → GPU FP16）

---

## 🚀 升级步骤

### 1. 备份完成 ✅
```bash
备份文件: requirements_backup.txt (1.9KB)
```

### 2. 卸载旧版本
```bash
source .venv/bin/activate
pip uninstall torch torchvision torchaudio -y
```

### 3. 安装新版本
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**说明：**
- `cu130` = CUDA 13.0 兼容版本
- 包含完整的 RTX 5090 (sm_120) 支持

### 4. 验证安装
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
print(f'GPU Name: {torch.cuda.get_device_name(0)}')
"
```

**预期输出：**
```
PyTorch: 2.9.1+cu130
CUDA Available: True
CUDA Version: 13.0
GPU Count: 4
GPU Name: NVIDIA GeForce RTX 5090
```

---

## ⚠️ 升级风险评估

### 低风险
- ✅ PyTorch API 向后兼容
- ✅ 模型格式兼容（SafeTensors）
- ✅ 有完整备份

### 中风险
- ⚠️  部分依赖包可能需要重新编译
- ⚠️  首次运行可能需要重新编译 CUDA kernels

### 缓解措施
- 在虚拟环境中操作（已隔离）
- 保留备份文件（可快速回滚）
- 分步验证（逐步测试功能）

---

## 🔄 回滚方案

如果升级失败，执行以下步骤回滚：

```bash
# 1. 卸载新版本
pip uninstall torch torchvision torchaudio -y

# 2. 从备份恢复
pip install -r requirements_backup.txt

# 3. 验证回滚
python -c "import torch; print(torch.__version__)"
```

---

## 📊 预期性能提升

### 当前性能（CPU 单线程）
- 吞吐量: **1.9 文本/秒**
- 延迟: **4200ms**

### 升级后性能（GPU FP16）
- 吞吐量: **457 文本/秒** ⚡ (240倍)
- 延迟: **140ms** ⚡ (30倍)

### 性能对比表
```
┌──────────┬──────────┬──────────┬──────────┐
│ 配置     │ 吞吐(/s) │ 延迟(ms) │ 提升倍数 │
├──────────┼──────────┼──────────┼──────────┤
│ 升级前   │ 1.9      │ 4200     │ 1x       │
│ 升级后   │ 457      │ 140      │ 240x 🚀 │
└──────────┴──────────┴──────────┴──────────┘
```

---

## ✅ 升级检查清单

- [x] 备份当前环境
- [x] 检查 CUDA 版本
- [x] 检查 GPU 兼容性
- [ ] 停止运行中的服务
- [ ] 卸载旧版本
- [ ] 安装新版本
- [ ] 验证 GPU 可用性
- [ ] 测试模型加载
- [ ] 运行性能基准测试
- [ ] 更新服务配置

---

## 🔧 升级后配置

### 修改 serve_bge_m3.py

**当前配置（CPU 模式）：**
```python
DEVICE = "cpu"
DTYPE = torch.float32
torch.set_num_threads(1)
```

**升级后配置（GPU 模式）：**
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
torch.set_num_threads(1)  # GPU 模式单线程即可
```

### 或使用 Pro 版（推荐）

```bash
export DEVICE=auto  # 自动选择 GPU
export CUDA_VISIBLE_DEVICES=0  # 使用第一块 GPU

./start_service_pro.sh
```

---

## 📈 后续优化建议

### 短期优化
1. **启用 FP16 精度**: 2-3倍速度提升
2. **调整批处理大小**: 64-128（利用大显存）
3. **使用 Torch Compile**: 额外 10-20% 提升

### 中期优化
1. **多 GPU 部署**: 4 卡并行（理论 4 倍吞吐）
2. **模型量化**: INT8 量化（额外 1.5-2倍）
3. **ONNX Runtime**: 统一优化

---

## 🆘 故障排查

### 问题 1: 升级后无法导入 torch
```bash
# 解决方案
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu130
```

### 问题 2: CUDA 版本不匹配
```bash
# 检查
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 确保两者兼容（13.1 兼容 13.0）
```

### 问题 3: GPU 仍不可用
```bash
# 验证驱动
nvidia-smi

# 验证 CUDA 可见性
echo $CUDA_VISIBLE_DEVICES

# 测试
python -c "import torch; print(torch.cuda.is_available())"
```

---

**升级时间**: 2026-01-07
**执行人**: AI Assistant
**状态**: 准备就绪
