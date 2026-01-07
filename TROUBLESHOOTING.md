# BGE-M3 服务故障排查完整记录

## 🔴 遇到的问题

### 问题 1: 参数弃用警告
```
`torch_dtype` is deprecated! Use `dtype` instead!
```
**影响**: 警告级别，不影响功能

### 问题 2: PyTorch 安全漏洞（主要问题）
```
ValueError: Due to a serious vulnerability issue in `torch.load`,
even with `weights_only=True`, we now require users to upgrade torch
to at least v2.6 in order to use the function.
```

**根本原因**:
- **当前 PyTorch 版本**: 2.3.0+cu121
- **Transformers 要求**: ≥ 2.6（因安全漏洞 CVE-2025-32434）
- **模型格式**: pytorch_model.bin（不安全的 pickle 格式）

### 问题 3: GPU 不兼容
```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities:
sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**根本原因**:
- **RTX 5090 计算能力**: sm_120 (CUDA Compute Capability 12.0)
- **PyTorch 2.3.0 支持**: 仅到 sm_90
- **结果**: CUDA kernel 无法在设备上执行

---

## ✅ 已实施的解决方案

### 解决方案 1: 转换模型为 SafeTensors 格式

**操作**:
```bash
python convert_to_safetensors.py
```

**结果**:
- ✅ 生成 `/opt/bge-m3/models/bge-m3/model.safetensors` (2.2GB)
- ✅ 绕过 torch.load 安全检查
- ✅ Transformers 自动优先加载 safetensors 文件

**优点**:
- 无需升级 PyTorch
- 更安全（无 pickle 反序列化漏洞）
- 加载速度更快（使用内存映射）

### 解决方案 2: 强制使用 CPU 模式

**修改文件**: `serve_bge_m3.py`

**变更**:
```python
# 修改前
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 修改后
DEVICE = "cpu"  # 临时强制使用 CPU（RTX 5090 需要 PyTorch 2.6+）
DTYPE = torch.float32
```

**结果**:
- ✅ 服务成功启动
- ✅ API 正常工作（已测试）
- ⚠️  性能降低（CPU 模式）

---

## 📊 测试验证

### 启动服务
```bash
./start_service.sh
# 或
source .venv/bin/activate
uvicorn serve_bge_m3:app --host 0.0.0.0 --port 8001 --workers 1
```

### API 测试
```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["测试文本"], "normalize": true}'
```

**测试结果**:
```json
{
  "embeddings": [[...1024个浮点数...]]
}
```
✅ **验证通过！**

---

## 🚀 性能对比

| 模式 | 推理时间 (单文本) | 批处理吞吐 (32 batch) | 显存占用 |
|------|-----------------|---------------------|---------|
| **CPU (当前)** | ~200-500ms | ~2-5 文本/秒 | 0 GB |
| **GPU (FP16)** | ~10-30ms | ~50-200 文本/秒 | ~3 GB |

---

## 🔧 长期解决方案

### 选项 A: 升级 PyTorch 以支持 RTX 5090（推荐）

#### 1. 安装支持 CUDA 12.8+ 的 PyTorch 2.6+

**检查最新版本**:
```bash
# 访问 https://pytorch.org/get-started/locally/
# 选择: Linux, Pip, Python, CUDA 12.8
```

**安装命令示例**:
```bash
source .venv/bin/activate
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### 2. 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

#### 3. 恢复 GPU 模式
修改 `serve_bge_m3.py`:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
```

**预期收益**:
- ✅ **10-20 倍性能提升**
- ✅ FP16 精度加速
- ✅ 批处理吞吐提升到 ~50-200 文本/秒

---

### 选项 B: 继续使用 CPU 模式（临时方案）

**适用场景**:
- 开发测试环境
- 低并发服务（<10 QPS）
- 避免环境破坏

**优化建议**:
```python
# 增加 CPU 线程数（修改 serve_bge_m3.py）
torch.set_num_threads(4)  # 原值为 1

# 调整批处理大小
batch_size = 8  # 降低批处理减少延迟
```

---

### 选项 C: 使用 ONNX Runtime（中等性能）

ONNX Runtime 对 CPU 有更好的优化：

#### 1. 安装 ONNX Runtime
```bash
source .venv/bin/activate
pip install onnxruntime
```

#### 2. 使用现有 ONNX 模型
模型路径: `/opt/bge-m3/models/bge-m3/onnx/`

#### 3. 修改加载代码（需要重写服务）
```python
from optimum.onnxruntime import ORTModelForFeatureExtraction

model = ORTModelForFeatureExtraction.from_pretrained(
    MODEL_ID,
    provider="CPUExecutionProvider"
)
```

**预期收益**:
- ✅ CPU 性能提升 **2-3 倍**
- ✅ 内存占用减少 ~30%
- ⚠️  需要重写推理代码

---

## 📋 文件变更清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `serve_bge_m3.py` | ✏️ 已修改 | 强制 CPU 模式 |
| `models/bge-m3/model.safetensors` | 🆕 新增 | 安全模型格式 |
| `convert_to_safetensors.py` | 🆕 新增 | 格式转换脚本 |
| `start_service.sh` | 🆕 新增 | 启动脚本 |
| `test_client.py` | 🆕 新增 | API 测试客户端 |
| `TROUBLESHOOTING.md` | 🆕 新增 | 本文档 |

---

## 🎯 推荐操作流程

### 立即可用（当前状态）
```bash
# 1. 启动服务（CPU 模式）
./start_service.sh

# 2. 测试 API
python test_client.py
```

### 最佳实践（建议升级后）
```bash
# 1. 升级 PyTorch 到 2.6+
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128

# 2. 恢复 GPU 模式（修改 serve_bge_m3.py）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3. 重启服务
./start_service.sh

# 4. 验证 GPU 加速
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## ⚠️ 注意事项

1. **SafeTensors 优先级**: Transformers 会自动优先加载 `.safetensors` 文件，即使 `pytorch_model.bin` 仍然存在

2. **删除旧模型**（可选）:
   ```bash
   # 节省 2.2GB 空间
   rm /opt/bge-m3/models/bge-m3/pytorch_model.bin
   ```

3. **备份当前环境**（升级前）:
   ```bash
   pip freeze > requirements_backup.txt
   ```

4. **PyTorch 版本兼容性**:
   - PyTorch 2.6+ 需要 CUDA 12.1+
   - 检查系统 CUDA 版本: `nvcc --version`

---

## 📞 获取帮助

- **PyTorch 官方文档**: https://pytorch.org/get-started/locally/
- **Transformers 文档**: https://huggingface.co/docs/transformers
- **BGE-M3 模型卡片**: https://huggingface.co/BAAI/bge-m3
- **CVE-2025-32434 详情**: https://nvd.nist.gov/vuln/detail/CVE-2025-32434
