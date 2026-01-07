# BGE-M3 快速启动指南

## ✅ 当前状态：可用（CPU 模式）

### 🚀 立即启动

```bash
cd /opt/bge-m3
./start_service.sh
```

服务将在 http://localhost:8001 启动

---

## 🧪 快速测试

```bash
# 方式1: 使用测试脚本
python test_client.py

# 方式2: 使用 curl
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["测试"], "normalize": true}'
```

---

## 📊 当前配置

| 项目 | 值 |
|------|-----|
| **运行模式** | CPU（临时） |
| **模型路径** | `/opt/bge-m3/models/bge-m3` |
| **模型格式** | SafeTensors（安全） |
| **PyTorch** | 2.3.0+cu121 |
| **端口** | 8001 |
| **向量维度** | 1024 |

---

## ⚡ 性能提升方案

### RTX 5090 用户必读！

**问题**: 当前 PyTorch 2.3.0 不支持 RTX 5090 GPU

**解决**: 升级到 PyTorch 2.6+

```bash
# 1. 激活环境
source .venv/bin/activate

# 2. 升级 PyTorch（访问官网获取最新命令）
# https://pytorch.org/get-started/locally/
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128

# 3. 修改 serve_bge_m3.py (第12-13行)
# 将：
#   DEVICE = "cpu"
# 改为：
#   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4. 重启服务
./start_service.sh
```

**性能对比**:
- CPU 模式: ~200-500ms/文本
- GPU 模式: ~10-30ms/文本 **(10-20倍提升)**

---

## 📖 API 使用示例

### Python 客户端

```python
import requests

response = requests.post(
    "http://localhost:8001/embed",
    json={
        "texts": ["文本1", "文本2", "文本3"],
        "normalize": True,      # 向量归一化
        "max_length": 512,      # 最大长度（支持到8192）
        "batch_size": 32        # 批处理大小
    }
)

embeddings = response.json()["embeddings"]
# embeddings: List[List[float]], 每个向量 1024 维
```

### 计算文本相似度

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)  # 已归一化，直接点积

# 获取嵌入
response = requests.post(..., json={
    "texts": ["机器学习", "深度学习"],
    "normalize": True
})

vecs = response.json()["embeddings"]
similarity = cosine_similarity(vecs[0], vecs[1])
print(f"相似度: {similarity:.4f}")  # 输出: 0.8523
```

---

## 📁 项目文件

```
/opt/bge-m3/
├── serve_bge_m3.py              # 主服务（已配置CPU模式）
├── start_service.sh             # 启动脚本 ⭐
├── test_client.py               # 测试客户端
├── convert_to_safetensors.py   # 格式转换工具（已执行）
├── QUICK_START.md              # 本文档
├── TROUBLESHOOTING.md          # 完整故障排查
└── models/bge-m3/
    ├── model.safetensors       # 安全模型格式（2.2GB）
    └── ...

```

---

## 🔧 常见问题

### Q1: 服务启动失败？
```bash
# 检查端口占用
lsof -i :8001

# 更换端口
uvicorn serve_bge_m3:app --port 8002
```

### Q2: API 返回 500 错误？
```bash
# 查看日志
tail -f /tmp/bge_service.log
```

### Q3: 如何停止服务？
```bash
pkill -f "uvicorn serve_bge_m3"
# 或按 Ctrl+C（前台运行时）
```

### Q4: 如何验证 GPU 是否工作？
```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
```

---

## 📚 更多资源

- **完整故障排查**: 查看 `TROUBLESHOOTING.md`
- **使用文档**: 查看 `USAGE.md`
- **API 文档**: 访问 http://localhost:8001/docs
- **模型详情**: https://huggingface.co/BAAI/bge-m3

---

**🎉 现在就开始使用吧！**

```bash
./start_service.sh
```
