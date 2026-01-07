# BGE-M3 本地服务使用指南

## ✅ 完成的修改

### 1. 模型路径配置
- **修改文件**: `serve_bge_m3.py`
- **变更内容**:
  - `MODEL_ID`: `"BAAI/bge-m3"` → `"/opt/bge-m3/models/bge-m3"`
  - 添加 `local_files_only=True` 参数，禁止网络下载

### 2. 本地模型验证
- ✅ PyTorch 权重: `pytorch_model.bin` (2.2GB)
- ✅ 配置文件: `config.json`
- ✅ 分词器: `tokenizer.json`, `sentencepiece.bpe.model`
- ✅ 所有必需文件齐全

---

## 🚀 启动服务

### 方式一: 使用启动脚本（推荐）
```bash
cd /opt/bge-m3
./start_service.sh
```

### 方式二: 手动命令
```bash
cd /opt/bge-m3
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 uvicorn serve_bge_m3:app --host 0.0.0.0 --port 8001 --workers 1
```

### 参数说明
- `CUDA_VISIBLE_DEVICES=0`: 使用 GPU 0
- `--host 0.0.0.0`: 监听所有网络接口
- `--port 8001`: 服务端口
- `--workers 1`: 单进程（推荐，避免模型重复加载）

---

## 🧪 测试服务

### 方式一: 使用测试脚本
```bash
# 启动服务后，在新终端运行：
cd /opt/bge-m3
python test_client.py
```

### 方式二: 使用 curl
```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["什么是机器学习?", "Machine learning is AI"],
    "output_type": "dense",
    "normalize": true,
    "max_length": 512
  }'
```

### 方式三: Python 代码
```python
import requests

response = requests.post(
    "http://localhost:8001/embed",
    json={
        "texts": ["测试文本1", "测试文本2"],
        "normalize": True
    }
)

embeddings = response.json()["embeddings"]
print(f"向量维度: {len(embeddings[0])}")  # 应输出 1024
```

---

## 📊 API 参考

### POST /embed

**请求体**:
```json
{
  "texts": ["文本1", "文本2"],      // 必需: 待嵌入的文本列表
  "output_type": "dense",           // 默认: "dense" (当前仅支持 dense)
  "normalize": true,                // 默认: true (向量归一化)
  "max_length": 512,                // 默认: 512 (最大支持 8192)
  "batch_size": 32                  // 默认: 32 (批处理大小)
}
```

**响应体**:
```json
{
  "embeddings": [
    [0.123, -0.456, ...],  // 1024 维向量
    [0.789, 0.234, ...]
  ]
}
```

---

## 🔧 性能优化建议

### 1. GPU 使用
- 默认启用 CUDA（如果可用）
- 自动使用 FP16 精度（节省显存 50%）
- CPU 模式自动降级为 FP32

### 2. 批处理配置
```python
# 小批量、低延迟
{"batch_size": 8, "max_length": 256}

# 大批量、高吞吐
{"batch_size": 64, "max_length": 512}

# 长文档处理
{"batch_size": 4, "max_length": 8192}
```

### 3. 多 GPU 部署
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 uvicorn serve_bge_m3:app --port 8001 --workers 1

# GPU 1
CUDA_VISIBLE_DEVICES=1 uvicorn serve_bge_m3:app --port 8002 --workers 1
```

---

## 🐛 故障排查

### 问题1: 模型加载失败
```
FileNotFoundError: /opt/bge-m3/models/bge-m3/config.json
```
**解决**: 检查模型文件是否存在
```bash
ls -lh /opt/bge-m3/models/bge-m3/
```

### 问题2: CUDA 内存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 减小 batch_size 或 max_length

### 问题3: 端口被占用
```
OSError: [Errno 98] Address already in use
```
**解决**: 更换端口或杀死占用进程
```bash
lsof -i :8001
kill -9 <PID>
```

---

## 📝 文件清单

- ✅ `serve_bge_m3.py` - 主服务文件（已修改）
- ✅ `start_service.sh` - 启动脚本（新增）
- ✅ `test_client.py` - 测试客户端（新增）
- ✅ `USAGE.md` - 本文档（新增）
- ✅ `models/bge-m3/` - 本地模型目录

---

## 🔗 相关资源

- 模型仓库: https://huggingface.co/BAAI/bge-m3
- FastAPI 文档: http://localhost:8001/docs (服务启动后访问)
- 模型论文: BGE M3-Embedding (Multi-lingual, Multi-functional, Multi-granularity)
