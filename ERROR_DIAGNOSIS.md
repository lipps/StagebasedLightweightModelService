# Call Analysis API 向量检索错误诊断

## 🔴 错误症状

### 错误日志摘要
```
ERROR | src.engines.vector_engine:448 | 向量检索失败:
Error executing plan: Internal error: Error finding id
```

**出现位置：**
- 破冰分析阶段（5次）
- 功能演绎分析阶段（5次）

**影响范围：**
- 破冰要点检测仍完成（命中3个）
- 功能演绎要点检测仍完成（命中3个）
- 整体流程成功完成（置信度 0.452）

---

## ✅ 已验证正常的组件

### 1. BGE-M3 嵌入服务
**状态**: ✅ **正常运行**

```bash
# 验证命令
ps aux | grep uvicorn
# 结果: PID 3676359 正在运行

# 测试 API
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["测试"], "normalize": true}'
# 结果: 成功返回 1024 维向量
```

**结论**: BGE-M3 服务运行正常，能够生成嵌入向量。

### 2. 文本预处理模块
**状态**: ✅ **正常工作**

```
INFO | src.processors.text_processor:361 | 内容分析: 总对话154, 销售82, 客户72, 未知0
INFO | src.processors.text_processor:500 | 时长计算: 共154个时间戳, 估算时长12.2分钟
```

---

## 🔍 问题根因分析

### 根本原因：向量数据库索引问题

**错误消息解读：**
```
Error executing plan: Internal error: Error finding id
```

这个错误表明：
1. 向量数据库执行查询计划时失败
2. 尝试查找某个 ID 时找不到对应的记录
3. 可能是索引损坏或数据不一致

### 可能的具体原因

#### 原因 1: 向量数据库中缺失索引数据
**症状：**
- 破冰/功能演绎的知识库向量已生成
- 但向量数据库中对应的 ID 不存在或损坏

**验证方法：**
```python
# 检查向量数据库集合
from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")
collection = Collection("your_collection_name")

# 查看集合统计
print(f"实体数量: {collection.num_entities}")
print(f"索引信息: {collection.index()}")
```

#### 原因 2: 向量数据库服务异常
**可能情况：**
- Milvus/Qdrant/Chroma 服务未启动
- 连接超时或网络问题
- 内存不足导致查询失败

**验证方法：**
```bash
# 检查 Milvus 服务
docker ps | grep milvus

# 检查日志
docker logs milvus-standalone 2>&1 | tail -50

# 检查资源使用
docker stats milvus-standalone --no-stream
```

#### 原因 3: ID 映射不一致
**可能情况：**
- 应用代码期望的 ID 格式与数据库中不匹配
- 数据插入时使用的 ID 生成策略不一致
- 部分数据被删除但索引未更新

---

## 🛠️ 修复方案

### 方案 A: 检查并修复向量数据库（推荐）

#### 步骤 1: 定位向量数据库配置

查找项目中的向量数据库配置文件：
```bash
# 查找配置文件
find /path/to/call-analysis-api -name "*.yaml" -o -name "*.yml" -o -name ".env" | xargs grep -l "vector\|milvus\|chroma"

# 查找向量引擎代码
find /path/to/call-analysis-api -name "vector_engine.py"
```

#### 步骤 2: 检查向量数据库服务状态

```bash
# 检查容器状态
docker ps -a | grep -E "(milvus|chroma|qdrant)"

# 重启向量数据库服务
docker restart <vector-db-container>

# 查看服务日志
docker logs -f <vector-db-container>
```

#### 步骤 3: 重建向量索引

如果发现数据损坏，重新构建索引：
```python
# 示例：重建 Milvus 索引
from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")
collection = Collection("knowledge_base")

# 删除旧索引
collection.drop_index()

# 重建索引
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()
```

#### 步骤 4: 重新导入知识库数据

```bash
# 假设有数据导入脚本
python scripts/import_knowledge_base.py \
  --embedding-service http://localhost:8001 \
  --vector-db milvus://localhost:19530
```

---

### 方案 B: 添加错误处理和降级机制

修改 `src/engines/vector_engine.py`（第448行附近）：

```python
def search(self, query_vector: List[float], top_k: int = 5):
    """向量检索"""
    try:
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k
        )
        return results
    except Exception as e:
        logger.error(f"向量检索失败: {e}")

        # 添加降级处理
        if "Error finding id" in str(e):
            logger.warning("ID查找失败，尝试重新加载集合")
            try:
                self.collection.load()  # 重新加载集合
                results = self.collection.search(...)  # 重试
                return results
            except Exception as retry_error:
                logger.error(f"重试失败: {retry_error}")
                return []  # 返回空结果

        return []  # 其他错误返回空结果
```

---

### 方案 C: 切换到基于文本的检索（临时方案）

如果向量检索持续失败，可以临时使用关键词匹配：

```python
# 在 src/processors/icebreak_processor.py 中添加降级逻辑
def detect_icebreak_points(self, dialogue: List[Dict]):
    """破冰要点检测"""

    # 尝试向量检索
    try:
        results = self.vector_engine.search(query_embedding, top_k=5)
    except Exception as e:
        logger.warning(f"向量检索失败，使用关键词匹配降级: {e}")

        # 降级到关键词匹配
        results = self._keyword_fallback_search(dialogue)

    return results

def _keyword_fallback_search(self, dialogue: List[Dict]):
    """基于关键词的降级检索"""
    keywords = ["破冰", "寒暄", "问候", "了解", "介绍"]
    matches = []

    for turn in dialogue:
        text = turn.get("text", "")
        if any(kw in text for kw in keywords):
            matches.append(turn)

    return matches
```

---

## 🔧 诊断命令清单

### 1. 检查 BGE-M3 服务
```bash
# 服务状态
ps aux | grep uvicorn

# 测试 API
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["测试文本"], "normalize": true}'

# 查看日志
tail -f /path/to/bge-m3/logs/service.log
```

### 2. 检查向量数据库
```bash
# Milvus 检查
docker ps | grep milvus
docker logs milvus-standalone 2>&1 | tail -50

# 连接测试
python -c "
from pymilvus import connections
connections.connect('default', host='localhost', port='19530')
print('连接成功')
"
```

### 3. 检查应用日志
```bash
# 查看完整日志
tail -200 /path/to/call-analysis-api/logs/app.log | grep -E "(ERROR|向量检索)"

# 查看向量引擎相关日志
grep "vector_engine" /path/to/call-analysis-api/logs/app.log | tail -50
```

### 4. 验证数据一致性
```python
# 检查向量数据库中的数据
from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")

# 列出所有集合
from pymilvus import utility
collections = utility.list_collections()
print(f"所有集合: {collections}")

# 检查具体集合
for coll_name in collections:
    coll = Collection(coll_name)
    print(f"{coll_name}: {coll.num_entities} 条记录")
```

---

## 📊 问题影响评估

### 当前影响

| 模块 | 状态 | 影响 |
|------|------|------|
| 文本预处理 | ✅ 正常 | 无影响 |
| 破冰分析 | ⚠️ 部分失败 | 仍能完成（命中3个） |
| 功能演绎 | ⚠️ 部分失败 | 仍能完成（命中3个） |
| 过程分析 | ✅ 正常 | 无影响 |
| 客户分析 | ✅ 正常 | 无影响 |
| 动作分析 | ✅ 正常 | 无影响 |
| 痛点量化 | ✅ 正常 | 无影响 |
| 深度分析 | ✅ 正常 | 无影响 |
| 通话总结 | ✅ 正常 | 无影响 |

**总体置信度**: 0.452（中等）

### 潜在风险

1. **准确性下降**: 向量检索失败可能导致知识库匹配不准确
2. **性能问题**: 如果持续重试失败的检索，可能增加延迟
3. **数据丢失**: 向量数据库损坏可能导致历史知识丢失

---

## 🚀 优先修复建议

### 优先级 1: 立即检查（高）
1. 检查向量数据库服务是否正常运行
2. 查看向量数据库日志确认具体错误
3. 验证 BGE-M3 服务与向量数据库的连接

### 优先级 2: 短期修复（中）
1. 重建受影响的向量索引
2. 添加详细的错误日志和监控
3. 实现降级处理机制

### 优先级 3: 长期优化（低）
1. 添加向量数据库健康检查
2. 实现自动索引修复
3. 添加性能监控和告警

---

## 📞 下一步行动

### 请提供以下信息以便深入诊断：

1. **项目路径**: Call Analysis API 的完整路径
   ```bash
   pwd  # 在项目根目录执行
   ```

2. **向量数据库类型**: 使用的是 Milvus、Chroma 还是其他？
   ```bash
   docker ps  # 查看运行的容器
   ```

3. **配置文件位置**: 向量数据库配置文件
   ```bash
   find . -name "*.yaml" -o -name ".env" | xargs grep -l vector
   ```

4. **向量引擎代码**:
   ```bash
   find . -name "vector_engine.py" -exec head -20 {} \;
   ```

---

**提供以上信息后，我可以为您提供更精准的修复方案！**
