# GPU 性能压测补齐与容量基线计划

## 1. 目标与背景
**目标**：补齐长文本、高并发场景下的性能指标，拆解 Tokenize 开销，确定生产环境的安全 Batch Size，并输出标准的容量基线。
**现状**：基础功能已验证，需进一步通过压测确定 RTX 5090 (或同级 GPU) 的性能边界。

## 2. 测试环境
- **硬件**: NVIDIA RTX 5090 (需 PyTorch 2.6+ 支持) / 替代测试 GPU
- **软件**: PyTorch (CUDA Mode), BGE-M3 Model (FP16/FP32)
- **工具**: `benchmark_gpu_comprehensive.py` (已集成全量测试用例)

## 3. 核心测试用例

### Case 1: 不同长度吞吐对比
**目的**: 评估显存带宽与计算能力的平衡点。
- **输入**: 固定长度文本 [256, 1024, 4096] tokens。
- **配置**: Batch Size = 32 (或根据显存调整)。
- **指标**: QPS (Queries Per Second), TPS (Tokens Per Second)。
- **预期**: 长文本 TPS 更高，但 QPS 下降；寻找 TPS 峰值点。

### Case 2: FP16 单卡安全 Batch (避免 OOM)
**目的**: 找到显存溢出边界，设定安全水位。
- **方法**: 二分查找 (Binary Search) 探测 OOM 临界值。
- **输入**: Max Length = [512, 1024, 2048, 4096, 8192]。
- **产出**: 
    - 极限 Batch Size ($B_{max}$)
    - **安全 Batch Size**: $B_{safe} = \lfloor B_{max} \times 0.85 \rfloor$

### Case 3: 混合长度 Tail Latency 观测
**目的**: 模拟真实线上流量，评估长尾延迟。
- **输入**: 混合分布 (例如 40% 短句 <128, 40% 中句 <512, 20% 长句 <2048)。
- **配置**: 持续压力测试 (Sustained Pressure)。
- **指标**: Avg Latency, **P95 Latency**, **P99 Latency**。
- **关注点**: 混合 batch 中，长文本是否会导致短文本的延迟显著增加 (Head-of-line blocking)。

### Case 4: Tokenize 占比拆解
**目的**: 识别 CPU 预处理瓶颈。
- **方法**: 记录总耗时 ($T_{total}$), Tokenize 耗时 ($T_{tok}$), Model Forward 耗时 ($T_{model}$)。
- **指标**: $Ratio_{tok} = T_{tok} / T_{total}$。
- **预期**: 短文本场景下 Tokenize 占比可能 > 30%，需考虑 CPU 优化或 C++ Tokenizer。

## 4. 执行计划

### 第一阶段：基准数据采集
执行全量基准测试脚本：
```bash
python benchmark_gpu_comprehensive.py
```
*注：该脚本已包含上述所有测试用例。*

### 第二阶段：数据分析与基线定义
根据 `benchmark_report.json` 输出，填写下表：

| 场景 | Max Length | 推荐 Batch | 极限 QPS | P99 延迟 (ms) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **短文本搜索** | 256 | $B_{safe\_256}$ | - | - | 典型搜索 Query |
| **通用场景** | 1024 | $B_{safe\_1024}$ | - | - | 标注文档/段落 |
| **长文档处理** | 4096 | $B_{safe\_4096}$ | - | - | 知识库构建 |

### 第三阶段：容量基线产出
定义 **单卡容量基线 (Capacity Baseline)**：
- **延迟敏感型 (Online)**: QPS @ P99 < 100ms
- **吞吐优先型 (Offline)**: QPS @ GPU Util > 90%

## 5. 风险预案
- **OOM**: 生产环境配置严格限制 `max_batch_size` 和 `max_length`。
- **超时**: 针对超长文本请求 (8192 tokens)，建议使用异步队列处理，避免阻塞实时请求。
- **PyTorch 版本**: 确保 PyTorch 版本 >= 2.6 以发挥 RTX 5090 性能，否则降级使用 CPU 多线程方案 (性能约为 GPU 的 1-5%)。
