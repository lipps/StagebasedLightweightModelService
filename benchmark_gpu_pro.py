#!/usr/bin/env python3
"""
BGE-M3 GPU 性能基准测试 - 增强版
覆盖场景：
  1. 不同长度文本批处理吞吐 (256/1024/4096 tokens)
  2. FP16 安全 batch size 探测 (OOM 边界)
  3. 混合长度场景 (短句+长句混批, tail latency)
  4. P95/P99 延迟指标
  5. Tokenize 时间占比拆解
"""

import gc
import time
import random
import statistics
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer, AutoModel

# ==================== 配置 ====================

MODEL_ID = "/opt/bge-m3/models/bge-m3"

# 目标 token 长度的文本生成模板
# 中文约 1.5 char/token, 英文约 4 char/token
TEXT_TEMPLATES = {
    "short": "这是一段简短的测试文本。",  # ~10 tokens
    "medium": "人工智能正在深刻改变着我们的生活方式，从智能手机到自动驾驶，从医疗诊断到金融分析。" * 3,  # ~50 tokens
    "long": ("大型语言模型是一种基于深度学习的自然语言处理技术，通过在海量文本数据上进行预训练，"
             "能够理解和生成人类语言，在问答、翻译、摘要、对话等任务上展现出强大的能力。") * 10,  # ~200 tokens
}


@dataclass
class BenchmarkResult:
    """单次测试结果"""
    name: str
    text_count: int
    batch_size: int
    max_length: int
    total_time_ms: float
    tokenize_time_ms: float
    encode_time_ms: float
    throughput: float  # texts/sec
    avg_latency_ms: float
    latencies: List[float] = field(default_factory=list)  # 每个 batch 的延迟
    memory_mb: float = 0.0
    peak_memory_mb: float = 0.0


@dataclass
class LatencyStats:
    """延迟统计"""
    min_ms: float
    max_ms: float
    avg_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float


def generate_texts_by_tokens(tokenizer, target_tokens: int, count: int) -> List[str]:
    """生成指定 token 长度的文本"""
    # 基础文本块
    base_text = ("这是一段用于性能测试的文本内容，包含了多种中文字符和标点符号。"
                 "人工智能、机器学习、深度学习、自然语言处理、计算机视觉等技术正在快速发展。")

    # 估算每个字符的平均 token 数
    sample_tokens = len(tokenizer.encode(base_text, add_special_tokens=False))
    chars_per_token = len(base_text) / sample_tokens

    # 生成目标长度的文本
    target_chars = int(target_tokens * chars_per_token * 1.1)  # 稍微多一点确保够长

    texts = []
    for i in range(count):
        # 重复基础文本直到达到目标长度
        repeated = (base_text + f"[{i}]") * (target_chars // len(base_text) + 1)
        text = repeated[:target_chars]
        texts.append(text)

    return texts


def generate_mixed_length_texts(tokenizer, count: int) -> Tuple[List[str], List[int]]:
    """生成混合长度的文本 (模拟真实场景)
    分布: 40% 短句(<128), 40% 中句(128-512), 20% 长句(512-2048)
    """
    texts = []
    lengths = []

    short_count = int(count * 0.4)
    medium_count = int(count * 0.4)
    long_count = count - short_count - medium_count

    # 短句
    short_texts = generate_texts_by_tokens(tokenizer, 64, short_count)
    texts.extend(short_texts)
    lengths.extend([64] * short_count)

    # 中句
    medium_texts = generate_texts_by_tokens(tokenizer, 256, medium_count)
    texts.extend(medium_texts)
    lengths.extend([256] * medium_count)

    # 长句
    long_texts = generate_texts_by_tokens(tokenizer, 1024, long_count)
    texts.extend(long_texts)
    lengths.extend([1024] * long_count)

    # 打乱顺序
    combined = list(zip(texts, lengths))
    random.shuffle(combined)
    texts, lengths = zip(*combined)

    return list(texts), list(lengths)


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean Pooling"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def compute_latency_stats(latencies: List[float]) -> LatencyStats:
    """计算延迟统计指标"""
    if not latencies:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0)

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    def percentile(p: float) -> float:
        idx = int(n * p / 100)
        return sorted_latencies[min(idx, n - 1)]

    return LatencyStats(
        min_ms=min(latencies),
        max_ms=max(latencies),
        avg_ms=statistics.mean(latencies),
        p50_ms=percentile(50),
        p90_ms=percentile(90),
        p95_ms=percentile(95),
        p99_ms=percentile(99),
        std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0
    )


@torch.inference_mode()
def benchmark_with_breakdown(
    tokenizer,
    model,
    texts: List[str],
    device: str,
    dtype: torch.dtype,
    max_length: int,
    batch_size: int,
    name: str = "test"
) -> BenchmarkResult:
    """带时间拆解的基准测试"""

    all_vecs = []
    batch_latencies = []
    total_tokenize_time = 0.0
    total_encode_time = 0.0

    # 重置显存统计
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_total = time.perf_counter()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_start = time.perf_counter()

        # ========== Tokenize 阶段 ==========
        tokenize_start = time.perf_counter()
        inputs = tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if device == "cuda":
            torch.cuda.synchronize()
        tokenize_end = time.perf_counter()
        total_tokenize_time += (tokenize_end - tokenize_start) * 1000

        # ========== Encode 阶段 ==========
        encode_start = time.perf_counter()
        out = model(**inputs, return_dict=True)
        vecs = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        if device == "cuda":
            torch.cuda.synchronize()
        encode_end = time.perf_counter()
        total_encode_time += (encode_end - encode_start) * 1000

        all_vecs.extend(vecs.detach().float().cpu().tolist())

        batch_end = time.perf_counter()
        batch_latencies.append((batch_end - batch_start) * 1000)

    if device == "cuda":
        torch.cuda.synchronize()

    end_total = time.perf_counter()
    total_time_ms = (end_total - start_total) * 1000

    # 显存统计
    memory_mb = 0.0
    peak_memory_mb = 0.0
    if device == "cuda":
        memory_mb = torch.cuda.memory_allocated() / 1024**2
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    count = len(all_vecs)
    throughput = count / (total_time_ms / 1000) if total_time_ms > 0 else 0

    return BenchmarkResult(
        name=name,
        text_count=count,
        batch_size=batch_size,
        max_length=max_length,
        total_time_ms=total_time_ms,
        tokenize_time_ms=total_tokenize_time,
        encode_time_ms=total_encode_time,
        throughput=throughput,
        avg_latency_ms=total_time_ms / count if count > 0 else 0,
        latencies=batch_latencies,
        memory_mb=memory_mb,
        peak_memory_mb=peak_memory_mb
    )


def find_safe_batch_size(
    tokenizer,
    model,
    device: str,
    dtype: torch.dtype,
    max_length: int,
    start_batch: int = 1,
    max_batch: int = 512
) -> Tuple[int, float]:
    """二分查找安全的最大 batch size (避免 OOM)

    返回: (safe_batch_size, peak_memory_mb)
    """
    if device != "cuda":
        return max_batch, 0.0

    print(f"\n  🔍 探测 max_length={max_length} 的安全 batch size...")

    # 生成测试文本
    test_texts = generate_texts_by_tokens(tokenizer, max_length, max_batch)

    safe_batch = start_batch
    safe_memory = 0.0

    # 二分查找
    low, high = start_batch, max_batch

    while low <= high:
        mid = (low + high) // 2

        try:
            # 清理显存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 尝试运行
            batch_texts = test_texts[:mid]
            inputs = tokenizer(
                batch_texts,
                padding="longest",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                out = model(**inputs, return_dict=True)
                vecs = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
                vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
                _ = vecs.cpu()

            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2

            # 成功，记录并尝试更大的 batch
            safe_batch = mid
            safe_memory = peak_mem
            print(f"    ✅ batch={mid} 成功，峰值显存 {peak_mem:.1f}MB")
            low = mid + 1

        except torch.cuda.OutOfMemoryError:
            print(f"    ❌ batch={mid} OOM")
            high = mid - 1
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    ⚠️  batch={mid} 错误: {e}")
            high = mid - 1

    # 安全边界：降低 10%
    safe_batch = int(safe_batch * 0.9)

    return safe_batch, safe_memory


def print_result_table(results: List[BenchmarkResult], title: str):
    """打印结果表格"""
    print(f"\n{'='*100}")
    print(f"  {title}")
    print('='*100)

    # 表头
    print(f"\n{'场景':<20} {'数量':>8} {'Batch':>8} {'MaxLen':>8} "
          f"{'吞吐(/s)':>12} {'延迟(ms)':>12} {'显存(MB)':>12}")
    print("-" * 100)

    for r in results:
        print(f"{r.name:<20} {r.text_count:>8} {r.batch_size:>8} {r.max_length:>8} "
              f"{r.throughput:>12.2f} {r.avg_latency_ms:>12.2f} {r.peak_memory_mb:>12.1f}")


def print_latency_breakdown(results: List[BenchmarkResult]):
    """打印延迟拆解"""
    print(f"\n{'='*100}")
    print("  时间拆解分析 (Tokenize vs Encode)")
    print('='*100)

    print(f"\n{'场景':<20} {'总时间(ms)':>12} {'Tokenize(ms)':>14} {'Encode(ms)':>12} "
          f"{'Tokenize%':>10} {'Encode%':>10}")
    print("-" * 100)

    for r in results:
        tok_pct = (r.tokenize_time_ms / r.total_time_ms * 100) if r.total_time_ms > 0 else 0
        enc_pct = (r.encode_time_ms / r.total_time_ms * 100) if r.total_time_ms > 0 else 0

        print(f"{r.name:<20} {r.total_time_ms:>12.2f} {r.tokenize_time_ms:>14.2f} "
              f"{r.encode_time_ms:>12.2f} {tok_pct:>9.1f}% {enc_pct:>9.1f}%")


def print_percentile_stats(results: List[BenchmarkResult]):
    """打印 P50/P90/P95/P99 延迟"""
    print(f"\n{'='*100}")
    print("  批次延迟分位数 (P50/P90/P95/P99)")
    print('='*100)

    print(f"\n{'场景':<20} {'Min(ms)':>10} {'P50(ms)':>10} {'P90(ms)':>10} "
          f"{'P95(ms)':>10} {'P99(ms)':>10} {'Max(ms)':>10} {'StdDev':>10}")
    print("-" * 100)

    for r in results:
        stats = compute_latency_stats(r.latencies)
        print(f"{r.name:<20} {stats.min_ms:>10.2f} {stats.p50_ms:>10.2f} "
              f"{stats.p90_ms:>10.2f} {stats.p95_ms:>10.2f} {stats.p99_ms:>10.2f} "
              f"{stats.max_ms:>10.2f} {stats.std_ms:>10.2f}")


def main():
    """主函数"""
    print("="*100)
    print("  BGE-M3 GPU 性能基准测试 - 增强版")
    print("="*100)

    # 检查 CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA 可用: {'✅ 是' if cuda_available else '❌ 否'}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存总量: {gpu_mem:.1f} GB")

    device = "cuda" if cuda_available else "cpu"
    dtype = torch.float16 if cuda_available else torch.float32

    # 加载模型
    print(f"\n加载模型 ({device.upper()}, {'FP16' if dtype == torch.float16 else 'FP32'})...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype, local_files_only=True)
    model.to(device)
    model.eval()
    print(f"  ✅ 模型加载完成 ({time.time() - start:.2f}s)")

    # 预热
    print("\n模型预热...")
    _ = benchmark_with_breakdown(
        tokenizer, model, ["预热测试"] * 8, device, dtype, 128, 8, "warmup"
    )
    print("  ✅ 预热完成")

    all_results = []

    # ==================== 测试 1: 不同长度文本吞吐 ====================
    print("\n" + "="*100)
    print("  测试 1: 不同长度文本批处理吞吐 (256/1024/4096 tokens)")
    print("="*100)

    length_configs = [
        {"target_tokens": 256, "count": 200, "batch_size": 64, "max_length": 512},
        {"target_tokens": 1024, "count": 100, "batch_size": 32, "max_length": 1536},
        {"target_tokens": 4096, "count": 50, "batch_size": 8, "max_length": 4608},
    ]

    length_results = []
    for cfg in length_configs:
        print(f"\n  📊 测试 ~{cfg['target_tokens']} tokens 文本...")
        texts = generate_texts_by_tokens(tokenizer, cfg["target_tokens"], cfg["count"])

        # 验证实际 token 长度
        sample_len = len(tokenizer.encode(texts[0], add_special_tokens=True))
        print(f"     实际样本长度: ~{sample_len} tokens")

        result = benchmark_with_breakdown(
            tokenizer, model, texts, device, dtype,
            cfg["max_length"], cfg["batch_size"],
            f"~{cfg['target_tokens']}tok"
        )
        length_results.append(result)
        print(f"     吞吐: {result.throughput:.2f}/s, 延迟: {result.avg_latency_ms:.2f}ms")

    all_results.extend(length_results)
    print_result_table(length_results, "不同长度文本吞吐对比")

    # ==================== 测试 2: 安全 Batch Size 探测 ====================
    if cuda_available:
        print("\n" + "="*100)
        print("  测试 2: FP16 安全 Batch Size 探测 (OOM 边界)")
        print("="*100)

        safe_batch_results = {}
        for max_len in [512, 1024, 2048, 4096]:
            safe_batch, peak_mem = find_safe_batch_size(
                tokenizer, model, device, dtype, max_len,
                start_batch=1, max_batch=256
            )
            safe_batch_results[max_len] = (safe_batch, peak_mem)

        print(f"\n  📋 安全 Batch Size 推荐 (含 10% 安全边界):")
        print(f"\n  {'MaxLength':>12} │ {'SafeBatch':>12} │ {'PeakMem(MB)':>14}")
        print(f"  {'-'*12}─┼─{'-'*12}─┼─{'-'*14}")
        for max_len, (batch, mem) in safe_batch_results.items():
            print(f"  {max_len:>12} │ {batch:>12} │ {mem:>14.1f}")

    # ==================== 测试 3: 混合长度场景 ====================
    print("\n" + "="*100)
    print("  测试 3: 混合长度场景 (真实分布: 40%短+40%中+20%长)")
    print("="*100)

    mixed_texts, mixed_lengths = generate_mixed_length_texts(tokenizer, 200)
    print(f"\n  文本分布: 短(~64tok): {mixed_lengths.count(64)}, "
          f"中(~256tok): {mixed_lengths.count(256)}, 长(~1024tok): {mixed_lengths.count(1024)}")

    # 测试不同 batch size 下的表现
    mixed_results = []
    for batch_size in [16, 32, 64]:
        result = benchmark_with_breakdown(
            tokenizer, model, mixed_texts, device, dtype,
            1536, batch_size, f"混合-B{batch_size}"
        )
        mixed_results.append(result)

    all_results.extend(mixed_results)
    print_result_table(mixed_results, "混合长度场景对比")

    # ==================== 测试 4: 高并发压力测试 (P95/P99) ====================
    print("\n" + "="*100)
    print("  测试 4: 高并发压力测试 (多轮迭代, 收集 P95/P99)")
    print("="*100)

    # 用中等长度文本进行多轮测试
    stress_texts = generate_texts_by_tokens(tokenizer, 256, 100)
    stress_results = []

    print("\n  运行 10 轮压力测试...")
    all_batch_latencies = []
    for round_idx in range(10):
        result = benchmark_with_breakdown(
            tokenizer, model, stress_texts, device, dtype,
            512, 32, f"压力测试-R{round_idx+1}"
        )
        stress_results.append(result)
        all_batch_latencies.extend(result.latencies)
        print(f"    轮次 {round_idx+1}: 吞吐 {result.throughput:.2f}/s")

    # 汇总统计
    combined_result = BenchmarkResult(
        name="压力测试-汇总",
        text_count=sum(r.text_count for r in stress_results),
        batch_size=32,
        max_length=512,
        total_time_ms=sum(r.total_time_ms for r in stress_results),
        tokenize_time_ms=sum(r.tokenize_time_ms for r in stress_results),
        encode_time_ms=sum(r.encode_time_ms for r in stress_results),
        throughput=sum(r.text_count for r in stress_results) / (sum(r.total_time_ms for r in stress_results) / 1000),
        avg_latency_ms=sum(r.total_time_ms for r in stress_results) / sum(r.text_count for r in stress_results),
        latencies=all_batch_latencies,
        memory_mb=stress_results[-1].memory_mb,
        peak_memory_mb=max(r.peak_memory_mb for r in stress_results)
    )

    print_percentile_stats([combined_result])

    # ==================== 汇总报告 ====================
    print("\n" + "="*100)
    print("  📊 汇总报告")
    print("="*100)

    print_result_table(all_results, "全部测试结果")
    print_latency_breakdown(all_results)
    print_percentile_stats(all_results)

    # 推荐配置
    print("\n" + "="*100)
    print("  💡 生产环境推荐配置")
    print("="*100)

    if cuda_available:
        print(f"""
  基于测试结果，推荐以下配置：

  ┌─────────────────────────────────────────────────────────────────┐
  │ 场景             │ MaxLength │ BatchSize │ 预期吞吐    │ 显存   │
  ├─────────────────────────────────────────────────────────────────┤
  │ 短文本高吞吐     │ 512       │ 64        │ ~{length_results[0].throughput:.0f}/s      │ <4GB   │
  │ 通用场景         │ 1024      │ 32        │ ~{length_results[1].throughput:.0f}/s      │ <8GB   │
  │ 长文本处理       │ 4096      │ 8         │ ~{length_results[2].throughput:.0f}/s       │ <16GB  │
  │ 混合真实场景     │ 1536      │ 32        │ ~{mixed_results[1].throughput:.0f}/s      │ <10GB  │
  └─────────────────────────────────────────────────────────────────┘

  ⚠️  注意事项：
  1. 生产环境建议 batch_size 设为测试安全值的 80%
  2. 混合长度场景使用 padding="longest" 可避免不必要的计算
  3. P99 延迟是 SLA 保障的关键指标
  4. 监控显存使用，预留 20% 缓冲应对峰值
""")
    else:
        print("\n  ⚠️  当前为 CPU 模式，建议启用 GPU 以获得更好性能")

    # 清理
    del model, tokenizer
    if cuda_available:
        torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "="*100)
    print("  ✅ 测试完成！")
    print("="*100)


if __name__ == "__main__":
    main()
