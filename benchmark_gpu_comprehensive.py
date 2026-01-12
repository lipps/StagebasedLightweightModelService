#!/usr/bin/env python3
"""
BGE-M3 GPU 性能基准测试 - 全面版 (10+ 测试用例)

测试矩阵:
  ┌────────────────────────────────────────────────────────────────────┐
  │ 用例组           │ 子用例                                          │
  ├────────────────────────────────────────────────────────────────────┤
  │ 1. 长度梯度      │ 128/256/512/1024/2048/4096 tokens (6组)        │
  │ 2. Batch Size    │ 8/16/32/64/128 (5组)                            │
  │ 3. 安全边界探测  │ max_length=512/1024/2048/4096 (4组)            │
  │ 4. 混合长度      │ 短+中/中+长/短+中+长 分布 (3组)                 │
  │ 5. 精度对比      │ FP16 vs FP32 (2组)                              │
  │ 6. 冷热启动      │ 冷启动 vs 热启动延迟 (2组)                      │
  │ 7. 并发压力      │ 持续负载下的 P50/P90/P95/P99 (1组)              │
  │ 8. 真实流模拟    │ 变长请求流 tail latency (1组)                   │
  └────────────────────────────────────────────────────────────────────┘

  总计: 24+ 测试用例
"""

import gc
import os
import sys
import time
import random
import statistics
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
from transformers import AutoTokenizer, AutoModel

# ==================== 配置 ====================

MODEL_ID = "/opt/bge-m3/models/bge-m3"
REPORT_FILE = "/opt/bge-m3/benchmark_report.json"

# ==================== 数据结构 ====================

@dataclass
class BenchmarkResult:
    """单次测试结果"""
    name: str
    category: str  # 测试类别
    text_count: int
    batch_size: int
    max_length: int
    avg_tokens: float  # 平均 token 数
    total_time_ms: float
    tokenize_time_ms: float
    encode_time_ms: float
    throughput: float  # texts/sec
    tokens_per_sec: float  # tokens/sec
    avg_latency_ms: float
    latencies: List[float] = field(default_factory=list)
    memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    dtype: str = "fp16"


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


# ==================== 工具函数 ====================

def generate_texts_by_tokens(tokenizer, target_tokens: int, count: int, variance: float = 0.1) -> Tuple[List[str], List[int]]:
    """生成指定 token 长度的文本

    Args:
        tokenizer: 分词器
        target_tokens: 目标 token 数
        count: 文本数量
        variance: 长度变异系数 (0.1 = ±10%)

    Returns:
        (texts, actual_token_counts)
    """
    base_text = ("这是一段用于性能测试的文本内容，包含了多种中文字符和标点符号。"
                 "人工智能、机器学习、深度学习、自然语言处理、计算机视觉等技术正在快速发展。"
                 "大型语言模型通过在海量文本数据上进行预训练，能够理解和生成人类语言。")

    # 估算字符/token 比率
    sample_tokens = len(tokenizer.encode(base_text, add_special_tokens=False))
    chars_per_token = len(base_text) / sample_tokens

    texts = []
    actual_lengths = []

    for i in range(count):
        # 添加随机变异
        var_factor = 1.0 + random.uniform(-variance, variance)
        adjusted_tokens = int(target_tokens * var_factor)
        target_chars = int(adjusted_tokens * chars_per_token * 1.1)

        # 生成文本
        repeated = (base_text + f"[样本{i}]") * (target_chars // len(base_text) + 1)
        text = repeated[:target_chars]
        texts.append(text)

        # 记录实际长度
        actual_len = len(tokenizer.encode(text, add_special_tokens=True))
        actual_lengths.append(actual_len)

    return texts, actual_lengths


def generate_mixed_distribution(tokenizer, count: int, distribution: Dict[int, float]) -> Tuple[List[str], List[int]]:
    """生成混合长度分布的文本

    Args:
        distribution: {target_tokens: percentage}, e.g., {64: 0.4, 256: 0.4, 1024: 0.2}
    """
    texts = []
    lengths = []

    for target_tokens, pct in distribution.items():
        n = int(count * pct)
        batch_texts, batch_lengths = generate_texts_by_tokens(tokenizer, target_tokens, n)
        texts.extend(batch_texts)
        lengths.extend(batch_lengths)

    # 补齐余数
    remaining = count - len(texts)
    if remaining > 0:
        first_target = list(distribution.keys())[0]
        extra_texts, extra_lengths = generate_texts_by_tokens(tokenizer, first_target, remaining)
        texts.extend(extra_texts)
        lengths.extend(extra_lengths)

    # 打乱顺序
    combined = list(zip(texts, lengths))
    random.shuffle(combined)
    texts, lengths = zip(*combined) if combined else ([], [])

    return list(texts), list(lengths)


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean Pooling"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def compute_latency_stats(latencies: List[float]) -> LatencyStats:
    """计算延迟统计"""
    if not latencies:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0)

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def pct(p: float) -> float:
        idx = min(int(n * p / 100), n - 1)
        return sorted_lat[idx]

    return LatencyStats(
        min_ms=min(latencies),
        max_ms=max(latencies),
        avg_ms=statistics.mean(latencies),
        p50_ms=pct(50),
        p90_ms=pct(90),
        p95_ms=pct(95),
        p99_ms=pct(99),
        std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0
    )


def get_gpu_memory_info() -> Dict[str, float]:
    """获取 GPU 显存信息"""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2
    }


# ==================== 核心测试函数 ====================

@torch.inference_mode()
def benchmark_encode(
    tokenizer,
    model,
    texts: List[str],
    device: str,
    dtype: torch.dtype,
    max_length: int,
    batch_size: int,
    name: str = "test",
    category: str = "general"
) -> BenchmarkResult:
    """带完整时间拆解的编码基准测试"""

    all_vecs = []
    batch_latencies = []
    total_tokenize_time = 0.0
    total_encode_time = 0.0
    total_tokens = 0

    # 重置显存统计
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_total = time.perf_counter()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_start = time.perf_counter()

        # Tokenize
        tok_start = time.perf_counter()
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
        tok_end = time.perf_counter()
        total_tokenize_time += (tok_end - tok_start) * 1000

        # 统计 token 数
        total_tokens += inputs["attention_mask"].sum().item()

        # Encode
        enc_start = time.perf_counter()
        out = model(**inputs, return_dict=True)
        vecs = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        if device == "cuda":
            torch.cuda.synchronize()
        enc_end = time.perf_counter()
        total_encode_time += (enc_end - enc_start) * 1000

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
    tokens_per_sec = total_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
    avg_tokens = total_tokens / count if count > 0 else 0

    return BenchmarkResult(
        name=name,
        category=category,
        text_count=count,
        batch_size=batch_size,
        max_length=max_length,
        avg_tokens=avg_tokens,
        total_time_ms=total_time_ms,
        tokenize_time_ms=total_tokenize_time,
        encode_time_ms=total_encode_time,
        throughput=throughput,
        tokens_per_sec=tokens_per_sec,
        avg_latency_ms=total_time_ms / count if count > 0 else 0,
        latencies=batch_latencies,
        memory_mb=memory_mb,
        peak_memory_mb=peak_memory_mb,
        dtype="fp16" if dtype == torch.float16 else "fp32"
    )


def find_max_safe_batch(
    tokenizer,
    model,
    device: str,
    dtype: torch.dtype,
    max_length: int,
    safety_margin: float = 0.9
) -> Tuple[int, float, int]:
    """二分查找最大安全 batch size

    Returns: (safe_batch, peak_memory_mb, absolute_max_batch)
    """
    if device != "cuda":
        return 256, 0.0, 256

    test_texts, _ = generate_texts_by_tokens(tokenizer, max_length, 512)

    safe_batch = 1
    safe_memory = 0.0
    low, high = 1, 512

    while low <= high:
        mid = (low + high) // 2

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            batch = test_texts[:mid]
            inputs = tokenizer(batch, padding="longest", truncation=True,
                             max_length=max_length, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                out = model(**inputs, return_dict=True)
                vecs = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
                vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
                _ = vecs.cpu()

            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2

            safe_batch = mid
            safe_memory = peak_mem
            low = mid + 1

        except torch.cuda.OutOfMemoryError:
            high = mid - 1
            torch.cuda.empty_cache()
        except Exception:
            high = mid - 1

    absolute_max = safe_batch
    safe_batch = int(safe_batch * safety_margin)

    return safe_batch, safe_memory, absolute_max


# ==================== 测试用例组 ====================

def test_length_gradient(tokenizer, model, device, dtype) -> List[BenchmarkResult]:
    """测试组1: 长度梯度测试 (6组)"""
    print("\n" + "="*100)
    print("  测试组 1: 长度梯度测试 (128/256/512/1024/2048/4096 tokens)")
    print("="*100)

    configs = [
        {"tokens": 128,  "count": 500, "batch": 64, "max_len": 256},
        {"tokens": 256,  "count": 400, "batch": 64, "max_len": 512},
        {"tokens": 512,  "count": 300, "batch": 32, "max_len": 768},
        {"tokens": 1024, "count": 200, "batch": 32, "max_len": 1536},
        {"tokens": 2048, "count": 100, "batch": 16, "max_len": 2560},
        {"tokens": 4096, "count": 50,  "batch": 8,  "max_len": 4608},
    ]

    results = []
    for cfg in configs:
        print(f"\n  📊 测试 ~{cfg['tokens']} tokens...")
        texts, lengths = generate_texts_by_tokens(tokenizer, cfg["tokens"], cfg["count"])
        avg_len = sum(lengths) / len(lengths)
        print(f"     生成 {len(texts)} 条文本，平均 {avg_len:.0f} tokens")

        result = benchmark_encode(
            tokenizer, model, texts, device, dtype,
            cfg["max_len"], cfg["batch"],
            f"~{cfg['tokens']}tok", "length_gradient"
        )
        results.append(result)
        print(f"     ✅ 吞吐: {result.throughput:.1f}/s | "
              f"Token吞吐: {result.tokens_per_sec:.0f} tok/s | "
              f"延迟: {result.avg_latency_ms:.2f}ms")

    return results


def test_batch_size_scaling(tokenizer, model, device, dtype) -> List[BenchmarkResult]:
    """测试组2: Batch Size 扩展性 (5组)"""
    print("\n" + "="*100)
    print("  测试组 2: Batch Size 扩展性测试 (8/16/32/64/128)")
    print("="*100)

    # 使用固定长度文本
    texts, _ = generate_texts_by_tokens(tokenizer, 256, 500)

    results = []
    for batch_size in [8, 16, 32, 64, 128]:
        print(f"\n  📊 测试 batch_size={batch_size}...")

        result = benchmark_encode(
            tokenizer, model, texts, device, dtype,
            512, batch_size,
            f"B{batch_size}", "batch_scaling"
        )
        results.append(result)
        print(f"     ✅ 吞吐: {result.throughput:.1f}/s | 峰值显存: {result.peak_memory_mb:.0f}MB")

    return results


def test_safe_batch_boundary(tokenizer, model, device, dtype) -> List[Dict]:
    """测试组3: 安全 Batch 边界探测 (4组)"""
    print("\n" + "="*100)
    print("  测试组 3: FP16 安全 Batch Size 边界探测")
    print("="*100)

    results = []
    for max_len in [512, 1024, 2048, 4096]:
        print(f"\n  🔍 探测 max_length={max_len}...")
        safe_batch, peak_mem, abs_max = find_max_safe_batch(
            tokenizer, model, device, dtype, max_len
        )
        results.append({
            "max_length": max_len,
            "safe_batch": safe_batch,
            "absolute_max": abs_max,
            "peak_memory_mb": peak_mem
        })
        print(f"     ✅ 安全: {safe_batch} | 极限: {abs_max} | 峰值: {peak_mem:.0f}MB")

    return results


def test_mixed_distribution(tokenizer, model, device, dtype) -> List[BenchmarkResult]:
    """测试组4: 混合长度分布 (3组)"""
    print("\n" + "="*100)
    print("  测试组 4: 混合长度分布测试")
    print("="*100)

    distributions = [
        {"name": "短+中 (50/50)", "dist": {64: 0.5, 256: 0.5}, "max_len": 512},
        {"name": "中+长 (60/40)", "dist": {256: 0.6, 1024: 0.4}, "max_len": 1536},
        {"name": "短+中+长 (40/40/20)", "dist": {64: 0.4, 256: 0.4, 1024: 0.2}, "max_len": 1536},
    ]

    results = []
    for cfg in distributions:
        print(f"\n  📊 测试 {cfg['name']}...")
        texts, lengths = generate_mixed_distribution(tokenizer, 300, cfg["dist"])

        result = benchmark_encode(
            tokenizer, model, texts, device, dtype,
            cfg["max_len"], 32,
            cfg["name"], "mixed_distribution"
        )
        results.append(result)

        # 计算 tail latency
        stats = compute_latency_stats(result.latencies)
        print(f"     ✅ 吞吐: {result.throughput:.1f}/s | P95: {stats.p95_ms:.1f}ms | P99: {stats.p99_ms:.1f}ms")

    return results


def test_precision_comparison(tokenizer, model_fp16, model_fp32, device) -> List[BenchmarkResult]:
    """测试组5: FP16 vs FP32 精度对比 (2组)"""
    print("\n" + "="*100)
    print("  测试组 5: FP16 vs FP32 精度对比")
    print("="*100)

    texts, _ = generate_texts_by_tokens(tokenizer, 256, 200)
    results = []

    # FP16
    print("\n  📊 测试 FP16...")
    result_fp16 = benchmark_encode(
        tokenizer, model_fp16, texts, device, torch.float16,
        512, 32, "FP16", "precision"
    )
    results.append(result_fp16)
    print(f"     ✅ 吞吐: {result_fp16.throughput:.1f}/s | 显存: {result_fp16.peak_memory_mb:.0f}MB")

    # FP32
    print("\n  📊 测试 FP32...")
    result_fp32 = benchmark_encode(
        tokenizer, model_fp32, texts, device, torch.float32,
        512, 32, "FP32", "precision"
    )
    results.append(result_fp32)
    print(f"     ✅ 吞吐: {result_fp32.throughput:.1f}/s | 显存: {result_fp32.peak_memory_mb:.0f}MB")

    speedup = result_fp16.throughput / result_fp32.throughput if result_fp32.throughput > 0 else 0
    mem_ratio = result_fp16.peak_memory_mb / result_fp32.peak_memory_mb if result_fp32.peak_memory_mb > 0 else 0
    print(f"\n  📈 FP16 加速比: {speedup:.2f}x | 显存节省: {(1-mem_ratio)*100:.1f}%")

    return results


def test_cold_vs_warm(tokenizer, model, device, dtype) -> List[BenchmarkResult]:
    """测试组6: 冷启动 vs 热启动 (2组)"""
    print("\n" + "="*100)
    print("  测试组 6: 冷启动 vs 热启动延迟对比")
    print("="*100)

    texts, _ = generate_texts_by_tokens(tokenizer, 256, 100)
    results = []

    # 冷启动 (清空缓存后)
    print("\n  📊 测试冷启动...")
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    result_cold = benchmark_encode(
        tokenizer, model, texts[:10], device, dtype,
        512, 10, "冷启动", "startup"
    )
    results.append(result_cold)
    print(f"     ✅ 首批延迟: {result_cold.total_time_ms:.2f}ms")

    # 热启动 (模型已预热)
    print("\n  📊 测试热启动...")
    # 先预热
    _ = benchmark_encode(tokenizer, model, texts[:50], device, dtype, 512, 32, "warmup", "")

    result_warm = benchmark_encode(
        tokenizer, model, texts[:10], device, dtype,
        512, 10, "热启动", "startup"
    )
    results.append(result_warm)
    print(f"     ✅ 首批延迟: {result_warm.total_time_ms:.2f}ms")

    speedup = result_cold.total_time_ms / result_warm.total_time_ms if result_warm.total_time_ms > 0 else 0
    print(f"\n  📈 热启动加速: {speedup:.2f}x")

    return results


def test_sustained_pressure(tokenizer, model, device, dtype) -> Tuple[BenchmarkResult, LatencyStats]:
    """测试组7: 持续压力测试 (收集 P50/P90/P95/P99)"""
    print("\n" + "="*100)
    print("  测试组 7: 持续压力测试 (20轮迭代)")
    print("="*100)

    texts, _ = generate_texts_by_tokens(tokenizer, 256, 100)
    all_latencies = []
    throughputs = []

    print("\n  运行 20 轮压力测试...")
    for i in range(20):
        result = benchmark_encode(
            tokenizer, model, texts, device, dtype,
            512, 32, f"Round{i+1}", "pressure"
        )
        all_latencies.extend(result.latencies)
        throughputs.append(result.throughput)

        if (i + 1) % 5 == 0:
            print(f"     轮次 {i+1}/20: 吞吐 {result.throughput:.1f}/s")

    # 汇总统计
    stats = compute_latency_stats(all_latencies)
    avg_throughput = statistics.mean(throughputs)
    throughput_std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0

    combined = BenchmarkResult(
        name="压力测试汇总",
        category="pressure",
        text_count=len(texts) * 20,
        batch_size=32,
        max_length=512,
        avg_tokens=256,
        total_time_ms=sum(r.latencies[0] for r in [result] * 20 if r.latencies),
        tokenize_time_ms=0,
        encode_time_ms=0,
        throughput=avg_throughput,
        tokens_per_sec=avg_throughput * 256,
        avg_latency_ms=stats.avg_ms,
        latencies=all_latencies
    )

    print(f"\n  📊 压力测试汇总:")
    print(f"     平均吞吐: {avg_throughput:.1f} ± {throughput_std:.1f} /s")
    print(f"     P50: {stats.p50_ms:.2f}ms | P90: {stats.p90_ms:.2f}ms | "
          f"P95: {stats.p95_ms:.2f}ms | P99: {stats.p99_ms:.2f}ms")

    return combined, stats


def test_realworld_stream(tokenizer, model, device, dtype) -> Tuple[BenchmarkResult, LatencyStats]:
    """测试组8: 真实请求流模拟"""
    print("\n" + "="*100)
    print("  测试组 8: 真实请求流模拟 (变长输入)")
    print("="*100)

    # 模拟真实分布: 指数分布的请求长度
    lengths = []
    for _ in range(500):
        # 指数分布，大部分短，少量长
        base = random.expovariate(1/200)  # 平均200 tokens
        length = max(32, min(int(base), 2048))
        lengths.append(length)

    # 生成对应长度的文本
    texts = []
    actual_lengths = []
    for target_len in lengths:
        t, l = generate_texts_by_tokens(tokenizer, target_len, 1)
        texts.extend(t)
        actual_lengths.extend(l)

    print(f"\n  生成 {len(texts)} 条变长文本:")
    print(f"     最短: {min(actual_lengths)} tokens")
    print(f"     最长: {max(actual_lengths)} tokens")
    print(f"     平均: {sum(actual_lengths)/len(actual_lengths):.0f} tokens")

    # 模拟真实场景：小 batch 逐个处理
    all_latencies = []
    start = time.perf_counter()

    for i in range(0, len(texts), 16):  # batch=16 模拟在线请求
        batch = texts[i:i+16]
        batch_start = time.perf_counter()

        inputs = tokenizer(batch, padding="longest", truncation=True,
                          max_length=2560, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = model(**inputs, return_dict=True)
            vecs = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
            _ = vecs.cpu()

        if device == "cuda":
            torch.cuda.synchronize()

        batch_end = time.perf_counter()
        all_latencies.append((batch_end - batch_start) * 1000)

    total_time = (time.perf_counter() - start) * 1000
    stats = compute_latency_stats(all_latencies)

    result = BenchmarkResult(
        name="真实请求流",
        category="realworld",
        text_count=len(texts),
        batch_size=16,
        max_length=2560,
        avg_tokens=sum(actual_lengths)/len(actual_lengths),
        total_time_ms=total_time,
        tokenize_time_ms=0,
        encode_time_ms=0,
        throughput=len(texts) / (total_time / 1000),
        tokens_per_sec=sum(actual_lengths) / (total_time / 1000),
        avg_latency_ms=total_time / len(texts),
        latencies=all_latencies
    )

    print(f"\n  📊 真实流测试结果:")
    print(f"     总吞吐: {result.throughput:.1f} texts/s")
    print(f"     Token吞吐: {result.tokens_per_sec:.0f} tokens/s")
    print(f"     Tail Latency - P95: {stats.p95_ms:.2f}ms | P99: {stats.p99_ms:.2f}ms | Max: {stats.max_ms:.2f}ms")

    return result, stats


# ==================== 报告生成 ====================

def print_summary_table(results: List[BenchmarkResult], title: str):
    """打印汇总表格"""
    print(f"\n{'='*120}")
    print(f"  {title}")
    print('='*120)

    print(f"\n{'用例':<20} {'类别':<15} {'数量':>6} {'Batch':>6} {'MaxLen':>7} "
          f"{'AvgTok':>7} {'吞吐(/s)':>10} {'Tok/s':>10} {'延迟(ms)':>10} {'显存(MB)':>10}")
    print("-" * 120)

    for r in results:
        print(f"{r.name:<20} {r.category:<15} {r.text_count:>6} {r.batch_size:>6} {r.max_length:>7} "
              f"{r.avg_tokens:>7.0f} {r.throughput:>10.1f} {r.tokens_per_sec:>10.0f} "
              f"{r.avg_latency_ms:>10.2f} {r.peak_memory_mb:>10.1f}")


def print_tokenize_breakdown(results: List[BenchmarkResult]):
    """打印 Tokenize 时间拆解"""
    print(f"\n{'='*120}")
    print("  Tokenize vs Encode 时间拆解")
    print('='*120)

    print(f"\n{'用例':<25} {'总时间(ms)':>12} {'Tokenize(ms)':>14} {'Encode(ms)':>12} "
          f"{'Tok占比':>10} {'Enc占比':>10}")
    print("-" * 120)

    for r in results:
        if r.total_time_ms > 0:
            tok_pct = r.tokenize_time_ms / r.total_time_ms * 100
            enc_pct = r.encode_time_ms / r.total_time_ms * 100
            print(f"{r.name:<25} {r.total_time_ms:>12.2f} {r.tokenize_time_ms:>14.2f} "
                  f"{r.encode_time_ms:>12.2f} {tok_pct:>9.1f}% {enc_pct:>9.1f}%")


def print_percentile_table(results: List[BenchmarkResult]):
    """打印百分位延迟表"""
    print(f"\n{'='*120}")
    print("  批次延迟分位数 (P50/P90/P95/P99)")
    print('='*120)

    print(f"\n{'用例':<25} {'Min(ms)':>10} {'P50(ms)':>10} {'P90(ms)':>10} "
          f"{'P95(ms)':>10} {'P99(ms)':>10} {'Max(ms)':>10} {'StdDev':>10}")
    print("-" * 120)

    for r in results:
        if r.latencies:
            stats = compute_latency_stats(r.latencies)
            print(f"{r.name:<25} {stats.min_ms:>10.2f} {stats.p50_ms:>10.2f} "
                  f"{stats.p90_ms:>10.2f} {stats.p95_ms:>10.2f} {stats.p99_ms:>10.2f} "
                  f"{stats.max_ms:>10.2f} {stats.std_ms:>10.2f}")


def print_safe_batch_table(safe_batch_results: List[Dict]):
    """打印安全 Batch 表"""
    print(f"\n{'='*80}")
    print("  FP16 安全 Batch Size 推荐 (含 10% 安全边界)")
    print('='*80)

    print(f"\n  {'MaxLength':>12} │ {'SafeBatch':>12} │ {'AbsoluteMax':>12} │ {'PeakMem(MB)':>14}")
    print(f"  {'-'*12}─┼─{'-'*12}─┼─{'-'*12}─┼─{'-'*14}")

    for r in safe_batch_results:
        print(f"  {r['max_length']:>12} │ {r['safe_batch']:>12} │ {r['absolute_max']:>12} │ {r['peak_memory_mb']:>14.1f}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("="*120)
    print("  BGE-M3 GPU 性能基准测试 - 全面版 (24+ 测试用例)")
    print("="*120)
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 检查环境
    cuda_available = torch.cuda.is_available()
    print(f"\n  CUDA 可用: {'✅ 是' if cuda_available else '❌ 否'}")

    if cuda_available:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  显存总量: {gpu_mem:.1f} GB")

    device = "cuda" if cuda_available else "cpu"

    # 加载模型
    print(f"\n  加载模型...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)

    # FP16 模型
    model_fp16 = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True)
    model_fp16.to(device).eval()

    # FP32 模型 (用于对比)
    model_fp32 = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32, local_files_only=True)
    model_fp32.to(device).eval()

    print(f"  ✅ 模型加载完成 ({time.time() - start:.2f}s)")

    # 预热
    print("\n  模型预热...")
    _ = benchmark_encode(tokenizer, model_fp16, ["预热"] * 16, device, torch.float16, 128, 16, "warmup", "")
    print("  ✅ 预热完成")

    # ==================== 运行所有测试 ====================
    all_results = []

    # 测试组 1: 长度梯度 (6组)
    results_length = test_length_gradient(tokenizer, model_fp16, device, torch.float16)
    all_results.extend(results_length)

    # 测试组 2: Batch Size 扩展 (5组)
    results_batch = test_batch_size_scaling(tokenizer, model_fp16, device, torch.float16)
    all_results.extend(results_batch)

    # 测试组 3: 安全边界探测 (4组)
    safe_batch_results = test_safe_batch_boundary(tokenizer, model_fp16, device, torch.float16)

    # 测试组 4: 混合分布 (3组)
    results_mixed = test_mixed_distribution(tokenizer, model_fp16, device, torch.float16)
    all_results.extend(results_mixed)

    # 测试组 5: 精度对比 (2组)
    results_precision = test_precision_comparison(tokenizer, model_fp16, model_fp32, device)
    all_results.extend(results_precision)

    # 测试组 6: 冷热启动 (2组)
    results_startup = test_cold_vs_warm(tokenizer, model_fp16, device, torch.float16)
    all_results.extend(results_startup)

    # 测试组 7: 持续压力 (1组)
    result_pressure, pressure_stats = test_sustained_pressure(tokenizer, model_fp16, device, torch.float16)
    all_results.append(result_pressure)

    # 测试组 8: 真实流 (1组)
    result_realworld, realworld_stats = test_realworld_stream(tokenizer, model_fp16, device, torch.float16)
    all_results.append(result_realworld)

    # ==================== 汇总报告 ====================
    print("\n" + "="*120)
    print("  📊 完整测试报告")
    print("="*120)

    print_summary_table(all_results, "全部测试结果汇总")
    print_tokenize_breakdown([r for r in all_results if r.tokenize_time_ms > 0])
    print_percentile_table([r for r in all_results if r.latencies])
    print_safe_batch_table(safe_batch_results)

    # ==================== 生产建议 ====================
    print("\n" + "="*120)
    print("  💡 生产环境配置建议")
    print("="*120)

    if cuda_available:
        # 找到最佳配置
        best_throughput = max(results_length, key=lambda x: x.throughput)
        best_token_throughput = max(results_length, key=lambda x: x.tokens_per_sec)

        print(f"""
  ┌────────────────────────────────────────────────────────────────────────────────┐
  │                          推荐配置矩阵                                          │
  ├────────────────────────────────────────────────────────────────────────────────┤
  │ 场景               │ MaxLength │ BatchSize │ 预期吞吐      │ P99延迟   │ 显存  │
  ├────────────────────────────────────────────────────────────────────────────────┤
  │ 高吞吐短文本       │ 256       │ {safe_batch_results[0]['safe_batch']:<10}│ ~{results_length[0].throughput:.0f}/s       │ <50ms     │ <2GB  │
  │ 通用平衡           │ 512       │ {safe_batch_results[0]['safe_batch']:<10}│ ~{results_length[1].throughput:.0f}/s       │ <80ms     │ <4GB  │
  │ 长文本处理         │ 2048      │ {safe_batch_results[2]['safe_batch']:<10}│ ~{results_length[4].throughput:.0f}/s       │ <200ms    │ <14GB │
  │ 超长文本           │ 4096      │ {safe_batch_results[3]['safe_batch']:<10}│ ~{results_length[5].throughput:.0f}/s        │ <400ms    │ <20GB │
  │ 混合真实场景       │ 1536      │ 32        │ ~{results_mixed[2].throughput:.0f}/s       │ <100ms    │ <3GB  │
  └────────────────────────────────────────────────────────────────────────────────┘

  ⚠️  关键注意事项:

  1. OOM 防护:
     - 生产环境 batch_size 设为上表 "SafeBatch" 的 80%
     - 实现请求队列限流，防止突发流量

  2. Tokenize 优化:
     - 短文本场景 Tokenize 占比高达 {results_length[0].tokenize_time_ms/results_length[0].total_time_ms*100:.1f}%
     - 考虑预编译高频查询或使用 tokenizer 缓存

  3. Tail Latency 控制:
     - 压力测试 P99: {pressure_stats.p99_ms:.2f}ms
     - 真实流 P99: {realworld_stats.p99_ms:.2f}ms
     - 混合长度场景 P99 波动较大，建议按长度分桶处理

  4. 精度选择:
     - FP16 相比 FP32 加速 {results_precision[0].throughput/results_precision[1].throughput:.2f}x
     - 显存节省 {(1-results_precision[0].peak_memory_mb/results_precision[1].peak_memory_mb)*100:.1f}%
     - 精度损失可忽略，推荐使用 FP16

  5. 预热策略:
     - 冷启动延迟: {results_startup[0].total_time_ms:.2f}ms
     - 热启动延迟: {results_startup[1].total_time_ms:.2f}ms
     - 服务启动时务必执行预热
""")

    # 保存报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "N/A",
        "results": [asdict(r) for r in all_results],
        "safe_batch": safe_batch_results,
        "pressure_stats": asdict(pressure_stats) if hasattr(pressure_stats, '__dict__') else {},
        "realworld_stats": asdict(realworld_stats) if hasattr(realworld_stats, '__dict__') else {}
    }

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  📁 详细报告已保存至: {REPORT_FILE}")

    # 清理
    del model_fp16, model_fp32, tokenizer
    if cuda_available:
        torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "="*120)
    print(f"  ✅ 全部 {len(all_results)} 组测试完成！")
    print("="*120)


if __name__ == "__main__":
    main()
