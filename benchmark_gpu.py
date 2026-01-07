#!/usr/bin/env python3
"""
BGE-M3 GPU 性能基准测试
对比 CPU 和 GPU 模式的性能差异
"""

import time
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import os

print("="*80)
print("  BGE-M3 GPU 性能基准测试")
print("="*80)

MODEL_ID = "/opt/bge-m3/models/bge-m3"

# 测试配置
TEST_CONFIGS = [
    {"name": "小批量短文本", "texts": ["这是一个测试文本"] * 10, "batch_size": 8, "max_length": 128},
    {"name": "中批量中文本", "texts": [f"这是第{i}个测试文本，用于评估性能" * 5 for i in range(50)], "batch_size": 32, "max_length": 256},
    {"name": "大批量长文本", "texts": [f"这是第{i}个较长的测试文本，包含更多内容用于全面评估系统性能表现" * 10 for i in range(100)], "batch_size": 64, "max_length": 512},
]


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean Pooling"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.inference_mode()
def encode(tokenizer, model, texts: List[str], device: str, dtype: torch.dtype,
           max_length: int, batch_size: int) -> List[List[float]]:
    """编码文本为向量"""
    all_vecs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model(**inputs, return_dict=True)
        vecs = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        all_vecs.extend(vecs.detach().float().cpu().tolist())

    return all_vecs


def benchmark(device: str, dtype: torch.dtype, threads: int = None):
    """运行基准测试"""
    print(f"\n{'='*80}")
    print(f"  测试模式: {device.upper()} ({'FP16' if dtype == torch.float16 else 'FP32'})")
    if threads:
        print(f"  CPU 线程数: {threads}")
    print("="*80)

    # 配置 CPU 线程
    if threads:
        torch.set_num_threads(threads)

    # 加载模型
    print("\n加载模型...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype, local_files_only=True)
    model.to(device)
    model.eval()
    load_time = time.time() - start
    print(f"  ✅ 模型加载完成 ({load_time:.2f}s)")

    # 模型预热
    print("\n模型预热...")
    _ = encode(tokenizer, model, ["预热测试"] * 4, device, dtype, 128, 4)
    if device == "cuda":
        torch.cuda.synchronize()
    print("  ✅ 预热完成")

    # 运行测试
    results = []
    for config in TEST_CONFIGS:
        print(f"\n{'─'*80}")
        print(f"  测试场景: {config['name']}")
        print(f"  文本数量: {len(config['texts'])}")
        print(f"  批处理大小: {config['batch_size']}")
        print(f"  最大长度: {config['max_length']}")
        print("─"*80)

        # 运行测试
        start = time.time()
        embeddings = encode(
            tokenizer, model, config['texts'], device, dtype,
            config['max_length'], config['batch_size']
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        # 计算指标
        count = len(embeddings)
        throughput = count / elapsed
        latency = elapsed * 1000
        avg_latency = latency / count

        print(f"\n  结果:")
        print(f"    总耗时: {latency:.2f}ms")
        print(f"    平均延迟: {avg_latency:.2f}ms/文本")
        print(f"    吞吐量: {throughput:.2f} 文本/秒")

        results.append({
            "name": config["name"],
            "count": count,
            "elapsed": elapsed,
            "throughput": throughput,
            "latency": latency
        })

    # 清理
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    """主函数"""
    all_results = {}

    # 检查 CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA 可用: {'✅ 是' if cuda_available else '❌ 否'}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")

    # 测试 1: CPU 单线程（原版配置）
    print("\n\n" + "="*80)
    print("  第 1 项测试: CPU 单线程（原版基线）")
    print("="*80)
    all_results["CPU 单线程"] = benchmark("cpu", torch.float32, threads=1)

    # 测试 2: CPU 多线程（Pro 版 CPU 优化）
    print("\n\n" + "="*80)
    print("  第 2 项测试: CPU 多线程（Pro 版优化）")
    print("="*80)
    all_results["CPU 多线程"] = benchmark("cpu", torch.float32, threads=8)

    # 测试 3: GPU FP32
    if cuda_available:
        print("\n\n" + "="*80)
        print("  第 3 项测试: GPU FP32")
        print("="*80)
        all_results["GPU FP32"] = benchmark("cuda", torch.float32)

        # 测试 4: GPU FP16（最优配置）
        print("\n\n" + "="*80)
        print("  第 4 项测试: GPU FP16（最优配置）")
        print("="*80)
        all_results["GPU FP16"] = benchmark("cuda", torch.float16)

    # 性能对比表
    print("\n\n" + "="*80)
    print("  性能对比总结")
    print("="*80)

    # 提取第二个测试场景的数据（中等规模）
    baseline_name = "CPU 单线程"
    baseline_throughput = all_results[baseline_name][1]["throughput"]

    print(f"\n基线（{baseline_name}）: {baseline_throughput:.2f} 文本/秒")
    print("\n" + "┌" + "─"*20 + "┬" + "─"*15 + "┬" + "─"*15 + "┬" + "─"*15 + "┐")
    print(f"│ {'配置':<18} │ {'吞吐(/s)':>13} │ {'延迟(ms)':>13} │ {'提升倍数':>13} │")
    print("├" + "─"*20 + "┼" + "─"*15 + "┼" + "─"*15 + "┼" + "─"*15 + "┤")

    for config_name, results in all_results.items():
        # 使用中等规模测试的数据
        result = results[1]
        throughput = result["throughput"]
        latency = result["latency"]
        speedup = throughput / baseline_throughput

        print(f"│ {config_name:<18} │ {throughput:>13.2f} │ {latency:>13.2f} │ {speedup:>12.2f}x │")

    print("└" + "─"*20 + "┴" + "─"*15 + "┴" + "─"*15 + "┴" + "─"*15 + "┘")

    # 显存使用（如果有 GPU）
    if cuda_available:
        print(f"\n显存使用:")
        print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  最大已分配: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

    print("\n" + "="*80)
    print("  测试完成！")
    print("="*80)

    # 推荐配置
    print("\n📊 推荐配置:")
    if cuda_available:
        best_config = max(all_results.items(), key=lambda x: x[1][1]["throughput"])
        print(f"  最佳性能: {best_config[0]} ({best_config[1][1]['throughput']:.2f} 文本/秒)")
        print(f"  🚀 性能提升: {best_config[1][1]['throughput'] / baseline_throughput:.1f}x")
    else:
        print("  推荐使用 CPU 多线程模式（无 GPU 可用）")


if __name__ == "__main__":
    main()
