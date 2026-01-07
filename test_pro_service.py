#!/usr/bin/env python3
"""
BGE-M3 Pro 版服务功能测试
测试所有新增特性和性能改进
"""

import requests
import time
import json
from typing import List

# 服务地址
BASE_URL = "http://localhost:8001"


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_health_check():
    """测试健康检查端点"""
    print_section("1. 健康检查测试")

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()

        data = response.json()
        print("✅ 健康检查成功")
        print(f"   状态: {data.get('status')}")
        print(f"   模型已加载: {data.get('model_loaded')}")
        print(f"   设备: {data.get('device')}")
        print(f"   PyTorch 版本: {data.get('torch_version')}")
        print(f"   CUDA 可用: {data.get('cuda_available')}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ 健康检查失败: {e}")
        return False


def test_stats():
    """测试统计信息端点"""
    print_section("2. 统计信息测试")

    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=5)
        response.raise_for_status()

        data = response.json()
        print("✅ 统计信息获取成功")
        for key, value in data.items():
            print(f"   {key}: {value}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ 统计信息获取失败: {e}")
        return False


def test_basic_embedding():
    """测试基础嵌入功能"""
    print_section("3. 基础嵌入测试")

    test_texts = ["这是一个测试文本", "Hello World"]

    try:
        start_time = time.time()

        response = requests.post(
            f"{BASE_URL}/embed",
            json={"texts": test_texts, "normalize": True},
            timeout=30
        )
        response.raise_for_status()

        elapsed = (time.time() - start_time) * 1000
        data = response.json()

        print("✅ 嵌入生成成功")
        print(f"   文本数量: {data.get('count')}")
        print(f"   向量维度: {data.get('dim')}")
        print(f"   服务器耗时: {data.get('time_ms'):.2f}ms")
        print(f"   客户端总耗时: {elapsed:.2f}ms")
        print(f"   平均耗时: {elapsed / data.get('count'):.2f}ms/文本")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ 嵌入生成失败: {e}")
        return False


def test_batch_performance():
    """测试批处理性能"""
    print_section("4. 批处理性能测试")

    # 生成测试数据
    test_sizes = [10, 50, 100]

    for size in test_sizes:
        texts = [f"这是第 {i+1} 个测试文本，用于评估批处理性能" for i in range(size)]

        try:
            start_time = time.time()

            response = requests.post(
                f"{BASE_URL}/embed",
                json={
                    "texts": texts,
                    "normalize": True,
                    "batch_size": 32
                },
                timeout=60
            )
            response.raise_for_status()

            elapsed = (time.time() - start_time) * 1000
            data = response.json()

            throughput = data.get('count') / (elapsed / 1000)

            print(f"✅ 批次大小 {size}:")
            print(f"   总耗时: {elapsed:.2f}ms")
            print(f"   服务器耗时: {data.get('time_ms'):.2f}ms")
            print(f"   平均耗时: {elapsed / size:.2f}ms/文本")
            print(f"   吞吐量: {throughput:.2f} 文本/秒")

        except requests.exceptions.RequestException as e:
            print(f"❌ 批次大小 {size} 测试失败: {e}")


def test_error_handling():
    """测试错误处理"""
    print_section("5. 错误处理测试")

    # 测试空文本列表
    print("测试 1: 空文本列表")
    try:
        response = requests.post(
            f"{BASE_URL}/embed",
            json={"texts": []},
            timeout=5
        )
        if response.status_code == 400:
            print("   ✅ 正确返回 400 错误")
            print(f"   错误信息: {response.json().get('detail')}")
        else:
            print(f"   ⚠️  返回状态码 {response.status_code}")
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")

    # 测试过大批次
    print("\n测试 2: 超大批次（1001 条文本）")
    try:
        large_texts = [f"文本{i}" for i in range(1001)]
        response = requests.post(
            f"{BASE_URL}/embed",
            json={"texts": large_texts},
            timeout=5
        )
        if response.status_code == 422:
            print("   ✅ 正确返回 422 验证错误")
        else:
            print(f"   ⚠️  返回状态码 {response.status_code}")
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")

    # 测试无效参数
    print("\n测试 3: 无效 max_length")
    try:
        response = requests.post(
            f"{BASE_URL}/embed",
            json={"texts": ["测试"], "max_length": 10000},
            timeout=5
        )
        if response.status_code == 422:
            print("   ✅ 正确返回 422 验证错误")
        else:
            print(f"   ⚠️  返回状态码 {response.status_code}")
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")


def test_vector_similarity():
    """测试向量相似度计算"""
    print_section("6. 向量相似度测试")

    test_pairs = [
        ("机器学习", "深度学习"),
        ("猫", "狗"),
        ("苹果", "橙子"),
        ("你好", "goodbye"),
    ]

    try:
        for text1, text2 in test_pairs:
            response = requests.post(
                f"{BASE_URL}/embed",
                json={"texts": [text1, text2], "normalize": True},
                timeout=10
            )
            response.raise_for_status()

            embeddings = response.json()["embeddings"]
            vec1, vec2 = embeddings[0], embeddings[1]

            # 计算余弦相似度（已归一化，直接点积）
            similarity = sum(a * b for a, b in zip(vec1, vec2))

            print(f"   '{text1}' vs '{text2}':")
            print(f"   相似度: {similarity:.4f}")

    except Exception as e:
        print(f"❌ 相似度测试失败: {e}")


def main():
    """主测试流程"""
    print(f"\n{'#'*60}")
    print(f"#  BGE-M3 Pro 版服务测试")
    print(f"#  服务地址: {BASE_URL}")
    print(f"{'#'*60}")

    # 测试计数
    tests = []

    # 1. 健康检查
    tests.append(("健康检查", test_health_check()))

    # 2. 统计信息
    tests.append(("统计信息", test_stats()))

    # 3. 基础嵌入
    tests.append(("基础嵌入", test_basic_embedding()))

    # 4. 批处理性能
    test_batch_performance()

    # 5. 错误处理
    test_error_handling()

    # 6. 向量相似度
    test_vector_similarity()

    # 测试总结
    print_section("测试总结")

    passed = sum(1 for _, result in tests if result)
    total = len(tests)

    print(f"通过测试: {passed}/{total}")
    print(f"失败测试: {total - passed}/{total}")

    if passed == total:
        print("\n✅ 所有核心测试通过！")
    else:
        print("\n⚠️  部分测试失败，请检查服务状态")

    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n测试失败: {e}")
