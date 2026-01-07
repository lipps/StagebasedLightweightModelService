#!/usr/bin/env python3
"""
BGE-M3 服务测试客户端
用法: python test_client.py
"""

import requests
import json

# 服务地址
SERVICE_URL = "http://localhost:8001/embed"

# 测试文本
test_texts = [
    "什么是机器学习？",
    "Machine learning is a subset of artificial intelligence.",
    "人工智能改变了我们的生活方式。"
]

# 构造请求
payload = {
    "texts": test_texts,
    "output_type": "dense",
    "normalize": True,
    "max_length": 512,
    "batch_size": 32
}

try:
    # 发送请求
    print("🚀 发送嵌入请求...")
    print(f"📝 文本数量: {len(test_texts)}")

    response = requests.post(SERVICE_URL, json=payload, timeout=30)
    response.raise_for_status()

    # 解析结果
    result = response.json()
    embeddings = result["embeddings"]

    print(f"✅ 成功获取嵌入向量!")
    print(f"📊 向量维度: {len(embeddings[0])}")
    print(f"📈 向量数量: {len(embeddings)}")
    print(f"\n前10个维度示例:")
    print(embeddings[0][:10])

except requests.exceptions.ConnectionError:
    print("❌ 连接失败! 请确保服务正在运行:")
    print("   CUDA_VISIBLE_DEVICES=0 uvicorn serve_bge_m3:app --host 0.0.0.0 --port 8001 --workers 1")
except Exception as e:
    print(f"❌ 错误: {e}")
