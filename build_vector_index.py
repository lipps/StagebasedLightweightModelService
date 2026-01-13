import torch
import json
import os
import time
from transformers import AutoTokenizer, AutoModel

# 配置
MODEL_ID = "/opt/bge-m3/models/bge-m3"
DATA_PATH = "data/standard_knowledge.json"
INDEX_PATH = "data/model_index.pt"

def mean_pooling(last_hidden_state, attention_mask):
    """Mean Pooling 聚合"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def build_index():
    # 强制使用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[*] 正在初始化模型 (设备: {device}, 精度: {dtype})...")
    
    start_time = time.time()
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(device)
    model.eval()
    
    print(f"[*] 模型加载完成，耗时: {time.time() - start_time:.2f}s")
    
    # 读取数据
    if not os.path.exists(DATA_PATH):
        print(f"[!] 错误: 数据文件 {DATA_PATH} 不存在")
        return
        
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    print(f"[*] 准备对 {len(texts)} 条话术进行向量化...")
    
    # 执行编码
    with torch.no_grad():
        inputs = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 模型推理
        out = model(**inputs)
        vectors = mean_pooling(out.last_hidden_state, inputs['attention_mask'])
        
        # L2 归一化 (关键: 归一化后点积即等同于余弦相似度)
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
    
    # 保存结果
    print(f"[*] 正在保存索引至 {INDEX_PATH}...")
    torch.save({
        'vectors': vectors.cpu(),  # 转为 CPU 存储，加载时可再入 GPU
        'metadata': data,
        'config': {
            'model': 'bge-m3',
            'dim': vectors.shape[1],
            'timestamp': time.time()
        }
    }, INDEX_PATH)
    
    print(f"[+] 索引构建成功！总耗时: {time.time() - start_time:.2f}s")
    print(f"    向量形状: {vectors.shape}")

if __name__ == "__main__":
    build_index()
