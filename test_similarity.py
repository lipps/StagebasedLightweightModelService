import torch
import sys
import os
from transformers import AutoTokenizer, AutoModel

# 确保能导入 src 目录
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.semantic_matcher import SemanticMatcher

# 配置
MODEL_ID = "/opt/bge-m3/models/bge-m3"
INDEX_PATH = "data/model_index.pt"

def mean_pooling(last_hidden_state, attention_mask):
    """Mean Pooling 聚合"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"[*] 正在初始化测试环境 (设备: {device})...")
    
    # 1. 加载模型（用于对 Query 进行实时向量化）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()
    
    # 2. 加载匹配引擎
    matcher = SemanticMatcher(INDEX_PATH, device=device)
    
    # 3. 定义测试用例
    test_queries = [
        "B点是买入信号吗？",
        "红壳子是什么意思？",
        "什么时候该卖？",
        "能不能当天买？",
        "今天天气不错",  # 预期：低分
        "什么是BS点功能？",
        "B点之后一定是最佳买点吗？",
        "下跌趋势形成是什么字母？"
    ]
    
    print("\n" + "="*80)
    print(f"{ '用户提问 (Query)':<30} | { '匹配意图':<20} | { '分数':<6} | {'命中状态'}")
    print("-" * 80)
    
    for query in test_queries:
        # 编码 Query
        with torch.no_grad():
            inputs = tokenizer([query], padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            q_vec = mean_pooling(out.last_hidden_state, inputs['attention_mask'])
            q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=1)
            
        # 在向量库中搜索 (阈值设为 0.5 以便观察低分结果)
        results = matcher.search(q_vec, top_k=1, threshold=0.5)
        
        if not results:
            print(f"{query:<30} | { '(无匹配)':<20} | { '-':<6} | ❌")
        else:
            res = results[0]
            # 判定标准：> 0.75 认为强相关，0.6-0.75 疑似相关，< 0.6 不相关
            score = res['score']
            status = "✅ 强相关" if score > 0.75 else ("⚠️ 疑似" if score > 0.65 else "❓ 弱相关")
            
            # 如果是无关输入，即使有最高分也应该是低分
            if query == "今天天气不错" and score < 0.6:
                status = "✅ 正确拦截"
                
            print(f"{query:<30} | {res['intent']:<20} | {score:<6} | {status}")
            # print(f"      [标准话术]: {res['text']}")

    print("="*80)
    print("\n[测试完成] 以上结果展示了语义泛化匹配能力。")

if __name__ == "__main__":
    test()
