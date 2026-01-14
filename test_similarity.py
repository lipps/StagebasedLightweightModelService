import torch
import sys
import os
import json
import glob
import re
from transformers import AutoTokenizer, AutoModel
from torch import nn

# 确保能导入 src 目录
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.semantic_matcher import SemanticMatcher

# ==================== 配置 ====================
MODEL_ID = "/opt/bge-m3/models/bge-m3"
INDEX_PATH = "data/model_index.pt"
BS_CHECKPOINT = "checkpoints_bs/epoch_5.pt"
DATA_HISTORY_DIR = "data_history"

# ==================== 模型定义 ====================
class BSLogisticRegression(nn.Module):
    """BS 点讲解识别分类器"""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        logit = self.linear(emb)
        return logit.squeeze(-1)

def mean_pooling(last_hidden_state, attention_mask):
    """Mean Pooling 聚合"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def clean_text(text):
    """清洗通话文本，去除时间戳和角色标识"""
    # 移除 [0:0:0]A: 这种格式
    text = re.sub(r'\\[\\d+:\\d+:\\d+\\][A-Z]:', '', text)
    # 替换 <br/> 为换行
    text = text.replace('<br/>', '\n')
    return text

# ==================== 测试逻辑 ====================
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"[*] 正在初始化 BS 点识别验证环境 (设备: {device})...")
    
    # 1. 加载 BGE-M3 模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
        model.eval()
    except Exception as e:
        print(f"[!] 加载 BGE 模型失败: {e}")
        return
    
    # 2. 加载语义匹配引擎
    if not os.path.exists(INDEX_PATH):
        print(f"[!] 索引文件不存在: {INDEX_PATH}")
        return
    matcher = SemanticMatcher(INDEX_PATH, device=device)
    
    # 3. 加载 BS 分类器
    if not os.path.exists(BS_CHECKPOINT):
        print(f"[!] BS 分类器模型不存在: {BS_CHECKPOINT}")
        return
    ckpt = torch.load(BS_CHECKPOINT, map_location=device)
    lr_model = BSLogisticRegression(ckpt['embedding_dim']).to(device).to(dtype)
    lr_model.load_state_dict({k: v.to(dtype) for k, v in ckpt['model_state_dict'].items()})
    lr_model.eval()
    t_high = ckpt.get('T_high', 0.5)
    
    # 4. 获取历史文件
    history_files = sorted(glob.glob(os.path.join(DATA_HISTORY_DIR, "*.json")))
    if not history_files:
        print(f"[!] 错误: 未在 {DATA_HISTORY_DIR} 找到通话记录文件")
        return

    print(f"[*] 开始分析 {len(history_files)} 个通话文件...\n")
    print("="*100)
    print(f"{ '文件名':<40} | {'BS点识别结果'}")
    print("-" * 100)

    for file_path in history_files:
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except:
                continue
        
        call_content = data.get("call_content", "")
        if not call_content:
            print(f"{file_name:<40} | ❌ 无通话内容")
            continue
            
        # 清洗并分段
        cleaned_content = clean_text(call_content)
        segments = [s.strip() for s in cleaned_content.split('\n') if s.strip()]
        
        bs_details = []
        
        # 批量推理
        batch_size = 8
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            with torch.no_grad():
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = model(**inputs)
                vecs = mean_pooling(out.last_hidden_state, inputs['attention_mask'])
                vecs_norm = torch.nn.functional.normalize(vecs, p=2, dim=1)
                
                # 分类预测
                logits = lr_model(vecs_norm)
                probs = torch.sigmoid(logits)
                
                for j, prob in enumerate(probs):
                    score_lr = prob.item()
                    # 语义匹配辅助验证
                    results = matcher.search(vecs_norm[j], top_k=1, threshold=0.7)
                    
                    if score_lr >= t_high or results:
                        hit_intent = results[0]['intent'] if results else "bs_general"
                        bs_details.append({
                            "text": batch[j][:50] + "..." if len(batch[j]) > 50 else batch[j],
                            "intent": hit_intent,
                            "prob": score_lr
                        })

        if bs_details:
            print(f"{file_name:<40} | ✅ 命中 {len(bs_details)} 处关键点")
            for detail in bs_details[:3]:  # 每个文件展示前3个命中点
                print(f"{ ' ':40} |   - [{detail['intent']}] {detail['text']}")
        else:
            print(f"{file_name:<40} | ⚪ 未讲解 BS 点")
        print("-" * 100)

    print("\n[分析完成] 验证完毕。")

if __name__ == "__main__":
    test()