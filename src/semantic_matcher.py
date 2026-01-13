import torch
import os

class SemanticMatcher:
    """
    语义匹配引擎
    基于余弦相似度实现向量检索
    """
    def __init__(self, index_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
            
        # 加载索引
        print(f"[*] 正在加载向量库: {index_path} (设备: {self.device})")
        checkpoint = torch.load(index_path, map_location=self.device)
        
        # 向量库矩阵 [N, 1024]
        self.vectors = checkpoint['vectors'].to(self.device)
        self.metadata = checkpoint['metadata']
        self.config = checkpoint.get('config', {})
        
        print(f"[*] 向量库加载完成，规模: {self.vectors.shape}")

    def search(self, query_vector, top_k=3, threshold=0.7):
        """
        在向量库中执行检索
        
        Args:
            query_vector: 输入向量 (torch.Tensor, shape=[1024] or [1, 1024])
            top_k: 返回最相似的 K 个结果
            threshold: 相似度阈值过滤
            
        Returns:
            List[Dict]: 匹配结果列表
        """
        # 统一格式为 [1, 1024]
        if not isinstance(query_vector, torch.Tensor):
            query_vector = torch.tensor(query_vector, dtype=torch.float32)
        
        query_vector = query_vector.to(self.device)
        if query_vector.ndim == 1:
            query_vector = query_vector.unsqueeze(0)
            
        # 确保精度匹配
        if query_vector.dtype != self.vectors.dtype:
            query_vector = query_vector.to(self.vectors.dtype)

        # 计算余弦相似度
        # 因为入库向量和 Query 向量都经过了 L2 归一化，所以 dot product 即等于 cosine similarity
        # [1, 1024] @ [1024, N] -> [1, N]
        with torch.no_grad():
            scores = torch.matmul(query_vector, self.vectors.T).squeeze(0)
        
        # 获取 Top-K
        k = min(top_k, len(self.vectors))
        top_scores, top_indices = torch.topk(scores, k=k)
        
        # 格式化输出并过滤阈值
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            if score >= threshold:
                results.append({
                    "score": round(score, 4),
                    "id": self.metadata[idx].get('id'),
                    "text": self.metadata[idx].get('text'),
                    "intent": self.metadata[idx].get('intent'),
                    "category": self.metadata[idx].get('category')
                })
        
        return results
