"""
train_bs_logreg_bge.py

功能：
- 使用 BGE-M3 作为冻结的 embedding 提取器
- 在其上训练一个逻辑回归头（线性层 + Sigmoid）做「BS 相关？」二分类
- 在验证集上自动扫描阈值，推荐 T_low / T_high
  - T_high：用于 "likely"
  - T_low ：用于 "maybe" 的下界（高召回优先）

依赖：
  pip install FlagEmbedding torch tqdm
"""

import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

# ==================== 配置区域 ====================

DATA_PATH = "data/bs_train.jsonl"  # 你的标注数据
SAVE_DIR = "checkpoints_bs"
os.makedirs(SAVE_DIR, exist_ok=True)

VALID_RATIO = 0.2        # 从数据中切 20% 做验证
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 1e-3
TARGET_RECALL_FOR_T_LOW = 0.9  # 选 T_low 时尽量满足的召回率
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==================== 数据集定义 ====================

class BSSegmentDataset(Dataset):
    """
    每条样本：
      text: str
      label: int (0/1)
    """
    def __init__(self, path: str):
        self.texts: List[str] = []
        self.labels: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                label = int(obj["label"])
                self.texts.append(text)
                self.labels.append(label)
        print(f"Loaded {len(self.texts)} samples from {path}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return texts, labels

# ==================== 模型：逻辑回归头 ====================

class BSLogisticRegression(nn.Module):
    """
    简单逻辑回归：emb -> Linear -> logit
    概率 = sigmoid(logit)
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: [batch, dim]
        logit = self.linear(emb)        # [batch, 1]
        return logit.squeeze(-1)        # [batch]

# ==================== 指标 & 阈值搜索 ====================

def precision_recall_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    """
    给定阈值，计算 P/R/F1
    """
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.9):
    """
    根据验证集的 y_true, y_prob 自动推荐：
      - T_high：最大 F1 对应的阈值
      - T_low ：在 Recall >= target_recall 条件下，Precision 最好的阈值
                如果找不到满足条件的，就选 Recall 最大的那个

    返回：
      T_low, T_high, metrics_low, metrics_high
    """
    thresholds = np.linspace(0.05, 0.95, num=19)  # 0.05, 0.10, ..., 0.95

    best_f1 = -1.0
    best_t_high = 0.5
    best_high_metrics = (0.0, 0.0, 0.0)

    best_t_low = 0.1
    best_low_metrics = (0.0, 0.0, 0.0)
    best_recall_for_low = -1.0

    for t in thresholds:
        p, r, f1 = precision_recall_f1(y_true, y_prob, t)
        # 更新 T_high：看 F1 最大
        if f1 > best_f1:
            best_f1 = f1
            best_t_high = t
            best_high_metrics = (p, r, f1)
        # 更新 T_low：优先满足 Recall >= target_recall，其次 Recall 最大
        if r >= target_recall:
            # 满足目标召回，取 Precision 高的
            if r > best_recall_for_low or (abs(r - best_recall_for_low) < 1e-6 and p > best_low_metrics[0]):
                best_recall_for_low = r
                best_t_low = t
                best_low_metrics = (p, r, f1)
        else:
            # 如果没有任何阈值满足 target_recall，最后再用 Recall 最大的兜底
            if best_recall_for_low < 0 and r > best_recall_for_low:
                best_recall_for_low = r
                best_t_low = t
                best_low_metrics = (p, r, f1)

    # 确保 T_low <= T_high，如果碰撞了就稍微调整一下
    if best_t_low > best_t_high:
        best_t_low = min(best_t_low, best_t_high)
        # 重新计算 low 的指标
        best_low_metrics = precision_recall_f1(y_true, y_prob, best_t_low)

    return best_t_low, best_t_high, best_low_metrics, best_high_metrics

# ==================== 主训练流程 ====================

def main():
    # 1) 加载数据
    dataset = BSSegmentDataset(DATA_PATH)
    if VALID_RATIO > 0:
        valid_size = int(len(dataset) * VALID_RATIO)
        train_size = len(dataset) - valid_size
        train_ds, valid_ds = random_split(dataset, [train_size, valid_size])
    else:
        train_ds = dataset
        valid_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )

    # 2) 加载 BGE-M3（冻结，只做 embedding）
    print("Loading BGE-M3 model...")
    # 使用本地模型路径
    LOCAL_MODEL_PATH = "/opt/bge-m3/models/bge-m3"
    bge_model = BGEM3FlagModel(
        LOCAL_MODEL_PATH,
        device=DEVICE,
        use_fp16=True,
    )
    test_emb = bge_model.encode(["test sentence"], return_dense=True)
    dense_vecs = test_emb["dense_vecs"]  # numpy [1, dim]
    embedding_dim = dense_vecs.shape[1]
    print(f"Embedding dim = {embedding_dim}")

    # 3) 定义逻辑回归头
    model = BSLogisticRegression(embedding_dim=embedding_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 4) 训练循环
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_samples = 0

        for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch} - train"):
            # 4.1 BGE-M3 提取 embedding
            with torch.no_grad():
                outputs = bge_model.encode(
                    texts,
                    batch_size=len(texts),
                    max_length=512,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                    normalize_embeddings=True,
                )
                dense_vecs = outputs["dense_vecs"]   # numpy [batch, dim]
                emb = torch.from_numpy(dense_vecs).to(DEVICE)

            labels = labels.to(DEVICE)  # float32 [batch]

            # 4.2 前向 + loss
            logits = model(emb)          # [batch]
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

        avg_loss = epoch_loss / max(num_samples, 1)
        print(f"[Epoch {epoch}] Train loss = {avg_loss:.4f}")

        # 5) 验证 + 阈值推荐
        if valid_loader is not None:
            model.eval()
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for texts, labels in tqdm(valid_loader, desc=f"Epoch {epoch} - valid"):
                    outputs = bge_model.encode(
                        texts,
                        batch_size=len(texts),
                        max_length=512,
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False,
                        normalize_embeddings=True,
                    )
                    dense_vecs = outputs["dense_vecs"]
                    emb = torch.from_numpy(dense_vecs).to(DEVICE)
                    logits = model(emb)           # [batch]
                    probs = torch.sigmoid(logits) # [batch]

                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(labels.numpy())

            y_prob = np.concatenate(all_probs, axis=0)
            y_true = np.concatenate(all_labels, axis=0).astype(int)

            # 基本指标（用 0.5 先算一个）
            p_05, r_05, f1_05 = precision_recall_f1(y_true, y_prob, threshold=0.5)
            print(
                f"[Epoch {epoch}] Valid @0.50: "
                f"precision={p_05:.4f}, recall={r_05:.4f}, f1={f1_05:.4f}"
            )

            # 自动找 T_low / T_high
            T_low, T_high, (p_low, r_low, f1_low), (p_high, r_high, f1_high) = \
                find_best_thresholds(y_true, y_prob, target_recall=TARGET_RECALL_FOR_T_LOW)

            print(
                f"[Epoch {epoch}] "
                f"Recommended T_low={T_low:.2f} "
                f"(P={p_low:.4f}, R={r_low:.4f}, F1={f1_low:.4f})"
            )
            print(
                f"[Epoch {epoch}] "
                f"Recommended T_high={T_high:.2f} "
                f"(P={p_high:.4f}, R={r_high:.4f}, F1={f1_high:.4f})"
            )

            # 6) 保存 checkpoint（附带阈值与简单指标）
            ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": embedding_dim,
                    "T_low": float(T_low),
                    "T_high": float(T_high),
                    "metrics": {
                        "p_05": float(p_05),
                        "r_05": float(r_05),
                        "f1_05": float(f1_05),
                        "P_low": float(p_low),
                        "R_low": float(r_low),
                        "F1_low": float(f1_low),
                        "P_high": float(p_high),
                        "R_high": float(r_high),
                        "F1_high": float(f1_high),
                    },
                },
                ckpt_path,
            )
            print(f"[Epoch {epoch}] Saved checkpoint to {ckpt_path}")
        else:
            # 没有验证集也可以简单保存一份
            ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": embedding_dim,
                },
                ckpt_path,
            )
            print(f"[Epoch {epoch}] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

