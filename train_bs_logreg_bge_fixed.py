#!/usr/bin/env python3
"""
train_bs_logreg_bge_fixed.py

修复版：直接使用 transformers 而非 FlagEmbedding（避免版本兼容问题）

功能：
- 使用 BGE-M3 作为冻结的 embedding 提取器
- 在其上训练一个逻辑回归头（线性层 + Sigmoid）做「BS 相关？」二分类
- 在验证集上自动扫描阈值，推荐 T_low / T_high
"""

import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ==================== 配置区域 ====================

DATA_PATH = "data/bs_train.jsonl"
SAVE_DIR = "checkpoints_bs"
MODEL_PATH = "/opt/bge-m3/models/bge-m3"

os.makedirs(SAVE_DIR, exist_ok=True)

VALID_RATIO = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 1e-3
TARGET_RECALL_FOR_T_LOW = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}, Dtype: {DTYPE}")

# ==================== 数据集定义 ====================

class BSSegmentDataset(Dataset):
    """每条样本：text: str, label: int (0/1)"""
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
        print(f"  Positive: {sum(self.labels)}, Negative: {len(self.labels) - sum(self.labels)}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return texts, labels

# ==================== BGE-M3 编码器 ====================

class BGEM3Encoder:
    """封装 BGE-M3 的 Dense 编码功能"""
    def __init__(self, model_path: str, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)

        print(f"Loading model from {model_path}...")
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
        self.model.to(device)
        self.model.eval()

        # 获取 embedding 维度
        self.embedding_dim = self.model.config.hidden_size
        print(f"Model loaded. Embedding dim = {self.embedding_dim}")

    def mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean Pooling"""
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.inference_mode()
    def encode(self, texts: List[str], max_length: int = 512, normalize: bool = True) -> torch.Tensor:
        """编码文本为向量"""
        inputs = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model(**inputs, return_dict=True)
        vecs = self.mean_pooling(out.last_hidden_state, inputs["attention_mask"])

        if normalize:
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)

        return vecs  # [batch, dim] on device

# ==================== 模型：逻辑回归头 ====================

class BSLogisticRegression(nn.Module):
    """简单逻辑回归：emb -> Linear -> logit"""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        logit = self.linear(emb)
        return logit.squeeze(-1)

# ==================== 指标 & 阈值搜索 ====================

def precision_recall_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    """给定阈值，计算 P/R/F1"""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.9):
    """
    自动推荐：
      - T_high：最大 F1 对应的阈值
      - T_low ：在 Recall >= target_recall 条件下，Precision 最好的阈值
    """
    thresholds = np.linspace(0.05, 0.95, num=19)

    best_f1 = -1.0
    best_t_high = 0.5
    best_high_metrics = (0.0, 0.0, 0.0)

    best_t_low = 0.1
    best_low_metrics = (0.0, 0.0, 0.0)
    best_recall_for_low = -1.0

    for t in thresholds:
        p, r, f1 = precision_recall_f1(y_true, y_prob, t)
        if f1 > best_f1:
            best_f1 = f1
            best_t_high = t
            best_high_metrics = (p, r, f1)
        if r >= target_recall:
            if r > best_recall_for_low or (abs(r - best_recall_for_low) < 1e-6 and p > best_low_metrics[0]):
                best_recall_for_low = r
                best_t_low = t
                best_low_metrics = (p, r, f1)
        else:
            if best_recall_for_low < 0 and r > best_recall_for_low:
                best_recall_for_low = r
                best_t_low = t
                best_low_metrics = (p, r, f1)

    if best_t_low > best_t_high:
        best_t_low = min(best_t_low, best_t_high)
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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn) if valid_ds else None

    # 2) 加载 BGE-M3
    print("\n" + "="*60)
    print("Loading BGE-M3 model...")
    print("="*60)
    bge_encoder = BGEM3Encoder(MODEL_PATH, DEVICE, DTYPE)
    embedding_dim = bge_encoder.embedding_dim

    # 3) 定义逻辑回归头
    model = BSLogisticRegression(embedding_dim=embedding_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"\nLogistic Head: {sum(p.numel() for p in model.parameters())} parameters")

    # 4) 训练循环
    print("\n" + "="*60)
    print("Training...")
    print("="*60)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - train")
        for texts, labels in pbar:
            # BGE-M3 提取 embedding (冻结)
            emb = bge_encoder.encode(texts, max_length=512, normalize=True)
            emb = emb.float()  # 确保 float32 用于训练

            labels = labels.to(DEVICE)

            # 前向 + loss
            logits = model(emb)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(num_samples, 1)
        print(f"\n[Epoch {epoch}] Train loss = {avg_loss:.4f}")

        # 5) 验证 + 阈值推荐
        if valid_loader is not None:
            model.eval()
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for texts, labels in tqdm(valid_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - valid"):
                    emb = bge_encoder.encode(texts, max_length=512, normalize=True)
                    emb = emb.float()
                    logits = model(emb)
                    probs = torch.sigmoid(logits)

                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(labels.numpy())

            y_prob = np.concatenate(all_probs, axis=0)
            y_true = np.concatenate(all_labels, axis=0).astype(int)

            # 基本指标
            p_05, r_05, f1_05 = precision_recall_f1(y_true, y_prob, threshold=0.5)
            print(f"\n{'─'*60}")
            print(f"[Epoch {epoch}] Valid @0.50: precision={p_05:.4f}, recall={r_05:.4f}, f1={f1_05:.4f}")

            # 自动找 T_low / T_high
            T_low, T_high, (p_low, r_low, f1_low), (p_high, r_high, f1_high) = \
                find_best_thresholds(y_true, y_prob, target_recall=TARGET_RECALL_FOR_T_LOW)

            print(f"[Epoch {epoch}] Recommended T_low={T_low:.2f}  (P={p_low:.4f}, R={r_low:.4f}, F1={f1_low:.4f})")
            print(f"[Epoch {epoch}] Recommended T_high={T_high:.2f} (P={p_high:.4f}, R={r_high:.4f}, F1={f1_high:.4f})")
            print(f"{'─'*60}\n")

            # 6) 保存 checkpoint
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
            ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch}.pt")
            torch.save(
                {"model_state_dict": model.state_dict(), "embedding_dim": embedding_dim},
                ckpt_path,
            )
            print(f"[Epoch {epoch}] Saved checkpoint to {ckpt_path}")

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    main()
