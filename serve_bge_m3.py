from typing import List, Literal
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel

# 让 Torch 少抢 CPU
torch.set_num_threads(1)

MODEL_ID = "/opt/bge-m3/models/bge-m3"
# 临时强制使用 CPU（RTX 5090 需要 PyTorch 2.6+ 才能支持）
DEVICE = "cpu"  # 原: "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# 从本地路径加载模型权重（优先使用 safetensors 格式）
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE, local_files_only=True)
model.to(DEVICE)
model.eval()

app = FastAPI(title="BGE-M3 Embedding Service (PyTorch dense only)")

class EmbedRequest(BaseModel):
    texts: List[str]
    output_type: Literal["dense", "sparse", "colbert"] = "dense"
    normalize: bool = True
    max_length: int = 512
    batch_size: int = 32

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, T, H]
    # attention_mask:    [B, T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                   # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                         # [B, 1]
    return summed / counts

@torch.inference_mode()
def encode_dense(texts: List[str], max_length: int, normalize: bool, batch_size: int) -> List[List[float]]:
    all_vecs: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        out = model(**inputs, return_dict=True)
        vecs = mean_pooling(out.last_hidden_state, inputs["attention_mask"])

        if normalize:
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)

        all_vecs.extend(vecs.detach().float().cpu().tolist())

    return all_vecs

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """
    输入：文本列表
    输出：dense 向量列表（只支持 PyTorch dense）
    """
    if req.output_type != "dense":
        raise NotImplementedError("当前服务仅提供 PyTorch dense embedding；sparse/colbert 请单独实现。")

    vecs = encode_dense(
        texts=req.texts,
        max_length=req.max_length,
        normalize=req.normalize,
        batch_size=req.batch_size,
    )
    return EmbedResponse(embeddings=vecs)

