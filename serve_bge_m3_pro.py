"""
BGE-M3 嵌入服务 - 生产级优化版本
特性：错误处理、日志、监控、健康检查、性能优化
"""

import os
import time
import logging
from typing import List, Literal, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, AutoModel

# ==================== 配置管理 ====================

class Config:
    """服务配置"""
    MODEL_ID = os.getenv("MODEL_PATH", "/opt/bge-m3/models/bge-m3")
    DEVICE = os.getenv("DEVICE", "auto")  # auto, cpu, cuda
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "128"))
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "8192"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # 性能配置
    ENABLE_WARMUP = os.getenv("ENABLE_WARMUP", "true").lower() == "true"
    CPU_THREADS = int(os.getenv("CPU_THREADS", "0"))  # 0 = auto

    @classmethod
    def get_device(cls) -> str:
        """智能设备选择"""
        if cls.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return cls.DEVICE

    @classmethod
    def get_dtype(cls) -> torch.dtype:
        """根据设备选择精度"""
        device = cls.get_device()
        return torch.float16 if device == "cuda" else torch.float32

    @classmethod
    def setup_cpu_threads(cls):
        """配置 CPU 线程数"""
        device = cls.get_device()
        if device == "cpu":
            if cls.CPU_THREADS > 0:
                threads = cls.CPU_THREADS
            else:
                threads = min(os.cpu_count() or 4, 8)
            torch.set_num_threads(threads)
            return threads
        else:
            torch.set_num_threads(1)
            return 1


# ==================== 日志配置 ====================

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== 全局状态 ====================

class ModelManager:
    """模型管理器"""
    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self.device: Optional[str] = None
        self.dtype: Optional[torch.dtype] = None
        self.is_ready = False

    def load(self):
        """加载模型"""
        if self.is_ready:
            logger.warning("模型已加载，跳过重复加载")
            return

        try:
            logger.info("开始加载模型...")
            start_time = time.time()

            # 设备和精度配置
            self.device = Config.get_device()
            self.dtype = Config.get_dtype()
            cpu_threads = Config.setup_cpu_threads()

            logger.info(f"设备配置: device={self.device}, dtype={self.dtype}, cpu_threads={cpu_threads}")

            # 加载分词器
            logger.info(f"加载分词器: {Config.MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.MODEL_ID,
                use_fast=True,
                local_files_only=True
            )

            # 加载模型
            logger.info(f"加载模型权重: {Config.MODEL_ID}")
            self.model = AutoModel.from_pretrained(
                Config.MODEL_ID,
                torch_dtype=self.dtype,
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时 {load_time:.2f}s")

            # 模型预热
            if Config.ENABLE_WARMUP:
                self.warmup()

            self.is_ready = True

        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            raise RuntimeError(f"模型加载失败: {e}")

    def warmup(self):
        """模型预热"""
        logger.info("开始模型预热...")
        try:
            dummy_texts = ["这是一个测试文本用于预热模型"] * 4
            _ = self.encode(dummy_texts, max_length=128, normalize=True, batch_size=4)
            logger.info("模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")

    def mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean Pooling 聚合"""
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.inference_mode()
    def encode(
        self,
        texts: List[str],
        max_length: int = 512,
        normalize: bool = True,
        batch_size: int = 32
    ) -> List[List[float]]:
        """编码文本为向量"""
        if not self.is_ready:
            raise RuntimeError("模型未加载或加载失败")

        all_vecs = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_idx = i // batch_size + 1
            batch = texts[i:i + batch_size]

            try:
                # 分词
                inputs = self.tokenizer(
                    batch,
                    padding="longest",  # 动态填充
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 模型推理
                out = self.model(**inputs, return_dict=True)
                vecs = self.mean_pooling(out.last_hidden_state, inputs["attention_mask"])

                # 归一化
                if normalize:
                    vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)

                # 转换为列表
                all_vecs.extend(vecs.detach().float().cpu().tolist())

                logger.debug(f"批次 {batch_idx}/{total_batches} 完成，大小 {len(batch)}")

            except Exception as e:
                logger.error(f"批次 {batch_idx} 推理失败: {e}")
                raise

        return all_vecs

    def unload(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_ready = False
        logger.info("模型已卸载")


# 全局模型管理器实例
model_manager = ModelManager()


# ==================== FastAPI 生命周期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    logger.info("应用启动中...")
    model_manager.load()
    logger.info("应用启动完成")

    yield

    # 关闭时卸载模型
    logger.info("应用关闭中...")
    model_manager.unload()
    logger.info("应用已关闭")


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="BGE-M3 Embedding Service (Pro)",
    description="生产级 BGE-M3 向量嵌入服务",
    version="2.0.0",
    lifespan=lifespan
)


# ==================== 数据模型 ====================

class EmbedRequest(BaseModel):
    """嵌入请求"""
    texts: List[str] = Field(..., description="待编码的文本列表")
    normalize: bool = Field(default=True, description="是否 L2 归一化")
    max_length: int = Field(default=512, ge=1, le=Config.MAX_LENGTH, description="最大序列长度")
    batch_size: int = Field(default=32, ge=1, le=Config.MAX_BATCH_SIZE, description="批处理大小")

    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("texts 不能为空")
        if len(v) > 1000:
            raise ValueError(f"单次请求最多 1000 条文本，当前 {len(v)} 条")
        for idx, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"texts[{idx}] 必须是字符串，当前类型 {type(text)}")
            if not text.strip():
                raise ValueError(f"texts[{idx}] 不能为空字符串")
        return v


class EmbedResponse(BaseModel):
    """嵌入响应"""
    embeddings: List[List[float]]
    count: int
    dim: int
    time_ms: float


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    device: str
    torch_version: str
    cuda_available: bool


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None


# ==================== 中间件 ====================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()

    # 记录请求
    logger.info(f"收到请求: {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000

        # 记录响应
        logger.info(
            f"请求完成: {request.method} {request.url.path} "
            f"状态={response.status_code} 耗时={process_time:.2f}ms"
        )

        # 添加响应头
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        return response

    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            f"请求失败: {request.method} {request.url.path} "
            f"错误={str(e)} 耗时={process_time:.2f}ms",
            exc_info=True
        )
        raise


# ==================== 异常处理 ====================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """处理参数验证错误"""
    logger.warning(f"参数验证失败: {exc}")
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="参数验证失败",
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """处理运行时错误"""
    logger.error(f"运行时错误: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="服务内部错误",
            detail=str(exc)
        ).dict()
    )


# ==================== API 端点 ====================

@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {
        "service": "BGE-M3 Embedding Service",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if model_manager.is_ready else "unhealthy",
        model_loaded=model_manager.is_ready,
        device=model_manager.device or "unknown",
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available()
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    """
    文本向量嵌入

    - **texts**: 文本列表（最多 1000 条）
    - **normalize**: 是否 L2 归一化（默认 True）
    - **max_length**: 最大序列长度（1-8192，默认 512）
    - **batch_size**: 批处理大小（1-128，默认 32）

    返回 1024 维向量
    """
    if not model_manager.is_ready:
        raise HTTPException(
            status_code=503,
            detail="模型未就绪，请稍后重试"
        )

    start_time = time.time()

    try:
        # 编码
        embeddings = model_manager.encode(
            texts=req.texts,
            max_length=req.max_length,
            normalize=req.normalize,
            batch_size=req.batch_size
        )

        # 计算耗时
        time_ms = (time.time() - start_time) * 1000

        # 统计信息
        count = len(embeddings)
        dim = len(embeddings[0]) if embeddings else 0

        logger.info(
            f"嵌入完成: count={count}, dim={dim}, "
            f"time={time_ms:.2f}ms, avg={time_ms/count:.2f}ms/text"
        )

        return EmbedResponse(
            embeddings=embeddings,
            count=count,
            dim=dim,
            time_ms=round(time_ms, 2)
        )

    except torch.cuda.OutOfMemoryError:
        logger.error("GPU 内存不足")
        raise HTTPException(
            status_code=500,
            detail="GPU 内存不足，请减小 batch_size 或 max_length"
        )
    except Exception as e:
        logger.error(f"嵌入失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"嵌入失败: {str(e)}"
        )


@app.get("/stats", response_model=dict)
async def get_stats():
    """获取服务统计信息"""
    stats = {
        "model_ready": model_manager.is_ready,
        "device": model_manager.device,
        "dtype": str(model_manager.dtype),
        "max_batch_size": Config.MAX_BATCH_SIZE,
        "max_length": Config.MAX_LENGTH,
    }

    if torch.cuda.is_available() and model_manager.device == "cuda":
        stats.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
        })

    return stats


# ==================== 主程序 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve_bge_m3_pro:app",
        host="0.0.0.0",
        port=8001,
        workers=1,
        log_level="info"
    )
