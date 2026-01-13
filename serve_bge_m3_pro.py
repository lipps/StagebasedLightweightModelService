"""
BGE-M3 嵌入服务 - 生产级优化版本
特性：错误处理、日志、监控、健康检查、性能优化
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Literal, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, AutoModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from src.semantic_matcher import SemanticMatcher

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

    @classmethod
    def validate(cls):
        """验证配置 (启动时快速失败)"""
        # 验证批次大小
        if cls.MAX_BATCH_SIZE < 1:
            raise ValueError(f"MAX_BATCH_SIZE 必须 >= 1，当前值: {cls.MAX_BATCH_SIZE}")
        if cls.MAX_BATCH_SIZE > 1024:
            raise ValueError(f"MAX_BATCH_SIZE 不应超过 1024，当前值: {cls.MAX_BATCH_SIZE}")

        # 验证最大长度
        if cls.MAX_LENGTH < 1:
            raise ValueError(f"MAX_LENGTH 必须 >= 1，当前值: {cls.MAX_LENGTH}")
        if cls.MAX_LENGTH > 8192:
            raise ValueError(f"MAX_LENGTH 不应超过 8192 (模型限制)，当前值: {cls.MAX_LENGTH}")

        # 验证模型路径
        if not os.path.exists(cls.MODEL_ID):
            raise FileNotFoundError(f"模型路径不存在: {cls.MODEL_ID}")

        # 验证模型文件
        has_safetensors = os.path.exists(os.path.join(cls.MODEL_ID, "model.safetensors"))
        has_pytorch = os.path.exists(os.path.join(cls.MODEL_ID, "pytorch_model.bin"))
        if not (has_safetensors or has_pytorch):
            raise FileNotFoundError(
                f"模型权重文件不存在: {cls.MODEL_ID}/model.safetensors 或 pytorch_model.bin"
            )

        # 验证分词器文件
        if not os.path.exists(os.path.join(cls.MODEL_ID, "tokenizer_config.json")):
            raise FileNotFoundError(f"分词器配置不存在: {cls.MODEL_ID}/tokenizer_config.json")

        # 验证设备配置
        device = cls.get_device()
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("设备配置为 CUDA，但 CUDA 不可用")

        logger.info(f"✅ 配置验证通过: device={device}, batch_size={cls.MAX_BATCH_SIZE}, max_length={cls.MAX_LENGTH}")


# ==================== 日志配置 ====================

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== Prometheus 指标 ====================

# 请求计数器
EMBED_REQUESTS_TOTAL = Counter(
    'embed_requests_total',
    'Total number of embed requests',
    ['status']  # success, error
)

# 请求延迟直方图
EMBED_LATENCY_SECONDS = Histogram(
    'embed_latency_seconds',
    'Embed request latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# 文本数量直方图
EMBED_TEXT_COUNT = Histogram(
    'embed_text_count',
    'Number of texts per request',
    buckets=[1, 5, 10, 20, 50, 100, 200, 500, 1000]
)

# 模型就绪状态
MODEL_READY_GAUGE = Gauge(
    'model_ready',
    'Whether the model is ready (1=ready, 0=not ready)'
)


# ==================== 全局状态 ====================

class IModelManager(ABC):
    """模型管理器抽象接口 (遵循依赖倒置原则)"""

    @abstractmethod
    def encode(
        self,
        texts: List[str],
        max_length: int = 512,
        normalize: bool = True,
        batch_size: int = 32
    ) -> List[List[float]]:
        """编码文本为向量"""
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """模型是否就绪"""
        pass


class ModelManager(IModelManager):
    """模型管理器实现"""
    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self.device: Optional[str] = None
        self.dtype: Optional[torch.dtype] = None
        self.matcher: Optional[SemanticMatcher] = None
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """模型是否就绪"""
        return self._is_ready

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
                dtype=self.dtype,
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时 {load_time:.2f}s")

            # 加载语义匹配引擎 (如果索引存在)
            index_path = "data/model_index.pt"
            if os.path.exists(index_path):
                try:
                    self.matcher = SemanticMatcher(index_path, device=self.device)
                    logger.info("语义匹配引擎加载完成")
                except Exception as e:
                    logger.error(f"语义匹配引擎加载失败: {e}")

            # 设置就绪标志 (在预热前设置,避免预热时检查失败)
            self._is_ready = True
            MODEL_READY_GAUGE.set(1)  # 更新 Prometheus 指标

            # 模型预热
            if Config.ENABLE_WARMUP:
                self.warmup()

        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            MODEL_READY_GAUGE.set(0)  # 更新 Prometheus 指标
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

        self._is_ready = False
        MODEL_READY_GAUGE.set(0)  # 更新 Prometheus 指标
        logger.info("模型已卸载")


# 全局模型管理器实例
model_manager = ModelManager()


# ==================== 依赖注入 ====================

def get_model_manager() -> IModelManager:
    """获取模型管理器实例 (依赖注入)"""
    return model_manager


# ==================== FastAPI 生命周期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时验证配置
    logger.info("应用启动中...")
    Config.validate()

    # 加载模型
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


class MatchRequest(BaseModel):
    """话术匹配请求"""
    text: str = Field(..., description="用户输入的提问或话术")
    top_k: int = Field(default=3, ge=1, le=10, description="返回候选结果数量")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")


class MatchResponse(BaseModel):
    """话术匹配响应"""
    best_match: Optional[dict] = None
    candidates: List[dict] = []
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
async def embed(
    req: EmbedRequest,
    manager: IModelManager = Depends(get_model_manager)
):
    """
    文本向量嵌入

    - **texts**: 文本列表（最多 1000 条）
    - **normalize**: 是否 L2 归一化（默认 True）
    - **max_length**: 最大序列长度（1-8192，默认 512）
    - **batch_size**: 批处理大小（1-128，默认 32）

    返回 1024 维向量
    """
    if not manager.is_ready:
        raise HTTPException(
            status_code=503,
            detail="模型未就绪，请稍后重试"
        )

    start_time = time.time()

    # 记录文本数量
    EMBED_TEXT_COUNT.observe(len(req.texts))

    try:
        # 编码
        with EMBED_LATENCY_SECONDS.time():
            embeddings = manager.encode(
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

        # 记录成功请求
        EMBED_REQUESTS_TOTAL.labels(status='success').inc()

        return EmbedResponse(
            embeddings=embeddings,
            count=count,
            dim=dim,
            time_ms=round(time_ms, 2)
        )

    except torch.cuda.OutOfMemoryError:
        logger.error("GPU 内存不足")
        EMBED_REQUESTS_TOTAL.labels(status='error').inc()  # 记录失败请求
        raise HTTPException(
            status_code=500,
            detail="GPU 内存不足，请减小 batch_size 或 max_length"
        )
    except Exception as e:
        logger.error(f"嵌入失败: {e}", exc_info=True)
        EMBED_REQUESTS_TOTAL.labels(status='error').inc()  # 记录失败请求
        raise HTTPException(
            status_code=500,
            detail=f"嵌入失败: {str(e)}"
        )


@app.post("/match", response_model=MatchResponse)
async def match(
    req: MatchRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    话术/意图匹配 (语义相似度)

    - **text**: 用户输入的文本
    - **top_k**: 返回候选数量 (默认 3)
    - **threshold**: 相似度阈值 (默认 0.7)

    基于向量余弦相似度在标准话术库中进行检索。
    """
    if not manager.is_ready:
        raise HTTPException(status_code=503, detail="模型未就绪")
    
    if not manager.matcher:
        raise HTTPException(status_code=501, detail="语义索引未加载，请检查 data/model_index.pt")

    start_time = time.time()

    try:
        # 1. 对用户输入进行编码 (使用现有的 encode 方法，取第一条结果)
        embeddings = manager.encode(
            texts=[req.text],
            max_length=512,
            normalize=True,
            batch_size=1
        )
        query_vec = torch.tensor(embeddings[0])

        # 2. 在向量库中检索
        results = manager.matcher.search(
            query_vector=query_vec,
            top_k=req.top_k,
            threshold=req.threshold
        )

        time_ms = (time.time() - start_time) * 1000
        
        best_match = results[0] if results else None
        
        return MatchResponse(
            best_match=best_match,
            candidates=results,
            time_ms=round(time_ms, 2)
        )

    except Exception as e:
        logger.error(f"匹配失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"匹配失败: {str(e)}")


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


@app.get("/metrics")
async def metrics():
    """
    Prometheus 指标端点

    暴露以下指标:
    - embed_requests_total: 请求总数 (按状态分类)
    - embed_latency_seconds: 请求延迟直方图
    - embed_text_count: 每次请求的文本数量直方图
    - model_ready: 模型就绪状态 (1=就绪, 0=未就绪)
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


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
