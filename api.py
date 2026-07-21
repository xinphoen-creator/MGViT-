"""
AnomalyNCD FastAPI 推理服务

启动方式:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --host 0.0.0.0 --port 8000

接口:
    POST /predict  — 上传图片，返回异常检测结果
    GET  /health   — 健康检查
"""

import time
import io
import os
import traceback
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ============================================================
# TODO: 以下 import 在实际集成时需要取消注释并调整路径
# ============================================================
# import torch
# import torch.nn.functional as F
# import cv2
# from models.AnomalyNCD import AnomalyNCD
# from models.modules._MEBin import MEBin
# from utils.general_utils import load_yaml

# ============================================================
# 全局配置
# ============================================================
# TODO: 实际使用时从环境变量或配置文件读取
MODEL_CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", "outputs/model.pt")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/AnomalyNCD.yaml")
DEVICE = "cuda:0"

# 类别名称映射（基于 AnomalyNCD 的 num_labeled_classes + num_unlabeled_classes）
# TODO: 实际使用时根据训练时的类别顺序填写
CATEGORY_NAMES = {
    0: "breakdown",
    1: "ablation",
    2: "fracture",
    3: "groove",
    # novel classes — 需要根据具体数据集填写
    # 4: "crack",
    # 5: "scratch",
    # ...
}


# ============================================================
# 模型加载（lifespan 生命周期）
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务启动时加载模型，关闭时释放资源。
    """
    # --- startup ---
    print("[startup] 加载模型...")
    # TODO: 实际加载逻辑
    # cfg = load_yaml(CONFIG_PATH)
    # args = load_args(cfg)
    # model = AnomalyNCD(args)
    # model.train_init()
    # checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
    # model.model.load_state_dict(checkpoint["model"])
    # model.model.eval()
    # app.state.model = model

    app.state.model = None  # TODO: 替换为实际模型实例
    app.state.ready = True
    print("[startup] 模型加载完成，服务就绪。")

    yield  # --- 服务运行中 ---

    # --- shutdown ---
    print("[shutdown] 释放模型资源...")
    app.state.model = None
    app.state.ready = False


app = FastAPI(
    title="AnomalyNCD Inference API",
    description="工业场景新异常类别发现推理服务",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# 辅助函数
# ============================================================
async def validate_image(file: UploadFile) -> Image.Image:
    """
    验证上传文件是否为有效图片，返回 PIL Image（RGB）。
    失败时抛出 HTTPException(400)。
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file.content_type}，请上传图片文件。",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="图片文件损坏或格式不支持，无法打开。",
        )

    return image


def preprocess_image(image: Image.Image):
    """
    预处理图片为模型输入格式。

    参数:
        image: PIL RGB Image

    返回:
        TODO: 返回预处理后的张量
    """
    # TODO: 实现 MEBin 二值化 + 子图裁剪 + resize 224×224
    # 参考原始代码 models/AnomalyNCD.py 中的 binarization() 和 load_datasets()
    # 步骤:
    #   1. 将 PIL Image 转为 numpy array (H, W, 3)
    #   2. 调用 MEBin.binarize_anomaly_maps() 生成二值 mask
    #   3. 调用 MEBin.crop_sub_image_mask() 裁剪异常子图
    #   4. 每个子图 resize 到 224×224 → 转为 tensor [N, 3, 224, 224]
    #   5. 对应的 mask 也 resize 到 224×224 → 转为 tensor [N, 1, 224, 224]
    #   6. 如果没有检测到异常子图，返回空或整张图作为 fallback
    raise NotImplementedError("TODO: 实现 MEBin 预处理逻辑")


def run_inference(model, image_tensor, mask_tensor):
    """
    模型推理。

    参数:
        model:  AnomalyNCD 模型实例
        image_tensor: [N, 3, 224, 224] 图像张量
        mask_tensor:  [N, 1, 224, 224] mask 张量

    返回:
        dict: {"is_anomaly": bool, "anomaly_score": float, "category": str}
    """
    # TODO: 实现推理逻辑
    # 参考原始代码 models/AnomalyNCD.py 中的 sub_image_predict() 和 region_merge_predict()
    # 步骤:
    #   1. model.model: MGViT(image_tensor, mask_tensor) → cls_token
    #   2. projector(cls_token) → logits (n_head 个输出)
    #   3. softmax(logits) → 取 labeled_classes 之后的概率作为 novel class 概率
    #   4. 取 argmax 得到类别 ID
    #   5. 如果有多个子图，使用 region_merge 策略合并为整图结果
    raise NotImplementedError("TODO: 实现模型推理逻辑")


# ============================================================
# API 路由
# ============================================================
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok" if app.state.ready else "not ready",
        "model_loaded": app.state.model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    异常检测推理接口。

    接收:
        - file: 上传的图片文件（支持 PNG / JPG / JPEG）

    返回:
        {
            "is_anomaly": bool,       # 是否检测到异常
            "anomaly_score": float,   # 异常分数 [0, 1]，越高越异常
            "category": str,          # 异常类别名称（正常时为 "normal"）
            "time_ms": float          # 推理耗时（毫秒）
        }
    """
    t_start = time.time()

    # ---- 1. 校验图片 ----
    try:
        image = await validate_image(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片处理失败: {str(e)}")

    # ---- 2. 预处理 ----
    try:
        # TODO: 取消注释并实现 preprocess_image
        # image_tensor, mask_tensor = preprocess_image(image)
        raise NotImplementedError("TODO: 预处理逻辑未实现")
    except NotImplementedError:
        raise HTTPException(
            status_code=500,
            detail="预处理逻辑尚未实现，请完成 preprocess_image() 中的 TODO。",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预处理失败: {str(e)}")

    # ---- 3. 模型推理 ----
    try:
        # TODO: 取消注释并实现 run_inference
        # result = run_inference(app.state.model, image_tensor, mask_tensor)
        raise NotImplementedError("TODO: 推理逻辑未实现")
    except NotImplementedError:
        raise HTTPException(
            status_code=500,
            detail="推理逻辑尚未实现，请完成 run_inference() 中的 TODO。",
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")

    # ---- 4. 构造响应 ----
    time_ms = (time.time() - t_start) * 1000

    return JSONResponse(
        content={
            "is_anomaly": False,      # TODO: 替换为 result["is_anomaly"]
            "anomaly_score": 0.0,     # TODO: 替换为 result["anomaly_score"]
            "category": "normal",     # TODO: 替换为 result["category"]
            "time_ms": round(time_ms, 2),
        }
    )


# ============================================================
# 本地调试入口
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )