"""
AnomalyNCD FastAPI 推理服务

启动方式:
    conda run -n anomaly_ncd --no-capture-output uvicorn api:app --host 0.0.0.0 --port 8000

接口:
    POST /predict  — 上传图片（可选 mask），返回分类推理结果（prototype-based）
    GET  /health   — 健康检查
"""

import traceback
from contextlib import asynccontextmanager

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from shared_utils.inference_utils import load_model, predict_single, InferenceState


# ============================================================
# 模型加载（lifespan 生命周期）
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] 开始加载模型 ...")
    state = load_model()
    print("[startup] 模型加载完成，挂载到 app.state ...")

    app.state.model = state.model
    app.state.args = state.args
    app.state.test_transform = state.test_transform
    app.state.device = state.device
    app.state.checkpoint_path = state.checkpoint_path
    app.state.class_names = state.class_names
    app.state.num_labeled_classes = state.num_labeled_classes
    app.state.num_unlabeled_classes = state.num_unlabeled_classes
    app.state.prototypes = state.prototypes
    app.state.model_loaded = True
    print("[startup] 服务就绪。")

    yield

    print("[shutdown] 服务关闭")


app = FastAPI(
    title="AnomalyNCD Service",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# API 路由
# ============================================================
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": app.state.model_loaded,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), mask_file: UploadFile = File(None)):
    try:
        if not app.state.model_loaded:
            raise HTTPException(
                status_code=503,
                detail="模型未加载，请检查配置文件和数据路径。",
            )

        # 1. 读取上传图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(
                status_code=400, detail="无法解码图片，请上传有效的 JPEG/PNG 文件。"
            )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # 2. 读取 mask（可选）
        mask_source = "all_ones_placeholder"
        mask_array = None
        if mask_file is not None:
            mask_contents = await mask_file.read()
            mask_nparr = np.frombuffer(mask_contents, np.uint8)
            mask_gray = cv2.imdecode(mask_nparr, cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                raise HTTPException(
                    status_code=400, detail="无法解码 mask 图片，请上传有效的 JPEG/PNG 文件。"
                )
            mask_array = mask_gray
            mask_source = "user_provided"

        # 3. 构建临时 InferenceState 供 predict_single 使用
        state = InferenceState()
        state.model = app.state.model
        state.test_transform = app.state.test_transform
        state.device = app.state.device
        state.class_names = app.state.class_names
        state.num_labeled_classes = app.state.num_labeled_classes
        state.num_unlabeled_classes = app.state.num_unlabeled_classes
        state.prototypes = app.state.prototypes

        # 4. 推理
        result = predict_single(state, image_pil, mask_array=mask_array)

        note = (
            "已使用用户提供的 anomaly map 作为 mask，分类基于 prototype 余弦相似度"
            if mask_source == "user_provided"
            else "未提供外部 anomaly map，mask 为全1占位，分类基于 prototype 余弦相似度"
        )

        return JSONResponse(
            content={
                "filename": file.filename,
                "image_score": result["image_score"],
                "predicted_class": result["predicted_class"],
                "is_anomaly": result["is_anomaly"],
                "class_probabilities": result["class_probabilities"],
                "loaded_checkpoint": app.state.checkpoint_path,
                "mask_source": mask_source,
                "note": note,
                "cosine_similarities": result["cosine_similarities"],
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[predict] error: {tb}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb},
        )


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
    )