"""
inference_utils.py — 模型加载、prototype 计算、单张推理的公共函数。

api.py 和 inference.py 均 import 此模块，避免重复实现。
"""

import os
import sys
import types
import glob
import re
import time

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# 将原始代码路径加入 sys.path（如已存在则跳过）
ORIGINAL_CODE_PATH = "/public/home/dongshuai_ht/AnomalyNCD-main-0.749"
if ORIGINAL_CODE_PATH not in sys.path:
    sys.path.insert(0, ORIGINAL_CODE_PATH)

from models.AnomalyNCD import AnomalyNCD
from utils.general_utils import load_yaml
from examples.anomalyncd_main import load_args
from datasets.data_utils import get_class_splits
from datasets.transform import get_transform
from datasets.dataset import Dataset_AnomalyNCD


# ============================================================
# args 构建
# ============================================================
def build_args_for_service():
    args = types.SimpleNamespace()

    args.dataset = "mvtec"
    args.category = "bottle"
    args.seed = 42
    args.dataset_path = "/public/home/dongshuai_ht/datasets/mvtec_ad/mvt_ad"
    args.anomaly_map_path = "/public/home/dongshuai_ht/datasets/musc_mvtec_anomaly_map"
    args.binary_data_path = "/public/home/dongshuai_ht/AnomalyNCD-main-0.749/outputs/mvtec_musc"
    args.crop_data_path = "/public/home/dongshuai_ht/AnomalyNCD-main-0.749/outputs/mvtec_musc_crop"
    args.base_data_path = "/public/home/dongshuai_ht/datasets/AeBAD_crop"

    args.config = os.path.join(ORIGINAL_CODE_PATH, "configs", "AnomalyNCD.yaml")
    args.runner_name = "AnomalyNCD_service"
    args.only_test = None
    args.checkpoint_path = None

    cfg = load_yaml(args.config)
    load_args(cfg, args)

    return args


# ============================================================
# checkpoint 扫描与 seed 提取
# ============================================================
def find_latest_checkpoint(category, outputs_dir):
    pattern = os.path.join(outputs_dir, f"*{category}*", "**", "model.pt")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for category '{category}' in {outputs_dir}"
        )
    latest = max(matches, key=os.path.getmtime)
    return latest


def extract_seed_from_path(checkpoint_path):
    m = re.search(r"seed(\d+)", checkpoint_path)
    if m:
        return int(m.group(1))
    return None


# ============================================================
# InferenceState — 装载所有推理所需的状态
# ============================================================
class InferenceState:
    def __init__(self):
        self.model = None
        self.args = None
        self.test_transform = None
        self.device = None
        self.checkpoint_path = None
        self.class_names = []
        self.num_labeled_classes = 0
        self.num_unlabeled_classes = 0
        self.prototypes = {}


# ============================================================
# 模型加载 + prototype 计算（api.py 和 inference.py 共用）
# ============================================================
def load_model(checkpoint_path=None):
    """
    加载 AnomalyNCD 模型并计算 prototype。

    参数:
        checkpoint_path: 可选，手动指定 checkpoint 路径。为 None 时自动扫描。

    返回:
        InferenceState 实例
    """
    state = InferenceState()
    args = build_args_for_service()

    # 切换到原始代码路径以加载配置
    original_cwd = os.getcwd()
    os.chdir(ORIGINAL_CODE_PATH)

    try:
        print(f"[load_model] args构建完成: dataset={args.dataset}, category={args.category}")

        # 1. 扫描或使用指定 checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(args.category, os.path.join(ORIGINAL_CODE_PATH, "outputs"))
        print(f"[load_model] checkpoint: {checkpoint_path}")

        # 2. 提取 seed
        seed = extract_seed_from_path(checkpoint_path)
        if seed is not None:
            args.seed = seed
            print(f"[load_model] seed: {seed}")

        # 3. 类别划分
        args = get_class_splits(args)
        print(
            f"[load_model] 类别划分: labeled={args.train_classes} ({len(args.train_classes)}), "
            f"unlabeled={args.unlabeled_classes} ({len(args.unlabeled_classes)})"
        )

        # 4. 实例化 AnomalyNCD
        anomaly_ncd = AnomalyNCD(args)
        anomaly_ncd.load_datasets = lambda: (None, None)
        anomaly_ncd.train_init()

        # 5. 加载 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=anomaly_ncd.device)
        missing, unexpected = anomaly_ncd.model.load_state_dict(
            checkpoint["model"], strict=False
        )
        print(f"[load_model] missing_keys: {missing}")
        print(f"[load_model] unexpected_keys: {unexpected}")
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint key不匹配: missing={missing}, unexpected={unexpected}"
            )

        # 6. 自检
        dummy_img = torch.zeros(1, 3, 224, 224).to(anomaly_ncd.device)
        dummy_mask = torch.ones(1, 1, 224, 224).to(anomaly_ncd.device)
        MGViT, projector = anomaly_ncd.model
        with torch.no_grad():
            cls_token = MGViT(dummy_img, dummy_mask)
            _, logits = projector(cls_token)
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        assert logits[0].shape == (1, num_classes), (
            f"Logits shape mismatch: {logits[0].shape} vs (1, {num_classes})"
        )
        print(f"[load_model] 自检通过: logits shape={logits[0].shape}")

        # 7. test_transform
        _, test_transform = get_transform(image_size=args.image_size, args=args)

        # 8. 计算 prototype
        print("[load_model] 开始计算 prototype ...")
        whole_dataset = Dataset_AnomalyNCD(
            source=args.crop_data_path,
            base_path=args.base_data_path,
            novel_class=args.category,
            transform=test_transform,
        )
        whole_dataset.target_transform = lambda x: 0
        class_names = list(args.train_classes) + list(args.unlabeled_classes)
        feat_dim = anomaly_ncd.model[0].embed_dim
        proto_sum = {c: torch.zeros(feat_dim).to(anomaly_ncd.device) for c in class_names}
        proto_count = {c: 0 for c in class_names}
        print(f"[load_model] 数据集样本数: {len(whole_dataset)}")
        MGViT_proto = anomaly_ncd.model[0]
        for i in range(len(whole_dataset)):
            img_tensor, mask_tensor = whole_dataset[i][0], whole_dataset[i][4]
            anomaly_label = whole_dataset.data_to_iterate[i][1]
            img_tensor = img_tensor.unsqueeze(0).to(anomaly_ncd.device)
            mask_tensor = mask_tensor.unsqueeze(0).to(anomaly_ncd.device)
            with torch.no_grad():
                cls_token = MGViT_proto(img_tensor, mask_tensor)
            cls_token = cls_token.squeeze(0)
            proto_sum[anomaly_label] += cls_token
            proto_count[anomaly_label] += 1
        prototypes = {}
        for c in class_names:
            if proto_count[c] > 0:
                proto = proto_sum[c] / proto_count[c]
                proto = proto / proto.norm(p=2, dim=-1, keepdim=True)
            else:
                proto = torch.zeros(feat_dim).to(anomaly_ncd.device)
            prototypes[c] = proto
            print(f"[load_model] prototype[{c}] count={proto_count[c]} norm={proto.norm():.4f}")
        print("[load_model] prototype 计算完成。")

        # 9. 填充 state
        state.model = anomaly_ncd.model
        state.args = args
        state.test_transform = test_transform
        state.device = anomaly_ncd.device
        state.checkpoint_path = checkpoint_path
        state.class_names = class_names
        state.num_labeled_classes = args.num_labeled_classes
        state.num_unlabeled_classes = args.num_unlabeled_classes
        state.prototypes = prototypes

    finally:
        os.chdir(original_cwd)

    return state


# ============================================================
# 单张推理（api.py 和 inference.py 共用）
# ============================================================
def predict_single(state, image_pil, mask_array=None):
    """
    对单张图片进行推理。

    参数:
        state:       InferenceState 实例
        image_pil:   PIL.Image，RGB 格式
        mask_array:  可选，numpy 2D array (H, W)，float32 [0, 1]，连续值 anomaly map

    返回:
        dict: {
            "predicted_class": str,
            "is_anomaly": bool,
            "image_score": float,
            "class_probabilities": dict,
            "cosine_similarities": dict,
            "elapsed_ms": float,
        }
    """
    t0 = time.time()

    # 1. 预处理
    image_tensor, _ = state.test_transform(image_pil, mask=None)
    image_tensor = image_tensor.unsqueeze(0).to(state.device)

    # 2. 构造 mask
    if mask_array is not None:
        mask_gray = cv2.resize(mask_array, (224, 224), interpolation=cv2.INTER_LINEAR)
        mask_gray = mask_gray.astype(np.float32) / 255.0
        mask = torch.from_numpy(mask_gray).unsqueeze(0).unsqueeze(0).to(state.device)
    else:
        mask = torch.ones(1, 1, 224, 224).to(state.device)

    # 3. 提取 CLS token
    MGViT, projector = state.model
    with torch.no_grad():
        cls_token = MGViT(image_tensor, mask)
        _, logits = projector(cls_token)

    cls_token = cls_token.squeeze(0)
    cls_token_norm = cls_token / (cls_token.norm(p=2, dim=-1, keepdim=True) + 1e-8)

    # 4. 余弦相似度
    sims = {}
    for c in state.class_names:
        proto = state.prototypes[c]
        sim = torch.dot(cls_token_norm, proto).item()
        sims[c] = sim

    sim_tensor = torch.tensor([sims[c] for c in state.class_names])
    sof_sim = F.softmax(sim_tensor * 10.0, dim=-1)

    class_probabilities = {
        state.class_names[i]: round(sof_sim[i].item(), 6)
        for i in range(len(state.class_names))
    }

    # 5. image_score
    n_labeled = state.num_labeled_classes
    sof_anomaly = sof_sim.clone()
    sof_anomaly[:n_labeled] = 0
    total = sof_anomaly[n_labeled:].sum()
    if total > 0:
        sof_anomaly[n_labeled:] /= total
    image_score = sof_anomaly[n_labeled:].max().item()

    # 6. 预测
    predicted_idx = sof_sim.argmax(dim=-1).item()
    predicted_class = state.class_names[predicted_idx]

    elapsed_ms = (time.time() - t0) * 1000.0

    return {
        "predicted_class": predicted_class,
        "is_anomaly": image_score > 0.5,
        "image_score": round(image_score, 6),
        "class_probabilities": class_probabilities,
        "cosine_similarities": {c: round(sims[c], 6) for c in state.class_names},
        "elapsed_ms": round(elapsed_ms, 1),
    }