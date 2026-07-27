"""
AnomalyNCD 命令行推理脚本

用法:
    python inference.py --image /path/to/image.png
    python inference.py --image /path/to/image.png --mask /path/to/musc_map.png
    python inference.py --image_dir /path/to/images/
    python inference.py --checkpoint /path/to/model.pt --image /path/to/image.png

依赖:
    python inference.py [options]
"""

import os
import sys
import argparse
import time

import numpy as np
import cv2
from PIL import Image

from shared_utils.inference_utils import load_model, predict_single


def main():
    parser = argparse.ArgumentParser(
        description="AnomalyNCD 命令行推理工具（prototype-based 余弦相似度分类）"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型权重路径，默认自动扫描 outputs 目录下最新的 checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="单张图片路径",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="批量推理：遍历文件夹内所有图片（png/jpg/jpeg）",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="（可选）MuSc anomaly map 路径，不传则用全 1 占位",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="训练集路径（用于计算 prototype），默认使用内置配置",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备，默认 cuda",
    )

    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("必须指定 --image 或 --image_dir")

    # 1. 加载模型 + 计算 prototype（只执行一次）
    print("=" * 60)
    print("加载模型 ...")
    t0 = time.time()
    state = load_model(checkpoint_path=args.checkpoint)
    print(f"模型加载耗时: {(time.time() - t0) * 1000:.1f} ms")
    print(f"checkpoint: {state.checkpoint_path}")
    print(f"类别数: {len(state.class_names)} (labeled={state.num_labeled_classes}, unlabeled={state.num_unlabeled_classes})")
    print(f"类别: {state.class_names}")
    print("=" * 60)

    # 2. 收集待推理图片列表
    image_paths = []
    if args.image:
        if not os.path.isfile(args.image):
            print(f"错误: 图片不存在: {args.image}")
            sys.exit(1)
        image_paths.append(args.image)
    if args.image_dir:
        if not os.path.isdir(args.image_dir):
            print(f"错误: 目录不存在: {args.image_dir}")
            sys.exit(1)
        for fname in sorted(os.listdir(args.image_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(args.image_dir, fname))
        if not image_paths:
            print(f"错误: 目录 {args.image_dir} 中没有 png/jpg/jpeg 图片")
            sys.exit(1)

    # 3. 加载 mask（可选）
    mask_array = None
    if args.mask:
        if not os.path.isfile(args.mask):
            print(f"错误: mask 文件不存在: {args.mask}")
            sys.exit(1)
        mask_array = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask_array is None:
            print(f"错误: 无法解码 mask 图片: {args.mask}")
            sys.exit(1)

    # 4. 逐张推理
    for img_path in image_paths:
        print()
        print(f"--- 推理: {img_path} ---")
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"  跳过: 无法解码 {img_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        result = predict_single(state, image_pil, mask_array=mask_array)

        print(f"预测类别: {result['predicted_class']}")
        print(f"是否异常: {result['is_anomaly']}")
        print(f"异常分数: {result['image_score']}")
        print(f"各类余弦相似度:")
        for c in state.class_names:
            print(f"  {c}: {result['cosine_similarities'][c]}")
        print(f"推理耗时: {result['elapsed_ms']} ms")


if __name__ == "__main__":
    main()