"""
AnomalyNCD 训练入口脚本

用法:
    python train.py --config configs/AnomalyNCD.yaml --gpu 0 --output_dir outputs/

说明:
    本脚本是对 AnomalyNCD-main-0.749/examples/anomalyncd_main.py 的薄封装。
    内部直接调用原仓库的 get_args() / load_args() / AnomalyNCD.main()，
    新增 --gpu 和 --output_dir 两个便捷参数。
"""

import os
import sys
import argparse

# 将原始代码路径加入 sys.path 并切换工作目录（原仓库所有路径均为相对路径）
ORIGINAL_CODE_PATH = "/public/home/dongshuai_ht/AnomalyNCD-main-0.749"
sys.path.insert(0, ORIGINAL_CODE_PATH)
os.chdir(ORIGINAL_CODE_PATH)

from examples.anomalyncd_main import get_args, load_args, set_seed
from models.AnomalyNCD import AnomalyNCD
from utils.general_utils import load_yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="AnomalyNCD 训练入口（封装 anomalyncd_main.py）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/AnomalyNCD.yaml",
        help="配置文件路径（相对于原仓库根目录）",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU 编号，默认 0。多卡用逗号分隔，如 0,1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="输出目录，会覆盖 config 中的 exp_root",
    )
    # 直接透传原仓库的剩余参数
    parser.add_argument(
        "--dataset", type=str, default="mvtec",
    )
    parser.add_argument(
        "--category", type=str, default="bottle",
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None,
    )
    parser.add_argument(
        "--anomaly_map_path", type=str, default=None,
    )
    parser.add_argument(
        "--binary_data_path", type=str, default=None,
    )
    parser.add_argument(
        "--crop_data_path", type=str, default=None,
    )
    parser.add_argument(
        "--base_data_path", type=str, default=None,
    )
    parser.add_argument(
        "--seed", type=int, default=None,
    )
    parser.add_argument(
        "--only_test", type=str, default=None,
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
    )
    parser.add_argument(
        "--runner_name", type=str, default="AnomalyNCD",
    )

    return parser.parse_args()


def main():
    # 1. 解析命令行参数
    wrapper_args = parse_args()

    # 2. 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = wrapper_args.gpu
    print(f"[train] CUDA_VISIBLE_DEVICES = {wrapper_args.gpu}")

    # 3. 构造 sys.argv 以透传给原仓库的 get_args()
    #    原仓库 get_args() 会调用 parser.parse_args() 读取 sys.argv，
    #    所以我们把参数拼成 argv 列表推给它。
    saved_argv = sys.argv[:]
    sys.argv = [
        sys.argv[0],
        "--config", wrapper_args.config,
        "--dataset", wrapper_args.dataset,
        "--category", wrapper_args.category,
    ]
    if wrapper_args.dataset_path:
        sys.argv += ["--dataset_path", wrapper_args.dataset_path]
    if wrapper_args.anomaly_map_path:
        sys.argv += ["--anomaly_map_path", wrapper_args.anomaly_map_path]
    if wrapper_args.binary_data_path:
        sys.argv += ["--binary_data_path", wrapper_args.binary_data_path]
    if wrapper_args.crop_data_path:
        sys.argv += ["--crop_data_path", wrapper_args.crop_data_path]
    if wrapper_args.base_data_path:
        sys.argv += ["--base_data_path", wrapper_args.base_data_path]
    if wrapper_args.seed is not None:
        sys.argv += ["--seed", str(wrapper_args.seed)]
    if wrapper_args.only_test:
        sys.argv += ["--only_test", wrapper_args.only_test]
    if wrapper_args.checkpoint_path:
        sys.argv += ["--checkpoint_path", wrapper_args.checkpoint_path]
    sys.argv += ["--runner_name", wrapper_args.runner_name]

    try:
        # 4. 调用原仓库的 get_args + load_args
        args = get_args()
        cfg = load_yaml(args.config)
        args = load_args(cfg, args)

        # 5. 用 --output_dir 覆盖 exp_root
        args.exp_root = wrapper_args.output_dir
        print(f"[train] exp_root = {args.exp_root}")

        # 6. 固定种子
        set_seed(args.seed)
        print(f"[train] seed = {args.seed}")

        # 7. 初始化模型并开始训练
        model = AnomalyNCD(args)
        model.main()

    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()