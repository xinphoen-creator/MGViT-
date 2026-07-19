# AnomalyNCD — 工业场景新异常类别发现

> 论文：AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios (CVPR 2025)
> 项目类型：论文型（兼顾少量工程适配）

---

## 1. 项目干什么

AnomalyNCD 解决的是工业场景下的**多类异常分类**问题——给定上游异常检测（AD）方法输出的异常分数图，自动将异常区域划分为不同类别（如裂纹、划痕、凹陷等），**无需人工标注异常类别**。

这是一个**新类发现（Novel Class Discovery, NCD）**任务：训练时有一组已知类别的异常（来自 AeBAD 数据集），模型需要在此基础上自动发现并聚类全新的异常类型。

**核心创新点**：

| 创新点 | 说明 |
|--------|------|
| **MEBin**（Main Element Binarization） | 自适应阈值将异常图二值化，再按连通域裁剪出异常子图，实现区域聚焦 |
| **MGViT**（Mask-Guided Vision Transformer） | 在 DINO ViT-B/8 基础上引入 mask-guided attention，让模型关注异常区域而非背景 |
| **区域合并策略**（Region Merging） | 将同一原图的多个子图预测合并为整张图的最终分类结果 |
| **兼容主流 AD 检测器** | 上游可使用 MuSc、CPR、PatchCore 等任意异常检测方法产生的异常图 |

**核心挑战**：异常不显著（区域小、弱）+ 弱语义（不同异常视觉差异小）。

---

## 2. 怎么安装环境

### 2.1 创建 conda 环境

```bash
conda create -n anomaly_ncd python=3.9
conda activate anomaly_ncd
```

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

### 2.3 关键依赖版本

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | 2.2.2 | 深度学习框架 |
| torchvision | 0.17.2 | 图像变换 + 预训练模型 |
| numpy | 1.24.4 | 数值计算 |
| opencv-python | 4.9.0.80 | 图像处理（MEBin 二值化、连通域分析） |
| scikit-learn | 1.3.2 | 聚类评估（NMI/ARI）+ t-SNE |
| scipy | 1.15.3 | 匈牙利匹配（linear_sum_assignment） |
| matplotlib | 3.7.5 | t-SNE 可视化 |
| pillow | 12.2.0 | 图像读写 |
| tqdm | 4.66.2 | 进度条 |
| PyYAML | 6.0.1 | 配置文件解析 |

**硬件要求**：单卡 GPU，显存建议 ≥ 12 GB（batch_size=32，224×224 输入）。

---

## 3. 怎么下载数据

### 3.1 所需数据集

| 数据集 | 用途 | 获取方式 |
|--------|------|----------|
| **MVTec AD** | 主力评估数据集，15 个产品类别 | [官网](https://www.mvtec.com/company/research/datasets/mvtec-ad) 下载 |
| **MTD**（Magnetic Tile Defect） | 磁瓦缺陷数据集，6 个缺陷类别 | [GitHub](https://github.com/abin24/Magnetic-tile-defect-datasets.) 下载 |
| **AeBAD** | 已标注基类数据（4 个类别：breakdown, ablation, fracture, groove） | 见论文 *Industrial anomaly detection with domain shift* (Computers in Industry, 2023)，需联系原作者获取|

### 3.2 异常分数图

需要先用上游 AD 方法（如 MuSc、CPR、PatchCore）对数据集生成异常分数图。代码中使用的默认路径为 `mvtec_musc_anomaly_map` 和 `mtd_musc_anomaly_map`。

### 3.3 目录结构要求

将数据按以下结构放在 `data/` 目录下：

```
data/
├── mvtec_anomaly_detection/          # MVTec AD 原始图像
│   └── {category}/                   # 如 bottle, cable, capsule ...
│       └── test/
│           └── {anomaly_type}/       # 如 broken_large, contamination ...
│               └── *.png
├── mvtec_musc_anomaly_map/           # 上游 AD 生成的异常分数图
│   └── {category}/
│       └── {anomaly_type}/
│           └── *.png
├── AeBAD_crop/                       # 已标注基类裁剪子图
│   └── {class_name}/                 # breakdown, ablation, fracture, groove
│       └── *.png
├── mtd_anomaly_detection/            # MTD 原始图像（可选）
│   └── test/
│       └── MTD/
│           └── {defect_type}/
│               └── *.jpg
└── mtd_musc_anomaly_map/             # MTD 异常分数图（可选）
    └── MTD/
        └── {defect_type}/
            └── *.png
```

**注意**：原始图像和异常分数图的文件名 basename 必须一致，否则代码会跳过不匹配的文件。

---

## 4. 怎么训练

### 4.1 配置文件

训练超参数由 YAML 配置文件控制，入口为 `configs/AnomalyNCD.yaml`。关键参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `binarization.sample_rate` | 4 | 下采样率 |
| `binarization.erode` | False | 是否腐蚀（MTD 细裂纹场景必须为 False） |
| `models.pretrained_backbone` | dino_vitb8 | DINO 预训练 ViT-Base，patch_size=8 |
| `models.n_views` | 2 | 每张图生成的对比学习视图数 |
| `models.n_head` | 4 | 聚类头数量 |
| `training.batch_size` | 32 | 批次大小 |
| `training.epochs` | 50 | 训练轮数 |
| `training.lr` | 0.003 | 初始学习率 |
| `loss.sup_weight` | 0.3 | 伪标签监督损失权重 |
| `loss.memax_weight` | 4 | 最大熵正则化权重（防止模型坍塌） |
| `loss.anomaly_thred` | 0.5 | 异常判定阈值 |

### 4.2 训练命令

**MVTec AD 数据集**（单类别训练示例）：

```bash
python examples/anomalyncd_main.py \
    --runner_name "mvtec_musc_crop" \
    --dataset "mvtec" \
    --category "bottle" \
    --dataset_path "data/mvtec_anomaly_detection" \
    --anomaly_map_path "data/mvtec_musc_anomaly_map" \
    --binary_data_path "data/mvtec_musc" \
    --crop_data_path "data/mvtec_musc_crop" \
    --base_data_path "data/AeBAD_crop"
```

**MVTec AD 全部 15 个类别**（批量训练）：

```bash
#!/bin/bash
gpu=0
categories=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" \
            "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")
for category in "${categories[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python examples/anomalyncd_main.py \
        --runner_name "mvtec_musc_crop" \
        --dataset "mvtec" \
        --category "$category" \
        --dataset_path "data/mvtec_anomaly_detection" \
        --anomaly_map_path "data/mvtec_musc_anomaly_map" \
        --binary_data_path "data/mvtec_musc" \
        --crop_data_path "data/mvtec_musc_crop" \
        --base_data_path "data/AeBAD_crop"
done
```

**MTD 数据集**：

```bash
python examples/anomalyncd_main.py \
    --runner_name "mtd_musc_crop" \
    --dataset "mtd" \
    --category "MTD" \
    --dataset_path "data/mtd_anomaly_detection/test" \
    --anomaly_map_path "data/mtd_musc_anomaly_map" \
    --binary_data_path "data/mtd_musc" \
    --crop_data_path "data/mtd_musc_crop" \
    --base_data_path "data/AeBAD_crop"
```

### 4.3 训练流程说明

1. **MEBin 预处理**：自动将异常图二值化、裁剪为 224×224 子图（一次性，保存在 `binary_data_path` 和 `crop_data_path`）
2. **模型训练**：50 个 epoch，使用 SGD 优化器 + CosineAnnealingLR，结合对比学习损失 + 蒸馏损失
3. **输出**：checkpoint 保存在 `outputs/{run_exp}/log/AnomalyNCD_{category}_*/checkpoints/model.pt`

---

## 5. 怎么推理

训练完成后，使用 `--only_test True` 参数加载 checkpoint 进行推理：

```bash
python examples/anomalyncd_main.py \
    --runner_name "mvtec_musc_crop" \
    --dataset "mvtec" \
    --category "bottle" \
    --dataset_path "data/mvtec_anomaly_detection" \
    --anomaly_map_path "data/mvtec_musc_anomaly_map" \
    --binary_data_path "data/mvtec_musc" \
    --crop_data_path "data/mvtec_musc_crop" \
    --base_data_path "data/AeBAD_crop" \
    --only_test True \
    --checkpoint_path "outputs/mvtec_musc_crop/log/AnomalyNCD_bottle_*/checkpoints"
```

推理分两步：
1. **子图级推理**（`sub_image_predict`）：对每个裁剪子图输出类别 ID
2. **图像级推理**（`region_merge_predict`）：将同一原图的多个子图预测合并为整张图的最终分类

---

## 6. 结果指标

### 6.1 评估指标说明

| 指标 | 全称 | 含义 |
|------|------|------|
| **NMI** | Normalized Mutual Information | 聚类结果与真实标签的归一化互信息 |
| **ARI** | Adjusted Rand Index | 聚类结果与真实标签的调整兰德指数 |
| **F1** | Micro F1 Score | 聚类结果的微观 F1 分数 |
| **AUPRO** | Area Under Per-Region-Overlap | 像素级异常定位指标 |

### 6.2 论文结果

**无监督设置**（仅用 AeBAD 作为 labeled data，novel data 无任何标签）：

| 数据集 | 方法 | NMI | ARI | F1 |
|--------|------|-----|-----|-----|
| MVTec AD | MuSc + AnomalyNCD | 0.613 | 0.526 | 0.712 |
| MTD | MuSc + AnomalyNCD | 0.268 | 0.228 | 0.509 |

**半监督设置**（AD 方法本身提供异常信息）：

| 数据集 | 方法 | NMI | ARI | F1 | AUPRO |
|--------|------|-----|-----|-----|-------|
| MVTec AD | CPR + AnomalyNCD | **0.736** | **0.674** | **0.805** | **0.964** |
| MTD | PatchCore + AnomalyNCD | 0.421 | 0.390 | 0.617 | 0.741 (PatchCore) |

### 6.3 工程指标

| 指标 | 值 |
|------|-----|
| 训练 epoch | 50 |
| 单类别训练时间 | 约 30 分钟 |
| 模型参数量 | ~86M（DINO ViT-B/8） |
| 显存需求 | 约 8-12 GB（batch_size=32） |
| 推理模式 | 支持 `--only_test` 独立推理 |
| 可复现性 | 固定全局种子 |

---

## 7. 常见报错怎么解决

### 7.1 CUDA Out of Memory

```
RuntimeError: CUDA out of memory.
```

**原因**：batch_size 太大或 GPU 显存不足。

**解决**：
- 在 `configs/AnomalyNCD.yaml` 中将 `training.batch_size` 从 32 降低到 16 或 8
- 确保训练时其他进程未占用 GPU 显存

### 7.2 cv2 / OpenCV 相关报错

```
ImportError: No module named cv2
```

或

```
cv2.error: ... is not a valid numeric value
```

**解决**：
- 确认 `opencv-python==4.9.0.80` 已安装：`pip show opencv-python`
- 如果版本不兼容，重新安装：`pip install opencv-python==4.9.0.80`

### 7.3 数据集路径错误

**现象**：训练时数据显示异常少，或日志中出现大量文件跳过。

**原因**：`dataset_path` 和 `anomaly_map_path` 中文件 basename 不匹配（如原始图是 `.jpg`，异常图是 `.png`），代码通过 basename 匹配会跳过不匹配的文件。

**解决**：
- 检查原始图像和异常图文件名是否一致（仅扩展名可不同）
- 检查目录结构是否符合第 3.3 节的要求
- 检查 `--category` 参数是否与实际目录名一致

### 7.4 MEBin 二值化失败

**现象**：threshold=255，est_anomaly_num=0，裁剪出整张图作为一张子图。

**原因**：`get_threshold` 找不到稳定区间，自适应阈值计算失败。

**解决**：
- 检查异常图是否正常（灰度图、像素值范围 0-255）
- 检查 `binarization.sample_rate` 和 `min_interval_len` 参数是否合理

### 7.5 erode 参数导致细小缺陷丢失

**现象**：MTD 数据集上聚类指标异常低（NMI 远低于预期）。

**原因**：`binarization.erode=True` 时，腐蚀操作会擦除 MTD 的细小裂纹（Crack）和磨损（Fray）。

**解决**：
- 在 `configs/AnomalyNCD.yaml` 中确保 `binarization.erode: False`
- 消融实验验证：`erode=False` 的 NMI=0.417 vs `erode=True` 的 NMI=0.405

### 7.6 已标注基类数据缺失

**现象**：训练正常进行但聚类效果差。

**原因**：`base_data_path`（AeBAD 数据）路径配置错误或数据缺失，导致无监督训练退化为纯无监督，失去伪标签校正能力。

**解决**：检查 `--base_data_path` 路径是否正确，AeBAD 数据是否包含 4 个类别子目录。

---

## 项目结构

```
AnomalyNCD-Project/
├── README.md               # 本文档
├── PROJECT_DEFINE.md       # 项目定义文档
├── requirements.txt        # Python 依赖
├── configs/                # 训练配置文件
├── models/                 # 模型代码
├── datasets/               # 数据处理
├── utils/                  # 工具函数
├── examples/               # 训练/推理入口
├── train.py                # 训练入口（待实现）
├── inference.py            # 推理入口（待实现）
└── api.py                  # API 服务（待实现）
```

---

## 引用

> AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios. CVPR 2025.