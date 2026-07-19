# PROJECT_DEFINE — AnomalyNCD

> 生成时间：2026-07-08
> 基于代码实际内容，不凭空编造。

---

## 1. 解决什么问题

**AnomalyNCD（Novel Anomaly Class Discovery）** 解决的是工业场景下的**多类异常分类**问题。具体来说：

- 给定一批来自上游异常检测（AD）方法输出的异常分数图（anomaly map），以及对应的原始图像，AnomalyNCD 需要自动将图像中的异常区域划分为不同的异常类别（如裂纹、划痕、凹陷等），而**不需要这些类别的标签**。
- 这是一个**新类发现（Novel Class Discovery, NCD）**问题：训练时，有一组已标注的异常类别（labeled/base data，来自 AeBAD 数据集），以及一组完全未标注的新异常类别（unlabeled/novel data，来自 MVTec AD 或 MTD 数据集）。模型需要在已标注数据的辅助下，自动发现并聚类新数据中的异常类别。
- 核心挑战：
  1. **异常不显著（non-prominence）**：异常区域在图中可能很小、很弱，难以被模型关注到。
  2. **弱语义（weak semantics）**：不同类型的异常在视觉上差异很小，区分难度大。

**代码入口**：[examples/anomalyncd_main.py](file:///public/home/dongshuai_ht/AnomalyNCD-main-0.749/examples/anomalyncd_main.py) → [models/AnomalyNCD.py](file:///public/home/dongshuai_ht/AnomalyNCD-main-0.749/models/AnomalyNCD.py)

---

## 2. 输入输出规格

### 2.1 输入

有两类输入数据：

| 数据角色 | 说明 | 格式 |
|---------|------|------|
| 原始图像 | 待检测的工业产品图像 | PNG/JPG，尺寸不固定（MVTec AD 900×900 左右，MTD 不固定） |
| 异常分数图 | 由上游任意 AD 方法产生的灰度图，每个像素值表示异常程度 | PNG，单通道灰度图，尺寸应与原始图像对应 |
| 已标注基类数据 | AeBAD 数据集裁剪后的子图，含 4 个已知异常类别（breakdown, ablation, fracture, groove） | PNG，裁剪后尺寸不固定，带 mask |
| 配置文件 | YAML 格式的超参数配置 | 见 [configs/AnomalyNCD.yaml](file:///public/home/dongshuai_ht/AnomalyNCD-main-0.749/configs/AnomalyNCD.yaml) |

**关键路径参数**（从 `scripts/anomalyncd.sh` 中可见）：

```
--dataset_path      原始图像路径，结构如 mvtec_anomaly_detection/{category}/test/{anomaly_type}/
--anomaly_map_path  异常分数图路径，结构如 mvtec_musc_anomaly_map/{category}/{anomaly_type}/
--base_data_path    已标注基类路径，如 AeBAD_crop/
--binary_data_path  MEBin 二值化输出路径
--crop_data_path    MEBin 裁剪输出路径
--category          待处理的单一产品类别名（如 bottle, cable 等）
```

**预处理后的模型输入规格**：

| 阶段 | 张量尺寸 | 说明 |
|------|---------|------|
| MEBin 二值化 | 原始图像尺寸 | 自适应阈值将异常图转为二值 mask |
| 子图裁剪 | 不固定 → resize 到 224×224 | 根据二值 mask 的连通域裁剪出异常子图，padding=10%，min_crop_size=10% |
| 模型输入 | `[B, 3, 224, 224]` 图像 + `[B, 1, 224, 224]` mask | 同时输入 RGB 图像和对应的二值 mask |
| 训练时 | 每张子图生成 2 个视图（n_views=2），尺寸仍为 224×224 | 随机裁剪 + 数据增强（颜色抖动、翻转、旋转、高斯模糊等） |

### 2.2 输出

| 输出项 | 类型 | 说明 |
|--------|------|------|
| 子图级聚类结果 | `numpy array (N,)` | 每个子图被分配到一个异常类别 ID（从 0 到 num_labeled_classes + num_unlabeled_classes - 1） |
| 图像级分类结果 | `numpy array (M,)` | 通过 region merging 策略，将同一原图的多个子图预测合并为整张图的最终类别 |
| 评估指标 | 浮点数 | NMI、ARI、F1（micro-averaged），写入 `metrics.csv` 和日志 |
| 模型 checkpoint | `model.pt` | 包含模型权重、优化器状态、epoch、loss_list 等 |
| t-SNE 可视化 | PNG 图片 | 自动生成特征分布的 t-SNE 图 |
| 中间产物 | 二值 mask 图、裁剪子图、裁剪 mask | 保存在 `binary_data_path` 和 `crop_data_path` 下 |

**输出文件路径结构**（从代码中 `init_experiment` 可见）：

```
outputs/
  {run_exp}/
    log/
      AnomalyNCD_{category}_(YYYY.MM.DD_HH:MM)/
        log.txt
        checkpoints/
          model.pt
    metrics.csv
```

---

## 3. 成功标准

### 3.1 论文指标（来自 README.md 中的论文结果表）

模型在 **MVTec AD** 和 **MTD** 两个数据集上评估，核心指标为：

| 指标 | 全称 | 含义 | 论文最佳值 (MVTec AD) |
|------|------|------|----------------------|
| **NMI** | Normalized Mutual Information | 聚类结果与真实标签的归一化互信息 | 0.736 (CPR+AnomalyNCD) |
| **ARI** | Adjusted Rand Index | 聚类结果与真实标签的调整兰德指数 | 0.674 (CPR+AnomalyNCD) |
| **F1** | Micro F1 Score | 聚类结果的微观 F1 分数 | 0.805 (CPR+AnomalyNCD) |
| **AUPRO** | Area Under Per-Region-Overlap | 像素级异常定位指标 | 0.964 (CPR+AnomalyNCD) |

**无监督设置**（仅用 AeBAD 作为 labeled data，novel data 无任何标签）：
- MVTec AD: NMI=0.613, ARI=0.526, F1=0.712 (MuSc+AnomalyNCD)
- MTD: NMI=0.268, ARI=0.228, F1=0.509 (MuSc+AnomalyNCD)

**半监督设置**（AD 方法本身提供了一些异常信息）：
- MVTec AD: NMI=0.736, ARI=0.674, F1=0.805 (CPR+AnomalyNCD)
- MTD: NMI=0.421, ARI=0.390, F1=0.617 (PatchCore+AnomalyNCD)

### 3.2 工程指标（从代码中推断）

| 指标 | 值 | 来源 |
|------|-----|------|
| 训练 epoch | 50 | 配置文件 `training.epochs` |
| 单类别训练时间 | 约 30 分钟 | 从 `final_results_3760.txt` 中日志时间戳推断（每个 task 约 30 min） |
| 模型参数量 | DINO ViT-Base: ~86M | 标准 ViT-B/8 |
| 显存需求 | 单卡 GPU，batch_size=32，224×224 输入 | 约 8-12 GB |
| 推理模式 | 支持 `--only_test` 独立推理 | 脚本 `anomalyncd_test.sh` |
| 可复现性 | 固定全局种子 `seed` | `set_seed()` 函数设置 random/numpy/torch/cudnn 全部种子 |
| 数据处理前置时间 | MEBin 二值化 + 裁剪（一次性） | 取决于数据集大小，约几分钟 |

---

## 4. 失败后果

### 4.1 如果模型表现不佳（聚类失败）

- **下游任务完全不可用**：AnomalyNCD 的输出是异常类别的聚类结果。如果聚类失败（NMI/ARI 接近 0），意味着模型无法区分不同类型的异常，所有异常被混为一谈，失去了"多类异常分类"的意义。
- **实际工业场景影响**：在产线质检中，不同异常类型需要不同的处理方式（如裂纹→报废，划痕→返工）。如果分类失败，无法指导后续处置决策。
- **论文场景**：指标不达标，无法支撑论文结论。

### 4.2 如果 MEBin 二值化失败

- 如果异常图的二值化阈值计算错误（如 `get_threshold` 找不到稳定区间），会返回 threshold=255, est_anomaly_num=0，导致裁剪出整张图作为一张子图，失去区域聚焦效果。
- 如果 `erode=True` 在 MTD 等细裂纹场景下使用，腐蚀操作会直接擦除细小缺陷，导致下游聚类完全失效。代码中已有消融实验验证：`ablation_erode_true_seed123` 的 NMI=0.405 vs `ablation_erode_false_seed123` 的 NMI=0.417。

### 4.3 如果输入路径配置错误

- `dataset_path` 和 `anomaly_map_path` 中文件名不一致（如原始图是 `.jpg`，异常图是 `.png`，或 basename 不匹配），代码通过 basename 匹配会跳过不匹配的文件，导致数据量减少。
- `base_data_path`（已标注基类 AeBAD）缺失会导致无监督训练变成纯无监督，失去伪标签校正能力。

### 4.4 如果部署到生产环境

- 当前代码无 ONNX/TensorRT 导出、无 REST API、无 Docker 化，直接部署不可行。
- 推理需要先运行 MEBin 预处理（CPU 密集型），再运行模型推理（GPU），整体延迟不适合实时产线。

---

## 5. 项目类型

**论文型**（兼顾少量工程适配）

判定依据：

- **核心目标**：复现论文 "AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios"（CVPR 2025）的实验结果。
- **代码结构**：以实验脚本驱动（`scripts/anomalyncd.sh`），通过命令行参数配置数据集路径和类别，逐个类别循环训练。
- **输出**：以 NMI/ARI/F1 等学术指标为核心，写入 `metrics.csv` 和日志文件，并自动生成 t-SNE 可视化。
- **工程化程度**：有基本的配置文件体系（YAML）、checkpoint 保存/加载、独立推理模式（`--only_test`），但无服务化封装、无 API 接口、无模型导出。
- **消融实验**：outputs 中包含 `ablation_erode_false` 和 `ablation_erode_true` 的消融实验记录，进一步印证论文实验性质。

---

## 6. 是否支持批量推理

**是，但有限制。**

- **训练脚本** `scripts/anomalyncd.sh`：通过 for 循环遍历所有类别（如 `bottle cable capsule ...`），逐类别串行训练。每个类别内部使用 `DataLoader` 进行批量推理，batch_size=32。
- **独立推理脚本** `scripts/anomalyncd_test.sh`：支持 `--only_test True` 参数，加载已训练的 checkpoint 进行推理。同样通过 for 循环遍历类别，每个类别批量推理。
- **限制**：
  - 每次只能处理一个类别（`--category` 参数），无法同时推理多个产品类别。
  - 推理前需要先完成 MEBin 二值化和子图裁剪预处理（一次性，代码会在 `main()` 中自动调用 `binarization()`）。
  - 子图级推理（`sub_image_predict`）和图像级推理（`region_merge_predict`）是分开的两个步骤，图像级推理需要将同一原图的多个子图预测合并。

---

## 7. 是否考虑部署

**否，当前代码未考虑部署。**

- 无模型导出功能（ONNX、TorchScript、TensorRT）。
- 无服务化封装（无 REST API、无 gRPC）。
- 无 Dockerfile 或容器化配置。
- 推理流程依赖 MEBin 预处理（CPU 密集的 OpenCV 操作），无法端到端 GPU 推理。
- 代码中硬编码了路径结构和文件系统操作（如 `os.makedirs`、`shutil.rmtree`），不适合生产环境。
- 配置文件中明确标注了"实验版本说明"和"回滚指南"，进一步表明这是实验代码而非生产代码。

**如需部署，需要额外工作**：
1. 将 MEBin 预处理逻辑封装为可在线调用的模块。
2. 导出模型为 ONNX/TensorRT 格式。
3. 封装推理 API（如 FastAPI）。
4. 处理多类别并发推理。
5. 添加健康检查、日志、监控等生产级能力。