# 热成像吸烟检测系统使用说明

## 概述

本系统包含四个核心脚本，用于基于热成像和姿态估计的综合吸烟行为检测：

1. **姿态估计脚本** (`pose_est_calc.py`)：对图片执行人体姿态估计，计算关键点距离指标并输出 JSON 文件
2. **经验阈值探索脚本** (`empirical_smk_thres.py`)：基于姿态估计结果，搜索最优分类阈值
3. **热成像K值训练器** (`thermal_distance_trainer.py`)：训练热成像检测的最优K值（距离阈值倍数）
4. **综合吸烟行为检测器** (`smoking_detector.py`)：结合热成像和姿态估计进行综合吸烟检测

---

## 一、姿态估计脚本

### 脚本位置

`/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/pose_est_calc.py`

### 输入数据配置

训练数据的图片目录配置在脚本的 `if __name__ == '__main__':` 块中：

| 变量 | 路径 | 用途 |
|------|------|------|
| `test_img_path` | `/home/projects/dongguan/Github/mmpose/tests/data/smoking_v1/images/video101_task1_011.jpg` | 单张测试图片路径 |
| `test_img_dirs[0]` | `/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态危险` | **训练用的危险样本目录**（吸烟行为） |
| `test_img_dirs[1]` | `/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态正常` | **训练用的正常样本目录**（非吸烟行为） |

### 功能说明

1. 对单张图片或指定目录下的所有 `.jpg` 图片执行人体 pose 估计。
2. 基于关键点计算每个人体实例的以下指标：
   - `nose_ear`：`nose` 到 `left_ear` / `right_ear` 的最大欧氏距离。
   - `nose_ear_norm`：上述距离按图像宽高归一化后的最大值。
   - `nose_wrist`：`nose` 到 `left_wrist` / `right_wrist` 的最小欧氏距离。
   - `nose_wrist_norm`：上述距离按图像宽高归一化后的最小值。
   - `nonorm_ratio`：`nose_ear / nose_wrist`。
   - `norm_ratio`：`nose_ear_norm / nose_wrist_norm`。
3. 保留可视化输出：在 MMPose 生成的骨架图上叠加每个人的距离信息，并在图像左下角写入当前帧的全局风险结果。
4. 将运行参数、输出路径、时间戳以及每张图片的逐人统计结果统一写入 JSON 文件。

### 输出位置

- **可视化图片目录**：`outputs/warning_vis`
- **JSON 输出路径**：`outputs/jsons/pose_metrics_xxx.json`

### JSON 顶层结构

| 字段 | 说明 |
|------|------|
| `IS_DIR` | 当前是否为目录批处理模式 |
| `predicted_path` | 当前处理的目标路径 |
| `params` | 本次运行使用的阈值参数 |
| `outputs` | 输出目录或文件的绝对路径映射 |
| `timestamp` | 本次运行生成 JSON 的时间戳 |
| `infoes` | 逐图片统计结果，键为图片文件名 |

### 单张图片记录结构

```json
{
  "img_path": "test_img_path_1",
  "instances": [
    {
      "person_id": 0,
      "nose_ear": 2.35,
      "nose_ear_norm": 3.6,
      "nose_wrist": 1.2,
      "nose_wrist_norm": 2.4,
      "nonorm_ratio": 1.958333,
      "norm_ratio": 1.5,
      "skipped": false,
      "risk": true,
      "norm_risk": true
    }
  ],
  "global_risk": true,
  "global_norm_risk": true,
  "skip_persons": 0
}
```

### 跳过逻辑

当满足以下条件之一时，该人体实例会被跳过：
- 满足 `score > keypoints_score_threshold` 的关键点数量少于 `min_keypoints`
- `nose` 置信度不足
- 左右手腕置信度都不足
- 左右肩膀置信度都不足

被跳过的实例相关距离字段为 `null`，并补充 `skipped: true` 和 `skip_reason` 字段。

---

## 二、经验阈值探索脚本

### 脚本位置

`/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/empirical_smk_thres.py`

### 输入数据配置

输入的 JSON 文件路径配置在脚本开头：

| 变量 | 路径格式 | 用途 |
|------|----------|------|
| `NORMAL_JSON` | `outputs/jsons/pose_metrics_姿态正常_<timestamp>.json` | 正常样本的姿态估计结果 |
| `WARNING_JSON` | `outputs/jsons/pose_metrics_姿态危险_<timestamp>.json` | 危险样本的姿态估计结果 |

**注意**：上述 JSON 文件由 `pose_est_calc.py` 处理对应目录后自动生成，文件名会包含时间戳。

### 功能说明

1. 读取两个由 `pose_est_calc.py` 生成的 JSON 文件，分别作为 `normal` 与 `warning` 两类样本。
2. 从 JSON 的 `infoes -> [img_name] -> instances` 中提取未跳过人体实例的指定指标：
   - `nonorm_ratio`
   - `norm_ratio`
3. 针对每个指标分别搜索单阈值分类规则，使两类样本尽可能分布在阈值两侧，并以分类准确率作为最优标准。
4. 将最优阈值、分类方向、混淆矩阵、统计信息及样本明细输出到终端，并可保存为汇总 JSON 文件。

### 阈值搜索逻辑

1. 对每个指标收集 `normal` 与 `warning` 的全部有效数值。
2. 根据所有去重后的数值生成候选阈值：
   - 最小值左侧一个边界
   - 相邻数值的中点
   - 最大值右侧一个边界
3. 对每个候选阈值同时评估两种分类方向：
   - `warning_if_ge`：`value >= threshold` 判为 `warning`
   - `warning_if_le`：`value <= threshold` 判为 `warning`
4. 以 `accuracy` 最大作为最优结果；若准确率相同，优先保留 `warning_if_ge`。

### 混淆矩阵说明

行表示真实类别，列表示预测类别：

|            | pred_normal | pred_warning |
|------------|-------------|--------------|
| true_normal | tn          | fp           |
| true_warning | fn         | tp           |

### 输出位置

- **汇总 JSON**：`outputs/jsons/empirical_threshold_summary_nose_shoulder_v2.json`
- **终端输出**：每个指标的最优阈值、准确率、2×2 混淆矩阵和样本统计信息

---

## 三、热成像K值训练器

### 脚本位置

`/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/thermal_distance_trainer.py`

### 功能说明

1. 训练最优的K值（距离阈值倍数）
2. K值定义：以鼻子为圆心，K×锚点距离为半径的区域
3. 判断该区域内是否有≥100°C的像素
4. 使用两类数据（normal/smoking）训练最优K值
5. **支持NPY格式热成像数据**

### 锚点类型

| 锚点类型 | 说明 |
|----------|------|
| `nose_ear` | 使用鼻子到耳朵的最大距离作为锚点 |
| `nose_shoulder` | 使用鼻子到双肩中心点的距离作为锚点 |

### 判断逻辑

- 在鼻子为圆心，K×锚点距离为半径的圆形区域内
- 是否存在温度≥100°C的像素
- 存在则标记为有热源

### 支持的数据目录结构

```
dataset/
├── rgb_v2_fullbase_check/
│   └── train/
│       ├── 姿态正常/          # RGB图片（PNG格式）
│       ├── 姿态危险/          # RGB图片（PNG格式）
│       ├── 嘴巴附近无热源/    # RGB图片（PNG格式）
│       ├── 嘴巴附近有热源/    # RGB图片（PNG格式）
│       └── ...
└── thermal_v2_fullbase_check/
    └── train/
        ├── 姿态正常/          # 热成像数据（NPY格式）
        ├── 姿态危险/          # 热成像数据（NPY格式）
        ├── 嘴巴附近无热源/    # 热成像数据（NPY格式）
        ├── 嘴巴附近有热源/    # 热成像数据（NPY格式）
        └── ...
```

**数据对应关系**：
- RGB图像格式：PNG
- 热成像数据格式：NPY（NumPy数组）
- **文件名相同**（不含扩展名）：如 `img001.png` 对应 `img001.npy`

### 配置说明

在脚本的 `if __name__ == '__main__':` 块中配置路径：

```python
# 大文件夹路径
BASE_DATA_DIR = "/path/to/your/dataset"

# 姿态估计JSON路径（由pose_est_calc.py生成）
NORMAL_JSON = os.path.join(BASE_DATA_DIR, "outputs/jsons/pose_metrics_姿态正常_xxx.json")
WARNING_JSON = os.path.join(BASE_DATA_DIR, "outputs/jsons/pose_metrics_姿态危险_xxx.json")

# 热成像数据目录
THERMAL_DATA_DIR = os.path.join(BASE_DATA_DIR, "thermal_v2_fullbase_check")

# 输出模型路径
OUTPUT_PATH = "outputs/thermal_K_model.json"

# 锚点类型
ANCHOR_TYPE = 'nose_ear'
```

### 训练策略说明

**核心逻辑**：只有同时满足以下两个条件才判定为 smoking：

```
smoking = (存在于姿态危险目录) AND (热成像检测到热源)
```

**详细说明**：

| 条件 | 说明 |
|------|------|
| 存在于姿态危险目录 | 图片文件存在于 `姿态危险/` 目录中 |
| 热成像检测到热源 | 鼻子附近50像素范围内存在≥100°C的像素 |

**标签判定真值表**：

| 存在于姿态危险目录 | 热成像检测热源 | 最终标签 |
|:----------------:|:------------:|:--------:|
| ✅ | ✅ | **smoking** |
| ✅ | ❌ | normal |
| ❌ | ✅ | normal |
| ❌ | ❌ | normal |

**重要说明**：
- **一个图片可以同时存在于多个目录中**（例如：同一张图片可以同时在 `姿态危险/` 和 `嘴巴附近有热源/` 目录中）
- 代码会自动检查图片是否存在于 `姿态危险/` 目录中（通过检查对应NPY文件是否存在）
- 如果图片在多个目录中都存在，代码只会处理一次，避免重复计数

### 输出位置

- **模型文件**：`outputs/thermal_K_model.json`

---

## 四、综合吸烟行为检测器

### 脚本位置

`/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/smoking_detector.py`

### 功能说明

1. 结合热成像数据和RGB姿态估计进行吸烟行为检测
2. 热成像检测：在手腕为圆心、K×锚点距离为半径的区域内，是否有≥100°C的像素
3. 姿态估计：计算手腕到鼻子的距离是否在危险范围内
4. **综合判断**：只有当两个条件同时满足时才判断为吸烟

### 判断逻辑

| 条件 | 说明 |
|------|------|
| 热成像条件 | 手腕圆形区域内存在≥100°C的像素（使用训练得到的K值） |
| 姿态条件 | 距离比值 >= 距离阈值（从`empirical_smk_thres.py`的训练结果） |
| 最终判断 | 热成像 AND 姿态 |

### 输入数据格式

| 数据类型 | 格式要求 |
|----------|----------|
| `thermal_matrix` | numpy.ndarray (float32)，形状 (1520, 2688, 1)，像素值为摄氏度 |
| `rgb_frame` | numpy.ndarray (uint8)，形状 (1520, 2688, 3)，BGR格式 |
| `pose_keypoints` | 姿态估计关键点，包含 nose, left_wrist, right_wrist 等 |

### 输出JSON结构

```json
{
    "timestamp": "2026-05-20_10-30-00",
    "frame_id": "0000001",
    "thermal_detection": {
        "has_heat_source": true,
        "max_temp": 105.3,
        "K_value_used": 1.5,
        "radius_used": 45.2,
        "anchor_distance": 30.1
    },
    "pose_detection": {
        "nose_wrist_distance": 15.2,
        "anchor_distance": 30.1,
        "ratio": 0.427,
        "is_dangerous": true,
        "threshold_used": 0.42
    },
    "smoking判定": {
        "is_smoking": true,
        "confidence": "high",
        "reason": "热源检测+距离危险"
    }
}
```

---

## 五、数据流程

```
原始数据                              训练阶段                              检测阶段
─────────────────────────────────────────────────────────────────────────────────────
RGB图像目录 ──┐
             ├──→ pose_est_calc.py ──→ pose_metrics_xxx.json ──→ empirical_smk_thres.py ──→ pose_threshold.json
热成像数据 ──┐│
            │└──→ thermal_distance_trainer.py ──→ thermal_K_model.json
            │
            └──→ smoking_detector.py ←── pose_threshold.json + thermal_K_model.json
                                         ↓
                                    综合吸烟检测结果
```

**训练数据目录**：
- **RGB正常样本**：`/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态正常`
- **RGB危险样本**：`/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态危险`
- **热成像数据目录**：需与RGB图像对应

## 六、运行流程

### 步骤 1：执行姿态估计

```bash
python pose_est_calc.py
```

该脚本会自动处理配置的图片目录，生成姿态估计结果 JSON 文件和可视化图片。

### 步骤 2：探索经验阈值

```bash
python empirical_smk_thres.py
```

该脚本读取步骤 1 生成的 JSON 文件，搜索最优分类阈值。

**典型输出示例**：
```
===== nonorm_ratio =====
预测规则: value >= threshold -> warning, else normal
best_threshold = 1.850000
accuracy       = 93.333333%
confusion matrix (rows=ground truth, cols=prediction)
  \prediction   normal  warning
true normal        8        1
true warning       1       20
normal stats   = count:9  min:1.200000  max:2.000000  mean:1.600000
warning stats  = count:21  min:1.700000  max:2.500000  mean:2.100000
```

### 步骤 3：训练热成像K值

```bash
python thermal_distance_trainer.py
```

该脚本基于热成像数据和姿态估计结果，训练最优的K值。

**典型输出示例**：
```
===== K值训练结果 =====
最优K值: 1.500
准确率: 95.00%
锚点类型: nose_ear

Normal样本: 45/50 正确
Smoking样本: 48/50 正确
```

### 步骤 4：运行综合检测

使用 `smoking_detector.py` 进行实时或批量检测。该脚本会加载训练好的K值和姿态阈值，对输入数据进行综合判断。

---

## 七、注意事项

1. **依赖环境**：确保已安装 MMPose 及其依赖项
2. **图片格式**：仅支持 `.jpg` 和 `.png` 格式图片
3. **路径配置**：运行前请确认脚本中的路径配置正确
4. **阈值文件**：经验阈值探索脚本会自动加载历史训练阈值，若无则使用默认值