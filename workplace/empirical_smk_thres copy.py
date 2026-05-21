"""
功能说明
1. 读取两个由 `pose_est_calc copy.py` 生成的 JSON 文件，分别作为 `normal` 与 `warning` 两类样本。
2. 从 JSON 的 `infoes -> [img_name] -> instances` 中提取未跳过人体实例的指定指标：
   - `nonorm_ratio`
   - `norm_ratio`
3. 针对每个指标分别搜索单阈值分类规则，使两类样本尽可能分布在阈值两侧，并以分类准确率作为最优标准。
4. 将最优阈值、分类方向、混淆矩阵、统计信息及样本明细输出到终端，并可保存为汇总 JSON 文件。

注：使用鼻子到双肩中心点距离作为锚点的新版本算法

样本提取逻辑
- 仅遍历 `infoes` 下每张图片的 `instances`。
- 当实例满足 `skipped = true` 时直接跳过，不参与阈值搜索。
- 当目标指标值为 `null` 时跳过。
- 保留每个有效样本的来源信息：
  - `label`
  - `img_name`
  - `img_path`
  - `person_id`
  - `value`

阈值搜索逻辑
- 对每个指标收集 `normal` 与 `warning` 的全部有效数值。
- 根据所有去重后的数值生成候选阈值：
  - 最小值左侧一个边界
  - 相邻数值的中点
  - 最大值右侧一个边界
- 对每个候选阈值同时评估两种分类方向：
  - `warning_if_ge`：`value >= threshold` 判为 `warning`
  - `warning_if_le`：`value <= threshold` 判为 `warning`
- 以 `accuracy` 最大作为最优结果；若准确率相同，优先保留 `warning_if_ge`。

混淆矩阵说明
- 行表示真实类别，列表示预测类别。
- 2×2 矩阵组织形式如下：
  - 第一行：`[tn, fp]`，对应 `true_normal`
  - 第二行：`[fn, tp]`，对应 `true_warning`
- 终端会输出 ASCII 形式的混淆矩阵，汇总 JSON 中也会保存 `labels` 与 `matrix` 字段。

汇总 JSON 结构
{
  "normal_json": "/path/to/normal.json",
  "warning_json": "/path/to/warning.json",
  "metrics": {
    "nonorm_ratio": {
      "best_threshold": 0.418946,
      "direction": "warning_if_ge",
      "accuracy": 0.933333,
      "confusion_matrix": {
        "labels": {
          "rows": ["true_normal", "true_warning"],
          "cols": ["pred_normal", "pred_warning"]
        },
        "matrix": [
          [8, 1],
          [1, 20]
        ],
        "tp": 20,
        "fn": 1,
        "fp": 1,
        "tn": 8,
        "total": 30
      },
      "normal_stats": {
        "count": 9,
        "min": 0.155497,
        "max": 0.667917,
        "mean": 0.347561
      },
      "warning_stats": {
        "count": 21,
        "min": 0.446815,
        "max": 0.705099,
        "mean": 0.577893
      },
      "normal_samples": [],
      "warning_samples": []
    }
  }
}

输出位置
- 输入 JSON：由常量 `NORMAL_JSON` 与 `WARNING_JSON` 指定。
- 汇总 JSON：`outputs/jsons/empirical_threshold_summary_nose_shoulder_v2.json`
- 终端输出：每个指标的最优阈值、准确率、2×2 混淆矩阵和样本统计信息。

运行流程
- 启动后先读取 `NORMAL_JSON` 与 `WARNING_JSON`。
- 依次处理 `nonorm_ratio` 与 `norm_ratio`。
- 每个指标分别完成样本提取、候选阈值生成、双方向评估与最优结果选择。
- 若 `SAVE_SUMMARY = True`，则将最终分析结果保存到汇总 JSON。
"""

import json
from pathlib import Path
from statistics import mean
from typing import List


NORMAL_JSON = Path(
    "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_normal_2026-05-20_01-07-34.json"
)
WARNING_JSON = Path(
    "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_warning_2026-05-20_01-07-34.json"
)
SAVE_SUMMARY = True


def load_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric_samples(data: dict, metric_name: str, label: str) -> List[dict]:
    samples = []
    infoes = data.get("infoes", {})

    for img_name, img_info in infoes.items():
        for instance in img_info.get("instances", []):
            if instance.get("skipped", False):
                continue

            metric_value = instance.get(metric_name)
            if metric_value is None:
                continue

            samples.append(
                {
                    "label": label,
                    "img_name": img_name,
                    "img_path": img_info.get("img_path"),
                    "person_id": instance.get("person_id"),
                    "value": float(metric_value),
                }
            )

    return samples


def get_candidate_thresholds(values: List[float]) -> List[float]:
    unique_values = sorted(set(values))
    if not unique_values:
        return []

    if len(unique_values) == 1:
        return unique_values

    candidates = []
    candidates.append(unique_values[0] - 1e-9)

    for left, right in zip(unique_values[:-1], unique_values[1:]):
        candidates.append((left + right) / 2.0)

    candidates.append(unique_values[-1] + 1e-9)
    return candidates


def evaluate_threshold(
    normal_values: List[float],
    warning_values: List[float],
    threshold: float,
    direction: str,
) -> dict:
    if direction == "warning_if_ge":
        normal_pred_warning = [v >= threshold for v in normal_values]
        warning_pred_warning = [v >= threshold for v in warning_values]
    elif direction == "warning_if_le":
        normal_pred_warning = [v <= threshold for v in normal_values]
        warning_pred_warning = [v <= threshold for v in warning_values]
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    tp = sum(warning_pred_warning)
    fn = len(warning_pred_warning) - tp
    fp = sum(normal_pred_warning)
    tn = len(normal_pred_warning) - fp

    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "threshold": threshold,
        "direction": direction,
        "accuracy": accuracy,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "total": total,
    }


def search_best_threshold(normal_values: List[float], warning_values: List[float]) -> dict:
    all_values = normal_values + warning_values
    candidates = get_candidate_thresholds(all_values)

    best_result = None
    for threshold in candidates:
        for direction in ("warning_if_ge", "warning_if_le"):
            result = evaluate_threshold(normal_values, warning_values, threshold, direction)
            if best_result is None:
                best_result = result
                continue

            if result["accuracy"] > best_result["accuracy"]:
                best_result = result
                continue

            if (
                result["accuracy"] == best_result["accuracy"]
                and direction == "warning_if_ge"
                and best_result["direction"] == "warning_if_le"
            ):
                best_result = result

    return best_result


def summarize_values(values: List[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
    }


def format_confusion_matrix(result: dict) -> str:
    return "\n".join([
        "confusion matrix (rows=ground truth, cols=prediction)",
        "  \prediction   normal  warning",
        f"true normal   {result['tn']:>6}  {result['fp']:>7}",
        f"true warning  {result['fn']:>6}  {result['tp']:>7}",
    ])


def format_result(metric_name: str, result: dict, normal_values: List[float], warning_values: List[float]) -> str:
    direction_text = {
        "warning_if_ge": "预测规则: value >= threshold -> warning, else normal",
        "warning_if_le": "预测规则: value <= threshold -> warning, else normal",
    }[result["direction"]]

    lines = [
        f"===== {metric_name} =====",
        direction_text,
        f"best_threshold = {result['threshold']:.6f}",
        f"accuracy       = {result['accuracy']:.6%}",
        format_confusion_matrix(result),
        f"normal stats   = count:{len(normal_values)}  min:{min(normal_values):.6f}  max:{max(normal_values):.6f}  mean:{mean(normal_values):.6f}",
        f"warning stats  = count:{len(warning_values)}  min:{min(warning_values):.6f}  max:{max(warning_values):.6f}  mean:{mean(warning_values):.6f}",
    ]
    return "\n".join(lines)


def main():
    normal_data = load_json(NORMAL_JSON)
    warning_data = load_json(WARNING_JSON)

    summary = {
        "normal_json": str(NORMAL_JSON),
        "warning_json": str(WARNING_JSON),
        "metrics": {},
    }

    for metric_name in ("nonorm_ratio", "norm_ratio"):
        normal_samples = extract_metric_samples(normal_data, metric_name, label="normal")
        warning_samples = extract_metric_samples(warning_data, metric_name, label="warning")

        normal_values = [sample["value"] for sample in normal_samples]
        warning_values = [sample["value"] for sample in warning_samples]

        if not normal_values or not warning_values:
            raise ValueError(f"{metric_name} 的可用样本为空，无法搜索阈值")

        best_result = search_best_threshold(normal_values, warning_values)
        print(format_result(metric_name, best_result, normal_values, warning_values))
        print()

        summary["metrics"][metric_name] = {
            "best_threshold": round(best_result["threshold"], 6),
            "direction": best_result["direction"],
            "accuracy": round(best_result["accuracy"], 6),
            "confusion_matrix": {
                "labels": {
                    "rows": ["true_normal", "true_warning"],
                    "cols": ["pred_normal", "pred_warning"]
                },
                "matrix": [
                    [best_result["tn"], best_result["fp"]],
                    [best_result["fn"], best_result["tp"]]
                ],
                "tp": best_result["tp"],
                "fn": best_result["fn"],
                "fp": best_result["fp"],
                "tn": best_result["tn"],
                "total": best_result["total"],
            },
            "normal_stats": summarize_values(normal_values),
            "warning_stats": summarize_values(warning_values),
            "normal_samples": normal_samples,
            "warning_samples": warning_samples,
        }

    if SAVE_SUMMARY:
        out_path = Path(__file__).resolve().parent / "outputs" / "jsons" / "empirical_threshold_summary_nose_shoulder_v2.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"summary saved to: {out_path}")


if __name__ == "__main__":
    main()
