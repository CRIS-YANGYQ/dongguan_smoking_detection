"""
鼻子附近热源阈值搜索脚本

功能说明
1. 只使用“嘴巴附近无热源”和“嘴巴附近有热源”两个热成像目录作为标签。
2. 只读取 pose_est_calc.py 生成的姿态估计 JSON，用于拿到关键点。
3. 对每张热成像图，计算:
   score = 鼻子到最近 >=100°C 热像素的距离 / 锚点距离
4. 自动搜索最优缩放阈值 threshold，使两类样本尽可能分开。

判定规则
- score < threshold: 判定为“鼻子附近有热源”
- score > threshold: 判定为“鼻子附近无热源”
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

HEAT_LABEL = "嘴巴附近有热源"
NO_HEAT_LABEL = "嘴巴附近无热源"


def load_pose_statistics(json_path: Path) -> Dict:
    """读取 pose_est_calc.py 生成的姿态估计 JSON。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_valid_instances(image_info: Dict) -> int:
    """统计一张图像里可用的姿态实例数量。"""
    count = 0
    for instance in image_info.get("instances", []):
        if instance.get("skipped", False):
            continue
        if instance.get("keypoints") is None:
            continue
        count += 1
    return count


def merge_pose_records(json_paths: Sequence[Path]) -> Dict[str, Dict]:
    """
    合并多个姿态 JSON。

    若同名样本重复出现，优先保留可用人体实例更多的那条记录。
    """
    merged: Dict[str, Dict] = {}

    for json_path in json_paths:
        data = load_pose_statistics(json_path)
        for image_name, image_info in data.get("infoes", {}).items():
            stem = Path(image_name).stem
            current_valid = count_valid_instances(image_info)
            existing = merged.get(stem)

            if existing is None or current_valid > count_valid_instances(existing):
                merged[stem] = image_info

    return merged


def load_thermal_npy(npy_path: Path) -> np.ndarray:
    """读取热成像 NPY，并统一为二维温度矩阵。"""
    thermal = np.load(npy_path)
    thermal = np.asarray(thermal)
    thermal = np.squeeze(thermal)

    if thermal.ndim != 2:
        raise ValueError(f"热成像矩阵维度异常: {npy_path}, shape={thermal.shape}")

    return thermal


def distance_between_points(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """计算二维欧氏距离。"""
    return float(np.linalg.norm(point_a[:2] - point_b[:2]))


def extract_anchor_distance(
    keypoints: np.ndarray,
    anchor_type: str = "nose_ear",
) -> float:
    """按指定锚点类型提取锚点距离。"""
    nose = keypoints[KEYPOINT_NAMES.index("nose"), :2]

    if anchor_type == "nose_ear":
        left_ear = keypoints[KEYPOINT_NAMES.index("left_ear"), :2]
        right_ear = keypoints[KEYPOINT_NAMES.index("right_ear"), :2]
        return max(
            distance_between_points(nose, left_ear),
            distance_between_points(nose, right_ear),
        )

    if anchor_type == "nose_shoulder":
        left_shoulder = keypoints[KEYPOINT_NAMES.index("left_shoulder"), :2]
        right_shoulder = keypoints[KEYPOINT_NAMES.index("right_shoulder"), :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        return distance_between_points(nose, shoulder_center)

    raise ValueError(f"未知的锚点类型: {anchor_type}")


def get_hot_points(thermal_matrix: np.ndarray, overheat_threshold: float) -> np.ndarray:
    """提取所有 >= overheat_threshold 的像素坐标，格式为 [y, x]。"""
    hot_y, hot_x = np.where(thermal_matrix >= overheat_threshold)
    if hot_y.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack((hot_y, hot_x)).astype(np.float32)


def compute_instance_score(
    hot_points: np.ndarray,
    keypoints: np.ndarray,
    instance: Dict,
    anchor_type: str,
) -> Optional[Dict]:
    """
    计算单个人体实例的 score。

    score = 鼻子到最近高温像素的距离 / 锚点距离
    """
    nose = keypoints[KEYPOINT_NAMES.index("nose"), :2]

    if anchor_type == "nose_ear" and instance.get("nose_ear") is not None:
        anchor_distance = float(instance["nose_ear"])
    else:
        anchor_distance = extract_anchor_distance(keypoints, anchor_type=anchor_type)

    if anchor_distance <= 0:
        return None

    if hot_points.size == 0:
        return {
            "anchor_distance": float(anchor_distance),
            "min_heat_distance": None,
            "score": float("inf"),
        }

    delta_y = hot_points[:, 0] - float(nose[1])
    delta_x = hot_points[:, 1] - float(nose[0])
    min_heat_distance = float(np.sqrt(delta_x * delta_x + delta_y * delta_y).min())

    return {
        "anchor_distance": float(anchor_distance),
        "min_heat_distance": float(min_heat_distance),
        "score": float(min_heat_distance / anchor_distance),
    }


def compute_image_score(
    thermal_path: Path,
    image_info: Dict,
    overheat_threshold: float,
    anchor_type: str,
) -> Optional[Dict]:
    """
    对一张热成像图像计算 image-level score。

    若一张图有多个人体实例，取 score 最小的实例，表示最接近鼻子的热源分布。
    """
    thermal_matrix = load_thermal_npy(thermal_path)
    hot_points = get_hot_points(thermal_matrix, overheat_threshold=overheat_threshold)
    max_temperature = float(np.max(thermal_matrix))

    best_instance_score = None

    for instance in image_info.get("instances", []):
        if instance.get("skipped", False):
            continue

        keypoints_data = instance.get("keypoints")
        if keypoints_data is None:
            continue

        keypoints = np.asarray(keypoints_data, dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[0] < len(KEYPOINT_NAMES):
            continue

        instance_score = compute_instance_score(
            hot_points=hot_points,
            keypoints=keypoints,
            instance=instance,
            anchor_type=anchor_type,
        )
        if instance_score is None:
            continue

        if best_instance_score is None or instance_score["score"] < best_instance_score["score"]:
            best_instance_score = {
                "person_id": instance.get("person_id"),
                "anchor_distance": instance_score["anchor_distance"],
                "min_heat_distance": instance_score["min_heat_distance"],
                "score": instance_score["score"],
                "hot_pixel_count": int(hot_points.shape[0]),
                "max_temperature": max_temperature,
            }

    return best_instance_score


def collect_samples(
    thermal_dir: Path,
    label: str,
    pose_records: Dict[str, Dict],
    overheat_threshold: float,
    anchor_type: str,
) -> Dict:
    """按目录标签收集样本分布。"""
    samples: List[Dict] = []
    skipped_no_pose = 0
    skipped_no_valid_pose = 0

    thermal_paths = sorted(thermal_dir.glob("*.npy"))

    for thermal_path in thermal_paths:
        image_info = pose_records.get(thermal_path.stem)
        if image_info is None:
            skipped_no_pose += 1
            continue

        score_info = compute_image_score(
            thermal_path=thermal_path,
            image_info=image_info,
            overheat_threshold=overheat_threshold,
            anchor_type=anchor_type,
        )
        if score_info is None:
            skipped_no_valid_pose += 1
            continue

        samples.append(
            {
                "label": label,
                "image_name": thermal_path.name,
                "image_stem": thermal_path.stem,
                "thermal_path": str(thermal_path),
                "person_id": score_info["person_id"],
                "anchor_distance": score_info["anchor_distance"],
                "min_heat_distance": score_info["min_heat_distance"],
                "score": score_info["score"],
                "hot_pixel_count": score_info["hot_pixel_count"],
                "max_temperature": score_info["max_temperature"],
            }
        )

    return {
        "label": label,
        "samples": samples,
        "stats": {
            "thermal_files": len(thermal_paths),
            "usable_samples": len(samples),
            "skipped_no_pose": skipped_no_pose,
            "skipped_no_valid_pose": skipped_no_valid_pose,
        },
    }


def summarize_scores(samples: Sequence[Dict]) -> Dict:
    """汇总 score 分布。"""
    finite_scores = sorted(
        float(sample["score"])
        for sample in samples
        if np.isfinite(sample["score"])
    )
    inf_count = sum(1 for sample in samples if not np.isfinite(sample["score"]))

    if not finite_scores:
        return {
            "count": len(samples),
            "finite_count": 0,
            "inf_count": inf_count,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }

    quantiles = np.quantile(np.asarray(finite_scores), [0.25, 0.5, 0.75])
    return {
        "count": len(samples),
        "finite_count": len(finite_scores),
        "inf_count": inf_count,
        "min": float(finite_scores[0]),
        "p25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "max": float(finite_scores[-1]),
    }


def build_threshold_candidates(scores: Iterable[float]) -> List[float]:
    """根据所有有限 score 构造候选阈值。"""
    finite_scores = sorted({float(score) for score in scores if np.isfinite(score)})
    if not finite_scores:
        return [0.0]

    candidates: List[float] = []
    first_score = finite_scores[0]

    if first_score > 0:
        candidates.append(first_score / 2.0)
    else:
        candidates.append(0.0)

    for left, right in zip(finite_scores, finite_scores[1:]):
        candidates.append((left + right) / 2.0)

    candidates.append(finite_scores[-1] + max(1e-6, finite_scores[-1] * 1e-6))
    return candidates


def evaluate_threshold(
    samples: Sequence[Dict],
    threshold: float,
    heat_error_weight: float = 1.0,
    no_heat_error_weight: float = 1.0,
) -> Dict:
    """评估给定阈值在样本上的分类效果。"""
    tp = fp = tn = fn = 0

    for sample in samples:
        is_heat_label = sample["label"] == HEAT_LABEL
        pred_heat = float(sample["score"]) < threshold

        if is_heat_label and pred_heat:
            tp += 1
        elif is_heat_label and not pred_heat:
            fn += 1
        elif (not is_heat_label) and pred_heat:
            fp += 1
        else:
            tn += 1

    total = len(samples)
    heat_count = tp + fn
    no_heat_count = tn + fp
    accuracy = (tp + tn) / total if total else 0.0
    heat_recall = tp / heat_count if heat_count else 0.0
    no_heat_recall = tn / no_heat_count if no_heat_count else 0.0
    balanced_accuracy = (heat_recall + no_heat_recall) / 2.0
    weighted_error = fn * heat_error_weight + fp * no_heat_error_weight
    weighted_total = heat_count * heat_error_weight + no_heat_count * no_heat_error_weight
    weighted_accuracy = 1.0 - (weighted_error / weighted_total) if weighted_total else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "weighted_accuracy": float(weighted_accuracy),
        "weighted_error": float(weighted_error),
        "balanced_accuracy": float(balanced_accuracy),
        "heat_recall": float(heat_recall),
        "no_heat_recall": float(no_heat_recall),
        "class_weights": {
            "heat_error_weight": float(heat_error_weight),
            "no_heat_error_weight": float(no_heat_error_weight),
        },
        "confusion_matrix": {
            "true_heat_pred_heat": tp,
            "true_heat_pred_no_heat": fn,
            "true_no_heat_pred_heat": fp,
            "true_no_heat_pred_no_heat": tn,
        },
    }


def search_best_threshold(samples: Sequence[Dict]) -> Dict:
    """搜索能最大限度分割两类目录分布的阈值。"""
    if not samples:
        raise ValueError("没有可用样本，无法搜索阈值。")

    scores = [float(sample["score"]) for sample in samples]
    candidates = build_threshold_candidates(scores)
    finite_scores = [score for score in scores if np.isfinite(score)]
    heat_count = sum(1 for sample in samples if sample["label"] == HEAT_LABEL)
    no_heat_count = sum(1 for sample in samples if sample["label"] == NO_HEAT_LABEL)

    if heat_count == 0 or no_heat_count == 0:
        raise ValueError("有热源和无热源样本都至少需要 1 个。")

    total = heat_count + no_heat_count
    heat_error_weight = total / (2.0 * heat_count)
    no_heat_error_weight = total / (2.0 * no_heat_count)

    best_result = None
    best_rank = None

    for threshold in candidates:
        result = evaluate_threshold(
            samples,
            threshold,
            heat_error_weight=heat_error_weight,
            no_heat_error_weight=no_heat_error_weight,
        )

        if finite_scores:
            margin = min(abs(threshold - score) for score in finite_scores)
        else:
            margin = 0.0

        rank = (
            result["weighted_accuracy"],
            result["balanced_accuracy"],
            margin,
            -threshold,
        )

        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_result = result

    return best_result


def sanitize_for_json(value):
    """将 inf 等非标准 JSON 数值转为可序列化形式。"""
    if isinstance(value, dict):
        return {key: sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "inf"
    return value


def save_result(output_path: Path, result: Dict) -> None:
    """保存结果 JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(result), f, ensure_ascii=False, indent=2)


def search_threshold_from_paths(
    pose_json_paths: Sequence[Path],
    no_heat_dir: Path,
    heat_dir: Path,
    output_path: Path,
    anchor_type: str = "nose_ear",
    overheat_threshold: float = 100.0,
) -> Dict:
    """对两个热成像目录搜索最优锚点距离缩放阈值。"""
    pose_records = merge_pose_records(pose_json_paths)

    no_heat_data = collect_samples(
        thermal_dir=no_heat_dir,
        label=NO_HEAT_LABEL,
        pose_records=pose_records,
        overheat_threshold=overheat_threshold,
        anchor_type=anchor_type,
    )
    heat_data = collect_samples(
        thermal_dir=heat_dir,
        label=HEAT_LABEL,
        pose_records=pose_records,
        overheat_threshold=overheat_threshold,
        anchor_type=anchor_type,
    )

    samples = no_heat_data["samples"] + heat_data["samples"]
    best_result = search_best_threshold(samples)

    result = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
        "anchor_type": anchor_type,
        "overheat_threshold": float(overheat_threshold),
        "score_definition": "score = 鼻子到最近>=100°C热像素的距离 / 锚点距离",
        "classification_rule": {
            "heat": "score < threshold",
            "no_heat": "score > threshold",
        },
        "best_threshold": best_result["threshold"],
        "metrics": {
            "accuracy": best_result["accuracy"],
            "weighted_accuracy": best_result["weighted_accuracy"],
            "weighted_error": best_result["weighted_error"],
            "balanced_accuracy": best_result["balanced_accuracy"],
            "heat_recall": best_result["heat_recall"],
            "no_heat_recall": best_result["no_heat_recall"],
            "class_weights": best_result["class_weights"],
            "confusion_matrix": best_result["confusion_matrix"],
        },
        "inputs": {
            "pose_json_paths": [str(path) for path in pose_json_paths],
            "no_heat_dir": str(no_heat_dir),
            "heat_dir": str(heat_dir),
        },
        "dataset_stats": {
            NO_HEAT_LABEL: {
                **no_heat_data["stats"],
                "score_summary": summarize_scores(no_heat_data["samples"]),
            },
            HEAT_LABEL: {
                **heat_data["stats"],
                "score_summary": summarize_scores(heat_data["samples"]),
            },
        },
        "samples": samples,
    }

    save_result(output_path, result)
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("鼻子附近热源缩放阈值搜索")
    print("=" * 60)

    # 姿态估计JSON路径（由pose_est_calc.py生成）
    POSE_JSON_PATHS = [
        Path("/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态正常_2026-05-26_00-50-48.json"),
        Path("/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态危险_2026-05-26_00-51-10.json"),
    ]

    NO_HEAT_DIR = Path("/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal/嘴巴附近无热源")
    HEAT_DIR = Path("/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal/嘴巴附近有热源")
    OUTPUT_PATH = Path("/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/thermal/nose_heat_threshold_search.json")

    ANCHOR_TYPE = "nose_ear"
    OVERHEAT_THRESHOLD = 100.0

    result = search_threshold_from_paths(
        pose_json_paths=POSE_JSON_PATHS,
        no_heat_dir=NO_HEAT_DIR,
        heat_dir=HEAT_DIR,
        output_path=OUTPUT_PATH,
        anchor_type=ANCHOR_TYPE,
        overheat_threshold=OVERHEAT_THRESHOLD,
    )

    print(f"锚点类型: {result['anchor_type']}")
    print(f"最优缩放阈值: {result['best_threshold']:.6f}")
    print(f"准确率: {result['metrics']['accuracy']:.2%}")
    print(f"平衡准确率: {result['metrics']['balanced_accuracy']:.2%}")
    print(f"有热源召回率: {result['metrics']['heat_recall']:.2%}")
    print(f"无热源召回率: {result['metrics']['no_heat_recall']:.2%}")
    print(f"结果已保存到: {OUTPUT_PATH}")
