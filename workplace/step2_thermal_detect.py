"""
功能说明
1. 基于 step1_pose_est 输出的 json，读取鼻子到两个锚点的像素距离和归一化距离：
   - nose_ear
   - nose_ear_norm
   - nose_shoulder
   - nose_shoulder_norm
2. 根据 thermal 输入目录加载热成像 npy 文件，并按图片文件名与姿态 json 一一对齐。
3. 先将 thermal 矩阵上采样到 RGB 图片分辨率，再判断鼻子附近热源。
4. 对四个锚点距离分别进行 K 值探索。类别不平衡时，搜索阶段按类别数目比例加权。
5. 输出四个 K 的最优结果，包括混淆矩阵、accuracy、recall、precision、F1-score。

判定规则
- 若 `鼻子到最近高温像素的距离 <= 锚点距离 * K`，则判定为“嘴巴附近有热源”。
- 对归一化距离，同样使用 `归一化热源距离 <= 归一化锚点距离 * K`。
anchor * K > min_dist 与step1相同
"""

from math import sqrt
from pathlib import Path
import json
import time

import cv2
import numpy as np


NO_HEAT_DIR = Path(
    "/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal/嘴巴附近无热源"
)
HEAT_DIR = Path(
    "/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal/嘴巴附近有热源"
)
POSE_JSON_PATHS = [
    Path(
        "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态正常_2026-05-26_12-20-08.json"
    ),
    Path(
        "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态危险_2026-05-26_12-20-08.json"
    ),
]
SEARCH_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "thermal"
    / "step2_thermal_search_results.json"
)
VIS_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "thermal_vis"
VIS_SUMMARY_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "thermal"
    / "step2_thermal_vis_summary.json"
)

OVERHEAT_THRESHOLD = 100.0
HEAT_LABEL = "嘴巴附近有热源"
NO_HEAT_LABEL = "嘴巴附近无热源"
THERMAL_W = 640
THERMAL_H = 512
TARGET_W = 2688
TARGET_H = 1520
RESIZE_TARGET = "homography_640x512_to_2688x1520"
RESIZE_INTERPOLATION = cv2.INTER_NEAREST
RESIZE_INTERPOLATION_NAME = "INTER_NEAREST"
THERMAL_BLEND_ALPHA = 0.35
THERMAL_TO_RGB_SCALE_X = TARGET_W / THERMAL_W
THERMAL_TO_RGB_SCALE_Y = TARGET_H / THERMAL_H
THERMAL_TO_RGB_TRANS_X = 0.5 * THERMAL_TO_RGB_SCALE_X - 0.5
THERMAL_TO_RGB_TRANS_Y = 0.5 * THERMAL_TO_RGB_SCALE_Y - 0.5
THERMAL_TO_RGB_HOMOGRAPHY = np.array(
    [
        [THERMAL_TO_RGB_SCALE_X, 0, THERMAL_TO_RGB_TRANS_X],
        [0, THERMAL_TO_RGB_SCALE_Y, THERMAL_TO_RGB_TRANS_Y],
        [0, 0, 1],
    ],
    dtype=np.float32,
)

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

METRIC_DEFINITIONS = {
    "K_nose_ear": {
        "display_name": "nose_ear",
        "anchor_field": "nose_ear",
        "distance_mode": "pixel",
    },
    "K_nose_ear_norm": {
        "display_name": "nose_ear_norm",
        "anchor_field": "nose_ear_norm",
        "distance_mode": "normalized",
    },
    "K_nose_shoulder": {
        "display_name": "nose_shoulder",
        "anchor_field": "nose_shoulder",
        "distance_mode": "pixel",
    },
    "K_nose_shoulder_norm": {
        "display_name": "nose_shoulder_norm",
        "anchor_field": "nose_shoulder_norm",
        "distance_mode": "normalized",
    },
}


def round_float(value, ndigits=6):
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return "inf"
    return round(float(value), ndigits)


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "inf"
    return value


def should_report_progress(index, total, report_every=50):
    if total <= 0:
        return False
    if index == 1 or index == total:
        return True
    if total <= 20:
        return True
    return index % report_every == 0


def print_progress(prefix, index, total, extra=""):
    percent = (index / total * 100.0) if total else 100.0
    suffix = f" | {extra}" if extra else ""
    print(f"[PROGRESS] {prefix}: {index}/{total} ({percent:.1f}%){suffix}")


def load_thermal_npy(npy_path: Path):
    thermal = np.load(npy_path)
    thermal = np.asarray(thermal)
    thermal = np.squeeze(thermal)
    if thermal.ndim != 2:
        raise ValueError(f"热成像矩阵维度异常: {npy_path}, shape={thermal.shape}")
    return thermal.astype(np.float32)


def count_valid_instances(image_info):
    count = 0
    for instance in image_info.get("instances", []):
        if instance.get("skipped", False):
            continue
        keypoints = instance.get("keypoints")
        if keypoints is None:
            continue
        count += 1
    return count


def merge_pose_records(json_paths):
    merged = {}
    for json_path in json_paths:
        data = load_json(json_path)
        for image_name, image_info in data.get("infoes", {}).items():
            stem = Path(image_name).stem
            current_valid = count_valid_instances(image_info)
            existing = merged.get(stem)
            if existing is None or current_valid > count_valid_instances(existing):
                merged[stem] = image_info
    return merged


def list_thermal_paths(thermal_dir: Path):
    npy_paths = sorted(thermal_dir.glob("*.npy"))
    if npy_paths:
        return npy_paths
    return sorted(path for path in thermal_dir.iterdir() if path.is_file())


def read_image_shape(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取对应 RGB 图片: {image_path}")
    img_h, img_w = image.shape[:2]
    return img_h, img_w


def align_thermal_to_rgb_resolution(thermal_matrix):
    thermal_h, thermal_w = thermal_matrix.shape[:2]
    if thermal_w != THERMAL_W or thermal_h != THERMAL_H:
        raise ValueError(
            f"热成像分辨率异常，期望 {THERMAL_W}x{THERMAL_H}，实际 {thermal_w}x{thermal_h}"
        )

    return cv2.warpPerspective(
        thermal_matrix,
        THERMAL_TO_RGB_HOMOGRAPHY,
        (TARGET_W, TARGET_H),
        flags=RESIZE_INTERPOLATION,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def get_hot_points(thermal_matrix, overheat_threshold):
    hot_y, hot_x = np.where(thermal_matrix >= overheat_threshold)
    if hot_y.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack((hot_y, hot_x)).astype(np.float32)


def distance_between_points(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def normalized_distance_between_points(p1, p2, h, w):
    norm_x_dist = (p1[0] - p2[0]) / w
    norm_y_dist = (p1[1] - p2[1]) / h
    return sqrt(norm_x_dist**2 + norm_y_dist**2)


def get_min_heat_distances(hot_points, nose_point, img_h, img_w):
    """
    计算鼻子坐标到最近高温像素点的像素距离和归一化距离
    参数:
        hot_points: 所有高温点的坐标数组，形状为(N, 2)，每行格式为[y坐标, x坐标]
        nose_point: 鼻子的坐标点，格式为[x坐标, y坐标]
        img_h: RGB图像的高度（像素）
        img_w: RGB图像的宽度（像素）
    返回:
        min_pixel_distance: 鼻子到最近高温点的像素距离，无高温点时返回None
        min_norm_distance: 鼻子到最近高温点的归一化距离（按图像对角线归一化），无高温点时返回None
    """
    # 无高温点时直接返回空值
    if hot_points.size == 0:
        return None, None

    # 计算所有高温点与鼻子点的坐标差值（hot_points存储格式为[y,x]，与nose_point的[x,y]对应转换）
    delta_y = hot_points[:, 0] - float(nose_point[1])
    delta_x = hot_points[:, 1] - float(nose_point[0])
    # 计算所有高温点到鼻子的欧氏像素距离
    pixel_distances = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    # 找到距离最近的高温点索引
    min_index = int(np.argmin(pixel_distances))
    # 提取最小像素距离
    min_pixel_distance = float(pixel_distances[min_index])

    # 提取最近高温点的坐标
    min_hot_y = float(hot_points[min_index, 0])
    min_hot_x = float(hot_points[min_index, 1])
    # 计算归一化距离
    min_norm_distance = float(
        normalized_distance_between_points(
            nose_point,
            [min_hot_x, min_hot_y],
            img_h,
            img_w,
        )
    )
    return min_pixel_distance, min_norm_distance


def compute_instance_metric_scores(hot_points, instance, img_h, img_w):
    keypoints = np.asarray(instance.get("keypoints"), dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] < len(KEYPOINT_NAMES):
        return {}

    nose = keypoints[KEYPOINT_NAMES.index("nose"), :2]
    min_pixel_distance, min_norm_distance = get_min_heat_distances(
        hot_points=hot_points,
        nose_point=nose,
        img_h=img_h,
        img_w=img_w,
    )

    metric_scores = {}
    for metric_name, metric_config in METRIC_DEFINITIONS.items():
        anchor_value = instance.get(metric_config["anchor_field"])
        if anchor_value is None:
            continue

        anchor_value = float(anchor_value)
        if anchor_value <= 0:
            continue

        if metric_config["distance_mode"] == "pixel":
            score = float("inf") if min_pixel_distance is None else min_pixel_distance / anchor_value
            min_distance = min_pixel_distance
        else:
            score = float("inf") if min_norm_distance is None else min_norm_distance / anchor_value
            min_distance = min_norm_distance

        metric_scores[metric_name] = {
            "person_id": instance.get("person_id"),
            "anchor_distance": float(anchor_value),
            "min_heat_distance": None if min_distance is None else float(min_distance),
            "score": float(score),
        }

    return metric_scores


def compute_image_metric_scores(thermal_path: Path, image_info, overheat_threshold):
    img_path = Path(image_info["img_path"])
    img_h, img_w = read_image_shape(img_path)
    if img_w != TARGET_W or img_h != TARGET_H:
        raise ValueError(
            f"RGB 图片分辨率异常，期望 {TARGET_W}x{TARGET_H}，实际 {img_w}x{img_h}"
        )
    thermal_matrix = load_thermal_npy(thermal_path)
    thermal_resized = align_thermal_to_rgb_resolution(thermal_matrix)
    hot_points = get_hot_points(thermal_resized, overheat_threshold=overheat_threshold)

    best_scores = {}
    for instance in image_info.get("instances", []):
        keypoints = instance.get("keypoints")
        if keypoints is None:
            continue

        instance_scores = compute_instance_metric_scores(
            hot_points=hot_points,
            instance=instance,
            img_h=img_h,
            img_w=img_w,
        )
        for metric_name, score_info in instance_scores.items():
            current_best = best_scores.get(metric_name)
            if current_best is None or score_info["score"] < current_best["score"]:
                best_scores[metric_name] = {
                    "person_id": score_info["person_id"],
                    "anchor_distance": score_info["anchor_distance"],
                    "min_heat_distance": score_info["min_heat_distance"],
                    "score": score_info["score"],
                }

    return {
        "image_size": {"h": int(img_h), "w": int(img_w)},
        "thermal_raw_shape": {
            "h": int(thermal_matrix.shape[0]),
            "w": int(thermal_matrix.shape[1]),
        },
        "hot_pixel_count": int(hot_points.shape[0]),
        "max_temperature": float(np.max(thermal_matrix)),
        "best_scores": best_scores,
    }


def collect_samples(thermal_dir: Path, label: str, pose_records, overheat_threshold):
    thermal_paths = list_thermal_paths(thermal_dir)
    samples_by_metric = {metric_name: [] for metric_name in METRIC_DEFINITIONS}
    skipped_no_pose = 0
    skipped_no_valid_pose = 0
    skipped_read_error = 0

    print(
        f"[INFO]: 开始收集样本 -> {label} | 目录: {thermal_dir} | 文件数: {len(thermal_paths)}"
    )

    for index, thermal_path in enumerate(thermal_paths, start=1):
        if should_report_progress(index, len(thermal_paths), report_every=50):
            print_progress(
                prefix=f"收集样本 {label}",
                index=index,
                total=len(thermal_paths),
                extra=thermal_path.name,
            )
        image_info = pose_records.get(thermal_path.stem)
        if image_info is None:
            skipped_no_pose += 1
            continue

        try:
            image_scores = compute_image_metric_scores(
                thermal_path=thermal_path,
                image_info=image_info,
                overheat_threshold=overheat_threshold,
            )
        except Exception as e:
            print(f"[WARNING]: 跳过样本 {thermal_path.name}, 原因: {e}")
            skipped_read_error += 1
            continue

        if not image_scores["best_scores"]:
            skipped_no_valid_pose += 1
            continue

        for metric_name, score_info in image_scores["best_scores"].items():
            samples_by_metric[metric_name].append(
                {
                    "label": label,
                    "image_name": thermal_path.name,
                    "image_stem": thermal_path.stem,
                    "thermal_path": str(thermal_path),
                    "img_path": image_info["img_path"],
                    "person_id": score_info["person_id"],
                    "anchor_distance": score_info["anchor_distance"],
                    "min_heat_distance": score_info["min_heat_distance"],
                    "score": score_info["score"],
                    "hot_pixel_count": image_scores["hot_pixel_count"],
                    "max_temperature": image_scores["max_temperature"],
                    "image_size": image_scores["image_size"],
                    "thermal_raw_shape": image_scores["thermal_raw_shape"],
                }
            )

    print(
        f"[INFO]: 样本收集完成 -> {label} | "
        f"可用: {sum(len(samples) for samples in samples_by_metric.values())} | "
        f"缺姿态: {skipped_no_pose} | 无有效姿态: {skipped_no_valid_pose} | "
        f"读取失败: {skipped_read_error}"
    )
    return {
        "label": label,
        "samples_by_metric": samples_by_metric,
        "stats": {
            "thermal_files": len(thermal_paths),
            "matched_pose_files": len(thermal_paths) - skipped_no_pose,
            "skipped_no_pose": skipped_no_pose,
            "skipped_no_valid_pose": skipped_no_valid_pose,
            "skipped_read_error": skipped_read_error,
            "usable_samples_by_metric": {
                metric_name: len(samples)
                for metric_name, samples in samples_by_metric.items()
            },
        },
    }


def summarize_scores(samples):
    finite_scores = sorted(
        float(sample["score"])
        for sample in samples
        if np.isfinite(float(sample["score"]))
    )
    inf_count = sum(1 for sample in samples if not np.isfinite(float(sample["score"])))

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

    quantiles = np.quantile(np.asarray(finite_scores, dtype=np.float32), [0.25, 0.5, 0.75])
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


def build_k_candidates(scores):
    finite_scores = sorted({float(score) for score in scores if np.isfinite(float(score))})
    if not finite_scores:
        return [0.0]

    candidates = []
    first_score = finite_scores[0]
    if first_score > 0:
        candidates.append(first_score / 2.0)
    else:
        candidates.append(0.0)

    for left, right in zip(finite_scores, finite_scores[1:]):
        candidates.append((left + right) / 2.0)

    candidates.append(finite_scores[-1] + max(1e-6, finite_scores[-1] * 1e-6))
    return candidates


def evaluate_threshold(samples, threshold, heat_error_weight=1.0, no_heat_error_weight=1.0):
    tp = fp = tn = fn = 0

    for sample in samples:
        is_heat_label = sample["label"] == HEAT_LABEL
        pred_heat = float(sample["score"]) <= threshold

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
    recall = tp / heat_count if heat_count else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    no_heat_recall = tn / no_heat_count if no_heat_count else 0.0
    balanced_accuracy = (recall + no_heat_recall) / 2.0
    weighted_error = fn * heat_error_weight + fp * no_heat_error_weight
    weighted_total = heat_count * heat_error_weight + no_heat_count * no_heat_error_weight
    weighted_accuracy = 1.0 - (weighted_error / weighted_total) if weighted_total else 0.0

    return {
        "best_K": float(threshold),
        "accuracy": float(accuracy),
        "recall": float(recall),
        "precision": float(precision),
        "f1_score": float(f1_score),
        "balanced_accuracy": float(balanced_accuracy),
        "weighted_accuracy": float(weighted_accuracy),
        "weighted_error": float(weighted_error),
        "class_weights": {
            "heat_error_weight": float(heat_error_weight),
            "no_heat_error_weight": float(no_heat_error_weight),
        },
        "confusion_matrix": {
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
        },
    }


def search_best_k(samples):
    if not samples:
        raise ValueError("没有可用样本，无法搜索 K。")

    heat_count = sum(1 for sample in samples if sample["label"] == HEAT_LABEL)
    no_heat_count = sum(1 for sample in samples if sample["label"] == NO_HEAT_LABEL)
    if heat_count == 0 or no_heat_count == 0:
        raise ValueError("有热源和无热源样本都至少需要 1 个。")

    total = heat_count + no_heat_count
    heat_error_weight = total / (2.0 * heat_count)
    no_heat_error_weight = total / (2.0 * no_heat_count)
    candidates = build_k_candidates([sample["score"] for sample in samples])
    print(
        f"[INFO]: 开始搜索 K | 样本数: {len(samples)} | 候选数: {len(candidates)} | "
        f"有热源: {heat_count} | 无热源: {no_heat_count}"
    )

    best_result = None
    best_rank = None
    for index, threshold in enumerate(candidates, start=1):
        if should_report_progress(index, len(candidates), report_every=200):
            print_progress(
                prefix="K搜索",
                index=index,
                total=len(candidates),
                extra=f"K={threshold:.6f}",
            )
        result = evaluate_threshold(
            samples,
            threshold,
            heat_error_weight=heat_error_weight,
            no_heat_error_weight=no_heat_error_weight,
        )

        finite_scores = [
            float(sample["score"]) for sample in samples if np.isfinite(float(sample["score"]))
        ]
        if finite_scores:
            margin = min(abs(threshold - score) for score in finite_scores)
        else:
            margin = 0.0

        rank = (
            result["weighted_accuracy"],
            result["balanced_accuracy"],
            result["f1_score"],
            margin,
            -threshold,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_result = result

    print(
        f"[INFO]: K搜索完成 | best_K={best_result['best_K']:.6f} | "
        f"accuracy={best_result['accuracy']:.2%} | "
        f"weighted_accuracy={best_result['weighted_accuracy']:.2%}"
    )
    return best_result


def save_result(output_path: Path, result):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(result), f, ensure_ascii=False, indent=2)


def load_best_thresholds(search_result):
    thresholds = {}
    for metric_name, metric_result in search_result["metric_results"].items():
        if metric_result.get("error"):
            thresholds[metric_name] = None
            continue
        thresholds[metric_name] = float(metric_result["metrics"]["best_K"])
    missing_thresholds = [
        metric_name for metric_name, threshold in thresholds.items() if threshold is None
    ]
    if missing_thresholds:
        raise ValueError(f"以下指标缺少可用 K 值: {missing_thresholds}")
    return thresholds


def create_thermal_heatmap(thermal_resized):
    min_value = float(np.min(thermal_resized))
    max_value = float(np.max(thermal_resized))
    if max_value > min_value:
        normalized = (thermal_resized - min_value) / (max_value - min_value)
    else:
        normalized = np.zeros_like(thermal_resized, dtype=np.float32)
    heatmap_uint8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


def overlay_thermal_on_rgb(rgb_image, thermal_resized, alpha=THERMAL_BLEND_ALPHA):
    heatmap = create_thermal_heatmap(thermal_resized)
    return cv2.addWeighted(rgb_image, 1.0 - alpha, heatmap, alpha, 0.0)


def normalized_radius_to_pixels(norm_radius, img_h, img_w):
    img_diag = sqrt(img_h**2 + img_w**2)
    return int(round(norm_radius * img_diag))


def draw_anchor1(canvas, metrics):
    nose = tuple(int(v) for v in metrics["nose"])
    left_ear = tuple(int(v) for v in metrics["left_ear"])
    right_ear = tuple(int(v) for v in metrics["right_ear"])
    label_pos = (
        int((nose[0] + left_ear[0] + right_ear[0]) / 3),
        int((nose[1] + left_ear[1] + right_ear[1]) / 3) - 8,
    )

    cv2.line(canvas, nose, left_ear, (0, 255, 0), 2)
    cv2.line(canvas, nose, right_ear, (0, 255, 0), 2)
    cv2.putText(
        canvas,
        "Anchor1",
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )


def draw_anchor2(canvas, metrics):
    nose = tuple(int(v) for v in metrics["nose"])
    left_shoulder = tuple(int(v) for v in metrics["left_shoulder"])
    right_shoulder = tuple(int(v) for v in metrics["right_shoulder"])
    shoulder_center = tuple(int(v) for v in metrics["shoulder_center"])
    label_pos = (
        int((nose[0] + shoulder_center[0]) / 2),
        int((nose[1] + shoulder_center[1]) / 2) - 8,
    )

    cv2.line(canvas, left_shoulder, right_shoulder, (255, 0, 255), 2)
    cv2.line(canvas, nose, shoulder_center, (255, 0, 255), 2)
    cv2.circle(canvas, shoulder_center, 4, (255, 0, 255), -1)
    cv2.putText(
        canvas,
        "Anchor2",
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2,
    )


def draw_thermal_alert_circles(canvas, metrics, thresholds, img_h, img_w):
    nose = tuple(int(v) for v in metrics["nose"])

    radius_nose_ear = int(round(metrics["anchor1"] * thresholds["K_nose_ear"]))
    radius_nose_ear_norm = normalized_radius_to_pixels(
        metrics["anchor1_norm"] * thresholds["K_nose_ear_norm"], img_h, img_w
    )
    radius_nose_shoulder = int(
        round(metrics["anchor2"] * thresholds["K_nose_shoulder"])
    )
    radius_nose_shoulder_norm = normalized_radius_to_pixels(
        metrics["anchor2_norm"] * thresholds["K_nose_shoulder_norm"], img_h, img_w
    )

    if radius_nose_ear > 0:
        cv2.circle(canvas, nose, radius_nose_ear, (0, 128, 0), 2)
    if radius_nose_ear_norm > 0:
        cv2.circle(canvas, nose, radius_nose_ear_norm, (144, 238, 144), 2)
    if radius_nose_shoulder > 0:
        cv2.circle(canvas, nose, radius_nose_shoulder, (128, 0, 128), 2)
    if radius_nose_shoulder_norm > 0:
        cv2.circle(canvas, nose, radius_nose_shoulder_norm, (255, 128, 255), 2)


def draw_hot_point(canvas, metrics):
    hot_point = metrics.get("nearest_hot_point")
    if hot_point is None:
        return
    hot_xy = tuple(int(v) for v in hot_point)
    nose = tuple(int(v) for v in metrics["nose"])
    cv2.circle(canvas, hot_xy, 5, (0, 0, 255), -1)
    cv2.line(canvas, nose, hot_xy, (0, 255, 255), 2)


def draw_person_text(canvas, metrics):
    person_text = (
        f"Heat:{metrics['min_heat_distance']:.1f}  "
        f"Heat_norm:{metrics['min_heat_distance_norm']:.3f}  "
        f"Anchor:{metrics['anchor1']:.3f}  "
        f"Anchor_norm:{metrics['anchor1_norm']:.3f}  "
        f"Risk:{metrics['risk1']}  Norm_Risk:{metrics['norm_risk1']}"
    )
    extra_text = (
        f"Anchor2:{metrics['anchor2']:.3f}  "
        f"Anchor2_norm:{metrics['anchor2_norm']:.3f}  "
        f"Risk2:{metrics['risk2']}  Norm_Risk2:{metrics['norm_risk2']}  "
        f"Max_Temp:{metrics['max_temperature']:.1f}C"
    )
    text_y = canvas.shape[0] - 60
    cv2.putText(
        canvas,
        person_text,
        (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        canvas,
        extra_text,
        (20, text_y + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )


def compute_instance_visual_metrics(hot_points, instance, thresholds, img_h, img_w, max_temperature):
    keypoints = np.asarray(instance.get("keypoints"), dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] < len(KEYPOINT_NAMES):
        return None

    nose = keypoints[KEYPOINT_NAMES.index("nose"), :2]
    left_ear = keypoints[KEYPOINT_NAMES.index("left_ear"), :2]
    right_ear = keypoints[KEYPOINT_NAMES.index("right_ear"), :2]
    left_shoulder = keypoints[KEYPOINT_NAMES.index("left_shoulder"), :2]
    right_shoulder = keypoints[KEYPOINT_NAMES.index("right_shoulder"), :2]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    min_heat_distance, min_heat_distance_norm = get_min_heat_distances(
        hot_points, nose, img_h, img_w
    )
    nearest_hot_point = None
    if hot_points.size != 0:
        delta_y = hot_points[:, 0] - float(nose[1])
        delta_x = hot_points[:, 1] - float(nose[0])
        min_index = int(np.argmin(np.sqrt(delta_x * delta_x + delta_y * delta_y)))
        nearest_hot_point = [
            float(hot_points[min_index, 1]),
            float(hot_points[min_index, 0]),
        ]

    anchor1 = instance.get("nose_ear")
    anchor1_norm = instance.get("nose_ear_norm")
    anchor2 = instance.get("nose_shoulder")
    anchor2_norm = instance.get("nose_shoulder_norm")

    if None in (anchor1, anchor1_norm, anchor2, anchor2_norm):
        return None

    anchor1 = float(anchor1)
    anchor1_norm = float(anchor1_norm)
    anchor2 = float(anchor2)
    anchor2_norm = float(anchor2_norm)

    min_heat_distance_value = float("inf") if min_heat_distance is None else float(min_heat_distance)
    min_heat_distance_norm_value = (
        float("inf") if min_heat_distance_norm is None else float(min_heat_distance_norm)
    )

    risk1 = min_heat_distance_value <= anchor1 * thresholds["K_nose_ear"]
    norm_risk1 = (
        min_heat_distance_norm_value <= anchor1_norm * thresholds["K_nose_ear_norm"]
    )
    risk2 = min_heat_distance_value <= anchor2 * thresholds["K_nose_shoulder"]
    norm_risk2 = (
        min_heat_distance_norm_value
        <= anchor2_norm * thresholds["K_nose_shoulder_norm"]
    )

    return {
        "person_id": instance.get("person_id"),
        "nose": [float(nose[0]), float(nose[1])],
        "left_ear": [float(left_ear[0]), float(left_ear[1])],
        "right_ear": [float(right_ear[0]), float(right_ear[1])],
        "left_shoulder": [float(left_shoulder[0]), float(left_shoulder[1])],
        "right_shoulder": [float(right_shoulder[0]), float(right_shoulder[1])],
        "shoulder_center": [float(shoulder_center[0]), float(shoulder_center[1])],
        "nearest_hot_point": nearest_hot_point,
        "min_heat_distance": min_heat_distance_value,
        "min_heat_distance_norm": min_heat_distance_norm_value,
        "anchor1": anchor1,
        "anchor1_norm": anchor1_norm,
        "anchor2": anchor2,
        "anchor2_norm": anchor2_norm,
        "risk1": bool(risk1),
        "norm_risk1": bool(norm_risk1),
        "risk2": bool(risk2),
        "norm_risk2": bool(norm_risk2),
        "max_temperature": float(max_temperature),
    }


def build_visualize_record(thermal_path, image_info, thresholds):
    img_path = Path(image_info["img_path"])
    rgb_image = cv2.imread(str(img_path))
    if rgb_image is None:
        raise FileNotFoundError(f"无法读取对应 RGB 图片: {img_path}")

    img_h, img_w = rgb_image.shape[:2]
    if img_w != TARGET_W or img_h != TARGET_H:
        raise ValueError(
            f"RGB 图片分辨率异常，期望 {TARGET_W}x{TARGET_H}，实际 {img_w}x{img_h}"
        )
    thermal_matrix = load_thermal_npy(thermal_path)
    thermal_resized = align_thermal_to_rgb_resolution(thermal_matrix)
    hot_points = get_hot_points(thermal_resized, overheat_threshold=OVERHEAT_THRESHOLD)
    canvas = overlay_thermal_on_rgb(rgb_image, thermal_resized)

    instance_records = []
    global_risk = False
    for instance in image_info.get("instances", []):
        if instance.get("skipped", False):
            continue
        if instance.get("keypoints") is None:
            continue

        metrics = compute_instance_visual_metrics(
            hot_points=hot_points,
            instance=instance,
            thresholds=thresholds,
            img_h=img_h,
            img_w=img_w,
            max_temperature=float(np.max(thermal_matrix)),
        )
        if metrics is None:
            continue

        draw_anchor1(canvas, metrics)
        draw_anchor2(canvas, metrics)
        draw_thermal_alert_circles(canvas, metrics, thresholds, img_h, img_w)
        draw_hot_point(canvas, metrics)
        draw_person_text(canvas, metrics)

        global_risk = global_risk or any(
            (metrics["risk1"], metrics["norm_risk1"], metrics["risk2"], metrics["norm_risk2"])
        )
        instance_records.append(metrics)

    return {
        "img_path": str(img_path),
        "thermal_path": str(thermal_path),
        "thermal_raw_shape": [int(thermal_matrix.shape[0]), int(thermal_matrix.shape[1])],
        "image_size": [int(img_h), int(img_w)],
        "hot_pixel_count": int(hot_points.shape[0]),
        "max_temperature": float(np.max(thermal_matrix)),
        "global_risk": bool(global_risk),
        "instances": instance_records,
        "canvas": canvas,
    }


def save_visualization_image(record, output_dir: Path):
    src_path = Path(record["img_path"])
    src_stem = src_path.stem
    src_suffix = src_path.suffix

    warning_dir = output_dir / "warning"
    normal_dir = output_dir / "normal"
    warning_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    target_dir = warning_dir if record["global_risk"] else normal_dir
    output_path = target_dir / f"{src_stem}_thermal_predicted{src_suffix}"
    cv2.imwrite(str(output_path), record["canvas"])
    return output_path


def visualize_thermal_dataset(pose_records, thresholds, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = {
        "inputs": {
            "pose_json_paths": [str(path) for path in POSE_JSON_PATHS],
            "no_heat_dir": str(NO_HEAT_DIR),
            "heat_dir": str(HEAT_DIR),
        },
        "thresholds": thresholds,
        "images": {},
    }

    total_visualize_files = sum(
        len(list_thermal_paths(thermal_dir)) for thermal_dir in (NO_HEAT_DIR, HEAT_DIR)
    )
    processed_visualize_files = 0
    print(
        f"[INFO]: 开始热成像可视化 | 输出目录: {output_dir} | 总文件数: {total_visualize_files}"
    )

    for thermal_dir in (NO_HEAT_DIR, HEAT_DIR):
        thermal_paths = list_thermal_paths(thermal_dir)
        print(
            f"[INFO]: 开始处理可视化目录 -> {thermal_dir.name} | 文件数: {len(thermal_paths)}"
        )
        for thermal_path in thermal_paths:
            processed_visualize_files += 1
            if should_report_progress(
                processed_visualize_files, total_visualize_files, report_every=50
            ):
                print_progress(
                    prefix="热成像可视化",
                    index=processed_visualize_files,
                    total=total_visualize_files,
                    extra=thermal_path.name,
                )
            image_info = pose_records.get(thermal_path.stem)
            if image_info is None:
                print(f"[WARNING]: 未找到姿态记录，跳过 {thermal_path.name}")
                continue
            try:
                record = build_visualize_record(thermal_path, image_info, thresholds)
                output_path = save_visualization_image(record, output_dir)
                record.pop("canvas")
                record["output_path"] = str(output_path)
                record["label_dir"] = thermal_dir.name
                all_records["images"][thermal_path.stem] = record
                print(f"[可视化]: 已保存到: {output_path}")
            except Exception as e:
                print(f"[WARNING]: 可视化失败 {thermal_path.name}, 原因: {e}")

    save_result(VIS_SUMMARY_OUTPUT_PATH, all_records)
    print(
        f"[INFO]: 热成像可视化完成 | 已处理: {processed_visualize_files}/{total_visualize_files}"
    )
    return all_records


def search_best_k_for_all_metrics(
    pose_json_paths,
    no_heat_dir: Path,
    heat_dir: Path,
    output_path: Path,
    overheat_threshold: float = 100.0,
):
    print("[INFO]: 开始合并姿态记录...")
    pose_records = merge_pose_records(pose_json_paths)
    print(f"[INFO]: 姿态记录合并完成 | 样本数: {len(pose_records)}")

    no_heat_data = collect_samples(
        thermal_dir=no_heat_dir,
        label=NO_HEAT_LABEL,
        pose_records=pose_records,
        overheat_threshold=overheat_threshold,
    )
    heat_data = collect_samples(
        thermal_dir=heat_dir,
        label=HEAT_LABEL,
        pose_records=pose_records,
        overheat_threshold=overheat_threshold,
    )

    metric_results = {}
    metric_items = list(METRIC_DEFINITIONS.items())
    for metric_index, (metric_name, metric_config) in enumerate(metric_items, start=1):
        print_progress(
            prefix="指标搜索",
            index=metric_index,
            total=len(metric_items),
            extra=metric_name,
        )
        metric_samples = (
            no_heat_data["samples_by_metric"][metric_name]
            + heat_data["samples_by_metric"][metric_name]
        )
        base_result = {
            "display_name": metric_config["display_name"],
            "anchor_field": metric_config["anchor_field"],
            "distance_mode": metric_config["distance_mode"],
            "sample_count": len(metric_samples),
            "score_definition": {
                "pixel": "score = 鼻子到最近高温像素的像素距离 / 像素锚点距离",
                "normalized": "score = 鼻子到最近高温像素的归一化距离 / 归一化锚点距离",
            }[metric_config["distance_mode"]],
            "classification_rule": "score <= K 判定为嘴巴附近有热源",
            "dataset_stats": {
                NO_HEAT_LABEL: {
                    "usable_samples": len(no_heat_data["samples_by_metric"][metric_name]),
                    "score_summary": summarize_scores(
                        no_heat_data["samples_by_metric"][metric_name]
                    ),
                },
                HEAT_LABEL: {
                    "usable_samples": len(heat_data["samples_by_metric"][metric_name]),
                    "score_summary": summarize_scores(
                        heat_data["samples_by_metric"][metric_name]
                    ),
                },
            },
            "samples": metric_samples,
        }

        try:
            best_result = search_best_k(metric_samples)
            base_result["metrics"] = best_result
        except Exception as e:
            base_result["error"] = str(e)

        metric_results[metric_name] = base_result

    result = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
        "overheat_threshold": float(overheat_threshold),
        "resize_target": RESIZE_TARGET,
        "resize_interpolation": RESIZE_INTERPOLATION_NAME,
        "inputs": {
            "pose_json_paths": [str(path) for path in pose_json_paths],
            "no_heat_dir": str(no_heat_dir),
            "heat_dir": str(heat_dir),
        },
        "dataset_stats": {
            NO_HEAT_LABEL: no_heat_data["stats"],
            HEAT_LABEL: heat_data["stats"],
        },
        "metric_results": metric_results,
    }
    save_result(output_path, result)
    print(f"[INFO]: K搜索结果已保存到: {output_path}")
    return result


def print_metric_result(metric_name, metric_result):
    if metric_result.get("error"):
        print(f"\n{'=' * 60}")
        print(f"指标: {metric_name} ({metric_result['distance_mode']})")
        print(f"[ERROR]: {metric_result['error']}")
        return

    metrics = metric_result["metrics"]
    confusion_matrix = metrics["confusion_matrix"]

    print(f"\n{'=' * 60}")
    print(f"指标: {metric_name} ({metric_result['distance_mode']})")
    print(f"最优 K: {metrics['best_K']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"F1-score: {metrics['f1_score']:.2%}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
    print(f"Weighted Accuracy: {metrics['weighted_accuracy']:.2%}")
    print("Confusion Matrix:")
    print(
        "  [[TP, FN], [FP, TN]] = "
        f"[[{confusion_matrix['tp']}, {confusion_matrix['fn']}], "
        f"[{confusion_matrix['fp']}, {confusion_matrix['tn']}]]"
    )


def main():
    print(f"{'=' * 60}")
    print("[INFO]: Step2 启动，开始执行 K 搜索...")
    result = search_best_k_for_all_metrics(
        pose_json_paths=POSE_JSON_PATHS,
        no_heat_dir=NO_HEAT_DIR,
        heat_dir=HEAT_DIR,
        output_path=SEARCH_OUTPUT_PATH,
        overheat_threshold=OVERHEAT_THRESHOLD,
    )
    print("[INFO]: K 搜索完成，开始加载最优阈值...")
    thresholds = load_best_thresholds(result)
    print(f"[INFO]: 最优阈值加载完成: {thresholds}")
    print("[INFO]: 开始重新合并姿态记录，用于可视化...")
    pose_records = merge_pose_records(POSE_JSON_PATHS)
    print("[INFO]: 姿态记录合并完成，开始逐图可视化...")
    visualize_thermal_dataset(
        pose_records=pose_records,
        thresholds=thresholds,
        output_dir=VIS_OUTPUT_DIR,
    )
    print("[INFO]: 可视化完成，开始输出汇总信息...")

    print(f"{'=' * 60}")
    print("step2 thermal detect")
    print(f"{'=' * 60}")
    print(f"[INFO]: 无热源目录: {NO_HEAT_DIR}")
    print(f"[INFO]: 有热源目录: {HEAT_DIR}")
    print(f"[INFO]: K搜索结果保存路径: {SEARCH_OUTPUT_PATH}")
    print(f"[INFO]: 可视化输出目录: {VIS_OUTPUT_DIR}")
    print(f"[INFO]: 可视化摘要保存路径: {VIS_SUMMARY_OUTPUT_PATH}")

    for metric_name, metric_result in result["metric_results"].items():
        print_metric_result(metric_name, metric_result)


if __name__ == "__main__":
    main()
