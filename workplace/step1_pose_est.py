"""
功能说明
1. 基于图片路径重新执行 MMPose 推理，支持单张图片与目录批处理。
2. 以 nose_ear 方案为主线，保留原有 `person_text` 展示逻辑与 JSON 主字段。
3. 同时计算 nose_shoulder 方案，并将其必要信息写入同一个 JSON。
4. 删去不必要的全局可视化，仅保留：
   - `person_text`
   - Anchor1（nose_ear）可视化
   - Anchor2（nose_shoulder）可视化
5. 可视化输出文件命名逻辑保持不变：`{src_stem}_pose_predicted{src_suffix}`。

anchor * K > min_dist
⇔ anchor / min_dist > 1/K
⇔ nonorm_ratio > 1/K
"""

from math import sqrt
from pathlib import Path
import json
import sys
import time

import cv2

PROJECT_ROOT = "/root/autodl-tmp/projects/dongguan/Github/mmpose"
sys.path.insert(0, PROJECT_ROOT)

from mmpose.apis import MMPoseInferencer


INPUT_PATHS = [
    "/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态危险",
    "/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态正常",
]
VIS_OUTPUT_DIR = "outputs/pose_est_vis"
JSON_OUTPUT_PATH = "outputs/jsons/pose_metrics.json"

NOSE_EAR_THRESHOLD_JSON = (
    Path(__file__).resolve().parent
    / "outputs"
    / "jsons"
    / "empirical_threshold_summary_nose_ear.json"
)
NOSE_SHOULDER_THRESHOLD_JSON = (
    Path(__file__).resolve().parent
    / "outputs"
    / "jsons"
    / "empirical_threshold_summary_nose_shoulder_v2.json"
)

DEFAULT_K_NOSE_EAR = 2.35
DEFAULT_K_NOSE_EAR_NORM = 3.6
DEFAULT_K_NOSE_SHOULDER = 1.5
DEFAULT_K_NOSE_SHOULDER_NORM = 2.0
KEYPOINTS_SCORE_THRESHOLD = 0.4
MIN_KEYPOINTS = 5

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


def distance_between_points(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def normalized_distance_between_points(p1, p2, h, w):
    norm_x_dist = (p1[0] - p2[0]) / w
    norm_y_dist = (p1[1] - p2[1]) / h
    return sqrt(norm_x_dist**2 + norm_y_dist**2)


def round_float(value, ndigits=6):
    if value is None:
        return None
    return round(float(value), ndigits)


def to_point(point):
    return [round_float(point[0]), round_float(point[1])]


def serialize_keypoints(keypoints):
    return [to_point(point) for point in keypoints]


def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metric_threshold(summary_path: Path, metric_name: str, default_value: float) -> float:
    if not summary_path.exists():
        print(f"[WARNING]: 未找到阈值文件 {summary_path}，使用默认值 {default_value}")
        return default_value

    data = load_json(summary_path)
    value = data.get("metrics", {}).get(metric_name, {}).get("best_threshold")
    if value is None:
        print(f"[WARNING]: 阈值文件 {summary_path} 中缺少 {metric_name}，使用默认值 {default_value}")
        return default_value

    return float(value)


def load_trained_thresholds():
    # pose_est_calc.py L208 / pose_est_calc copy.py L199
    # risk = anchor_nose_ear * K_nose_ear > min_dist
    
    # anchor * K > min_dist
    # ⇔ anchor / min_dist > 1/K
    # ⇔ nonorm_ratio > 1/K
    thresholds = {
        "K_nose_ear": load_metric_threshold(
            NOSE_EAR_THRESHOLD_JSON, "nonorm_ratio", 1/DEFAULT_K_NOSE_EAR
        ),
        "K_nose_ear_norm": load_metric_threshold(
            NOSE_EAR_THRESHOLD_JSON, "norm_ratio", 1/DEFAULT_K_NOSE_EAR_NORM
        ),
        "K_nose_shoulder": load_metric_threshold(
            NOSE_SHOULDER_THRESHOLD_JSON,
            "nonorm_ratio",
            1/DEFAULT_K_NOSE_SHOULDER,
        ),
        "K_nose_shoulder_norm": load_metric_threshold(
            NOSE_SHOULDER_THRESHOLD_JSON,
            "norm_ratio",
            1/DEFAULT_K_NOSE_SHOULDER_NORM,
        ),
    }

    print("[INFO]: 已加载阈值")
    print(f"[INFO]: 阈值详情:")
    for key, value in thresholds.items():
        print(f"        {key} = {value}")
    return thresholds


def list_image_paths(input_path: Path):
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        image_paths = [
            path
            for path in sorted(input_path.iterdir())
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        return image_paths
    raise FileNotFoundError(f"输入路径不存在: {input_path}")


def build_json_output_path(base_output_path: str, input_path: Path, timestamp: str) -> Path:
    base_path = Path(base_output_path)
    if input_path.is_dir():
        postfix = input_path.name
    else:
        postfix = input_path.stem
    return Path(f"{base_path.with_suffix('')}_{postfix}_{timestamp}.json")


def get_keypoint_scores(person):
    return [float(score) for score in person.get("keypoint_scores", [])]


def get_required_scores(scores, keypoint_names, required_names):
    return [scores[keypoint_names.index(name)] for name in required_names]


def build_skip_record(person_id, keypoints, skip_reason):
    return {
        "person_id": person_id,
        "nose_ear": None,
        "nose_ear_norm": None,
        "nose_wrist": None,
        "nose_wrist_norm": None,
        "nonorm_ratio": None,
        "norm_ratio": None,
        "skipped": True,
        "skip_reason": skip_reason,
        "risk": None,
        "norm_risk": None,
        "keypoints": serialize_keypoints(keypoints),
    }


def evaluate_common_skip(scores):
    qualified_keypoints_cnt = sum(
        1 for score in scores if score > KEYPOINTS_SCORE_THRESHOLD
    )
    if qualified_keypoints_cnt < MIN_KEYPOINTS:
        return f"关键点数量不足{MIN_KEYPOINTS}个"

    nose_score, left_wrist_score, right_wrist_score = get_required_scores(
        scores, KEYPOINT_NAMES, ["nose", "left_wrist", "right_wrist"]
    )
    nose_skip = nose_score < KEYPOINTS_SCORE_THRESHOLD
    wrist_skip = (
        left_wrist_score < KEYPOINTS_SCORE_THRESHOLD
        and right_wrist_score < KEYPOINTS_SCORE_THRESHOLD
    )
    if nose_skip or wrist_skip:
        return f"关键点score低于{KEYPOINTS_SCORE_THRESHOLD}"
    return None


def evaluate_nose_ear_metrics(keypoints, img_h, img_w, thresholds, scores):
    left_ear_score, right_ear_score = get_required_scores(
        scores, KEYPOINT_NAMES, ["left_ear", "right_ear"]
    )
    ear_skip = (
        left_ear_score < KEYPOINTS_SCORE_THRESHOLD
        and right_ear_score < KEYPOINTS_SCORE_THRESHOLD
    )
    if ear_skip:
        return {
            "skipped": True,
            "skip_reason": f"关键点score低于{KEYPOINTS_SCORE_THRESHOLD}",
        }

    nose = keypoints[KEYPOINT_NAMES.index("nose")]
    left_wrist = keypoints[KEYPOINT_NAMES.index("left_wrist")]
    right_wrist = keypoints[KEYPOINT_NAMES.index("right_wrist")]
    left_ear = keypoints[KEYPOINT_NAMES.index("left_ear")]
    right_ear = keypoints[KEYPOINT_NAMES.index("right_ear")]

    anchor_nose_ear = max(
        distance_between_points(nose, left_ear),
        distance_between_points(nose, right_ear),
    )
    max_norm_anchor_nose_ear = max(
        normalized_distance_between_points(nose, left_ear, img_h, img_w),
        normalized_distance_between_points(nose, right_ear, img_h, img_w),
    )
    dist_l = distance_between_points(nose, left_wrist)
    dist_r = distance_between_points(nose, right_wrist)
    min_dist = min(dist_l, dist_r)
    norm_dist_l = normalized_distance_between_points(nose, left_wrist, img_h, img_w)
    norm_dist_r = normalized_distance_between_points(nose, right_wrist, img_h, img_w)
    min_norm_dist = min(norm_dist_l, norm_dist_r)

    nonorm_ratio = anchor_nose_ear / min_dist if min_dist > 0 else None
    norm_ratio = max_norm_anchor_nose_ear / min_norm_dist if min_norm_dist > 0 else None
    risk = anchor_nose_ear * thresholds["K_nose_ear"] > min_dist
    norm_risk = max_norm_anchor_nose_ear * thresholds["K_nose_ear_norm"] > min_norm_dist

    return {
        "skipped": False,
        "skip_reason": None,
        "nose_ear": round_float(anchor_nose_ear),
        "nose_ear_norm": round_float(max_norm_anchor_nose_ear),
        "nose_wrist": round_float(min_dist),
        "nose_wrist_norm": round_float(min_norm_dist),
        "nonorm_ratio": round_float(nonorm_ratio) if nonorm_ratio is not None else None,
        "norm_ratio": round_float(norm_ratio) if norm_ratio is not None else None,
        "risk": bool(risk),
        "norm_risk": bool(norm_risk),
        "dist_l": float(dist_l),
        "dist_r": float(dist_r),
        "norm_dist_l": float(norm_dist_l),
        "norm_dist_r": float(norm_dist_r),
        "anchor_value": float(anchor_nose_ear),
        "anchor_norm_value": float(max_norm_anchor_nose_ear),
        "nose": nose,
        "left_ear": left_ear,
        "right_ear": right_ear,
    }


def evaluate_nose_shoulder_metrics(keypoints, img_h, img_w, thresholds, scores):
    left_shoulder_score, right_shoulder_score = get_required_scores(
        scores, KEYPOINT_NAMES, ["left_shoulder", "right_shoulder"]
    )
    shoulder_skip = (
        left_shoulder_score < KEYPOINTS_SCORE_THRESHOLD
        and right_shoulder_score < KEYPOINTS_SCORE_THRESHOLD
    )
    if shoulder_skip:
        return {
            "skipped": True,
            "skip_reason": f"关键点score低于{KEYPOINTS_SCORE_THRESHOLD}",
        }

    nose = keypoints[KEYPOINT_NAMES.index("nose")]
    left_wrist = keypoints[KEYPOINT_NAMES.index("left_wrist")]
    right_wrist = keypoints[KEYPOINT_NAMES.index("right_wrist")]
    left_shoulder = keypoints[KEYPOINT_NAMES.index("left_shoulder")]
    right_shoulder = keypoints[KEYPOINT_NAMES.index("right_shoulder")]

    shoulder_center = [
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    ]
    anchor_nose_shoulder = distance_between_points(nose, shoulder_center)
    norm_anchor_nose_shoulder = normalized_distance_between_points(
        nose, shoulder_center, img_h, img_w
    )
    dist_l = distance_between_points(nose, left_wrist)
    dist_r = distance_between_points(nose, right_wrist)
    min_dist = min(dist_l, dist_r)
    norm_dist_l = normalized_distance_between_points(nose, left_wrist, img_h, img_w)
    norm_dist_r = normalized_distance_between_points(nose, right_wrist, img_h, img_w)
    min_norm_dist = min(norm_dist_l, norm_dist_r)

    nonorm_ratio = anchor_nose_shoulder / min_dist if min_dist > 0 else None
    norm_ratio = (
        norm_anchor_nose_shoulder / min_norm_dist if min_norm_dist > 0 else None
    )
    risk = anchor_nose_shoulder * thresholds["K_nose_shoulder"] > min_dist
    norm_risk = (
        norm_anchor_nose_shoulder * thresholds["K_nose_shoulder_norm"] > min_norm_dist
    )

    return {
        "skipped": False,
        "skip_reason": None,
        "nose_shoulder": round_float(anchor_nose_shoulder),
        "nose_shoulder_norm": round_float(norm_anchor_nose_shoulder),
        "nose_wrist_nose_shoulder": round_float(min_dist),
        "nose_wrist_norm_nose_shoulder": round_float(min_norm_dist),
        "nonorm_ratio_nose_shoulder": (
            round_float(nonorm_ratio) if nonorm_ratio is not None else None
        ),
        "norm_ratio_nose_shoulder": (
            round_float(norm_ratio) if norm_ratio is not None else None
        ),
        "skipped_nose_shoulder": False,
        "skip_reason_nose_shoulder": None,
        "risk_nose_shoulder": bool(risk),
        "norm_risk_nose_shoulder": bool(norm_risk),
        "nose": nose,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "shoulder_center": shoulder_center,
    }


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


def normalized_radius_to_pixels(norm_radius, img_h, img_w):
    # 归一化距离无法无损还原为单一像素半径，这里用图像对角线做近似，
    # 便于将归一化阈值区域直观投影到原图中供人工核验。
    img_diag = sqrt(img_h**2 + img_w**2)
    return int(round(norm_radius * img_diag))


def draw_anchor1_alert_circles(canvas, metrics, thresholds, img_h, img_w):
    nose = tuple(int(v) for v in metrics["nose"])
    # 经验阈值 T 来自 ratio = anchor / wrist_distance，因此警报半径应为 anchor / T。
    radius_nonorm = int(round(metrics["anchor_value"] / thresholds["K_nose_ear"]))
    radius_norm = normalized_radius_to_pixels(
        metrics["anchor_norm_value"] / thresholds["K_nose_ear_norm"],
        img_h,
        img_w,
    )

    if radius_nonorm > 0:
        cv2.circle(canvas, nose, radius_nonorm, (0, 128, 0), 2)
    if radius_norm > 0:
        cv2.circle(canvas, nose, radius_norm, (144, 238, 144), 2)


def draw_anchor2_alert_circles(canvas, metrics, thresholds, img_h, img_w):
    nose = tuple(int(v) for v in metrics["nose"])
    radius_nonorm = int(
        round(metrics["nose_shoulder"] / thresholds["K_nose_shoulder"])
    )
    radius_norm = normalized_radius_to_pixels(
        metrics["nose_shoulder_norm"] / thresholds["K_nose_shoulder_norm"],
        img_h,
        img_w,
    )

    if radius_nonorm > 0:
        cv2.circle(canvas, nose, radius_nonorm, (128, 0, 128), 2)
    if radius_norm > 0:
        cv2.circle(canvas, nose, radius_norm, (255, 128, 255), 2)


def draw_person_text(canvas, metrics):
    nose = metrics["nose"]
    person_text = (
        f"L:{metrics['dist_l']:.1f}  R:{metrics['dist_r']:.1f}  "
        f"L_norm:{metrics['norm_dist_l']:.3f}  R_norm:{metrics['norm_dist_r']:.3f}  "
        f"Anchor:{metrics['anchor_value']:.3f}  "
        f"Anchor_norm:{metrics['anchor_norm_value']:.3f}  "
        f"Risk:{metrics['risk']}  Norm_Risk:{metrics['norm_risk']}"
    )
    nose_x, nose_y = int(nose[0]), int(nose[1])
    text_x = nose_x
    text_y = nose_y - 10 if nose_y - 10 > 20 else nose_y + 20

    # 将文本放置在左下角，使用更大的加粗字体
    text_x = 20  # 距离左边缘20像素
    text_y = canvas.shape[0] - 30  # 距离底部30像素
    cv2.putText(
        canvas,
        person_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,  # 增大字体尺寸
        (255, 0, 0),
        2,  # 增加线条宽度实现加粗效果
    )


def visualize_pose_distances(img_path, instances, canvas, output_dir, thresholds):
    src_path = Path(img_path)
    src_stem = src_path.stem
    src_suffix = src_path.suffix

    img_h, img_w = canvas.shape[:2]
    global_risk = False
    global_norm_risk = False
    global_risk_nose_shoulder = False
    global_norm_risk_nose_shoulder = False
    skip_cnt = 0
    skip_cnt_nose_shoulder = 0
    instance_records = []

    for person_id, person in enumerate(instances):
        keypoints = person["keypoints"]
        scores = get_keypoint_scores(person)
        common_skip_reason = evaluate_common_skip(scores)

        if common_skip_reason is not None:
            print(f"[LOG]: 跳过该人:{person_id},{common_skip_reason}")
            base_record = build_skip_record(person_id, keypoints, common_skip_reason)
            base_record.update(
                {
                    "nose_shoulder": None,
                    "nose_shoulder_norm": None,
                    "nose_wrist_nose_shoulder": None,
                    "nose_wrist_norm_nose_shoulder": None,
                    "nonorm_ratio_nose_shoulder": None,
                    "norm_ratio_nose_shoulder": None,
                    "skipped_nose_shoulder": True,
                    "skip_reason_nose_shoulder": common_skip_reason,
                    "risk_nose_shoulder": None,
                    "norm_risk_nose_shoulder": None,
                }
            )
            instance_records.append(base_record)
            skip_cnt += 1
            skip_cnt_nose_shoulder += 1
            continue

        nose_ear_metrics = evaluate_nose_ear_metrics(
            keypoints, img_h, img_w, thresholds, scores
        )
        nose_shoulder_metrics = evaluate_nose_shoulder_metrics(
            keypoints, img_h, img_w, thresholds, scores
        )

        if nose_ear_metrics["skipped"]:
            skip_cnt += 1
        else:
            global_risk = global_risk or nose_ear_metrics["risk"]
            global_norm_risk = global_norm_risk or nose_ear_metrics["norm_risk"]
            draw_anchor1_alert_circles(
                canvas, nose_ear_metrics, thresholds, img_h, img_w
            )
            draw_person_text(canvas, nose_ear_metrics)
            draw_anchor1(canvas, nose_ear_metrics)

        if nose_shoulder_metrics["skipped"]:
            skip_cnt_nose_shoulder += 1
        else:
            global_risk_nose_shoulder = (
                global_risk_nose_shoulder or nose_shoulder_metrics["risk_nose_shoulder"]
            )
            global_norm_risk_nose_shoulder = (
                global_norm_risk_nose_shoulder
                or nose_shoulder_metrics["norm_risk_nose_shoulder"]
            )
            draw_anchor2_alert_circles(
                canvas, nose_shoulder_metrics, thresholds, img_h, img_w
            )
            draw_anchor2(canvas, nose_shoulder_metrics)

        merged_record = {
            "person_id": person_id,
            "nose_ear": nose_ear_metrics.get("nose_ear"),
            "nose_ear_norm": nose_ear_metrics.get("nose_ear_norm"),
            "nose_wrist": nose_ear_metrics.get("nose_wrist"),
            "nose_wrist_norm": nose_ear_metrics.get("nose_wrist_norm"),
            "nonorm_ratio": nose_ear_metrics.get("nonorm_ratio"),
            "norm_ratio": nose_ear_metrics.get("norm_ratio"),
            "skipped": nose_ear_metrics["skipped"],
            "skip_reason": nose_ear_metrics.get("skip_reason"),
            "risk": nose_ear_metrics.get("risk"),
            "norm_risk": nose_ear_metrics.get("norm_risk"),
            "keypoints": serialize_keypoints(keypoints),
            "nose_shoulder": nose_shoulder_metrics.get("nose_shoulder"),
            "nose_shoulder_norm": nose_shoulder_metrics.get("nose_shoulder_norm"),
            "nose_wrist_nose_shoulder": nose_shoulder_metrics.get(
                "nose_wrist_nose_shoulder"
            ),
            "nose_wrist_norm_nose_shoulder": nose_shoulder_metrics.get(
                "nose_wrist_norm_nose_shoulder"
            ),
            "nonorm_ratio_nose_shoulder": nose_shoulder_metrics.get(
                "nonorm_ratio_nose_shoulder"
            ),
            "norm_ratio_nose_shoulder": nose_shoulder_metrics.get(
                "norm_ratio_nose_shoulder"
            ),
            "skipped_nose_shoulder": nose_shoulder_metrics["skipped"],
            "skip_reason_nose_shoulder": nose_shoulder_metrics.get("skip_reason"),
            "risk_nose_shoulder": nose_shoulder_metrics.get("risk_nose_shoulder"),
            "norm_risk_nose_shoulder": nose_shoulder_metrics.get(
                "norm_risk_nose_shoulder"
            ),
        }
        instance_records.append(merged_record)

        print(
            "[LOG]: 检测该人[{}]的耳锚点风险状态: {}  耳锚点归一化风险状态: {}  "
            "肩锚点风险状态: {}  肩锚点归一化风险状态: {}".format(
                person_id,
                nose_ear_metrics.get("risk"),
                nose_ear_metrics.get("norm_risk"),
                nose_shoulder_metrics.get("risk_nose_shoulder"),
                nose_shoulder_metrics.get("norm_risk_nose_shoulder"),
            )
        )

    out_path = Path(output_dir)
    warning_out_path = out_path / "warning"
    normal_out_path = out_path / "normal"
    warning_out_path.mkdir(parents=True, exist_ok=True)
    normal_out_path.mkdir(parents=True, exist_ok=True)

    has_any_risk = any(
        (
            global_risk,
            global_norm_risk,
            global_risk_nose_shoulder,
            global_norm_risk_nose_shoulder,
        )
    )
    target_out_path = warning_out_path if has_any_risk else normal_out_path
    out_file = target_out_path / f"{src_stem}_pose_predicted{src_suffix}"
    cv2.imwrite(str(out_file), canvas)
    print(f"[可视化]: 已保存到: {out_file}")
    return {
        "img_path": str(src_path),
        "instances": instance_records,
        "global_risk": global_risk,
        "global_norm_risk": global_norm_risk,
        "skip_persons": skip_cnt,
        "global_risk_nose_shoulder": global_risk_nose_shoulder,
        "global_norm_risk_nose_shoulder": global_norm_risk_nose_shoulder,
        "skip_persons_nose_shoulder": skip_cnt_nose_shoulder,
    }


def build_json_records(predicted_path, thresholds, timestamp, current_json_output_path):
    return {
        "IS_DIR": Path(predicted_path).is_dir(),
        "predicted_path": str(predicted_path),
        "params": {
            "K_nose_ear": thresholds["K_nose_ear"],
            "K_nose_ear_norm": thresholds["K_nose_ear_norm"],
            "K_nose_shoulder": thresholds["K_nose_shoulder"],
            "K_nose_shoulder_norm": thresholds["K_nose_shoulder_norm"],
            "keypoints_score_threshold": KEYPOINTS_SCORE_THRESHOLD,
            "min_keypoints": MIN_KEYPOINTS,
        },
        "outputs": {
            VIS_OUTPUT_DIR: str(Path(VIS_OUTPUT_DIR).resolve()),
            str(current_json_output_path): str(current_json_output_path.resolve()),
        },
        "timestamp": timestamp,
        "infoes": {},
    }


def run_for_input_path(inferencer, input_path: Path, thresholds, timestamp):
    image_paths = list_image_paths(input_path)
    current_json_output_path = build_json_output_path(
        JSON_OUTPUT_PATH, input_path, timestamp
    )
    json_records = build_json_records(
        input_path, thresholds, timestamp, current_json_output_path
    )

    print(f"\n{'=' * 60}")
    print(f"开始处理路径: {input_path}")
    print(f"图片数量: {len(image_paths)}")
    print(f"{'=' * 60}\n")

    for img_idx, img_path in enumerate(image_paths):
        print(f"\n[LOG]: Iter IMG {img_idx}/{len(image_paths)} -> {img_path}")
        results = inferencer(str(img_path), show=False, return_vis=True)
        for result in results:
            instances = result["predictions"][0]
            if not result.get("visualization"):
                raise ValueError(f"未获取到 MMPose 可视化结果: {img_path}")
            # MMPose 返回 RGB 图像，这里转成 OpenCV 使用的 BGR 画布继续叠加文本和锚点。
            canvas = cv2.cvtColor(result["visualization"][0], cv2.COLOR_RGB2BGR)
            frame_record = visualize_pose_distances(
                img_path=img_path,
                instances=instances,
                canvas=canvas,
                output_dir=VIS_OUTPUT_DIR,
                thresholds=thresholds,
            )
            json_records["infoes"][img_path.name] = frame_record

    current_json_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(current_json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)
    print(f"[JSON]: 已保存到: {current_json_output_path}")


def main():
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    thresholds = load_trained_thresholds()
    inferencer = MMPoseInferencer("human")

    for raw_input_path in INPUT_PATHS:
        run_for_input_path(inferencer, Path(raw_input_path), thresholds, timestamp)


if __name__ == "__main__":
    main()
