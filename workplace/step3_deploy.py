# 根据Step1和Step2可用的阈值，留有两个场景的超参条件，
# 条件一是基于姿态估计模型的预测，分别为四个锚点的选择以及K值的宏定义（我直接输入就行）以及step1的其他超参，例如KEYPOINTS_SCORE_THRESHOLD = 0.4 MIN_KEYPOINTS = 5
# 条件二是热成像检测，分别为四个锚点的选择+最简单的高温检测（画面中最高温超过100度，则检测为预警），高温阈值定义以及K值的宏定义以及
# 只有当一和二都预警，才报警可能有人吸烟。

# 待检测的图像可选择为单一文件或者一个目录，检测后可使用超参选择是否show，是否保存检测图像，以及画布上是否绘制类似Step1和Step2的person_text
# 输入和输出目录作为可填写的内容

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import argparse
import sys
from typing import Iterable, Optional

import cv2
import numpy as np

PROJECT_ROOT = "/root/autodl-tmp/projects/dongguan/Github/mmpose"
sys.path.insert(0, PROJECT_ROOT)

from mmpose.apis import MMPoseInferencer


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


def list_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return [
            path
            for path in sorted(input_path.iterdir())
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    raise FileNotFoundError(f"输入路径不存在: {input_path}")


def get_keypoint_scores(person: dict) -> list[float]:
    return [float(score) for score in person.get("keypoint_scores", [])]


def distance_between_points(p1, p2) -> float:
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def normalized_distance_between_points(p1, p2, h: int, w: int) -> float:
    norm_x_dist = (p1[0] - p2[0]) / float(w)
    norm_y_dist = (p1[1] - p2[1]) / float(h)
    return sqrt(norm_x_dist**2 + norm_y_dist**2)


def normalized_radius_to_pixels(norm_radius: float, img_h: int, img_w: int) -> int:
    img_diag = sqrt(float(img_h) ** 2 + float(img_w) ** 2)
    return int(round(float(norm_radius) * img_diag))


def evaluate_common_skip(
    scores: list[float],
    keypoints_score_threshold: float,
    min_keypoints: int,
) -> Optional[str]:
    qualified_keypoints_cnt = sum(1 for score in scores if float(score) > keypoints_score_threshold)
    if qualified_keypoints_cnt < min_keypoints:
        return f"关键点数量不足{min_keypoints}个"

    nose_score = scores[KEYPOINT_NAMES.index("nose")]
    left_wrist_score = scores[KEYPOINT_NAMES.index("left_wrist")]
    right_wrist_score = scores[KEYPOINT_NAMES.index("right_wrist")]
    nose_skip = float(nose_score) < keypoints_score_threshold
    wrist_skip = (
        float(left_wrist_score) < keypoints_score_threshold
        and float(right_wrist_score) < keypoints_score_threshold
    )
    if nose_skip or wrist_skip:
        return f"关键点score低于{keypoints_score_threshold}"
    return None


@dataclass(frozen=True)
class PoseAnchors:
    nose: np.ndarray
    left_ear: np.ndarray
    right_ear: np.ndarray
    left_shoulder: np.ndarray
    right_shoulder: np.ndarray
    shoulder_center: np.ndarray
    left_wrist: np.ndarray
    right_wrist: np.ndarray

    nose_ear: float
    nose_ear_norm: float
    nose_shoulder: float
    nose_shoulder_norm: float

    nose_wrist_min: float
    nose_wrist_min_norm: float


def compute_pose_anchors(keypoints: np.ndarray, img_h: int, img_w: int) -> PoseAnchors:
    nose = keypoints[KEYPOINT_NAMES.index("nose"), :2].astype(np.float32)
    left_ear = keypoints[KEYPOINT_NAMES.index("left_ear"), :2].astype(np.float32)
    right_ear = keypoints[KEYPOINT_NAMES.index("right_ear"), :2].astype(np.float32)
    left_shoulder = keypoints[KEYPOINT_NAMES.index("left_shoulder"), :2].astype(np.float32)
    right_shoulder = keypoints[KEYPOINT_NAMES.index("right_shoulder"), :2].astype(np.float32)
    shoulder_center = ((left_shoulder + right_shoulder) / 2.0).astype(np.float32)
    left_wrist = keypoints[KEYPOINT_NAMES.index("left_wrist"), :2].astype(np.float32)
    right_wrist = keypoints[KEYPOINT_NAMES.index("right_wrist"), :2].astype(np.float32)

    nose_ear = float(
        max(
            distance_between_points(nose, left_ear),
            distance_between_points(nose, right_ear),
        )
    )
    nose_ear_norm = float(
        max(
            normalized_distance_between_points(nose, left_ear, img_h, img_w),
            normalized_distance_between_points(nose, right_ear, img_h, img_w),
        )
    )
    nose_shoulder = float(distance_between_points(nose, shoulder_center))
    nose_shoulder_norm = float(
        normalized_distance_between_points(nose, shoulder_center, img_h, img_w)
    )

    dist_l = float(distance_between_points(nose, left_wrist))
    dist_r = float(distance_between_points(nose, right_wrist))
    nose_wrist_min = float(min(dist_l, dist_r))
    norm_dist_l = float(normalized_distance_between_points(nose, left_wrist, img_h, img_w))
    norm_dist_r = float(normalized_distance_between_points(nose, right_wrist, img_h, img_w))
    nose_wrist_min_norm = float(min(norm_dist_l, norm_dist_r))

    return PoseAnchors(
        nose=nose,
        left_ear=left_ear,
        right_ear=right_ear,
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
        shoulder_center=shoulder_center,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        nose_ear=nose_ear,
        nose_ear_norm=nose_ear_norm,
        nose_shoulder=nose_shoulder,
        nose_shoulder_norm=nose_shoulder_norm,
        nose_wrist_min=nose_wrist_min,
        nose_wrist_min_norm=nose_wrist_min_norm,
    )


def draw_anchor_lines(canvas: np.ndarray, anchors: PoseAnchors, metric: str) -> None:
    nose = tuple(int(v) for v in anchors.nose.tolist())
    if metric.startswith("nose_ear"):
        left_ear = tuple(int(v) for v in anchors.left_ear.tolist())
        right_ear = tuple(int(v) for v in anchors.right_ear.tolist())
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
        return

    left_shoulder = tuple(int(v) for v in anchors.left_shoulder.tolist())
    right_shoulder = tuple(int(v) for v in anchors.right_shoulder.tolist())
    shoulder_center = tuple(int(v) for v in anchors.shoulder_center.tolist())
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


def draw_pose_alert_circle(
    canvas: np.ndarray, anchors: PoseAnchors, pose_metric: str, pose_k: float, img_h: int, img_w: int
) -> int:
    nose = tuple(int(v) for v in anchors.nose.tolist())
    if pose_metric.endswith("_norm"):
        radius = normalized_radius_to_pixels(getattr(anchors, pose_metric) * float(pose_k), img_h, img_w)
    else:
        radius = int(round(getattr(anchors, pose_metric) * float(pose_k)))
    if radius > 0:
        cv2.circle(canvas, nose, radius, (0, 0, 255), 2)
    return radius


def draw_text_line(
    canvas: np.ndarray,
    text: str,
    line_index: int,
    color: tuple[int, int, int] = (255, 0, 0),
    font_scale: float = 0.7,
    thickness: int = 2,
) -> None:
    x = 20
    base_y = canvas.shape[0] - 20
    y = base_y - (line_index * 28)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def resolve_thermal_for_rgb(rgb_path: Path, thermal_input: Optional[Path]) -> Optional[Path]:
    if thermal_input is None:
        return None
    if thermal_input.is_file():
        return thermal_input
    if thermal_input.is_dir():
        candidate = thermal_input / f"{rgb_path.stem}.npy"
        if candidate.exists():
            return candidate
        matches = sorted(thermal_input.glob(f"{rgb_path.stem}.*"))
        if matches:
            return matches[0]
        return None
    return None


def load_thermal_matrix(thermal_path: Path) -> np.ndarray:
    thermal = np.load(str(thermal_path))
    thermal = np.asarray(thermal)
    thermal = np.squeeze(thermal)
    if thermal.ndim != 2:
        raise ValueError(f"热成像矩阵维度异常: {thermal_path}, shape={thermal.shape}")
    return thermal.astype(np.float32)


def align_thermal_to_rgb_resolution(thermal_matrix: np.ndarray, rgb_h: int, rgb_w: int) -> np.ndarray:
    thermal_h, thermal_w = thermal_matrix.shape[:2]
    scale_x = float(rgb_w) / float(thermal_w)
    scale_y = float(rgb_h) / float(thermal_h)
    trans_x = 0.5 * scale_x - 0.5
    trans_y = 0.5 * scale_y - 0.5
    homography = np.array(
        [[scale_x, 0.0, trans_x], [0.0, scale_y, trans_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return cv2.warpPerspective(
        thermal_matrix,
        homography,
        (int(rgb_w), int(rgb_h)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def create_thermal_heatmap(thermal_resized: np.ndarray) -> np.ndarray:
    min_value = float(np.min(thermal_resized))
    max_value = float(np.max(thermal_resized))
    if max_value > min_value:
        normalized = (thermal_resized - min_value) / (max_value - min_value)
    else:
        normalized = np.zeros_like(thermal_resized, dtype=np.float32)
    heatmap_uint8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


def overlay_thermal_on_rgb(rgb_image: np.ndarray, thermal_resized: np.ndarray, alpha: float) -> np.ndarray:
    heatmap = create_thermal_heatmap(thermal_resized)
    return cv2.addWeighted(rgb_image, 1.0 - float(alpha), heatmap, float(alpha), 0.0)


def get_hot_points(thermal_resized: np.ndarray, overheat_threshold: float) -> np.ndarray:
    hot_y, hot_x = np.where(thermal_resized >= float(overheat_threshold))
    if hot_y.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack((hot_y, hot_x)).astype(np.float32)


def get_min_heat_distances(
    hot_points: np.ndarray, nose_point_xy: np.ndarray, img_h: int, img_w: int
) -> tuple[float, float, Optional[np.ndarray]]:
    if hot_points.size == 0:
        return float("inf"), float("inf"), None

    delta_y = hot_points[:, 0] - float(nose_point_xy[1])
    delta_x = hot_points[:, 1] - float(nose_point_xy[0])
    pixel_distances = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    min_index = int(np.argmin(pixel_distances))
    min_pixel_distance = float(pixel_distances[min_index])

    min_hot_y = float(hot_points[min_index, 0])
    min_hot_x = float(hot_points[min_index, 1])
    min_norm_distance = float(
        normalized_distance_between_points(
            nose_point_xy,
            np.array([min_hot_x, min_hot_y], dtype=np.float32),
            img_h,
            img_w,
        )
    )
    nearest_hot_xy = np.array([min_hot_x, min_hot_y], dtype=np.float32)
    return min_pixel_distance, min_norm_distance, nearest_hot_xy


def save_image_grouped(
    canvas: np.ndarray, output_dir: Path, group_name: str, is_warning: bool, src_path: Path, suffix: str
) -> Path:
    warning_dir = output_dir / group_name / "warning"
    normal_dir = output_dir / group_name / "normal"
    warning_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    target_dir = warning_dir if is_warning else normal_dir
    out_path = target_dir / f"{src_path.stem}_{suffix}{src_path.suffix}"
    cv2.imwrite(str(out_path), canvas)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_path", type=str, 
                        default='/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态正常/000045_20260521_112200.png', 
                        help="RGB图片路径（单文件或目录）")
    parser.add_argument("--thermal_path", type=str, 
                        default='/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal/嘴巴附近无热源/000045_20260521_112200.npy', 
                        help="热成像npy路径（单文件或目录，可为空）")
    parser.add_argument("--output_dir", type=str, default="outputs/step3_deploy", help="输出目录")

    pose_metric_options = ["nose_ear", "nose_ear_norm", "nose_shoulder", "nose_shoulder_norm"]
    thermal_metric_options = ["nose_ear", "nose_ear_norm", "nose_shoulder", "nose_shoulder_norm"]
    thermal_mode_options = ["overheat_and_near_nose", "overheat_only"]
    parser.add_argument("--pose_metric", type=str, default="nose_ear", choices=pose_metric_options)
    parser.add_argument("--pose_k", type=float, default=2.35)
    parser.add_argument("--thermal_metric", type=str, default="nose_ear", choices=thermal_metric_options)
    parser.add_argument("--thermal_k", type=float, default=1.5)
    parser.add_argument("--overheat_threshold", type=float, default=100.0)
    parser.add_argument(
        "--thermal_mode",
        type=str,
        default="overheat_only",
        choices=thermal_mode_options,
    )

    parser.add_argument("--keypoints_score_threshold", type=float, default=0.4)
    parser.add_argument("--min_keypoints", type=int, default=5)

    parser.add_argument("--thermal_blend_alpha", type=float, default=0.35)
    parser.add_argument("--draw_person_text", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rgb_input = Path(args.rgb_path)
    thermal_input = None if args.thermal_path is None else Path(args.thermal_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inferencer = MMPoseInferencer("human")
    image_paths = list_image_paths(rgb_input)
    if not image_paths:
        raise ValueError(f"未找到可用图片: {rgb_input}")
    file_len = len(image_paths)
    for img_idx, img_path in enumerate(image_paths):
        print(f"\n[Progress]: {img_idx}/{file_len}")
        results = inferencer(str(img_path), show=False, return_vis=True)
        pose_result = None
        for result in results:
            pose_result = result
            break
        if pose_result is None:
            raise ValueError(f"未获取到 MMPose 推理结果: {img_path}")
        if not pose_result.get("visualization"):
            raise ValueError(f"未获取到 MMPose 可视化结果: {img_path}")

        pose_canvas = cv2.cvtColor(pose_result["visualization"][0], cv2.COLOR_RGB2BGR)
        rgb_image = cv2.imread(str(img_path))
        if rgb_image is None:
            raise FileNotFoundError(f"无法读取RGB图片: {img_path}")
        img_h, img_w = rgb_image.shape[:2]

        instances = pose_result["predictions"][0]

        cond1_global = False
        valid_persons = 0
        for person_id, person in enumerate(instances):
            keypoints = np.asarray(person.get("keypoints", []), dtype=np.float32)
            if keypoints.ndim != 2 or keypoints.shape[0] < len(KEYPOINT_NAMES):
                continue
            scores = get_keypoint_scores(person)
            if len(scores) < len(KEYPOINT_NAMES):
                continue
            skip_reason = evaluate_common_skip(
                scores, args.keypoints_score_threshold, args.min_keypoints
            )
            if skip_reason is not None:
                continue

            anchors = compute_pose_anchors(keypoints, img_h=img_h, img_w=img_w)
            if args.pose_metric.endswith("_norm"):
                wrist_distance = anchors.nose_wrist_min_norm
            else:
                wrist_distance = anchors.nose_wrist_min
            risk = bool(getattr(anchors, args.pose_metric) * float(args.pose_k) > float(wrist_distance))
            cond1_global = cond1_global or risk
            valid_persons += 1

            draw_anchor_lines(pose_canvas, anchors, args.pose_metric)
            circle_radius = draw_pose_alert_circle(
                pose_canvas, anchors, args.pose_metric, args.pose_k, img_h, img_w
            )
            if args.draw_person_text:
                color = (0, 0, 255) if risk else (255, 0, 0)
                draw_text_line(
                    pose_canvas,
                    (
                        f"[Cond1][P{person_id}] metric={args.pose_metric}  "
                        f"K={args.pose_k:.3f}  anchor={getattr(anchors, args.pose_metric):.3f}  "
                        f"wrist={wrist_distance:.3f}  r={circle_radius}  warn={risk}"
                    ),
                    line_index=person_id,
                    color=color,
                    font_scale=0.65,
                    thickness=2,
                )

        if args.draw_person_text:
            draw_text_line(
                pose_canvas,
                f"[Cond1] persons={valid_persons}  global_warn={cond1_global}",
                line_index=len(instances) + 1,
                color=(0, 0, 255) if cond1_global else (255, 0, 0),
                font_scale=0.75,
                thickness=2,
            )

        thermal_canvas = None
        cond2_global = False
        thermal_path = resolve_thermal_for_rgb(img_path, thermal_input)
        thermal_max_temp = None
        if thermal_path is not None and thermal_path.exists():
            thermal_raw = load_thermal_matrix(thermal_path)
            thermal_max_temp = float(np.max(thermal_raw))
            thermal_resized = align_thermal_to_rgb_resolution(thermal_raw, rgb_h=img_h, rgb_w=img_w)
            hot_points = get_hot_points(thermal_resized, args.overheat_threshold)
            thermal_canvas = overlay_thermal_on_rgb(rgb_image, thermal_resized, args.thermal_blend_alpha)

            is_overheat = bool(thermal_max_temp >= float(args.overheat_threshold))
            if args.thermal_mode == "overheat_only":
                cond2_global = bool(is_overheat)
            else:
                thermal_valid_persons = 0
                any_near_nose_risk = False
                for person_id, person in enumerate(instances):
                    keypoints = np.asarray(person.get("keypoints", []), dtype=np.float32)
                    if keypoints.ndim != 2 or keypoints.shape[0] < len(KEYPOINT_NAMES):
                        continue
                    scores = get_keypoint_scores(person)
                    if len(scores) < len(KEYPOINT_NAMES):
                        continue
                    skip_reason = evaluate_common_skip(
                        scores, args.keypoints_score_threshold, args.min_keypoints
                    )
                    if skip_reason is not None:
                        continue

                    thermal_valid_persons += 1
                    anchors = compute_pose_anchors(keypoints, img_h=img_h, img_w=img_w)
                    min_heat_dist, min_heat_dist_norm, nearest_hot_xy = get_min_heat_distances(
                        hot_points, anchors.nose, img_h=img_h, img_w=img_w
                    )
                    if args.thermal_metric.endswith("_norm"):
                        heat_distance = float(min_heat_dist_norm)
                        radius_px = normalized_radius_to_pixels(
                            float(getattr(anchors, args.thermal_metric)) * float(args.thermal_k),
                            img_h,
                            img_w,
                        )
                    else:
                        heat_distance = float(min_heat_dist)
                        radius_px = int(
                            round(float(getattr(anchors, args.thermal_metric)) * float(args.thermal_k))
                        )

                    risk = bool(
                        is_overheat
                        and (
                            heat_distance
                            <= float(getattr(anchors, args.thermal_metric)) * float(args.thermal_k)
                        )
                    )
                    any_near_nose_risk = any_near_nose_risk or risk

                    draw_anchor_lines(thermal_canvas, anchors, args.thermal_metric)
                    nose_xy = tuple(int(v) for v in anchors.nose.tolist())
                    if radius_px > 0:
                        cv2.circle(thermal_canvas, nose_xy, radius_px, (0, 0, 255), 2)
                    if nearest_hot_xy is not None:
                        hot_xy = tuple(int(v) for v in nearest_hot_xy.tolist())
                        cv2.circle(thermal_canvas, hot_xy, 5, (0, 0, 255), -1)
                        cv2.line(thermal_canvas, nose_xy, hot_xy, (0, 255, 255), 2)

                    if args.draw_person_text:
                        color = (0, 0, 255) if risk else (255, 0, 0)
                        draw_text_line(
                            thermal_canvas,
                            (
                                f"[Cond2][P{person_id}] metric={args.thermal_metric}  "
                                f"K={args.thermal_k:.3f}  heat={heat_distance:.3f}  "
                                f"anchor={getattr(anchors, args.thermal_metric):.3f}  "
                                f"maxT={thermal_max_temp:.1f}  warn={risk}"
                            ),
                            line_index=person_id,
                            color=color,
                            font_scale=0.65,
                            thickness=2,
                        )

                cond2_global = bool(
                    is_overheat and (any_near_nose_risk if thermal_valid_persons else True)
                )

            if args.draw_person_text:
                draw_text_line(
                    thermal_canvas,
                    (
                        f"[Cond2] mode={args.thermal_mode}  global_warn={cond2_global}  "
                        f"overheat={is_overheat}  maxT={thermal_max_temp:.1f}  "
                        f"hot_pixels={int(hot_points.shape[0])}  "
                        f"overheat_threshold={args.overheat_threshold:.1f}"
                    ),
                    line_index=len(instances) + 1,
                    color=(0, 0, 255) if cond2_global else (255, 0, 0),
                    font_scale=0.75,
                    thickness=2,
                )
        else:
            thermal_canvas = rgb_image.copy()
            if args.draw_person_text:
                draw_text_line(
                    thermal_canvas,
                    f"[Cond2] 未找到热成像文件: {thermal_input} (stem={img_path.stem}) -> cond2_warn=False",
                    line_index=0,
                    color=(255, 0, 0),
                    font_scale=0.7,
                    thickness=2,
                )

        smoking_alarm = bool(cond1_global and cond2_global)
        final_canvas = thermal_canvas.copy()
        top_color = (0, 0, 255) if smoking_alarm else (0, 255, 0)
        cv2.putText(
            final_canvas,
            f"SMOKING_ALARM={smoking_alarm}  POSE_WARN={cond1_global}  THERMAL_WARN={cond2_global}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            top_color,
            3,
        )

        if args.save:
            pose_out = save_image_grouped(
                pose_canvas, output_dir, "condition1_pose", cond1_global, img_path, "cond1_pose_predicted"
            )
            thermal_out = save_image_grouped(
                thermal_canvas,
                output_dir,
                "condition2_thermal",
                cond2_global,
                img_path,
                "cond2_thermal_predicted",
            )
            final_out = save_image_grouped(
                final_canvas, output_dir, "smoking_alarm", smoking_alarm, img_path, "smoking_alarm_predicted"
            )
            print(f"[Cond1 可视化]: {pose_out}")
            print(f"[Cond2 可视化]: {thermal_out}")
            print(f"[最终 可视化]: {final_out}")

        if args.show:
            cv2.imshow("condition1_pose", pose_canvas)
            cv2.imshow("condition2_thermal", thermal_canvas)
            cv2.imshow("smoking_alarm", final_canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
