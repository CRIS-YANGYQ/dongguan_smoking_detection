# 输入一张图片和对应的thermal，尝试resize代码，并保存到指定目录下
import argparse
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def load_thermal_npy(path: Path) -> np.ndarray:
    matrix = np.load(str(path))
    if matrix.ndim == 3 and matrix.shape[-1] == 1:
        matrix = matrix[:, :, 0]
    if matrix.ndim != 2:
        raise ValueError(f"thermal npy 需要是二维矩阵，实际 shape={matrix.shape}")
    return matrix.astype(np.float32, copy=False)


def load_pose_img_path_from_json(pose_json_path: Path, stem: str) -> Optional[Path]:
    data = json.loads(pose_json_path.read_text(encoding="utf-8"))
    infoes = data.get("infoes", {})
    for image_name, image_info in infoes.items():
        if Path(image_name).stem != stem:
            continue
        img_path = image_info.get("img_path")
        if not img_path:
            return None
        return Path(img_path)
    return None


def thermal_to_colormap(thermal: np.ndarray) -> np.ndarray:
    thermal_norm = cv2.normalize(thermal, None, 0, 255, cv2.NORM_MINMAX)
    thermal_u8 = thermal_norm.astype(np.uint8)
    return cv2.applyColorMap(thermal_u8, cv2.COLORMAP_JET)


def overlay(bgr: np.ndarray, thermal_bgr: np.ndarray, alpha: float) -> np.ndarray:
    if bgr.shape[:2] != thermal_bgr.shape[:2]:
        raise ValueError(
            f"overlay 分辨率不一致: rgb={bgr.shape[:2]} thermal={thermal_bgr.shape[:2]}"
        )
    return cv2.addWeighted(bgr, 1.0 - alpha, thermal_bgr, alpha, 0)


def build_homography_scale_only(
    thermal_w: int, thermal_h: int, target_w: int, target_h: int
) -> np.ndarray:
    sx = target_w / thermal_w
    sy = target_h / thermal_h
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)


def build_homography_span_aligned(
    thermal_w: int, thermal_h: int, target_w: int, target_h: int
) -> np.ndarray:
    sx = (target_w - 1) / (thermal_w - 1)
    sy = (target_h - 1) / (thermal_h - 1)
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)


def build_homography_half_pixel_aligned(
    thermal_w: int, thermal_h: int, target_w: int, target_h: int
) -> np.ndarray:
    sx = target_w / thermal_w
    sy = target_h / thermal_h
    tx = 0.5 * sx - 0.5
    ty = 0.5 * sy - 0.5
    return np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=np.float32)


def warp_thermal(
    thermal: np.ndarray, H: np.ndarray, target_w: int, target_h: int
) -> np.ndarray:
    return cv2.warpPerspective(
        thermal,
        H,
        (target_w, target_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def project_points(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"points_xy 需要是 Nx2, 实际 shape={points_xy.shape}")
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float32)
    pts = np.concatenate([points_xy.astype(np.float32), ones], axis=1)
    proj = (H @ pts.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def parse_float_list(text: str) -> List[float]:
    if not text:
        return []
    ratios: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ratios.append(float(part))
    return ratios


def unique_values(values: List[float]) -> List[float]:
    seen = set()
    out: List[float] = []
    for v in values:
        key = round(float(v), 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(v))
    return out


def center_crop_by_ratios_and_delta(
    thermal: np.ndarray, w_ratio: float, h_ratio: float, delta_x: float, delta_y: float
) -> np.ndarray:
    if not (0 < w_ratio <= 1) or not (0 < h_ratio <= 1):
        raise ValueError(f"crop ratio 需要在 (0, 1]，实际 w={w_ratio} h={h_ratio}")
    h, w = thermal.shape[:2]
    crop_w = int(round(w * w_ratio))
    crop_h = int(round(h * h_ratio))
    crop_w = max(1, min(w, crop_w))
    crop_h = max(1, min(h, crop_h))
    center_x = (w - 1) / 2.0 + float(delta_x)
    center_y = (h - 1) / 2.0 - float(delta_y)
    x0 = int(round(center_x - crop_w / 2.0))
    y0 = int(round(center_y - crop_h / 2.0))
    x0 = max(0, min(w - crop_w, x0))
    y0 = max(0, min(h - crop_h, y0))
    return thermal[y0 : y0 + crop_h, x0 : x0 + crop_w]


def draw_debug_markers(
    canvas_bgr: np.ndarray,
    H: np.ndarray,
    thermal_w: int,
    thermal_h: int,
    label: str,
) -> np.ndarray:
    out = canvas_bgr.copy()
    src_points = np.array(
        [
            [0.0, 0.0],
            [0.0, float(thermal_h - 1)],
            [float(thermal_w - 1), 0.0],
            [float(thermal_w - 1), float(thermal_h - 1)],
            [(thermal_w - 1) / 2.0, (thermal_h - 1) / 2.0],
            [thermal_w / 2.0 - 0.5, thermal_h / 2.0 - 0.5],
        ],
        dtype=np.float32,
    )
    dst_points = project_points(H, src_points)
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    names = ["TL", "BL", "TR", "BR", "C_idx", "C_pix"]
    for (x, y), color, name in zip(dst_points, colors, names):
        cv2.circle(out, (int(round(x)), int(round(y))), 8, color, thickness=-1)
        cv2.putText(
            out,
            name,
            (int(round(x)) + 10, int(round(y)) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def save_case(
    out_dir: Path,
    name: str,
    rgb_bgr: np.ndarray,
    thermal_aligned: np.ndarray,
    H: Optional[np.ndarray],
    thermal_w: int,
    thermal_h: int,
    alpha: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    heat_bgr = thermal_to_colormap(thermal_aligned)
    over = overlay(rgb_bgr, heat_bgr, alpha=alpha)

    if H is not None:
        over = draw_debug_markers(over, H, thermal_w=thermal_w, thermal_h=thermal_h, label=name)

    cv2.imwrite(str(out_dir / f"{name}_overlay.jpg"), over)
    cv2.imwrite(str(out_dir / f"{name}_thermal_colormap.jpg"), heat_bgr)


def main():
    # /root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态危险/000092_20260521_112221.png
    # /root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态危险/000042_20260521_111425.png
    # /root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态正常/000045_20260521_112200.png /root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal/嘴巴附近无热源/000045_20260521_112200.npy
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str,
                        default='/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态正常/000045_20260521_112200.png')
    parser.add_argument("--thermal", type=str, 
                        default='/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal/嘴巴附近无热源/000045_20260521_112200.npy')
    parser.add_argument("--pose-json", type=str, 
                        default='/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态正常_2026-05-26_12-20-08.json')
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "outputs" / "thermal_resize_test" / "000045_20260521_112200"
        ),
    )
    parser.add_argument("--thermal-w", type=int, default=640)
    parser.add_argument("--thermal-h", type=int, default=512)
    parser.add_argument("--target-w", type=int, default=2688)
    parser.add_argument("--target-h", type=int, default=1520)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--crop-w-ratios", type=str, default="1.0,0.9,0.8,0.7")
    parser.add_argument("--crop-h-ratios", type=str, default="1.0,0.9,0.8,0.7")
    parser.add_argument("--delta-xs", type=str, default="0")
    parser.add_argument("--delta-ys", type=str, default="0")
    args = parser.parse_args()

    thermal_path = Path(args.thermal)
    if not thermal_path.exists():
        raise FileNotFoundError(f"thermal 不存在: {thermal_path}")

    rgb_path = Path(args.rgb) if args.rgb else None
    if rgb_path is None:
        if not args.pose_json:
            raise ValueError("未提供 --rgb 时，必须提供 --pose-json 用于从姿态 json 推断 img_path")
        rgb_path = load_pose_img_path_from_json(Path(args.pose_json), thermal_path.stem)
        if rgb_path is None:
            raise FileNotFoundError(
                f"无法从 pose json 找到对应图片: stem={thermal_path.stem} json={args.pose_json}"
            )

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB 图片不存在: {rgb_path}")

    rgb_bgr = cv2.imread(str(rgb_path))
    if rgb_bgr is None:
        raise FileNotFoundError(f"无法读取 RGB 图片: {rgb_path}")

    thermal = load_thermal_npy(thermal_path)
    if thermal.shape[:2] != (args.thermal_h, args.thermal_w):
        raise ValueError(
            f"thermal 分辨率异常: 期望 {(args.thermal_h, args.thermal_w)} 实际 {thermal.shape[:2]}"
        )

    if rgb_bgr.shape[1] != args.target_w or rgb_bgr.shape[0] != args.target_h:
        raise ValueError(
            f"RGB 分辨率异常: 期望 {(args.target_h, args.target_w)} 实际 {rgb_bgr.shape[:2]}"
        )

    out_dir = Path(args.out_dir)

    crop_w_ratios = unique_values(parse_float_list(args.crop_w_ratios))
    crop_h_ratios = unique_values(parse_float_list(args.crop_h_ratios))
    if not crop_w_ratios:
        crop_w_ratios = [1.0]
    if not crop_h_ratios:
        crop_h_ratios = [1.0]

    delta_xs = unique_values(parse_float_list(args.delta_xs))
    delta_ys = unique_values(parse_float_list(args.delta_ys))
    if not delta_xs:
        delta_xs = [0.0]
    if not delta_ys:
        delta_ys = [0.0]

    for w_ratio in crop_w_ratios:
        for h_ratio in crop_h_ratios:
            for delta_x in delta_xs:
                for delta_y in delta_ys:
                    thermal_crop = center_crop_by_ratios_and_delta(
                        thermal,
                        w_ratio=w_ratio,
                        h_ratio=h_ratio,
                        delta_x=delta_x,
                        delta_y=delta_y,
                    )
                    crop_h, crop_w = thermal_crop.shape[:2]
                    tag = f"w{w_ratio:.2f}_h{h_ratio:.2f}__dx{delta_x:+.1f}_dy{delta_y:+.1f}"

                    thermal_resize = cv2.resize(
                        thermal_crop,
                        (args.target_w, args.target_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    save_case(
                        out_dir,
                        f"{tag}__resize_inter_nearest",
                        rgb_bgr,
                        thermal_resize,
                        H=None,
                        thermal_w=crop_w,
                        thermal_h=crop_h,
                        alpha=args.alpha,
                    )

                    H_scale = build_homography_scale_only(
                        thermal_w=crop_w,
                        thermal_h=crop_h,
                        target_w=args.target_w,
                        target_h=args.target_h,
                    )
                    thermal_warp_scale = warp_thermal(
                        thermal_crop, H_scale, args.target_w, args.target_h
                    )
                    print(f"[H scale_only {tag}]\n{H_scale}")
                    save_case(
                        out_dir,
                        f"{tag}__warp_scale_only",
                        rgb_bgr,
                        thermal_warp_scale,
                        H=H_scale,
                        thermal_w=crop_w,
                        thermal_h=crop_h,
                        alpha=args.alpha,
                    )

                    H_span = build_homography_span_aligned(
                        thermal_w=crop_w,
                        thermal_h=crop_h,
                        target_w=args.target_w,
                        target_h=args.target_h,
                    )
                    thermal_warp_span = warp_thermal(
                        thermal_crop, H_span, args.target_w, args.target_h
                    )
                    print(f"[H span_aligned {tag}]\n{H_span}")
                    save_case(
                        out_dir,
                        f"{tag}__warp_span_aligned",
                        rgb_bgr,
                        thermal_warp_span,
                        H=H_span,
                        thermal_w=crop_w,
                        thermal_h=crop_h,
                        alpha=args.alpha,
                    )

                    H_half = build_homography_half_pixel_aligned(
                        thermal_w=crop_w,
                        thermal_h=crop_h,
                        target_w=args.target_w,
                        target_h=args.target_h,
                    )
                    thermal_warp_half = warp_thermal(
                        thermal_crop, H_half, args.target_w, args.target_h
                    )
                    print(f"[H half_pixel_aligned {tag}]\n{H_half}")
                    save_case(
                        out_dir,
                        f"{tag}__warp_half_pixel_aligned",
                        rgb_bgr,
                        thermal_warp_half,
                        H=H_half,
                        thermal_w=crop_w,
                        thermal_h=crop_h,
                        alpha=args.alpha,
                    )

    print(f"[DONE] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
