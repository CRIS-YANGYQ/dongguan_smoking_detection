# thermal numpy file to image (resolution fixed)
import cv2
import numpy as np
from pathlib import Path


def load_thermal_npy(path: Path) -> np.ndarray:
    matrix = np.load(str(path))
    if matrix.ndim == 3 and matrix.shape[-1] == 1:
        matrix = matrix[:, :, 0]
    if matrix.ndim != 2:
        raise ValueError(f"thermal npy 需要是二维矩阵，实际 shape={matrix.shape}")
    return matrix.astype(np.float32, copy=False)


def thermal_to_colormap(thermal: np.ndarray) -> np.ndarray:
    thermal_norm = cv2.normalize(thermal, None, 0, 255, cv2.NORM_MINMAX)
    thermal_u8 = thermal_norm.astype(np.uint8)
    return cv2.applyColorMap(thermal_u8, cv2.COLORMAP_JET)


if __name__ == "__main__":
    npy_path = "/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal/嘴巴附近无热源/000045_20260521_112200.npy"

    thermal = load_thermal_npy(npy_path)

    print(f"原始热成像分辨率: {thermal.shape[1]}x{thermal.shape[0]} (宽x高)")
    print(f"数据类型: {thermal.dtype}")
    print(f"温度范围: {thermal.min():.2f}°C - {thermal.max():.2f}°C")

    thermal_colored = thermal_to_colormap(thermal)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "thermal_visualization.png"

    cv2.imwrite(str(output_path), thermal_colored)
    print(f"可视化图像已保存到: {output_path}")