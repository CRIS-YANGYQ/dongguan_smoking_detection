import cv2
import numpy as np
import os
import glob
from utils.warp_scale_only_overlay import warp_scale_only_overlay

# rgb_img_path = '/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态正常/000045_20260521_112200.png'
# thermal_img_path = '/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal/嘴巴附近无热源/000045_20260521_112200.npy'

rgb_img_path = '/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态危险/000042_20260521_111425.png'
thermal_img_path = '/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal/嘴巴附近无热源/000042_20260521_111425.npy'

matrices_dir = 'outputs/matrices'
vis_dir = 'outputs/vis'

os.makedirs(matrices_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# 提取RGB图片的文件名作为前缀
rgb_filename = os.path.basename(rgb_img_path).split('.')[0]
thermal_aligned_save_path = os.path.join(vis_dir, f'{rgb_filename}_thermal_aligned_homography.png')
thermal_rgb_overlay_save_path = os.path.join(vis_dir, f'{rgb_filename}_thermal_rgb_overlay_homography.png')

use_existing_matrix = True

if __name__ == '__main__':
    # 热成像中的4个点 [左上, 右上, 右下, 左下]
    
    # Thermal
    # 1. 300 , 315
    # 2. 349 , 314
    # 3. 342 , 222
    # 4. 327 , 179
    
    # RGB
    # 1. 1306 , 1201 左脚尖
    # 2. 1709 , 1221 右脚尖
    # 3. 1651 , 461 烟脚（点燃的一头）
    # 4. 1551 , 127 鼻头
    if use_existing_matrix:
        matrix_files = sorted(glob.glob(os.path.join(matrices_dir, 'H_*.npy')))
        if not matrix_files:
            raise FileNotFoundError(f"No H matrix found in {matrices_dir}")
        homography_path = matrix_files[-1]
        H = np.load(homography_path)
        print(f"Loaded existing Homography matrix from {homography_path}")
    else:
        # === 你手动提供的N个点，比如4个对应点（标定板四角）===
        pts_thermal = np.array([
            [300, 315],
            [349, 314],
            [342, 222],
            [327, 179]
        ], dtype=np.float32)

        pts_rgb = np.array([
            [1306, 1201],
            [1704, 1221],
            [1653, 464],
            [1551, 127]
        ], dtype=np.float32)

        # 计算单应变换矩阵
        H, status = cv2.findHomography(pts_thermal, pts_rgb, method=cv2.RANSAC)

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        homography_path = os.path.join(matrices_dir, f'H_{timestamp}.npy')
        np.save(homography_path, H)
        print(f"Saved Homography matrix to {homography_path}")

    rgb_img = cv2.imread(rgb_img_path)
    thermal_img = np.load(thermal_img_path)
    h, w = rgb_img.shape[:2]

    thermal_aligned = cv2.warpPerspective(thermal_img, H, (w, h))
    if thermal_aligned.dtype != np.uint8:
        thermal_aligned_vis = cv2.normalize(thermal_aligned, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    else:
        thermal_aligned_vis = thermal_aligned

    _, overlay = warp_scale_only_overlay(thermal_aligned, rgb_img, colormap=cv2.COLORMAP_JET, alpha=0.5)

    cv2.imwrite(thermal_aligned_save_path, thermal_aligned_vis)
    cv2.imwrite(thermal_rgb_overlay_save_path, overlay)