import cv2
import numpy as np
import os
import glob
from utils.warp_scale_only_overlay import warp_scale_only_overlay
from tqdm import tqdm

rgb_img_dir = '/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/test/data/rgb/'
thermal_img_dir = '/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/test/data/thermal/'

matrices_dir = 'outputs/matrices'
vis_dir = 'outputs/vis/homography'

os.makedirs(matrices_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)



use_existing_matrix = True

# 热成像中的4个点 [左上, 右上, 右下, 左下]

PTS_THERMAL = np.array([
    [300, 315],
    [349, 314],
    [341, 221],
    [327, 179]
], dtype=np.float32)

PTS_RGB = np.array([
    [1306, 1201],
    [1709, 1221],
    [1653, 464],
    [1551, 127]
], dtype=np.float32)

def search_and_load_matrix_file(matrices_dir):
    """
    Search for and load a numpy matrix file.

    Args:
        matrices_dir: Directory to search for matrix files

    Returns:
        numpy array: The loaded matrix
    """
    matrix_files = sorted(glob.glob(os.path.join(matrices_dir, 'Homography_*.npy')))
    if not matrix_files:
        raise FileNotFoundError(f"No Homography matrix found in {matrices_dir}")
    # 加载最新的M矩阵
    homography_path = matrix_files[-1]
    H = np.load(homography_path)
    print(f"Loaded existing Homography matrix from {homography_path}")
    return H

def compute_matrix_from_pts(pts_thermal, pts_rgb):
    """
    Compute the homography matrix from two sets of points.

    Args:
        pts_thermal: Thermal points in the reference frame
        pts_rgb: RGB points in the target frame

    Returns:
        numpy array: The computed homography matrix
    """
    return cv2.getHomography(pts_thermal, pts_rgb)

if __name__ == '__main__':
    homography_matrix = None
    if use_existing_matrix:
        homography_matrix = search_and_load_matrix_file(matrices_dir)
    else:
        # 计算单应变换矩阵
        homography_matrix, status = cv2.findHomography(PTS_THERMAL, PTS_RGB, method=cv2.RANSAC)

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        homography_path = os.path.join(matrices_dir, f'Homography_{timestamp}.npy')
        np.save(homography_path, homography_matrix)
        print(f"Saved Homography matrix to {homography_path}")
    
    rgb_img_path_lst = glob.glob(os.path.join(rgb_img_dir, '*.png'))
    thermal_img_path_lst = glob.glob(os.path.join(thermal_img_dir, '*.npy'))
    rgb_img_path_lst = sorted(rgb_img_path_lst)
    thermal_img_path_lst = sorted(thermal_img_path_lst)
    for rgb_img_path, thermal_img_path in tqdm(zip(rgb_img_path_lst, thermal_img_path_lst), desc="Processing images with Homography"):

        rgb_img = cv2.imread(rgb_img_path)
        thermal_img = np.load(thermal_img_path)
        h, w = rgb_img.shape[:2]

        thermal_aligned = cv2.warpPerspective(thermal_img, homography_matrix, (w, h))
        if thermal_aligned.dtype != np.uint8:
            thermal_aligned_vis = cv2.normalize(thermal_aligned, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        else:
            thermal_aligned_vis = thermal_aligned

        _, overlay = warp_scale_only_overlay(thermal_aligned, rgb_img, colormap=cv2.COLORMAP_JET, alpha=0.5)
        
        # 提取RGB图片的文件名作为前缀
        rgb_filename = os.path.basename(rgb_img_path).split('.')[0]
        thermal_aligned_save_path = os.path.join(vis_dir, f'{rgb_filename}_thermal_aligned_homography.png')
        thermal_rgb_overlay_save_path = os.path.join(vis_dir, f'{rgb_filename}_thermal_rgb_overlay_homography.png')

        cv2.imwrite(thermal_aligned_save_path, thermal_aligned_vis)
        cv2.imwrite(thermal_rgb_overlay_save_path, overlay)