import cv2
import numpy as np


def warp_scale_only_overlay(thermal_img, rgb_img, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Upsample lower resolution image to match higher resolution and create overlay visualization.

    Args:
        thermal_img: Thermal image (grayscale)
        rgb_img: Reference RGB image
        colormap: OpenCV colormap for thermal image visualization
        alpha: Blending factor for overlay (0=rgb only, 1=thermal only)

    Returns:
        thermal_scaled: Upsampled thermal image (same size as rgb_img)
        overlay: Blended visualization of RGB and upsampled thermal
    """
    if len(thermal_img.shape) == 2:
        thermal_gray = thermal_img
    else:
        thermal_gray = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)

    if thermal_gray.dtype != np.uint8:
        thermal_gray = cv2.normalize(thermal_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    h_thermal, w_thermal = thermal_gray.shape[:2]
    h_rgb, w_rgb = rgb_img.shape[:2]

    if h_thermal != h_rgb or w_thermal != w_rgb:
        thermal_scaled = cv2.resize(thermal_gray, (w_rgb, h_rgb), interpolation=cv2.INTER_LINEAR)
    else:
        thermal_scaled = thermal_gray

    thermal_colored = cv2.applyColorMap(thermal_scaled, colormap)
    overlay = cv2.addWeighted(rgb_img, 1 - alpha, thermal_colored, alpha, 0)

    return thermal_scaled, overlay