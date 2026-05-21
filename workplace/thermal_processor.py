import numpy as np
import cv2
from pathlib import Path
import json
import time


class ThermalProcessor:
    def __init__(self, overheat_threshold: float = 100.0):
        self.overheat_threshold = overheat_threshold

    def detect_overheat_regions(self, thermal_matrix: np.ndarray) -> dict:
        """
        检测热成像矩阵中的过热区域
        
        Args:
            thermal_matrix: 热成像温度矩阵，形状为 (height, width, 1)，类型为 float32
                          每个像素值代表真实摄氏度
            
        Returns:
            dict: 包含过热区域信息的字典
                  - has_overheat: 是否存在过热区域
                  - max_temperature: 矩阵中的最大温度
                  - avg_temperature: 矩阵中的平均温度
                  - overheat_mask: 过热区域的二值掩码
                  - overheat_bboxes: 过热区域的边界框列表 [x1, y1, x2, y2]
        """
        if thermal_matrix is None:
            return {
                'has_overheat': False,
                'max_temperature': None,
                'avg_temperature': None,
                'overheat_mask': None,
                'overheat_bboxes': []
            }

        max_temp = np.max(thermal_matrix)
        avg_temp = np.mean(thermal_matrix)
        
        overheat_mask = (thermal_matrix >= self.overheat_threshold).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(
            overheat_mask.squeeze(), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        overheat_bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                overheat_bboxes.append([x, y, x + w, y + h])
        
        return {
            'has_overheat': len(overheat_bboxes) > 0,
            'max_temperature': float(max_temp),
            'avg_temperature': float(avg_temp),
            'overheat_mask': overheat_mask,
            'overheat_bboxes': overheat_bboxes
        }

    def extract_temperature_in_bbox(self, thermal_matrix: np.ndarray, bbox: list) -> dict:
        """
        从热成像矩阵中提取指定边界框内的温度信息
        
        Args:
            thermal_matrix: 热成像温度矩阵
            bbox: 边界框 [x1, y1, x2, y2]
            
        Returns:
            dict: 包含温度统计信息的字典
        """
        x1, y1, x2, y2 = bbox
        roi = thermal_matrix[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {
                'max_temperature': None,
                'avg_temperature': None,
                'min_temperature': None,
                'is_overheat': False
            }
        
        max_temp = np.max(roi)
        avg_temp = np.mean(roi)
        min_temp = np.min(roi)
        
        return {
            'max_temperature': float(max_temp),
            'avg_temperature': float(avg_temp),
            'min_temperature': float(min_temp),
            'is_overheat': max_temp >= self.overheat_threshold
        }

    def visualize_thermal_on_rgb(self, rgb_frame: np.ndarray, thermal_info: dict, 
                                output_path: str = None) -> np.ndarray:
        """
        在RGB图像上可视化热成像检测结果
        
        Args:
            rgb_frame: RGB图像（BGR格式）
            thermal_info: 热成像检测信息
            output_path: 输出路径，为None时不保存
            
        Returns:
            np.ndarray: 可视化后的图像
        """
        canvas = rgb_frame.copy()
        
        for bbox in thermal_info['overheat_bboxes']:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            temp_info = self.extract_temperature_in_bbox(None, bbox)
            text = f"Max: {thermal_info['max_temperature']:.1f}C"
            cv2.putText(canvas, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        global_text = f"Overheat: {thermal_info['has_overheat']} | Max: {thermal_info['max_temperature']:.1f}C | Avg: {thermal_info['avg_temperature']:.1f}C"
        h, w = canvas.shape[:2]
        cv2.putText(canvas, global_text, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, canvas)
        
        return canvas

    def process_thermal_frame(self, thermal_matrix: np.ndarray, 
                             rgb_frame: np.ndarray = None, 
                             save_visualization: bool = False,
                             output_prefix: str = 'thermal_output') -> dict:
        """
        完整处理一帧热成像数据
        
        Args:
            thermal_matrix: 热成像温度矩阵
            rgb_frame: 对应的RGB图像（可选）
            save_visualization: 是否保存可视化结果
            output_prefix: 输出文件前缀
            
        Returns:
            dict: 包含完整处理结果的字典
        """
        result = {
            'timestamp': time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
            'overheat_threshold': self.overheat_threshold,
            'thermal_shape': thermal_matrix.shape if thermal_matrix is not None else None,
            'detection': self.detect_overheat_regions(thermal_matrix)
        }
        
        if save_visualization and rgb_frame is not None:
            vis_path = f'{output_prefix}_{result["timestamp"]}.jpg'
            self.visualize_thermal_on_rgb(rgb_frame, result['detection'], vis_path)
            result['visualization_path'] = vis_path
        
        return result


def create_thermal_processor(overheat_threshold: float = 100.0) -> ThermalProcessor:
    """
    创建热成像处理器实例
    
    Args:
        overheat_threshold: 过热阈值，默认100摄氏度
        
    Returns:
        ThermalProcessor: 热成像处理器实例
    """
    return ThermalProcessor(overheat_threshold=overheat_threshold)


def thermal_to_pose_input(thermal_matrix: np.ndarray, 
                         overheat_threshold: float = 100.0) -> dict:
    """
    将热成像数据转换为姿态估计可用的输入格式
    
    Args:
        thermal_matrix: 热成像温度矩阵
        overheat_threshold: 过热阈值
        
    Returns:
        dict: 包含热成像特征的字典，可作为姿态估计的输入
    """
    processor = ThermalProcessor(overheat_threshold=overheat_threshold)
    detection = processor.detect_overheat_regions(thermal_matrix)
    
    pose_input = {
        'thermal_features': {
            'has_overheat': detection['has_overheat'],
            'max_temperature': detection['max_temperature'],
            'avg_temperature': detection['avg_temperature'],
            'overheat_bboxes': detection['overheat_bboxes'],
            'threshold_used': overheat_threshold
        }
    }
    
    return pose_input


if __name__ == '__main__':
    np.random.seed(42)
    mock_thermal = np.random.uniform(25.0, 45.0, (1520, 2688, 1)).astype(np.float32)
    mock_thermal[700:900, 1200:1400] = np.random.uniform(95.0, 120.0, (200, 200, 1)).astype(np.float32)
    
    mock_rgb = np.random.randint(0, 255, (1520, 2688, 3)).astype(np.uint8)
    
    processor = ThermalProcessor(overheat_threshold=100.0)
    result = processor.process_thermal_frame(mock_thermal, mock_rgb, save_visualization=True)
    
    print(f"热成像处理完成")
    print(f"是否检测到过热: {result['detection']['has_overheat']}")
    print(f"最高温度: {result['detection']['max_temperature']:.1f}C")
    print(f"平均温度: {result['detection']['avg_temperature']:.1f}C")
    print(f"过热区域数量: {len(result['detection']['overheat_bboxes'])}")
    
    pose_input = thermal_to_pose_input(mock_thermal)
    print("\n转换为姿态估计输入格式:")
    print(json.dumps(pose_input, indent=2))