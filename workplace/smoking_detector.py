"""
综合吸烟行为检测器 - 基于K值的热成像检测

功能说明
1. 结合热成像数据（thermal_matrix）和RGB姿态估计进行吸烟行为检测
2. 热成像检测：在手腕为圆心、K×锚点距离为半径的区域内，是否有≥100°C的像素
3. 姿态估计：计算手腕到鼻子的距离是否在危险范围内
4. 综合判断：只有当两个条件同时满足时才判断为吸烟

判断逻辑
- 热成像条件：手腕圆形区域内存在≥100°C的像素（使用训练得到的K值）
- 姿态条件：距离比值 >= 距离阈值（从empirical_smk_thres.py的训练结果）
- 最终判断：热成像 AND 姿态

锚点类型
- Nose-Ear: 使用 nose-ear 距离作为锚点
- Nose-Shoulder: 使用 nose-shoulder 距离作为锚点

输入数据格式
- thermal_matrix: numpy.ndarray (float32), 形状 (1520, 2688, 1), 像素值为摄氏度
- rgb_frame: numpy.ndarray (uint8), 形状 (1520, 2688, 3), BGR格式
- pose_keypoints: 姿态估计关键点，包含 nose, left_wrist, right_wrist 等

输出JSON结构
{
    "timestamp": "2026-05-20_10-30-00",
    "frame_id": "0000001",
    "thermal_detection": {
        "has_heat_source": true,
        "wrist_max_temp": 105.3,
        "K_value_used": 1.5,
        "radius_used": 45.2,
        "anchor_distance": 30.1
    },
    "pose_detection": {
        "nose_wrist_distance": 15.2,
        "anchor_distance": 30.1,
        "ratio": 0.427,
        "is_dangerous": true,
        "threshold_used": 0.42
    },
    "smoking判定": {
        "is_smoking": true,
        "confidence": "high",
        "reason": "热源检测+距离危险"
    }
}
"""

import numpy as np
import cv2
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Optional


class ThermalDetector:
    """热成像检测器 - 检测圆形区域内是否有高温"""
    
    def __init__(self, overheat_threshold: float = 100.0):
        """
        初始化热成像检测器
        
        Args:
            overheat_threshold: 过热温度阈值（固定为100°C）
        """
        self.overheat_threshold = overheat_threshold
    
    def detect_heat_in_circle_region(self, thermal_matrix: np.ndarray,
                                    center: np.ndarray,
                                    radius: float) -> Dict:
        """
        检测圆形区域内是否有高温（≥100°C）
        
        Args:
            thermal_matrix: 热成像矩阵
            center: 圆心坐标 (x, y)
            radius: 半径
            
        Returns:
            dict: 检测结果
        """
        if thermal_matrix is None:
            return {
                'has_heat': False,
                'max_temperature': None,
                'radius_used': radius,
                'center': center if center is not None else None
            }
        
        h, w = thermal_matrix.shape[:2]
        cx, cy = int(center[0]), int(center[1])
        
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        circle_mask = dist_from_center <= radius
        
        roi_temps = thermal_matrix[circle_mask]
        
        if roi_temps.size == 0:
            return {
                'has_heat': False,
                'max_temperature': None,
                'radius_used': float(radius),
                'center': [float(center[0]), float(center[1])]
            }
        
        max_temp = np.max(roi_temps)
        has_heat = max_temp >= self.overheat_threshold
        
        return {
            'has_heat': bool(has_heat),
            'max_temperature': float(max_temp),
            'radius_used': float(radius),
            'center': [float(center[0]), float(center[1])]
        }
    
    def detect_overheat_regions(self, thermal_matrix: np.ndarray) -> dict:
        """
        检测热成像矩阵中的过热区域
        
        Args:
            thermal_matrix: 热成像温度矩阵，形状为 (height, width, 1)，类型为 float32
                          每个像素值代表真实摄氏度
            
        Returns:
            dict: 包含过热区域信息的字典
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


class SmokingBehaviorDetector:
    """吸烟行为检测器 - 基于K值的综合检测"""
    
    def __init__(self, 
                 K_value: float = 1.5,
                 pose_threshold: float = 0.42,
                 overheat_threshold: float = 100.0,
                 anchor_type: str = 'nose_ear'):
        """
        初始化吸烟行为检测器
        
        Args:
            K_value: 距离倍数阈值（从训练得到）
            pose_threshold: 姿态距离比值阈值（从训练得到）
            overheat_threshold: 过热检测阈值（固定为100°C）
            anchor_type: 锚点类型 'nose_ear' 或 'nose_shoulder'
        """
        self.K_value = K_value
        self.pose_threshold = pose_threshold
        self.anchor_type = anchor_type
        self.thermal_detector = ThermalDetector(overheat_threshold=overheat_threshold)
        
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """计算两点之间的欧氏距离"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_anchor_distance(self, keypoints: np.ndarray) -> float:
        """
        计算锚点距离
        
        Args:
            keypoints: 姿态关键点
            
        Returns:
            float: 锚点距离
        """
        nose = keypoints[self.keypoint_names.index('nose')]
        
        if self.anchor_type == 'nose_ear':
            left_ear = keypoints[self.keypoint_names.index('left_ear')]
            right_ear = keypoints[self.keypoint_names.index('right_ear')]
            dist_left = self.calculate_distance(nose, left_ear)
            dist_right = self.calculate_distance(nose, right_ear)
            return max(dist_left, dist_right)
        elif self.anchor_type == 'nose_shoulder':
            left_shoulder = keypoints[self.keypoint_names.index('left_shoulder')]
            right_shoulder = keypoints[self.keypoint_names.index('right_shoulder')]
            shoulder_center = (left_shoulder + right_shoulder) / 2
            return self.calculate_distance(nose, shoulder_center)
        else:
            raise ValueError(f"未知的锚点类型: {self.anchor_type}")
    
    def detect_thermal_in_wrist_circle(self, thermal_matrix: np.ndarray, 
                                      keypoints: np.ndarray,
                                      anchor_distance: float) -> Dict:
        """
        检测手腕圆形区域内的热源（使用K值）
        
        Args:
            thermal_matrix: 热成像矩阵
            keypoints: 姿态关键点
            anchor_distance: 锚点距离
            
        Returns:
            dict: 手腕区域温度检测结果
        """
        left_wrist = keypoints[self.keypoint_names.index('left_wrist')]
        right_wrist = keypoints[self.keypoint_names.index('right_wrist')]
        
        radius = self.K_value * anchor_distance
        
        left_result = self.thermal_detector.detect_heat_in_circle_region(
            thermal_matrix, left_wrist, radius
        )
        right_result = self.thermal_detector.detect_heat_in_circle_region(
            thermal_matrix, right_wrist, radius
        )
        
        has_heat_source = left_result['has_heat'] or right_result['has_heat']
        max_temp = max(
            left_result['max_temperature'] if left_result['max_temperature'] is not None else 0,
            right_result['max_temperature'] if right_result['max_temperature'] is not None else 0
        )
        
        return {
            'has_heat_source': has_heat_source,
            'left_wrist_result': left_result,
            'right_wrist_result': right_result,
            'max_temp': float(max_temp) if max_temp > 0 else None,
            'K_value_used': self.K_value,
            'radius_used': float(radius),
            'anchor_distance': float(anchor_distance)
        }
    
    def detect_pose_risk(self, keypoints: np.ndarray, 
                       keypoint_scores: np.ndarray,
                       image_h: int, image_w: int,
                       anchor_distance: float) -> Dict:
        """
        基于姿态估计检测吸烟风险
        
        Args:
            keypoints: 关键点坐标
            keypoint_scores: 关键点置信度
            image_h: 图像高度
            image_w: 图像宽度
            anchor_distance: 锚点距离
            
        Returns:
            dict: 姿态风险检测结果
        """
        nose = keypoints[self.keypoint_names.index('nose')]
        left_wrist = keypoints[self.keypoint_names.index('left_wrist')]
        right_wrist = keypoints[self.keypoint_names.index('right_wrist')]
        
        nose_wrist_l = self.calculate_distance(nose, left_wrist)
        nose_wrist_r = self.calculate_distance(nose, right_wrist)
        nose_wrist_min = min(nose_wrist_l, nose_wrist_r)
        
        if nose_wrist_min > 0:
            ratio = anchor_distance / nose_wrist_min
        else:
            ratio = float('inf')
        
        is_dangerous = ratio >= self.pose_threshold
        
        return {
            'anchor_distance': float(anchor_distance),
            'nose_wrist_distance': float(nose_wrist_min),
            'ratio': float(ratio) if ratio != float('inf') else None,
            'is_dangerous': is_dangerous,
            'threshold_used': self.pose_threshold
        }
    
    def infer_smoking_behavior(self, 
                              thermal_matrix: np.ndarray,
                              keypoints: np.ndarray,
                              keypoint_scores: Optional[np.ndarray] = None,
                              image_shape: Tuple[int, int] = (1520, 2688),
                              frame_id: str = "0000000") -> Dict:
        """
        综合推断吸烟行为
        
        Args:
            thermal_matrix: 热成像矩阵
            keypoints: 姿态关键点
            keypoint_scores: 关键点置信度（可选）
            image_shape: 图像尺寸 (h, w)
            frame_id: 帧ID
            
        Returns:
            dict: 完整的检测结果
        """
        h, w = image_shape
        
        anchor_distance = self.calculate_anchor_distance(keypoints)
        
        thermal_result = self.detect_thermal_in_wrist_circle(
            thermal_matrix, keypoints, anchor_distance
        )
        
        pose_result = self.detect_pose_risk(
            keypoints, keypoint_scores, h, w, anchor_distance
        )
        
        is_smoking = thermal_result['has_heat_source'] and pose_result['is_dangerous']
        
        if is_smoking:
            if thermal_result['max_temp'] and thermal_result['max_temp'] > 120:
                confidence = "high"
            else:
                confidence = "medium"
            reason = "热源检测+距离危险"
        elif thermal_result['has_heat_source']:
            confidence = "low"
            reason = "仅热源检测"
        elif pose_result['is_dangerous']:
            confidence = "low"
            reason = "仅距离危险"
        else:
            confidence = "none"
            reason = "无异常"
        
        result = {
            'timestamp': time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
            'frame_id': frame_id,
            'anchor_type': self.anchor_type,
            'thermal_detection': thermal_result,
            'pose_detection': pose_result,
            'smoking判定': {
                'is_smoking': is_smoking,
                'confidence': confidence,
                'reason': reason
            }
        }
        
        return result
    
    def visualize_detection(self, rgb_frame: np.ndarray, 
                          detection_result: Dict,
                          keypoints: np.ndarray,
                          output_path: str = None) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            rgb_frame: RGB图像
            detection_result: 检测结果
            keypoints: 关键点
            output_path: 输出路径
            
        Returns:
            np.ndarray: 可视化图像
        """
        canvas = rgb_frame.copy()
        h, w = canvas.shape[:2]
        
        nose = keypoints[self.keypoint_names.index('nose')]
        left_wrist = keypoints[self.keypoint_names.index('left_wrist')]
        right_wrist = keypoints[self.keypoint_names.index('right_wrist')]
        
        thermal_info = detection_result['thermal_detection']
        pose_info = detection_result['pose_detection']
        smoking判定 = detection_result['smoking判定']
        
        radius = thermal_info['radius_used']
        for wrist in [left_wrist, right_wrist]:
            cx, cy = int(wrist[0]), int(wrist[1])
            color = (0, 0, 255) if thermal_info['has_heat_source'] else (0, 255, 0)
            cv2.circle(canvas, (cx, cy), int(radius), color, 2)
        
        if smoking判定['is_smoking']:
            color = (0, 0, 255)
            status_text = "SMOKING DETECTED"
        elif thermal_info['has_heat_source'] or pose_info['is_dangerous']:
            color = (0, 255, 255)
            status_text = "WARNING"
        else:
            color = (0, 255, 0)
            status_text = "NORMAL"
        
        text = f"{status_text} | Temp:{thermal_info['max_temp']:.1f}°C | Ratio:{pose_info['ratio']:.3f}"
        cv2.putText(canvas, text, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        detail_text = f"Thermal:{thermal_info['has_heat_source']} Pose:{pose_info['is_dangerous']} K:{thermal_info['K_value_used']:.2f}"
        cv2.putText(canvas, detail_text, (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, canvas)
        
        return canvas


def load_threshold_model(model_path: str) -> Dict:
    """
    加载训练好的阈值模型
    """
    with open(model_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_detector_from_models(thermal_K_model_path: str = None,
                               pose_model_path: str = None,
                               anchor_type: str = 'nose_ear') -> SmokingBehaviorDetector:
    """
    从训练好的模型创建检测器
    
    Args:
        thermal_K_model_path: K值模型路径
        pose_model_path: 姿态阈值模型路径
        anchor_type: 锚点类型
        
    Returns:
        SmokingBehaviorDetector: 检测器实例
    """
    K_value = 1.5
    pose_threshold = 0.42
    
    if thermal_K_model_path and Path(thermal_K_model_path).exists():
        thermal_model = load_threshold_model(thermal_K_model_path)
        K_value = thermal_model.get('best_K', 1.5)
    
    if pose_model_path and Path(pose_model_path).exists():
        pose_model = load_threshold_model(pose_model_path)
        pose_threshold = pose_model.get('best_threshold', 0.42)
    
    return SmokingBehaviorDetector(
        K_value=K_value,
        pose_threshold=pose_threshold,
        anchor_type=anchor_type
    )


if __name__ == '__main__':
    np.random.seed(42)
    
    thermal_matrix = np.random.uniform(25.0, 45.0, (1520, 2688, 1)).astype(np.float32)
    
    mock_keypoints = np.array([
        [1344, 760, 0.9],
        [1320, 745, 0.85],
        [1368, 745, 0.88],
        [1305, 755, 0.82],
        [1383, 755, 0.80],
        [1150, 850, 0.92],
        [1538, 850, 0.91],
        [1100, 950, 0.88],
        [1588, 950, 0.87],
        [1280, 780, 0.85],
        [1408, 780, 0.84],
        [1180, 1100, 0.90],
        [1508, 1100, 0.89],
        [1200, 1350, 0.80],
        [1488, 1350, 0.79],
        [1220, 1600, 0.75],
        [1468, 1600, 0.74]
    ])
    
    detector = SmokingBehaviorDetector(
        K_value=1.5,
        pose_threshold=0.42,
        anchor_type='nose_ear'
    )
    
    thermal_matrix[750:810, 1350:1450] = np.random.uniform(100.0, 120.0, (60, 100, 1)).astype(np.float32)
    
    result = detector.infer_smoking_behavior(
        thermal_matrix=thermal_matrix,
        keypoints=mock_keypoints,
        keypoint_scores=mock_keypoints[:, 2],
        image_shape=(1520, 2688),
        frame_id="test_0001"
    )
    
    print("=" * 60)
    print("吸烟行为检测结果（基于K值）")
    print("=" * 60)
    print(f"帧ID: {result['frame_id']}")
    print(f"时间戳: {result['timestamp']}")
    print(f"锚点类型: {result['anchor_type']}")
    print()
    print("【热成像检测】")
    print(f"  热源检测: {result['thermal_detection']['has_heat_source']}")
    print(f"  最高温度: {result['thermal_detection']['max_temp']:.1f}°C")
    print(f"  K值: {result['thermal_detection']['K_value_used']:.2f}")
    print(f"  检测半径: {result['thermal_detection']['radius_used']:.1f}")
    print(f"  锚点距离: {result['thermal_detection']['anchor_distance']:.1f}")
    print()
    print("【姿态检测】")
    print(f"  锚点距离: {result['pose_detection']['anchor_distance']:.2f}")
    print(f"  鼻腕距离: {result['pose_detection']['nose_wrist_distance']:.2f}")
    print(f"  距离比值: {result['pose_detection']['ratio']:.4f}")
    print(f"  危险判定: {result['pose_detection']['is_dangerous']}")
    print(f"  使用阈值: {result['pose_detection']['threshold_used']:.4f}")
    print()
    print("【综合判定】")
    print(f"  吸烟行为: {result['smoking判定']['is_smoking']}")
    print(f"  置信度: {result['smoking判定']['confidence']}")
    print(f"  判定依据: {result['smoking判定']['reason']}")
    print("=" * 60)
    
    output_json = 'outputs/smoking_detection_result.json'
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_json}")