"""
综合吸烟行为检测脚本

功能说明
1. 结合热成像数据（thermal_matrix）和RGB姿态估计进行吸烟行为检测
2. 热成像检测：判断手腕区域是否存在高温（热源）
3. 姿态估计：计算手腕到鼻子的距离是否在危险范围内
4. 综合判断：只有当两个条件同时满足时才判断为吸烟

判断逻辑
- 热成像条件：手腕区域温度 >= 热源阈值（训练得到的阈值）
- 姿态条件：距离比值 >= 距离阈值（从empirical_smk_thres.py的训练结果）
- 最终判断：热成像 AND 姿态

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
        "wrist_max_temp": 78.5,
        "threshold_used": 65.0
    },
    "pose_detection": {
        "nose_wrist_distance": 15.2,
        "nose_ear_distance": 35.6,
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

使用方法
# 1. 准备训练好的阈值模型（thermal_threshold 和 pose_threshold）
# 2. 调用 infer_smoking_behavior(thermal_matrix, rgb_frame, keypoints, thresholds)
"""

import numpy as np
import cv2
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Optional
from thermal_processor import ThermalProcessor


class SmokingBehaviorDetector:
    def __init__(self, 
                 thermal_threshold: float = 65.0,
                 pose_threshold: float = 0.42,
                 overheat_threshold: float = 100.0):
        """
        初始化吸烟行为检测器
        
        Args:
            thermal_threshold: 热源温度阈值（从训练得到）
            pose_threshold: 姿态距离比值阈值（从训练得到）
            overheat_threshold: 过热检测阈值（默认100°C）
        """
        self.thermal_threshold = thermal_threshold
        self.pose_threshold = pose_threshold
        self.thermal_processor = ThermalProcessor(overheat_threshold=overheat_threshold)
        
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """计算两点之间的欧氏距离"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_normalized_distance(self, p1: np.ndarray, p2: np.ndarray, 
                                     h: int, w: int) -> float:
        """计算两点之间的归一化距离"""
        norm_x = (p1[0] - p2[0]) / w
        norm_y = (p1[1] - p2[1]) / h
        return np.sqrt(norm_x**2 + norm_y**2)
    
    def detect_thermal_in_wrist_region(self, thermal_matrix: np.ndarray, 
                                      keypoints: np.ndarray,
                                      wrist_margin: int = 50) -> Dict:
        """
        检测手腕区域的热源
        
        Args:
            thermal_matrix: 热成像矩阵
            keypoints: 姿态关键点
            wrist_margin: 手腕区域扩展边距
            
        Returns:
            dict: 手腕区域温度检测结果
        """
        left_wrist = keypoints[self.keypoint_names.index('left_wrist')]
        right_wrist = keypoints[self.keypoint_names.index('right_wrist')]
        
        left_wrist_roi = self._extract_roi(thermal_matrix, left_wrist, wrist_margin)
        right_wrist_roi = self._extract_roi(thermal_matrix, right_wrist, wrist_margin)
        
        left_max_temp = np.max(left_wrist_roi) if left_wrist_roi.size > 0 else 0
        right_max_temp = np.max(right_wrist_roi) if right_wrist_roi.size > 0 else 0
        
        max_temp = max(left_max_temp, right_max_temp)
        has_heat_source = max_temp >= self.thermal_threshold
        
        return {
            'has_heat_source': has_heat_source,
            'left_wrist_max_temp': float(left_max_temp),
            'right_wrist_max_temp': float(right_max_temp),
            'max_temp': float(max_temp),
            'threshold_used': self.thermal_threshold
        }
    
    def _extract_roi(self, thermal_matrix: np.ndarray, 
                    point: np.ndarray, 
                    margin: int) -> np.ndarray:
        """提取感兴趣区域"""
        x, y = int(point[0]), int(point[1])
        h, w = thermal_matrix.shape[:2]
        
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + margin)
        y2 = min(h, y + margin)
        
        return thermal_matrix[y1:y2, x1:x2]
    
    def detect_pose_risk(self, keypoints: np.ndarray, 
                       keypoint_scores: np.ndarray,
                       image_h: int, image_w: int,
                       score_threshold: float = 0.4) -> Dict:
        """
        基于姿态估计检测吸烟风险
        
        Args:
            keypoints: 关键点坐标
            keypoint_scores: 关键点置信度
            image_h: 图像高度
            image_w: 图像宽度
            
        Returns:
            dict: 姿态风险检测结果
        """
        nose = keypoints[self.keypoint_names.index('nose')]
        left_wrist = keypoints[self.keypoint_names.index('left_wrist')]
        right_wrist = keypoints[self.keypoint_names.index('right_wrist')]
        left_ear = keypoints[self.keypoint_names.index('left_ear')]
        right_ear = keypoints[self.keypoint_names.index('right_ear')]
        
        nose_ear_l = self.calculate_distance(nose, left_ear)
        nose_ear_r = self.calculate_distance(nose, right_ear)
        nose_ear_max = max(nose_ear_l, nose_ear_r)
        
        nose_wrist_l = self.calculate_distance(nose, left_wrist)
        nose_wrist_r = self.calculate_distance(nose, right_wrist)
        nose_wrist_min = min(nose_wrist_l, nose_wrist_r)
        
        if nose_wrist_min > 0:
            ratio = nose_ear_max / nose_wrist_min
        else:
            ratio = float('inf')
        
        is_dangerous = ratio >= self.pose_threshold
        
        return {
            'nose_ear_distance': float(nose_ear_max),
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
        
        thermal_result = self.detect_thermal_in_wrist_region(thermal_matrix, keypoints)
        
        pose_result = self.detect_pose_risk(keypoints, keypoint_scores, h, w)
        
        is_smoking = thermal_result['has_heat_source'] and pose_result['is_dangerous']
        
        if is_smoking:
            if thermal_result['max_temp'] > self.thermal_threshold * 1.2:
                confidence = "high"
                reason = "热源检测+距离危险"
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
        nose_x, nose_y = int(nose[0]), int(nose[1])
        
        thermal_info = detection_result['thermal_detection']
        pose_info = detection_result['pose_detection']
        smoking判定 = detection_result['smoking判定']
        
        if smoking判定['is_smoking']:
            color = (0, 0, 255)
            status_text = "SMOKING DETECTED"
        elif thermal_info['has_heat_source'] or pose_info['is_dangerous']:
            color = (0, 255, 255)
            status_text = "WARNING"
        else:
            color = (0, 255, 0)
            status_text = "NORMAL"
        
        text = f"{status_text} | Temp:{thermal_info['max_temp']:.1f}C | Ratio:{pose_info['ratio']:.3f}"
        cv2.putText(canvas, text, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        detail_text = f"Thermal:{thermal_info['has_heat_source']} Pose:{pose_info['is_dangerous']}"
        cv2.putText(canvas, detail_text, (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, canvas)
        
        return canvas


def load_threshold_model(model_path: str) -> Dict:
    """
    加载训练好的阈值模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        dict: 模型参数
    """
    with open(model_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_detector_from_models(thermal_model_path: str = None,
                               pose_model_path: str = None) -> SmokingBehaviorDetector:
    """
    从训练好的模型创建检测器
    
    Args:
        thermal_model_path: 热成像阈值模型路径
        pose_model_path: 姿态阈值模型路径
        
    Returns:
        SmokingBehaviorDetector: 检测器实例
    """
    thermal_threshold = 65.0
    pose_threshold = 0.42
    
    if thermal_model_path and Path(thermal_model_path).exists():
        thermal_model = load_threshold_model(thermal_model_path)
        thermal_threshold = thermal_model.get('best_threshold', 65.0)
    
    if pose_model_path and Path(pose_model_path).exists():
        pose_model = load_threshold_model(pose_model_path)
        pose_threshold = pose_model.get('best_threshold', 0.42)
    
    return SmokingBehaviorDetector(
        thermal_threshold=thermal_threshold,
        pose_threshold=pose_threshold
    )


def infer_from_provider(rgb_frame: np.ndarray,
                       thermal_matrix: np.ndarray,
                       timestamp: float,
                       keypoints: np.ndarray,
                       keypoint_scores: np.ndarray,
                       detector: SmokingBehaviorDetector,
                       save_visualization: bool = False,
                       output_dir: str = 'outputs/smoking_detection') -> Dict:
    """
    从ThermalRGBProvider获取数据后进行推理
    
    Args:
        rgb_frame: RGB图像 (1520, 2688, 3)
        thermal_matrix: 热成像矩阵 (1520, 2688, 1)
        timestamp: 时间戳
        keypoints: 姿态关键点
        keypoint_scores: 关键点置信度
        detector: 检测器实例
        save_visualization: 是否保存可视化
        output_dir: 输出目录
        
    Returns:
        dict: 检测结果
    """
    frame_id = f"{int(timestamp)}"
    
    result = detector.infer_smoking_behavior(
        thermal_matrix=thermal_matrix,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        image_shape=rgb_frame.shape[:2],
        frame_id=frame_id
    )
    
    if save_visualization:
        output_path = f"{output_dir}/{frame_id}_detection.jpg"
        detector.visualize_detection(rgb_frame, result, keypoints, output_path)
        result['visualization_path'] = output_path
    
    return result


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
        thermal_threshold=65.0,
        pose_threshold=0.42
    )
    
    thermal_matrix[750:810, 1250:1410] = np.random.uniform(70.0, 90.0, (60, 160, 1)).astype(np.float32)
    
    result = detector.infer_smoking_behavior(
        thermal_matrix=thermal_matrix,
        keypoints=mock_keypoints,
        keypoint_scores=mock_keypoints[:, 2],
        image_shape=(1520, 2688),
        frame_id="test_0001"
    )
    
    print("=" * 60)
    print("吸烟行为检测结果")
    print("=" * 60)
    print(f"帧ID: {result['frame_id']}")
    print(f"时间戳: {result['timestamp']}")
    print()
    print("【热成像检测】")
    print(f"  热源检测: {result['thermal_detection']['has_heat_source']}")
    print(f"  最高温度: {result['thermal_detection']['max_temp']:.1f}°C")
    print(f"  左手腕温度: {result['thermal_detection']['left_wrist_max_temp']:.1f}°C")
    print(f"  右手腕温度: {result['thermal_detection']['right_wrist_max_temp']:.1f}°C")
    print(f"  使用阈值: {result['thermal_detection']['threshold_used']:.1f}°C")
    print()
    print("【姿态检测】")
    print(f"  鼻耳距离: {result['pose_detection']['nose_ear_distance']:.2f}")
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