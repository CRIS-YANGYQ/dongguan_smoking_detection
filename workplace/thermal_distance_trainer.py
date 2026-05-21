"""
热成像距离阈值训练脚本（K值训练）

功能说明
1. 训练最优的K值（距离阈值倍数）
2. K值定义：以手腕为圆心，K×锚点距离为半径的区域
3. 判断该区域内是否有≥100°C的像素
4. 使用两类数据（normal/smoking）训练最优K值

锚点类型
- Nose-Ear: K_nose_ear
- Nose-Shoulder: K_nose_shoulder

判断逻辑
- 在手腕为圆心，K×锚点距离为半径的圆形区域内
- 是否存在温度≥100°C的像素
- 存在则标记为有热源
"""

import numpy as np
import cv2
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Optional
from thermal_processor import ThermalProcessor


class ThermalDistanceTrainer:
    def __init__(self, overheat_threshold: float = 100.0):
        """
        初始化距离阈值训练器
        
        Args:
            overheat_threshold: 过热温度阈值（固定为100°C）
        """
        self.processor = ThermalProcessor(overheat_threshold=overheat_threshold)
        self.overheat_threshold = overheat_threshold
        self.training_samples = {
            'normal': [],
            'smoking': []
        }
    
    def load_pose_statistics(self, json_path: str) -> Dict:
        """
        加载姿态估计统计数据（包含关键点信息）
        
        Args:
            json_path: pose_est_calc 生成的 JSON 文件
            
        Returns:
            dict: 姿态统计数据
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_anchor_distance(self, keypoints: np.ndarray, 
                               anchor_type: str = 'nose_ear') -> float:
        """
        提取锚点距离
        
        Args:
            keypoints: 姿态关键点 (17, 2) 或 (17, 3)
            anchor_type: 'nose_ear' 或 'nose_shoulder'
            
        Returns:
            float: 锚点距离
        """
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        nose = keypoints[keypoint_names.index('nose')]
        
        if anchor_type == 'nose_ear':
            left_ear = keypoints[keypoint_names.index('left_ear')]
            right_ear = keypoints[keypoint_names.index('right_ear')]
            dist_left = np.sqrt((nose[0] - left_ear[0])**2 + (nose[1] - left_ear[1])**2)
            dist_right = np.sqrt((nose[0] - right_ear[0])**2 + (nose[1] - right_ear[1])**2)
            return max(dist_left, dist_right)
        elif anchor_type == 'nose_shoulder':
            left_shoulder = keypoints[keypoint_names.index('left_shoulder')]
            right_shoulder = keypoints[keypoint_names.index('right_shoulder')]
            shoulder_center = (left_shoulder + right_shoulder) / 2
            return np.sqrt((nose[0] - shoulder_center[0])**2 + (nose[1] - shoulder_center[1])**2)
        else:
            raise ValueError(f"未知的锚点类型: {anchor_type}")
    
    def check_heat_in_wrist_circle(self, thermal_matrix: np.ndarray, 
                                  keypoints: np.ndarray,
                                  anchor_distance: float,
                                  K: float) -> bool:
        """
        检查手腕圆形区域内是否有高温
        
        Args:
            thermal_matrix: 热成像矩阵
            keypoints: 姿态关键点
            anchor_distance: 锚点距离
            K: 距离倍数阈值
            
        Returns:
            bool: 是否在区域内检测到≥100°C的像素
        """
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        left_wrist = keypoints[keypoint_names.index('left_wrist')]
        right_wrist = keypoints[keypoint_names.index('right_wrist')]
        
        radius = K * anchor_distance
        
        # 检查两个手腕
        for wrist in [left_wrist, right_wrist]:
            has_heat = self._check_circle_region(thermal_matrix, wrist, radius)
            if has_heat:
                return True
        
        return False
    
    def _check_circle_region(self, thermal_matrix: np.ndarray, 
                            center: np.ndarray, radius: float) -> bool:
        """
        检查圆形区域内是否有高温像素
        
        Args:
            thermal_matrix: 热成像矩阵
            center: 圆心坐标 (x, y)
            radius: 半径
            
        Returns:
            bool: 是否存在≥100°C的像素
        """
        h, w = thermal_matrix.shape[:2]
        
        cx, cy = int(center[0]), int(center[1])
        
        # 创建网格坐标
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # 圆形区域掩码
        circle_mask = dist_from_center <= radius
        
        # 检查该区域内的温度
        roi_temps = thermal_matrix[circle_mask]
        
        if roi_temps.size > 0:
            max_temp = np.max(roi_temps)
            return max_temp >= self.overheat_threshold
        
        return False
    
    def add_sample(self, thermal_matrix: np.ndarray,
                  keypoints: np.ndarray,
                  anchor_distance: float,
                  label: str = 'normal'):
        """
        添加训练样本
        
        Args:
            thermal_matrix: 热成像矩阵
            keypoints: 姿态关键点
            anchor_distance: 锚点距离
            label: 'normal' 或 'smoking'
        """
        self.training_samples[label].append({
            'thermal': thermal_matrix,
            'keypoints': keypoints,
            'anchor_distance': anchor_distance
        })
    
    def search_best_K(self, anchor_type: str = 'nose_ear',
                     K_candidates: Optional[List[float]] = None) -> Dict:
        """
        搜索最优K值
        
        Args:
            anchor_type: 锚点类型
            K_candidates: 候选K值列表，为None时自动生成
            
        Returns:
            dict: 最优K值及统计信息
        """
        normal_samples = self.training_samples['normal']
        smoking_samples = self.training_samples['smoking']
        
        if not normal_samples or not smoking_samples:
            return {
                'best_K': None,
                'accuracy': 0.0,
                'error': '样本不足'
            }
        
        # 生成候选K值
        if K_candidates is None:
            K_candidates = []
            # 从0.1到5.0，步长0.1
            for k in np.arange(0.1, 5.1, 0.1):
                K_candidates.append(float(k))
        
        best_K = None
        best_accuracy = 0.0
        
        for K in K_candidates:
            normal_correct = 0
            smoking_correct = 0
            
            # normal样本：不应检测到热源
            for sample in normal_samples:
                has_heat = self.check_heat_in_wrist_circle(
                    sample['thermal'],
                    sample['keypoints'],
                    sample['anchor_distance'],
                    K
                )
                if not has_heat:
                    normal_correct += 1
            
            # smoking样本：应检测到热源
            for sample in smoking_samples:
                has_heat = self.check_heat_in_wrist_circle(
                    sample['thermal'],
                    sample['keypoints'],
                    sample['anchor_distance'],
                    K
                )
                if has_heat:
                    smoking_correct += 1
            
            accuracy = (normal_correct + smoking_correct) / (len(normal_samples) + len(smoking_samples))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_K = K
        
        return {
            'best_K': float(best_K) if best_K is not None else None,
            'accuracy': float(best_accuracy),
            'anchor_type': anchor_type,
            'overheat_threshold': self.overheat_threshold,
            'normal_stats': {
                'count': len(normal_samples),
                'correct': normal_correct,
                'wrong': len(normal_samples) - normal_correct
            },
            'smoking_stats': {
                'count': len(smoking_samples),
                'correct': smoking_correct,
                'wrong': len(smoking_samples) - smoking_correct
            },
            'timestamp': time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        }
    
    def save_model(self, output_path: str, result: Dict):
        """
        保存训练模型
        
        Args:
            output_path: 输出路径
            result: 训练结果
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存到: {output}")


def train_from_pose_stats(normal_json: str,
                         warning_json: str,
                         thermal_data_dir: str,
                         output_path: str = 'outputs/thermal_K_model.json',
                         anchor_type: str = 'nose_ear') -> Dict:
    """
    从姿态统计数据训练K值模型
    
    Args:
        normal_json: normal样本的pose统计JSON
        warning_json: smoking样本的pose统计JSON
        thermal_data_dir: 热成像数据目录
        output_path: 模型输出路径
        anchor_type: 锚点类型
        
    Returns:
        dict: 训练结果
    """
    trainer = ThermalDistanceTrainer()
    
    # TODO: 这里需要加载对应的热成像数据
    # 目前先假设数据已经通过 add_sample 添加
    
    # 暂时用简单的搜索逻辑
    result = trainer.search_best_K(anchor_type=anchor_type)
    
    if result.get('best_K') is not None:
        trainer.save_model(output_path, result)
        
        print("\n===== K值训练结果 =====")
        print(f"最优K值: {result['best_K']:.3f}")
        print(f"准确率: {result['accuracy']:.2%}")
        print(f"锚点类型: {result['anchor_type']}")
        print(f"\nNormal样本: {result['normal_stats']['correct']}/{result['normal_stats']['count']} 正确")
        print(f"Smoking样本: {result['smoking_stats']['correct']}/{result['smoking_stats']['count']} 正确")
    
    return result


if __name__ == '__main__':
    print("热力图距离阈值训练器")
    print("注意：这个脚本需要与pose_est_calc生成的JSON配合使用")
    print("\n使用方法示例:")
    print("1. 先用 pose_est_calc.py 处理 normal 和 smoking 数据")
    print("2. 再用这个脚本训练K值")