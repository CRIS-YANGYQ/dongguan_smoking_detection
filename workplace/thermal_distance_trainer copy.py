
"""
热成像距离阈值训练脚本（K值训练）

功能说明
1. 训练最优的K值（距离阈值倍数）
2. K值定义：以鼻子为圆心，K×锚点距离为半径的区域
3. 判断该区域内是否有≥100°C的像素
4. 使用两类数据（normal/smoking）训练最优K值

锚点类型
- Nose-Ear: K_nose_ear
- Nose-Shoulder: K_nose_shoulder

判断逻辑
- 在鼻子为圆心，K×锚点距离为半径的圆形区域内
- 是否存在温度≥100°C的像素
- 存在则标记为有热源
"""

import numpy as np
import cv2
from pathlib import Path
import json
import time
import os
from typing import List, Dict, Tuple, Optional

try:
    from thermal_processor import ThermalProcessor
    HAS_THERMAL_PROCESSOR = True
except ImportError:
    HAS_THERMAL_PROCESSOR = False
    print("[WARNING] thermal_processor 模块未找到，将使用内置热成像处理逻辑")


class ThermalDistanceTrainer:
    def __init__(self, overheat_threshold: float = 100.0):
        """
        初始化距离阈值训练器
        
        Args:
            overheat_threshold: 过热温度阈值（固定为100°C）
        """
        self.processor = None
        if HAS_THERMAL_PROCESSOR:
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
    
    def load_thermal_npy(self, npy_path: str) -> np.ndarray:
        """
        加载NPY格式的热成像数据
        
        Args:
            npy_path: NPY文件路径
            
        Returns:
            np.ndarray: 热成像矩阵，像素值为摄氏度
        """
        try:
            thermal_data = np.load(npy_path)
            if len(thermal_data.shape) == 2:
                thermal_data = thermal_data[..., np.newaxis]
            return thermal_data
        except Exception as e:
            print(f"[ERROR] 加载热成像文件失败 {npy_path}: {e}")
            return None
    
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
    
    def check_heat_in_nose_circle(self, thermal_matrix: np.ndarray, 
                                 keypoints: np.ndarray,
                                 anchor_distance: float,
                                 K: float) -> bool:
        """
        检查鼻子圆形区域内是否有高温
        
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
        
        nose = keypoints[keypoint_names.index('nose')]
        
        radius = K * anchor_distance
        
        return self._check_circle_region(thermal_matrix, nose, radius)
    
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
        
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        circle_mask = dist_from_center <= radius
        
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
    
    def has_heat_near_nose(self, thermal_matrix: np.ndarray, keypoints: np.ndarray, threshold: float = 100.0) -> bool:
        """
        检查鼻子附近是否有高温热源
        
        Args:
            thermal_matrix: 热成像矩阵
            keypoints: 姿态关键点
            threshold: 温度阈值（默认100°C）
            
        Returns:
            bool: 鼻子附近是否有高温
        """
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        nose = keypoints[keypoint_names.index('nose')]
        nose_x, nose_y = int(nose[0]), int(nose[1])
        
        h, w = thermal_matrix.shape[:2]
        search_radius = 50
        
        y_min = max(0, nose_y - search_radius)
        y_max = min(h, nose_y + search_radius)
        x_min = max(0, nose_x - search_radius)
        x_max = min(w, nose_x + search_radius)
        
        roi = thermal_matrix[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return False
        
        max_temp = np.max(roi)
        return max_temp >= threshold
    
    def load_samples_from_data(self, 
                              rgb_pose_json,
                              thermal_base_dir,
                              pose_danger_dir='姿态危险',
                              heat_source_dir='嘴巴附近有热源',
                              pose_normal_dir='姿态正常',
                              no_heat_dir='嘴巴附近无热源'):
        """
        从数据目录加载训练样本
        
        训练策略：只有同时满足以下两个条件才判定为 smoking：
        1. 图片存在于"姿态危险"目录中
        2. 热成像检测到鼻子附近有≥100°C的像素
        
        注意：一个图片可以同时存在于多个目录中（如同时在"姿态危险"和"嘴巴附近有热源"）
        
        Args:
            rgb_pose_json: 姿态估计JSON数据
            thermal_base_dir: 热成像数据基础目录
            pose_danger_dir: 姿态危险目录名
            heat_source_dir: 有热源目录名
            pose_normal_dir: 姿态正常目录名
            no_heat_dir: 无热源目录名
            
        Returns:
            Tuple[int, int]: (normal样本数, smoking样本数)
        """
        normal_count = 0
        smoking_count = 0
        processed_images = set()
        
        infoes = rgb_pose_json.get('infoes', {})
        
        for img_name, img_info in infoes.items():
            img_stem = os.path.splitext(img_name)[0]
            img_path = img_info.get('img_path', '')
            
            img_dir = os.path.basename(os.path.dirname(img_path))
            
            if img_dir not in [pose_danger_dir, heat_source_dir, pose_normal_dir, no_heat_dir]:
                print(f"[WARNING] 未知类别目录: {img_dir}，跳过")
                continue
            
            if img_stem in processed_images:
                continue
            
            thermal_npy_path_danger = os.path.join(thermal_base_dir, pose_danger_dir, f'{img_stem}.npy')
            thermal_npy_path_heat = os.path.join(thermal_base_dir, heat_source_dir, f'{img_stem}.npy')
            thermal_npy_path_normal = os.path.join(thermal_base_dir, pose_normal_dir, f'{img_stem}.npy')
            thermal_npy_path_noheat = os.path.join(thermal_base_dir, no_heat_dir, f'{img_stem}.npy')
            
            thermal_matrix = None
            has_pose_danger = os.path.exists(thermal_npy_path_danger)
            has_heat_source = False
            
            if has_pose_danger:
                thermal_matrix = self.load_thermal_npy(thermal_npy_path_danger)
            elif os.path.exists(thermal_npy_path_heat):
                thermal_matrix = self.load_thermal_npy(thermal_npy_path_heat)
            elif os.path.exists(thermal_npy_path_normal):
                thermal_matrix = self.load_thermal_npy(thermal_npy_path_normal)
            elif os.path.exists(thermal_npy_path_noheat):
                thermal_matrix = self.load_thermal_npy(thermal_npy_path_noheat)
            
            if thermal_matrix is None:
                print(f"[WARNING] 未找到热成像文件: {img_stem}")
                continue
            
            instances = img_info.get('instances', [])
            for instance in instances:
                if instance.get('skipped', False):
                    continue
                
                keypoints_data = instance.get('keypoints')
                if keypoints_data is None:
                    continue
                
                keypoints = np.array(keypoints_data)
                
                anchor_distance = instance.get('nose_ear')
                if anchor_distance is None:
                    anchor_distance = self.extract_anchor_distance(keypoints)
                
                detected_heat = self.has_heat_near_nose(thermal_matrix, keypoints)
                
                if has_pose_danger and detected_heat:
                    self.add_sample(thermal_matrix, keypoints, anchor_distance, 'smoking')
                    smoking_count += 1
                else:
                    self.add_sample(thermal_matrix, keypoints, anchor_distance, 'normal')
                    normal_count += 1
            
            processed_images.add(img_stem)
        
        print(f"[INFO] 加载完成: normal={normal_count}, smoking={smoking_count}")
        return normal_count, smoking_count
    
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
        
        if K_candidates is None:
            K_candidates = []
            for k in np.arange(0.1, 5.1, 0.1):
                K_candidates.append(float(k))
        
        best_K = None
        best_accuracy = 0.0
        
        for K in K_candidates:
            normal_correct = 0
            smoking_correct = 0
            
            for sample in normal_samples:
                has_heat = self.check_heat_in_nose_circle(
                    sample['thermal'],
                    sample['keypoints'],
                    sample['anchor_distance'],
                    K
                )
                if not has_heat:
                    normal_correct += 1
            
            for sample in smoking_samples:
                has_heat = self.check_heat_in_nose_circle(
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
    
    训练策略：只有同时满足以下两个条件才判定为 smoking：
    1. 图片存在于"姿态危险"目录中
    2. 热成像检测到鼻子附近有≥100°C的像素
    
    注意：一个图片可以同时存在于多个目录中（如同时在"姿态危险"和"嘴巴附近有热源"）
    
    Args:
        normal_json: normal样本的pose统计JSON（姿态正常、嘴巴附近无热源）
        warning_json: smoking样本的pose统计JSON（姿态危险、嘴巴附近有热源）
        thermal_data_dir: 热成像数据目录
        output_path: 模型输出路径
        anchor_type: 锚点类型
        
    Returns:
        dict: 训练结果
    """
    trainer = ThermalDistanceTrainer()
    
    print(f"[INFO] 加载姿态估计数据...")
    normal_data = trainer.load_pose_statistics(normal_json)
    warning_data = trainer.load_pose_statistics(warning_json)
    
    print(f"[INFO] 加载热成像样本...")
    print(f"[INFO] 训练策略: smoking = (存在于姿态危险目录) AND (热成像检测到热源)")
    
    trainer.load_samples_from_data(normal_data, thermal_data_dir,
                                  pose_danger_dir='姿态危险',
                                  heat_source_dir='嘴巴附近有热源',
                                  pose_normal_dir='姿态正常',
                                  no_heat_dir='嘴巴附近无热源')
    
    trainer.load_samples_from_data(warning_data, thermal_data_dir,
                                  pose_danger_dir='姿态危险',
                                  heat_source_dir='嘴巴附近有热源',
                                  pose_normal_dir='姿态正常',
                                  no_heat_dir='嘴巴附近无热源')
    
    print(f"[INFO] 搜索最优K值...")
    result = trainer.search_best_K(anchor_type=anchor_type)
    
    if result.get('best_K') is not None:
        trainer.save_model(output_path, result)
        
        print("\n===== K值训练结果 =====")
        print(f"训练策略: smoking = (危险类别) AND (热成像检测到热源)")
        print(f"最优K值: {result['best_K']:.3f}")
        print(f"准确率: {result['accuracy']:.2%}")
        print(f"锚点类型: {result['anchor_type']}")
        print(f"过热阈值: {result['overheat_threshold']}°C")
        print(f"\nNormal样本: {result['normal_stats']['correct']}/{result['normal_stats']['count']} 正确")
        print(f"Smoking样本: {result['smoking_stats']['correct']}/{result['smoking_stats']['count']} 正确")
    
    return result


if __name__ == '__main__':
    print("="*60)
    print("热力图距离阈值训练器")
    print("="*60)
    print("注意：请先配置下面的路径参数")
    print("="*60)
    
    # ==================== 配置以下路径 ====================
    # 大文件夹路径
    BASE_DATA_DIR = "/root/autodl-tmp/projects/dongguan/dataset"
    
    # 姿态估计JSON路径（由pose_est_calc.py生成）
    NORMAL_JSON = "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态正常_2026-05-25_15-09-50.json"
    WARNING_JSON = "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态危险_2026-05-25_15-12-22.json"
    
    # 热成像数据目录（包含各类别子目录）
    THERMAL_DATA_DIR = "/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal"
    
    # 输出模型路径
    OUTPUT_PATH = "outputs/thermal/thermal_K_model.json"
    
    # 锚点类型: 'nose_ear' 或 'nose_shoulder'
    ANCHOR_TYPE = 'nose_ear'
    # ==================== 配置结束 ====================
    
    print(f"\n当前配置:")
    print(f"  BASE_DATA_DIR: {BASE_DATA_DIR}")
    print(f"  NORMAL_JSON: {NORMAL_JSON}")
    print(f"  WARNING_JSON: {WARNING_JSON}")
    print(f"  THERMAL_DATA_DIR: {THERMAL_DATA_DIR}")
    print(f"  OUTPUT_PATH: {OUTPUT_PATH}")
    print(f"  ANCHOR_TYPE: {ANCHOR_TYPE}")
    
    # 检查文件是否存在
    print("\n[INFO] 检查数据文件...")
    if not os.path.exists(NORMAL_JSON):
        print(f"[ERROR] NORMAL_JSON不存在: {NORMAL_JSON}")
    if not os.path.exists(WARNING_JSON):
        print(f"[ERROR] WARNING_JSON不存在: {WARNING_JSON}")
    if not os.path.exists(THERMAL_DATA_DIR):
        print(f"[ERROR] THERMAL_DATA_DIR不存在: {THERMAL_DATA_DIR}")
    
    print("\n[INFO] 开始训练...")
    result = train_from_pose_stats(
        normal_json=NORMAL_JSON,
        warning_json=WARNING_JSON,
        thermal_data_dir=THERMAL_DATA_DIR,
        output_path=OUTPUT_PATH,
        anchor_type=ANCHOR_TYPE
    )
    
    if result.get('error'):
        print(f"[ERROR] 训练失败: {result['error']}")
