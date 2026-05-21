import numpy as np
import cv2
from pathlib import Path
import json
import time
from typing import List, Tuple, Dict
from thermal_processor import ThermalProcessor


class ThermalThresholdTrainer:
    def __init__(self, overheat_threshold: float = 100.0):
        self.processor = ThermalProcessor(overheat_threshold=overheat_threshold)
        self.temperature_ranges = {
            'normal': [],
            'smoking': []
        }
    
    def extract_temperature_stats(self, thermal_matrix: np.ndarray, 
                                 bbox: List[int] = None) -> Dict[str, float]:
        """
        从热成像矩阵中提取温度统计信息
        
        Args:
            thermal_matrix: 热成像温度矩阵
            bbox: 可选的ROI区域 [x1, y1, x2, y2]，如果为None则使用整个矩阵
            
        Returns:
            dict: 包含温度统计信息
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi = thermal_matrix[y1:y2, x1:x2]
        else:
            roi = thermal_matrix
        
        if roi.size == 0:
            return {
                'max': None,
                'min': None,
                'mean': None,
                'median': None,
                'std': None
            }
        
        return {
            'max': float(np.max(roi)),
            'min': float(np.min(roi)),
            'mean': float(np.mean(roi)),
            'median': float(np.median(roi)),
            'std': float(np.std(roi))
        }
    
    def load_thermal_data(self, data_dir: str, bbox: List[int] = None,
                         thermal_suffix: str = '.npy') -> List[Dict]:
        """
        加载目录中的热成像数据
        
        Args:
            data_dir: 数据目录路径（包含 thermal/ 子目录）
            bbox: 可选的ROI区域 [x1, y1, x2, y2]
            thermal_suffix: 热成像文件后缀，默认 '.npy'
            
        Returns:
            list: 每个样本的温度统计信息
        """
        data_path = Path(data_dir)
        thermal_path = data_path / 'thermal'
        
        if not thermal_path.exists():
            thermal_path = data_path
        
        if not thermal_path.exists():
            print(f"警告: 目录不存在 {thermal_path}")
            return []
        
        thermal_files = list(thermal_path.glob(f'*{thermal_suffix}'))
        
        samples = []
        for thermal_file in thermal_files:
            try:
                thermal_matrix = np.load(str(thermal_file))
                stats = self.extract_temperature_stats(thermal_matrix, bbox)
                samples.append({
                    'file': str(thermal_file),
                    'stats': stats
                })
            except Exception as e:
                print(f"加载失败 {thermal_file}: {e}")
                continue
        
        return samples
    
    def add_training_sample(self, thermal_matrix: np.ndarray, 
                           bbox: List[int] = None,
                           label: str = 'normal'):
        """
        添加训练样本
        
        Args:
            thermal_matrix: 热成像矩阵
            bbox: 可选的ROI区域
            label: 标签，'normal' 或 'smoking'
        """
        stats = self.extract_temperature_stats(thermal_matrix, bbox)
        
        if label == 'normal':
            self.temperature_ranges['normal'].append(stats)
        elif label == 'smoking':
            self.temperature_ranges['smoking'].append(stats)
    
    def search_best_threshold(self, metric: str = 'max') -> Dict:
        """
        搜索最优温度阈值
        
        Args:
            metric: 用于分类的温度指标，'max', 'mean', 'median'
            
        Returns:
            dict: 最优阈值及相关统计信息
        """
        normal_values = [s['stats'][metric] for s in self.temperature_ranges['normal'] 
                        if s['stats'][metric] is not None]
        smoking_values = [s['stats'][metric] for s in self.temperature_ranges['smoking'] 
                        if s['stats'][metric] is not None]
        
        if not normal_values or not smoking_values:
            return {
                'best_threshold': None,
                'accuracy': 0.0,
                'error': '样本不足'
            }
        
        all_values = sorted(normal_values + smoking_values)
        
        thresholds = []
        thresholds.append(np.mean(normal_values))
        thresholds.append(np.mean(smoking_values))
        
        for i in range(len(all_values) - 1):
            thresholds.append((all_values[i] + all_values[i + 1]) / 2)
        
        best_threshold = None
        best_accuracy = 0.0
        best_direction = None
        
        for threshold in thresholds:
            for direction in ['smoking_if_ge', 'smoking_if_le']:
                if direction == 'smoking_if_ge':
                    normal_correct = sum(1 for v in normal_values if v < threshold)
                    smoking_correct = sum(1 for v in smoking_values if v >= threshold)
                else:
                    normal_correct = sum(1 for v in normal_values if v > threshold)
                    smoking_correct = sum(1 for v in smoking_values if v <= threshold)
                
                accuracy = (normal_correct + smoking_correct) / (len(normal_values) + len(smoking_values))
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
                    best_direction = direction
        
        return {
            'best_threshold': float(best_threshold),
            'direction': best_direction,
            'accuracy': float(best_accuracy),
            'metric_used': metric,
            'normal_stats': {
                'count': len(normal_values),
                'mean': float(np.mean(normal_values)),
                'std': float(np.std(normal_values)),
                'min': float(np.min(normal_values)),
                'max': float(np.max(normal_values))
            },
            'smoking_stats': {
                'count': len(smoking_values),
                'mean': float(np.mean(smoking_values)),
                'std': float(np.std(smoking_values)),
                'min': float(np.min(smoking_values)),
                'max': float(np.max(smoking_values))
            },
            'normal_values': [float(v) for v in normal_values],
            'smoking_values': [float(v) for v in smoking_values]
        }
    
    def train(self, metric: str = 'max') -> Dict:
        """
        训练并返回最优阈值
        
        Args:
            metric: 用于分类的温度指标
            
        Returns:
            dict: 训练结果
        """
        result = self.search_best_threshold(metric)
        result['timestamp'] = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        return result
    
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
    
    def load_model(self, model_path: str) -> Dict:
        """
        加载训练模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            dict: 模型参数
        """
        with open(model_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def train_from_directories(normal_dir: str, smoking_dir: str,
                           output_path: str = 'outputs/thermal_threshold_model.json',
                           metric: str = 'max') -> Dict:
    """
    从两个目录训练温度阈值模型
    
    Args:
        normal_dir: 正常（无烟）样本目录
        smoking_dir: 吸烟（有烟）样本目录
        output_path: 模型输出路径
        metric: 用于分类的温度指标
        
    Returns:
        dict: 训练结果
    """
    trainer = ThermalThresholdTrainer()
    
    normal_samples = trainer.load_thermal_data(normal_dir)
    smoking_samples = trainer.load_thermal_data(smoking_dir)
    
    print(f"加载完成:")
    print(f"  正常样本: {len(normal_samples)} 个")
    print(f"  吸烟样本: {len(smoking_samples)} 个")
    
    trainer.temperature_ranges['normal'] = normal_samples
    trainer.temperature_ranges['smoking'] = smoking_samples
    
    result = trainer.train(metric=metric)
    
    trainer.save_model(output_path, result)
    
    print("\n===== 训练结果 =====")
    print(f"最优阈值: {result['best_threshold']:.2f}C")
    print(f"分类方向: {result['direction']}")
    print(f"准确率: {result['accuracy']:.2%}")
    print(f"\n正常样本温度范围: {result['normal_stats']['min']:.1f} - {result['normal_stats']['max']:.1f}C")
    print(f"吸烟样本温度范围: {result['smoking_stats']['min']:.1f} - {result['smoking_stats']['max']:.1f}C")
    
    return result


if __name__ == '__main__':
    normal_dir = "/root/autodl-tmp/projects/dongguan/dataset/thermal/normal"
    smoking_dir = "/root/autodl-tmp/projects/dongguan/dataset/thermal/smoking"
    output_path = "outputs/thermal_threshold_model.json"
    
    result = train_from_directories(
        normal_dir=normal_dir,
        smoking_dir=smoking_dir,
        output_path=output_path,
        metric='max'
    )