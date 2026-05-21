"""
评估脚本：计算吸烟检测模型的准确率（Nose-Shoulder锚点版本）

功能：
1. 读取 pose_est_calc copy.py 生成的统计 JSON
2. 使用 empirical_smk_thres copy.py 训练的阈值进行评估
3. 输出混淆矩阵、准确率、精确率、召回率等指标
"""

import json
from pathlib import Path
import glob


def find_latest_json_files():
    """查找最新的normal和warning JSON文件"""
    jsons_dir = Path(__file__).parent / "outputs" / "jsons"
    
    # 查找normal文件
    normal_files = glob.glob(str(jsons_dir / "pose_metrics_normal_*.json"))
    warning_files = glob.glob(str(jsons_dir / "pose_metrics_warning_*.json"))
    
    if not normal_files or not warning_files:
        raise ValueError("未找到JSON文件")
    
    # 按修改时间排序
    normal_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    warning_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    return normal_files[0], warning_files[0]


def load_thresholds(threshold_file: str) -> dict:
    """加载训练好的阈值"""
    with open(threshold_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'nonorm_ratio': data['metrics']['nonorm_ratio']['best_threshold'],
        'norm_ratio': data['metrics']['norm_ratio']['best_threshold']
    }


def evaluate_model(normal_json: str, warning_json: str, thresholds: dict, metric: str = 'nonorm_ratio') -> dict:
    """评估模型效果"""
    # 加载数据
    with open(normal_json, 'r', encoding='utf-8') as f:
        normal_data = json.load(f)
    
    with open(warning_json, 'r', encoding='utf-8') as f:
        warning_data = json.load(f)
    
    threshold = thresholds[metric]
    
    # 提取样本
    normal_samples = []
    for img_info in normal_data['infoes'].values():
        for instance in img_info['instances']:
            if not instance.get('skipped', False) and instance.get(metric) is not None:
                normal_samples.append(instance[metric])
    
    warning_samples = []
    for img_info in warning_data['infoes'].values():
        for instance in img_info['instances']:
            if not instance.get('skipped', False) and instance.get(metric) is not None:
                warning_samples.append(instance[metric])
    
    # 计算混淆矩阵
    # 规则: ratio >= threshold -> warning
    tn = sum(1 for v in normal_samples if v < threshold)  # 正常判为正常
    fp = sum(1 for v in normal_samples if v >= threshold) # 正常判为吸烟
    fn = sum(1 for v in warning_samples if v < threshold) # 吸烟判为正常
    tp = sum(1 for v in warning_samples if v >= threshold) # 吸烟判为吸烟
    
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'threshold': threshold,
        'metric': metric,
        'total_samples': total,
        'normal_count': len(normal_samples),
        'warning_count': len(warning_samples),
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        },
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def format_result(result: dict) -> str:
    """格式化评估结果"""
    lines = [
        f"\n{'='*60}",
        f"评估结果 ({result['metric']}) - Nose-Shoulder锚点",
        f"{'='*60}",
        f"使用阈值: {result['threshold']:.6f}",
        f"样本总数: {result['total_samples']} (normal:{result['normal_count']}, warning:{result['warning_count']})",
        f"",
        f"混淆矩阵:",
        f"          预测",
        f"          normal  warning",
        f"真实 normal    {result['confusion_matrix']['tn']:>6}  {result['confusion_matrix']['fp']:>7}",
        f"真实 warning   {result['confusion_matrix']['fn']:>6}  {result['confusion_matrix']['tp']:>7}",
        f"",
        f"评估指标:",
        f"  准确率 (Accuracy): {result['accuracy']:.2%}",
        f"  精确率 (Precision): {result['precision']:.2%}",
        f"  召回率 (Recall):    {result['recall']:.2%}",
        f"  F1分数 (F1):       {result['f1']:.2%}",
        f"",
        f"说明:",
        f"  - tn: 真阴性 (正常样本正确识别)",
        f"  - tp: 真阳性 (吸烟样本正确识别)",
        f"  - fn: 假阴性 (吸烟样本漏检)",
        f"  - fp: 假阳性 (正常样本误判)",
        f"{'='*60}"
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    # 自动查找最新的JSON文件
    try:
        normal_json, warning_json = find_latest_json_files()
        print(f"\n找到最新文件:")
        print(f"  Normal: {Path(normal_json).name}")
        print(f"  Warning: {Path(warning_json).name}")
    except Exception as e:
        print(f"自动查找失败: {e}")
        # 如果自动查找失败，使用默认路径
        normal_json = "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_normal_2026-05-20_01-07-34.json"
        warning_json = "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_warning_2026-05-20_01-07-34.json"
        print(f"使用默认路径")
    
    threshold_file = "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/empirical_threshold_summary_nose_shoulder_v2.json"
    
    print("\n正在加载训练阈值 (Nose-Shoulder)...")
    thresholds = load_thresholds(threshold_file)
    print(f"已加载阈值: nonorm_ratio={thresholds['nonorm_ratio']}, norm_ratio={thresholds['norm_ratio']}")
    
    print("\n正在评估 nonorm_ratio 指标...")
    result1 = evaluate_model(normal_json, warning_json, thresholds, 'nonorm_ratio')
    print(format_result(result1))
    
    print("\n正在评估 norm_ratio 指标...")
    result2 = evaluate_model(normal_json, warning_json, thresholds, 'norm_ratio')
    print(format_result(result2))
    
    # 比较两个指标
    print("\n" + "="*60)
    print("指标对比 (Nose-Shoulder)")
    print("="*60)
    print(f"{'指标':<12} {'nonorm_ratio':>12} {'norm_ratio':>12}")
    print(f"{'阈值':<12} {result1['threshold']:>12.4f} {result2['threshold']:>12.4f}")
    print(f"{'准确率':<12} {result1['accuracy']:>12.2%} {result2['accuracy']:>12.2%}")
    print(f"{'精确率':<12} {result1['precision']:>12.2%} {result2['precision']:>12.2%}")
    print(f"{'召回率':<12} {result1['recall']:>12.2%} {result2['recall']:>12.2%}")
    print(f"{'F1':<12} {result1['f1']:>12.2%} {result2['f1']:>12.2%}")
    print("="*60)