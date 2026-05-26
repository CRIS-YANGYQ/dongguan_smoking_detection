"""
功能说明
1. 对单张图片或指定目录下的所有 `.jpg` 图片执行人体 pose 估计。
2. 基于关键点计算每个人体实例的以下指标：
   - `nose_shoulder`：`nose` 到双肩中心点的欧氏距离。
   - `nose_shoulder_norm`：上述距离按图像宽高归一化后的值。
   - `nose_wrist`：`nose` 到 `left_wrist` / `right_wrist` 的最小欧氏距离。
   - `nose_wrist_norm`：上述距离按图像宽高归一化后的最小值。
   - `nonorm_ratio`：`nose_shoulder / nose_wrist`。
   - `norm_ratio`：`nose_shoulder_norm / nose_wrist_norm`。
3. 保留可视化输出：在 MMPose 生成的骨架图上叠加每个人的距离信息，并在图像左下角写入当前帧的全局风险结果。
4. 将运行参数、输出路径、时间戳以及每张图片的逐人统计结果统一写入 JSON 文件。

跳过逻辑
- 当满足 `score > keypoints_score_threshold` 的关键点数量少于 `min_keypoints` 时，跳过该人体实例。
- 当 `nose` 置信度不足，或左右手腕都不足，或左右肩膀都不足时，也跳过该人体实例。
- 被跳过的人体实例仍会写入 JSON，相关距离字段为 `null`，并补充：
  - `skipped`
  - `skip_reason`
- 未跳过的人体实例除距离与比值外，还会记录：
  - `risk`
  - `norm_risk`

JSON 顶层结构
- `IS_DIR`：当前是否为目录批处理模式。
- `predicted_path`：当前处理的目标路径；目录模式下为 `test_img_dir`，单图模式下为 `test_img_path`。
- `params`：本次运行使用的阈值参数，包括 `K_nose_shoulder`、`K_nose_shoulder_norm`、`keypoints_score_threshold`、`min_keypoints`。
- `outputs`：输出目录或文件的绝对路径映射。
- `timestamp`：本次运行生成 JSON 的时间戳。
- `infoes`：逐图片统计结果，键为图片文件名，值为该图片的检测记录。

单张图片记录结构
{
  "img_path": "test_img_path_1",
  "instances": [
    {
      "person_id": 0,
      "nose_shoulder": 2.35,
      "nose_shoulder_norm": 3.6,
      "nose_wrist": 1.2,
      "nose_wrist_norm": 2.4,
      "nonorm_ratio": 1.958333,
      "norm_ratio": 1.5,
      "skipped": false,
      "risk": true,
      "norm_risk": true
    },
    {
      "person_id": 1,
      "nose_shoulder": null,
      "nose_shoulder_norm": null,
      "nose_wrist": null,
      "nose_wrist_norm": null,
      "nonorm_ratio": null,
      "norm_ratio": null,
      "skipped": true,
      "skip_reason": "关键点数量不足5个"
    }
  ],
  "global_risk": true,
  "global_norm_risk": true,
  "skip_persons": 1
}

完整 JSON 结构示例
{
  "IS_DIR": true,
  "predicted_path": "/path/to/images",
  "params": {
    "K_nose_shoulder": 2.35,
    "K_nose_shoulder_norm": 3.6,
    "keypoints_score_threshold": 0.4,
    "min_keypoints": 5
  },
  "outputs": {
    "outputs": "/abs/path/to/outputs",
    "outputs/warning_vis": "/abs/path/to/outputs/warning_vis",
    "outputs/jsons/pose_metrics_xxx.json": "/abs/path/to/outputs/jsons/pose_metrics_xxx.json"
  },
  "timestamp": "2026-05-12_22-21-56",
  "infoes": {
    "img1.jpg": {
      "img_path": "/path/to/img1.jpg",
      "instances": [],
      "global_risk": false,
      "global_norm_risk": false,
      "skip_persons": 0
    }
  }
}

输出位置
- 可视化图片目录：`outputs/warning_vis`
- JSON 基础输出路径：`outputs/jsons/pose_metrics.json`
- 当 `IS_DIR = True` 时，实际 JSON 文件名会自动追加目录名和时间戳，例如：`pose_metrics_normal_2026-05-12_22-21-56.json`

运行流程
- `IS_DIR = False` 时处理单张图片 `test_img_path`。
- `IS_DIR = True` 时遍历目录 `test_img_dir` 下所有 `.jpg` 图片。
- 运行开始时会生成 `timestamp`；目录模式下会基于目录名和时间戳动态拼接 JSON 输出文件名。
"""


from math import sqrt
from pathlib import Path
import json
import os
import cv2
import numpy as np
import time
import sys
PROJECT_ROOT = "/root/autodl-tmp/projects/dongguan/Github/mmpose"
sys.path.insert(0, PROJECT_ROOT)
from mmpose.apis import MMPoseInferencer

def distance_between_points(p1, p2):
    """计算两点之间的距离"""
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def normalized_distance_between_points(p1, p2, h, w):
    """计算两点之间的归一化距离"""
    norm_x_dist = (p1[0] - p2[0]) / w
    norm_y_dist = (p1[1] - p2[1]) / h
    norm_dist = sqrt(norm_x_dist ** 2 + norm_y_dist ** 2)
    return norm_dist

def visualize_pose_distances(img_path, instances, keypoint_names, inferencer_out_dir, output_dir):
    """在 inferencer 已保存的骨架图上，仅在 nose 头顶标注左右手腕距离后保存。"""

    src_path = Path(img_path)
    src_stem = src_path.stem
    src_suffix = src_path.suffix

    vis_img_path = Path(inferencer_out_dir) / 'visualizations' / f'{src_stem}{src_suffix}'
    canvas = cv2.imread(str(vis_img_path))
    if canvas is None:
        raise FileNotFoundError(f'无法读取 inferencer 可视化图: {vis_img_path}')

    h, w = canvas.shape[:2]

    global_risk = False
    global_norm_risk = False
    skip_cnt = 0
    instance_records = []
    for person_id, person in enumerate(instances):
        keypoints = person['keypoints']

        scores = person.get('keypoint_scores', [])
        quealified_keypoints_cnt = sum([1 for score in scores if score > keypoints_score_threshold])

        keypoints_name_lst = ['nose', 'left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder']
        score_lst = [scores[keypoint_names.index(keypoint_name)] for keypoint_name in keypoints_name_lst]
        nose_skip = score_lst[0] < keypoints_score_threshold
        wrist_skip = score_lst[1] < keypoints_score_threshold and score_lst[2] < keypoints_score_threshold
        shoulder_skip = score_lst[3] < keypoints_score_threshold and score_lst[4] < keypoints_score_threshold

        skip_reason = None
        if quealified_keypoints_cnt < min_keypoints:
            skip_reason = f'关键点数量不足{min_keypoints}个'
        elif nose_skip or wrist_skip or shoulder_skip:
            skip_reason = f'关键点score低于{keypoints_score_threshold}'

        if skip_reason is not None:
            print(f'[LOG]: 跳过该人:{person_id},{skip_reason}')
            instance_records.append({
                'person_id': person_id,
                'nose_shoulder': None,
                'nose_shoulder_norm': None,
                'nose_wrist': None,
                'nose_wrist_norm': None,
                'nonorm_ratio': None,
                'norm_ratio': None,
                'skipped': True,
                'skip_reason': skip_reason,
                'keypoints': keypoints
            })
            skip_cnt += 1
            continue


        nose = keypoints[keypoint_names.index('nose')]
        left_wrist = keypoints[keypoint_names.index('left_wrist')]
        right_wrist = keypoints[keypoint_names.index('right_wrist')]
        left_shoulder = keypoints[keypoint_names.index('left_shoulder')]
        right_shoulder = keypoints[keypoint_names.index('right_shoulder')]

        shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
        anchor_nose_shoulder = distance_between_points(nose, shoulder_center)
        norm_anchor_nose_shoulder = normalized_distance_between_points(nose, shoulder_center, h, w)
        dist_l = distance_between_points(nose, left_wrist)
        dist_r = distance_between_points(nose, right_wrist)
        min_dist = min(dist_l, dist_r)
        norm_dist_l = normalized_distance_between_points(nose, left_wrist, h, w)
        norm_dist_r = normalized_distance_between_points(nose, right_wrist, h, w)
        min_norm_dist = min(norm_dist_l, norm_dist_r)

        nonorm_ratio = anchor_nose_shoulder / min_dist if min_dist > 0 else None
        norm_ratio = norm_anchor_nose_shoulder / min_norm_dist if min_norm_dist > 0 else None
        risk = anchor_nose_shoulder * K_nose_shoulder > min_dist
        norm_risk = norm_anchor_nose_shoulder * K_nose_shoulder_norm > min_norm_dist

        if risk and not global_risk:
            global_risk = True
        if norm_risk and not global_norm_risk:
            global_norm_risk = True

        instance_records.append({
            'person_id': person_id,
            'nose_shoulder': round(float(anchor_nose_shoulder), 6),
            'nose_shoulder_norm': round(float(norm_anchor_nose_shoulder), 6),
            'nose_wrist': round(float(min_dist), 6),
            'nose_wrist_norm': round(float(min_norm_dist), 6),
            'nonorm_ratio': round(float(nonorm_ratio), 6) if nonorm_ratio is not None else None,
            'norm_ratio': round(float(norm_ratio), 6) if norm_ratio is not None else None,
            'skipped': False,
            'risk': risk,
            'norm_risk': norm_risk,
            'keypoints': keypoints
            
        })

        person_text = f'L:{dist_l:.1f}  R:{dist_r:.1f}  L_norm:{norm_dist_l:.3f}  R_norm:{norm_dist_r:.3f}  Anchor:{anchor_nose_shoulder:.3f}  Anchor_norm:{norm_anchor_nose_shoulder:.3f}  Risk:{risk}  Norm_Risk:{norm_risk}'
        nose_x, nose_y = int(nose[0]), int(nose[1])
        text_x = nose_x
        text_y = nose_y - 10 if nose_y - 10 > 20 else nose_y + 20

        cv2.putText(
            canvas, person_text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
        )
        print(f'[LOG]: 检测该人:{person_id}的风险状态: {risk}  归一化风险状态: {norm_risk}')

    global_txt = f'Risk: {global_risk}  Norm Risk: {global_norm_risk}  Skip {skip_cnt} persons'
    cv2.putText(
        canvas, global_txt, (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4
    )
    cv2.putText(
        canvas, global_txt, (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f'{src_stem}_pose_predicted{src_suffix}'
    cv2.imwrite(str(out_file), canvas)
    print(f'[可视化]: 已保存到: {out_file}')
    return {
        'img_path': str(src_path),
        'instances': instance_records,
        'global_risk': global_risk,
        'global_norm_risk': global_norm_risk,
        'skip_persons': skip_cnt,
    }



def load_trained_thresholds(threshold_file: str = None) -> dict:
    """
    从训练结果文件加载训练好的阈值
    
    Args:
        threshold_file: 阈值文件路径，如果为None则使用默认路径
        
    Returns:
        dict: 包含训练好的阈值
    """
    if threshold_file is None:
        threshold_file = str(Path(__file__).resolve().parent / "outputs" / "jsons" / "empirical_threshold_summary_nose_shoulder_v2.json")
    
    if Path(threshold_file).exists():
        with open(threshold_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        K_nose_shoulder = data['metrics'].get('nonorm_ratio', {}).get('best_threshold', 1.5)
        K_nose_shoulder_norm = data['metrics'].get('norm_ratio', {}).get('best_threshold', 2.0)
        print(f"[INFO]: 已从 {threshold_file} 加载训练阈值")
        print(f"        K_nose_shoulder = {K_nose_shoulder}")
        print(f"        K_nose_shoulder_norm = {K_nose_shoulder_norm}")
    else:
        print(f"[WARNING]: 未找到阈值文件 {threshold_file}，使用默认值")
        K_nose_shoulder = 1.5
        K_nose_shoulder_norm = 2.0
    
    return {
        'K_nose_shoulder': K_nose_shoulder,
        'K_nose_shoulder_norm': K_nose_shoulder_norm
    }


if __name__ == '__main__':
    test_img_path = "/home/projects/dongguan/Github/mmpose/tests/data/smoking_v1/images/video101_task1_011.jpg"
    # 批量处理模式：同时处理多个目录
    test_img_dirs = [
        "/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态危险",
        "/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb/姿态正常"
    ]
    inferencer_out_dir = 'outputs'
    vis_output_dir = 'outputs/warning_vis'
    json_output_path = 'outputs/jsons/pose_metrics.json'
    IS_DIR = True
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    thresholds = load_trained_thresholds()
    K_nose_shoulder = thresholds['K_nose_shoulder']
    K_nose_shoulder_norm = thresholds['K_nose_shoulder_norm']
    keypoints_score_threshold = 0.4
    min_keypoints = 5
    
    inferencer = MMPoseInferencer('human')
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # 批量处理每个目录
    for test_img_dir in test_img_dirs:
        print(f'\n{"="*60}')
        print(f'开始处理目录: {test_img_dir}')
        print(f'{"="*60}\n')
        
        json_records = {}
        json_records['IS_DIR'] = IS_DIR
        json_records['predicted_path'] = test_img_dir
        json_records['params'] = {
            'K_nose_shoulder': K_nose_shoulder,
            'K_nose_shoulder_norm': K_nose_shoulder_norm,
            'keypoints_score_threshold': keypoints_score_threshold,
            'min_keypoints': min_keypoints,
        }
        json_records['timestamp'] = timestamp
        json_records['infoes'] = {}
        
        postfix = test_img_dir.split('/')[-1]
        current_json_output_path = f"{os.path.splitext(json_output_path)[0]}_{postfix}_{timestamp}.json"
        json_records['outputs'] = {
            inferencer_out_dir: str(Path(inferencer_out_dir).resolve()),
            vis_output_dir: str(Path(vis_output_dir).resolve()),
            current_json_output_path: str(Path(current_json_output_path).resolve()),
        }
        
        img_path_lst = [str(Path(test_img_dir) / path) for path in os.listdir(test_img_dir) if path.endswith('.jpg') or path.endswith('.png')]
        img_path_lst = sorted(img_path_lst)
        img_len = len(img_path_lst)
        for img_idx, img_path in enumerate(img_path_lst):
            print(f"\n[LOG]: Iter IMG {img_idx}/{img_len}")
            results = inferencer(img_path, show=False, out_dir=inferencer_out_dir)
            for result in results:
                
                instances = result['predictions'][0]
                frame_record = visualize_pose_distances(
                    img_path=img_path,
                    instances=instances,
                    keypoint_names=keypoint_names,
                    inferencer_out_dir=inferencer_out_dir,
                    output_dir=vis_output_dir
                )
                json_records['infoes'][Path(img_path).name] = frame_record
        
        json_output = Path(current_json_output_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(json_records, f, ensure_ascii=False, indent=2)
        print(f'\n[JSON]: {postfix} 目录处理完成，保存到: {json_output}')
    
    print(f'\n{"="*60}')
    print(f'所有目录处理完成！')
    print(f'{"="*60}')
    print(f'\n生成的文件：')
    for test_img_dir in test_img_dirs:
        postfix = test_img_dir.split('/')[-1]
        current_json_output_path = f"{os.path.splitext(json_output_path)[0]}_{postfix}_{timestamp}.json"
        print(f'  - {current_json_output_path}')
    print(f'\n运行以下命令更新路径并分析结果：')
    print(f'  python "empirical_smk_thres copy.py"')
