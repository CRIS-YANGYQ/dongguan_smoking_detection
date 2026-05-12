"""
功能说明
1. 对单张图片或指定目录下的所有 `.jpg` 图片执行人体 pose 估计。
2. 基于关键点计算每个人体实例的以下指标：
   - `nose_ear`：`nose` 到 `left_ear` / `right_ear` 的最大欧氏距离。
   - `nose_ear_norm`：上述距离按图像宽高归一化后的最大值。
   - `nose_wrist`：`nose` 到 `left_wrist` / `right_wrist` 的最小欧氏距离。
   - `nose_wrist_norm`：上述距离按图像宽高归一化后的最小值。
   - `nonorm_ratio`：`nose_ear / nose_wrist`。
   - `norm_ratio`：`nose_ear_norm / nose_wrist_norm`。
3. 保留可视化输出：在 MMPose 生成的骨架图上叠加每个人的距离信息，并在图像左下角写入当前帧的全局风险结果。
4. 将运行参数、输出路径、时间戳以及每张图片的逐人统计结果统一写入 JSON 文件。

跳过逻辑
- 当满足 `score > keypoints_score_threshold` 的关键点数量少于 `min_keypoints` 时，跳过该人体实例。
- 当 `nose` 置信度不足，或左右手腕都不足，或左右耳朵都不足时，也跳过该人体实例。
- 被跳过的人体实例仍会写入 JSON，相关距离字段为 `null`，并补充：
  - `skipped`
  - `skip_reason`
- 未跳过的人体实例除距离与比值外，还会记录：
  - `risk`
  - `norm_risk`

JSON 顶层结构
- `IS_DIR`：当前是否为目录批处理模式。
- `predicted_path`：当前处理的目标路径；目录模式下为 `test_img_dir`，单图模式下为 `test_img_path`。
- `params`：本次运行使用的阈值参数，包括 `K_nose_ear`、`K_nose_ear_norm`、`keypoints_score_threshold`、`min_keypoints`。
- `outputs`：输出目录或文件的绝对路径映射。
- `timestamp`：本次运行生成 JSON 的时间戳。
- `infoes`：逐图片统计结果，键为图片文件名，值为该图片的检测记录。

单张图片记录结构
{
  "img_path": "test_img_path_1",
  "instances": [
    {
      "person_id": 0,
      "nose_ear": 2.35,
      "nose_ear_norm": 3.6,
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
      "nose_ear": null,
      "nose_ear_norm": null,
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
    "K_nose_ear": 2.35,
    "K_nose_ear_norm": 3.6,
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
# 文件开头手动加父目录到 sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# 是上一级目录的mmpose.apis文件
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
    # 检查满足阈值要求的关键点个数,如果少于min_keypoints,则跳过该人的吸烟动作检测
    
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
        
        # 检查满足阈值要求的关键点个数,如果少于min_keypoints,则跳过该人的吸烟动作检测
        scores = person.get('keypoint_scores', [])
        quealified_keypoints_cnt = sum([1 for score in scores if score > keypoints_score_threshold]) # quealified_keypoints_cnt记录超过keypoints_score_threshold(合格)的关键点数

        # 得到nose left_wrist right_wrist left_ear right_ear的score列表
        keypoints_name_lst = ['nose', 'left_wrist', 'right_wrist', 'left_ear', 'right_ear']
        score_lst = [scores[keypoint_names.index(keypoint_name)] for keypoint_name in keypoints_name_lst]
        # 检查keypoint score是否都超过keypoints_score_threshold
        nose_skip = score_lst[0] < keypoints_score_threshold # 鼻子低于阈值才跳过
        wrist_skip = score_lst[1] < keypoints_score_threshold and score_lst[2] < keypoints_score_threshold #两个都低于阈值才跳过,防止侧面人物被直接跳过
        ear_skip = score_lst[3] < keypoints_score_threshold and score_lst[4] < keypoints_score_threshold #两个都低于阈值才跳过,防止侧面人物被直接跳过

        skip_reason = None
        if quealified_keypoints_cnt < min_keypoints:
            skip_reason = f'关键点数量不足{min_keypoints}个'
        elif nose_skip or wrist_skip or ear_skip:
            skip_reason = f'关键点score低于{keypoints_score_threshold}'

        if skip_reason is not None:
            print(f'[LOG]: 跳过该人:{person_id},{skip_reason}')
            instance_records.append({
                'person_id': person_id,
                'nose_ear': None,
                'nose_ear_norm': None,
                'nose_wrist': None,
                'nose_wrist_norm': None,
                'nonorm_ratio': None,
                'norm_ratio': None,
                'skipped': True,
                'skip_reason': skip_reason,
            })
            skip_cnt += 1
            continue
        
        
        nose = keypoints[keypoint_names.index('nose')]
        left_wrist = keypoints[keypoint_names.index('left_wrist')] # 左手腕坐标
        right_wrist = keypoints[keypoint_names.index('right_wrist')] # 右手腕坐标
        left_ear = keypoints[keypoint_names.index('left_ear')] # 左耳坐标
        right_ear = keypoints[keypoint_names.index('right_ear')] # 右耳坐标
        
        # 最大鼻耳距离（最推荐，专门针对面部动作）作为平移不变性的锚点
        # 锚点计算方式： Anchor = max( distance(nose, left_ear), distance(nose, right_ear) )
        anchor_nose_ear = max(distance_between_points(nose, left_ear), distance_between_points(nose, right_ear)) # 最大鼻耳距离
        max_norm_anchor_nose_ear = max(normalized_distance_between_points(nose, left_ear, h, w), normalized_distance_between_points(nose, right_ear, h, w)) # 最大归一化鼻耳距离
        # 最小手腕鼻子距离
        dist_l = distance_between_points(nose, left_wrist)
        dist_r = distance_between_points(nose, right_wrist)
        min_dist = min(dist_l, dist_r)
        # 最小手腕鼻子距离归一化
        norm_dist_l = normalized_distance_between_points(nose, left_wrist, h, w)
        norm_dist_r = normalized_distance_between_points(nose, right_wrist, h, w)
        min_norm_dist = min(norm_dist_l, norm_dist_r)

        nonorm_ratio = anchor_nose_ear / min_dist if min_dist > 0 else None
        norm_ratio = max_norm_anchor_nose_ear / min_norm_dist if min_norm_dist > 0 else None
        risk = anchor_nose_ear * K_nose_ear > min_dist #TODO 如果涉及到的keypoint(比如鼻子手腕不满足keypoint阈值),则标记risk unknown?
        norm_risk = max_norm_anchor_nose_ear * K_nose_ear_norm > min_norm_dist

        if risk and not global_risk:
            global_risk = True
        if norm_risk and not global_norm_risk:
            global_norm_risk = True
            
        # 可视化标注文本
        instance_records.append({
            'person_id': person_id,
            'nose_ear': round(float(anchor_nose_ear), 6),
            'nose_ear_norm': round(float(max_norm_anchor_nose_ear), 6),
            'nose_wrist': round(float(min_dist), 6),
            'nose_wrist_norm': round(float(min_norm_dist), 6),
            'nonorm_ratio': round(float(nonorm_ratio), 6) if nonorm_ratio is not None else None,
            'norm_ratio': round(float(norm_ratio), 6) if norm_ratio is not None else None,
            'skipped': False,
            'risk': risk,
            'norm_risk': norm_risk,
        })

        person_text = f'L:{dist_l:.1f}  R:{dist_r:.1f}  L_norm:{norm_dist_l:.3f}  R_norm:{norm_dist_r:.3f}  Anchor:{anchor_nose_ear:.3f}  Anchor_norm:{max_norm_anchor_nose_ear:.3f}  Risk:{risk}  Norm_Risk:{norm_risk}'
        nose_x, nose_y = int(nose[0]), int(nose[1])
        text_x = nose_x
        text_y = nose_y - 10 if nose_y - 10 > 20 else nose_y + 20

        cv2.putText(
            canvas, person_text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
        )
        print(f'[LOG]: 检测该人:{person_id}的风险状态: {risk}  归一化风险状态: {norm_risk}')
    # 每个人体实例的检测结束,更新该帧全局风险状态
    global_txt = f'Risk: {global_risk}  Norm Risk: {global_norm_risk}  Skip {skip_cnt} persons'
    # 文本放在图片左下角，增加边距避免贴边
    # 加粗字体：通过两次绘制实现伪加粗
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



if __name__ == '__main__':
    test_img_path = "/home/projects/dongguan/Github/mmpose/tests/data/smoking_v1/images/video101_task1_011.jpg"
    test_img_dir = "/root/autodl-tmp/projects/dongguan/dataset/classified_test/normal"
    inferencer_out_dir = 'outputs'
    vis_output_dir = 'outputs/warning_vis'
    json_output_path = 'outputs/jsons/pose_metrics.json'
    IS_DIR = True
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    if IS_DIR:
        postfix = test_img_dir.split('/')[-1]
        json_output_path = f"{os.path.splitext(json_output_path)[0]}_{postfix}_{timestamp}.json"
    
    K_nose_ear = 2.35 # K_nose_ear越大锚点阈值越大,因此报警概率越高
    K_nose_ear_norm = 3.6 # K_nose_ear_norm越大归一化阈值越大,因此报警概率越高
    keypoints_score_threshold = 0.4 # 关键点 score 阈值, 超过阈值的关键点少于 min_keypoints 个就不再检测吸烟动作.keypoints_score_threshold越大,关键点筛选越严格
    min_keypoints = 5 # 最小关键点数, 若满足关键点 score 阈值的关键点数目小于该值就不再检测吸烟动作.min_keypoints越大,关键点筛选越严格
    
    
    inferencer = MMPoseInferencer('human')
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    # 初始化json_records，记录运行的元数据
    json_records = {}
    json_records['IS_DIR'] = IS_DIR
    json_records['predicted_path'] = test_img_dir if IS_DIR else test_img_path
    json_records['params'] = {
        'K_nose_ear': K_nose_ear,
        'K_nose_ear_norm': K_nose_ear_norm,
        'keypoints_score_threshold': keypoints_score_threshold,
        'min_keypoints': min_keypoints,
    }
    json_records['outputs'] = {
        inferencer_out_dir: str(Path(inferencer_out_dir).resolve()), # abs path
        vis_output_dir: str(Path(vis_output_dir).resolve()), # abs path
        json_output_path: str(Path(json_output_path).resolve()), # abs path
    }
    json_records['timestamp'] = timestamp
    json_records['infoes'] = {}
    

    if not IS_DIR:
        results = inferencer(test_img_path, show=False, out_dir=inferencer_out_dir)

        for result in results:
            instances = result['predictions'][0]


            frame_record = visualize_pose_distances(
                img_path=test_img_path,
                instances=instances,
                keypoint_names=keypoint_names,
                inferencer_out_dir=inferencer_out_dir,
                output_dir=vis_output_dir
            )
            json_records['infoes'][Path(test_img_path).name] = frame_record
            
    else:
        img_path_lst = [str(Path(test_img_dir) / path) for path in os.listdir(test_img_dir) if path.endswith('.jpg')]
        # print(img_path_lst)
        for img_path in img_path_lst:
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

    json_output = Path(json_output_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)
    print(f'[JSON]: 已保存到: {json_output}')
