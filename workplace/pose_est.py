from mmpose.apis import MMPoseInferencer

img_path = 'tests/data/coco/000000000785.jpg'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('human')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)

# 对于该帧 score > 0.3 的关键点少于 3~5 个就不再检测该人的吸烟动作 
# keypoints_score_threshold min_keypoints超参进行约束

# TODO 推荐方案二：鼻子到双肩中心点的距离（颈部长度的延伸）
# shoulder_center = (left_shoulder + right_shoulder) / 2
# Anchor = distance(nose, shoulder_center)

# TODO 增加 Y 轴高度限制： 抽烟时，手腕通常在嘴巴附近，也就是在眼睛下方、肩膀上方。可以增加条件：wrist.y > eye.y (假设 y 轴向下为正) 且 wrist.y < shoulder.y，这样能过滤掉撩头发、摸头顶的动作。
# Anchor = distance(nose, shoulder_center)
from math import sqrt
from pathlib import Path
import os
import cv2
import numpy as np
# 文件开头手动加父目录到 sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# 是上一级目录的mmpose.apis文件
from mmpose.apis import MMPoseInferencer
# 
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
    SKIP = 0
    for person_id, person in enumerate(instances):
        keypoints = person['keypoints']
        
        # 检查满足阈值要求的关键点个数,如果少于min_keypoints,则跳过该人的吸烟动作检测
        scores = person.get('keypoint_scores', [])
        quealified_keypoints_cnt = sum([1 for score in scores if score > keypoints_score_threshold]) # quealified_keypoints_cnt记录超过keypoints_score_threshold(合格)的关键点数
        if quealified_keypoints_cnt < min_keypoints:# 跳过该人的吸烟动作检测,开始下一个人的检测
            print(f'[LOG]: 跳过该人:{person_id},关键点数量不足{min_keypoints}个')
            SKIP += 1
            continue
        # 得到nose left_wrist right_wrist left_ear right_ear的score列表
        keypoints_name_lst = ['nose', 'left_wrist', 'right_wrist', 'left_ear', 'right_ear']
        score_lst = [scores[keypoint_names.index(keypoint_name)] for keypoint_name in keypoints_name_lst]
        # 检查keypoint score是否都超过keypoints_score_threshold
        nose_skip = score_lst[0] < keypoints_score_threshold # 鼻子低于阈值才跳过
        wrist_skip = score_lst[1] < keypoints_score_threshold and score_lst[2] < keypoints_score_threshold #两个都低于阈值才跳过,防止侧面人物被直接跳过
        ear_skip = score_lst[3] < keypoints_score_threshold and score_lst[4] < keypoints_score_threshold #两个都低于阈值才跳过,防止侧面人物被直接跳过
        if nose_skip or wrist_skip or ear_skip:
            print(f'[LOG]: 跳过该人:{person_id},五个关键点中存在低于{keypoints_score_threshold}的score')
            SKIP += 1
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

        risk = True if anchor_nose_ear * K_nose_ear > min_dist else False #TODO 如果涉及到的keypoint(比如鼻子手腕不满足keypoint阈值),则标记risk unknown?
        norm_risk = True if max_norm_anchor_nose_ear * K_nose_ear_norm > min_norm_dist else False
        
        # 更新全局
        if risk and not global_risk:
            global_risk = True
        if norm_risk and not global_norm_risk:
            global_norm_risk = True
        
        # 可视化标注文本
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
    global_txt = f'Risk: {global_risk}  Norm Risk: {global_norm_risk}  Skip {SKIP} persons'
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



if __name__ == '__main__':
    test_img_path = "/home/projects/dongguan/Github/mmpose/tests/data/smoking_v1/images/video101_task1_011.jpg"
    test_img_dir = "/root/autodl-tmp/projects/dongguan/dataset/sync_records/test/姿态RGB/正常"
    inferencer_out_dir = 'outputs'
    vis_output_dir = 'outputs/warning_vis'
    IS_DIR = True
    K_nose_ear = 2.35 # K_nose_ear越大锚点阈值越大,因此报警概率越高
    K_nose_ear_norm = 3.6 # K_nose_ear_norm越大归一化阈值越大,因此报警概率越高
    keypoints_score_threshold = 0.4 # 关键点 score 阈值, 超过阈值的关键点少于 min_keypoints 个就不再检测吸烟动作.keypoints_score_threshold越大,关键点筛选越严格
    min_keypoints = 5 # 最小关键点数, 小于该值就不再检测吸烟动作.min_keypoints越大,关键点筛选越严格
    
    
    inferencer = MMPoseInferencer('human')
    
    
    if not IS_DIR:
        results = inferencer(test_img_path, show=False, out_dir=inferencer_out_dir)
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        for result in results:
            instances = result['predictions'][0]

            
            
            # for person_id, person in enumerate(instances):
            #     keypoints = person['keypoints']
            #     scores = person.get('keypoint_scores', [])
                # xy_dict = {
                #     name: {
                #         'x': float(keypoints[i][0]),
                #         'y': float(keypoints[i][1]),
                #         'score': float(scores[i]) if i < len(scores) else None
                #     }
                #     for i, name in enumerate(keypoint_names)
                # }
                # nose_pos = xy_dict['nose']
                # left_wrist_pos = xy_dict['left_wrist']
                # right_wrist_pos = xy_dict['right_wrist']
                # left_wrist_nose_dist = distance_between_points(
                #     nose_pos, left_wrist_pos
                # )
                # right_wrist_nose_dist = distance_between_points(
                #     nose_pos, right_wrist_pos
                # )

                # nose_wrist_min_dist = min(
                #     left_wrist_nose_dist,
                #     right_wrist_nose_dist
                # # )

                # print(f'person {person_id}: nose:{xy_dict["nose"]}, '
                #       f'left_wrist:{xy_dict["left_wrist"]}, '
                #       f'right_wrist:{xy_dict["right_wrist"]}, '
                #       f'nose_wrist_min_dist:{nose_wrist_min_dist:.4f}')

            visualize_pose_distances(
                img_path=test_img_path,
                instances=instances,
                keypoint_names=keypoint_names,
                inferencer_out_dir=inferencer_out_dir,
                output_dir=vis_output_dir
            )
            
    else:
        img_path_lst = [str(Path(test_img_dir) / path) for path in os.listdir(test_img_dir) if path.endswith('.jpg')]
        # print(img_path_lst)
        for img_path in img_path_lst:
            results = inferencer(img_path, show=False, out_dir=inferencer_out_dir)
            keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]

            for result in results:
                instances = result['predictions'][0]
                
                visualize_pose_distances(
                    img_path=img_path,
                    instances=instances,
                    keypoint_names=keypoint_names,
                    inferencer_out_dir=inferencer_out_dir,
                    output_dir=vis_output_dir
                )


"""
对该目录下的所有图片进行pose估计，使用json记录每一张图中每一个人类实例记录
（1）记录nose到ear的最大距离以及归一化最大距离，（2）记录nose到wrist的最小距离以及归一化最小距离，随即计算每一个人类实例的(1)/(2)（包括距离和归一化距离）并记录该比值至json文件
json示例：
{
    "[img_filename_1]":
    {
        "img_path": "test_img_path_1",
        "instances": [
            {
                "person_id": 0,
                "nose_ear": 2.35,
                "nose_ear_norm": 3.6,
                "nose_wrist": 1.2,
                "nose_wrist_norm": 2.4,
                "nonorm_ratio": 0.x,
                "norm_ratio": 0.x,
            },
            {
                "person_id": 1,
                "nose_ear": 2.35,
                "nose_ear_norm": 3.6,
                "nose_wrist": 1.2,
                "nose_wrist_norm": 2.4,
                "nonorm_ratio": 0.x,
                "norm_ratio": 0.x,
            },
        ]
    },
    "[img_filename_2]":
    {
        "img_path": "test_img_path_2",
}
"""