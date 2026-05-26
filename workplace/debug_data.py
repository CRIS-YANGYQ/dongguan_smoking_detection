
import json
import os

# 加载JSON
NORMAL_JSON = "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态正常_2026-05-25_15-09-50.json"
THERMAL_DIR = "/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal"

print("="*60)
print("调试：检查数据匹配情况")
print("="*60)

with open(NORMAL_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

infoes = data.get('infoes', {})
print(f"\n[INFO] JSON中共有 {len(infoes)} 个图片")

# 统计有效样本（不是skipped的）
valid_instances_count = 0
skipped_instances_count = 0
for img_name, img_info in infoes.items():
    instances = img_info.get('instances', [])
    for instance in instances:
        if instance.get('skipped', False):
            skipped_instances_count += 1
        else:
            valid_instances_count += 1

print(f"[INFO] 有效样本（未被skip）: {valid_instances_count}")
print(f"[INFO] 被跳过的样本（skipped）: {skipped_instances_count}")

# 检查热成像文件是否存在
found_count = 0
not_found_count = 0
not_found_samples = []

for img_name, img_info in infoes.items():
    img_stem = os.path.splitext(img_name)[0]
    
    # 尝试查找热成像文件
    found = False
    for root, dirs, files in os.walk(THERMAL_DIR):
        if f'{img_stem}.npy' in files:
            found = True
            break
    
    if found:
        found_count += 1
    else:
        not_found_count += 1
        if len(not_found_samples) < 10:  # 只记录前10个
            not_found_samples.append(img_stem)

print(f"\n[INFO] 找到热成像文件: {found_count}")
print(f"[INFO] 未找到热成像文件: {not_found_count}")

if not_found_samples:
    print(f"\n[WARNING] 前10个未找到的文件名：")
    for name in not_found_samples:
        print(f"  - {name}")

print("\n" + "="*60)
print("结论：")
print("="*60)

if valid_instances_count == 0:
    print("❌ 问题：JSON中的所有样本都被标记为 skipped（关键点数量不足）")
    print("   这意味着这些图片的姿态估计算法没有检测到足够的关键点")
    
if not_found_count > 0 and valid_instances_count > 0:
    print(f"⚠️  警告：{not_found_count} 个样本缺少对应的热成像文件")
    print("   这可能是因为output1、2、3合并后，有些文件确实不存在")
    
if found_count > 0 and valid_instances_count > 0:
    print(f"✓ 成功：{found_count} 个样本可以找到对应的热成像文件")
    print("   可以开始训练！")
