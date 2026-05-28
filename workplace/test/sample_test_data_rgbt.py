# 随机采样N个RGBT对放在data子目录下，要求rgb子目录的文件一一对应thermal的子文件
import os
import json
import random
import shutil
from pathlib import Path

N = 20
OUTPUT_DIR = Path("/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/test/data")
RGB_DIR = OUTPUT_DIR / "rgb"
THERMAL_DIR = OUTPUT_DIR / "thermal"

JSON_FILES = [
    "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态正常_2026-05-26_12-20-08.json",
    "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态危险_2026-05-26_12-20-08.json",
]

RGB_BASE_DIR = Path("/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/rgb")
THERMAL_BASE_DIR = Path("/root/autodl-tmp/projects/dongguan/dataset/sync_records/pictures/thermal")

def get_all_rgb_files():
    rgb_files = set()
    for json_file in JSON_FILES:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for filename in data.get('infoes', {}).keys():
            rgb_files.add(filename)
    return list(rgb_files)

def find_thermal_file(rgb_filename):
    base_name = Path(rgb_filename).stem
    for ext in ["", "_norm"]:
        npy_file = THERMAL_BASE_DIR / "姿态正常" / f"{base_name}.npy"
        if npy_file.exists():
            return npy_file
        npy_file = THERMAL_BASE_DIR / "姿态危险" / f"{base_name}.npy"
        if npy_file.exists():
            return npy_file
    return None

def main():
    RGB_DIR.mkdir(parents=True, exist_ok=True)
    THERMAL_DIR.mkdir(parents=True, exist_ok=True)

    all_rgb_files = get_all_rgb_files()
    print(f"Total RGB files found: {len(all_rgb_files)}")

    sampled_files = random.sample(all_rgb_files, min(N, len(all_rgb_files)))
    print(f"Sampled {len(sampled_files)} files")

    success_count = 0
    for rgb_filename in sampled_files:
        rgb_path = None
        for json_file in JSON_FILES:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            img_path = data.get('infoes', {}).get(rgb_filename, {}).get('img_path')
            if img_path and os.path.exists(img_path):
                rgb_path = img_path
                break

        if rgb_path is None:
            print(f"RGB file not found for: {rgb_filename}")
            continue

        thermal_path = find_thermal_file(rgb_filename)
        if thermal_path is None:
            print(f"Thermal file not found for: {rgb_filename}")
            continue

        shutil.copy(rgb_path, RGB_DIR / rgb_filename)
        shutil.copy(thermal_path, THERMAL_DIR / f"{Path(rgb_filename).stem}.npy")
        success_count += 1

    print(f"Successfully copied {success_count} RGBT pairs to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()