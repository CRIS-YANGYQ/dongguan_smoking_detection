"""
功能说明
1. 读取两组结构一致的姿态统计 JSON：
   - nose_ear：由 `pose_est_calc.py` 生成
   - nose_shoulder：由 `pose_est_calc copy.py` 生成
2. 以 nose_ear 的 `pose_metrics_*.json` 为基础，保持原有 JSON 结构不变：
   - 顶层仍为 `IS_DIR`、`predicted_path`、`params`、`outputs`、`timestamp`、`infoes`
   - 每张图片记录仍为 `img_path`、`instances`、`global_risk`、`global_norm_risk`、`skip_persons`
3. 仅在原有结构上追加 nose_shoulder 的必要字段，便于一个 JSON 同时保存两套姿态结果。
4. 分别输出 normal 与 warning 的合并版 `pose_metrics` JSON。
"""

import copy
import json
from pathlib import Path


NOSE_EAR_NORMAL_JSON = Path(
    "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态正常_2026-05-26_00-50-48.json"
)
NOSE_EAR_WARNING_JSON = Path(
    "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态危险_2026-05-26_00-51-10.json"
)
NOSE_SHOULDER_NORMAL_JSON = Path(
    "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态正常_2026-05-26_00-49-59.json"
)
NOSE_SHOULDER_WARNING_JSON = Path(
    "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态危险_2026-05-26_00-49-59.json"
)

MERGED_NORMAL_JSON = Path(
    "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态正常_merged.json"
)
MERGED_WARNING_JSON = Path(
    "/root/autodl-tmp/projects/dongguan/Github/mmpose/workplace/outputs/jsons/pose_metrics_姿态危险_merged.json"
)

PAIR_CONFIGS = (
    {
        "name": "姿态正常",
        "base_json": NOSE_EAR_NORMAL_JSON,
        "extra_json": NOSE_SHOULDER_NORMAL_JSON,
        "output_json": MERGED_NORMAL_JSON,
    },
    {
        "name": "姿态危险",
        "base_json": NOSE_EAR_WARNING_JSON,
        "extra_json": NOSE_SHOULDER_WARNING_JSON,
        "output_json": MERGED_WARNING_JSON,
    },
)

INSTANCE_EXTRA_FIELDS = (
    "nose_shoulder",
    "nose_shoulder_norm",
    "nose_wrist",
    "nose_wrist_norm",
    "nonorm_ratio",
    "norm_ratio",
    "skipped",
    "skip_reason",
    "risk",
    "norm_risk",
)


def load_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_outputs(base_outputs: dict, output_json: Path) -> dict:
    merged_outputs = {}
    for key, value in base_outputs.items():
        if str(value).endswith(".json"):
            continue
        merged_outputs[key] = value

    relative_output_key = str(Path("outputs") / "jsons" / output_json.name)
    merged_outputs[relative_output_key] = str(output_json.resolve())
    return merged_outputs


def merge_params(base_params: dict, extra_params: dict) -> dict:
    merged_params = copy.deepcopy(base_params)
    if "K_nose_shoulder" in extra_params:
        merged_params["K_nose_shoulder"] = extra_params["K_nose_shoulder"]
    if "K_nose_shoulder_norm" in extra_params:
        merged_params["K_nose_shoulder_norm"] = extra_params["K_nose_shoulder_norm"]
    return merged_params


def get_instance_key(instance: dict, fallback_idx: int) -> tuple:
    person_id = instance.get("person_id")
    if person_id is not None:
        return ("person_id", person_id)
    return ("index", fallback_idx)


def merge_instance(base_instance: dict, extra_instance: dict) -> dict:
    merged_instance = copy.deepcopy(base_instance)

    for field in INSTANCE_EXTRA_FIELDS:
        if field not in extra_instance:
            continue

        if field.startswith("nose_shoulder"):
            merged_key = field
        else:
            merged_key = f"{field}_nose_shoulder"
        merged_instance[merged_key] = extra_instance[field]

    return merged_instance


def merge_frame(base_frame: dict, extra_frame: dict, img_name: str) -> dict:
    merged_frame = copy.deepcopy(base_frame)
    merged_frame["global_risk_nose_shoulder"] = extra_frame.get("global_risk")
    merged_frame["global_norm_risk_nose_shoulder"] = extra_frame.get("global_norm_risk")
    merged_frame["skip_persons_nose_shoulder"] = extra_frame.get("skip_persons")

    base_instances = base_frame.get("instances", [])
    extra_instances = extra_frame.get("instances", [])

    extra_instance_map = {
        get_instance_key(instance, idx): instance for idx, instance in enumerate(extra_instances)
    }

    merged_instances = []
    for idx, base_instance in enumerate(base_instances):
        key = get_instance_key(base_instance, idx)
        if key not in extra_instance_map:
            raise ValueError(f"{img_name} 中缺少匹配的 nose_shoulder 实例: {key}")
        merged_instances.append(merge_instance(base_instance, extra_instance_map[key]))

    if len(merged_instances) != len(extra_instances):
        raise ValueError(
            f"{img_name} 中实例数量不一致: nose_ear={len(base_instances)}, "
            f"nose_shoulder={len(extra_instances)}"
        )

    merged_frame["instances"] = merged_instances
    return merged_frame


def merge_pose_json(base_data: dict, extra_data: dict, output_json: Path) -> dict:
    merged_data = copy.deepcopy(base_data)
    merged_data["params"] = merge_params(base_data.get("params", {}), extra_data.get("params", {}))
    merged_data["outputs"] = build_outputs(base_data.get("outputs", {}), output_json)

    base_infoes = base_data.get("infoes", {})
    extra_infoes = extra_data.get("infoes", {})

    missing_images = sorted(set(base_infoes) - set(extra_infoes))
    extra_images = sorted(set(extra_infoes) - set(base_infoes))
    if missing_images or extra_images:
        raise ValueError(
            "两个 JSON 的图片集合不一致: "
            f"nose_ear独有={missing_images[:5]}, nose_shoulder独有={extra_images[:5]}"
        )

    merged_infoes = {}
    for img_name, base_frame in base_infoes.items():
        merged_infoes[img_name] = merge_frame(base_frame, extra_infoes[img_name], img_name)

    merged_data["infoes"] = merged_infoes
    return merged_data


def main() -> None:
    for pair in PAIR_CONFIGS:
        print(f"开始合并: {pair['name']}")
        print(f"  nose_ear      = {pair['base_json']}")
        print(f"  nose_shoulder = {pair['extra_json']}")

        base_data = load_json(pair["base_json"])
        extra_data = load_json(pair["extra_json"])
        merged_data = merge_pose_json(base_data, extra_data, pair["output_json"])

        save_json(merged_data, pair["output_json"])
        print(f"  merged json   = {pair['output_json']}")
        print()


if __name__ == "__main__":
    main()
