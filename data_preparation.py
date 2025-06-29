import json
import os
import glob
import random
import shutil

OUTPUT_DIR = "Data"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")


def split_dataset(dataset_dir, train_ratio=0.8):
    prompt_files = [f for f in os.listdir(dataset_dir) if f.endswith("_prompt.json")]
    scene_prefixes = [f.replace("_prompt.json", "") for f in prompt_files]
    random.shuffle(scene_prefixes)
    split_idx = int(len(scene_prefixes) * train_ratio)
    train_scenes = scene_prefixes[:split_idx]
    val_scenes = scene_prefixes[split_idx:]
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    for scene in train_scenes:
        prompt_file = f"{scene}_prompt.json"
        shutil.move(
            os.path.join(dataset_dir, prompt_file), os.path.join(TRAIN_DIR, prompt_file)
        )
        images = [
            f
            for f in os.listdir(dataset_dir)
            if f.startswith(scene + "_") and f.endswith(".png")
        ]
        for img in images:
            shutil.move(os.path.join(dataset_dir, img), os.path.join(TRAIN_DIR, img))
    for scene in val_scenes:
        prompt_file = f"{scene}_prompt.json"
        shutil.move(
            os.path.join(dataset_dir, prompt_file), os.path.join(VAL_DIR, prompt_file)
        )
        images = [
            f
            for f in os.listdir(dataset_dir)
            if f.startswith(scene + "_") and f.endswith(".png")
        ]
        for img in images:
            shutil.move(os.path.join(dataset_dir, img), os.path.join(VAL_DIR, img))
    print(
        f"Split {len(scene_prefixes)} scenes into {len(train_scenes)} train and {len(val_scenes)} val scenes."
    )


def create_annotations(split_dir):
    output_jsonl = os.path.join(split_dir, "annotations.jsonl")
    prompt_files = sorted(glob.glob(os.path.join(split_dir, "*_prompt.json")))
    annotations = []
    for prompt_file in prompt_files:
        prefix = os.path.basename(prompt_file).replace("_prompt.json", "")
        with open(prompt_file, "r") as f:
            prompt_data = json.load(f)
        prompt_text = prompt_data["prompts"]["all"]
        images = sorted(glob.glob(os.path.join(split_dir, f"{prefix}_*.png")))
        for img_path in images:
            img_name = os.path.basename(img_path)
            annotations.append(
                {
                    "image": img_name,
                    "prefix": "Describe this image.",
                    "suffix": prompt_text,
                }
            )
    with open(output_jsonl, "w") as out_f:
        for ann in annotations:
            out_f.write(json.dumps(ann) + "\n")


split_dataset("output", train_ratio=0.8)

create_annotations(TRAIN_DIR)
create_annotations(VAL_DIR)
