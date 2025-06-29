import os
import json
from PIL import Image
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor



SYSTEM_MESSAGE = "You are a helpful assistant that is describing scenes in images."


def format_data(image_directory_path, entry):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_directory_path + "/" + entry["image"],
                },
                {
                    "type": "text",
                    "text": entry["prefix"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": entry["suffix"]}],
        },
    ]


class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self):
        entries = []
        with open(self.jsonl_file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry["image"])
        image = Image.open(image_path)
        return image, entry, format_data(self.image_directory_path, entry)
    
class TrainCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        _, _, examples = zip(*batch)
        texts = [self.processor.apply_chat_template(example, tokenize=False) for example in examples]
        image_inputs = [process_vision_info(example)[0] for example in examples]
        model_inputs = self.processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )
        labels = model_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        if isinstance(self.processor, Qwen2_5_VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]

        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        return (
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["pixel_values"],
            model_inputs["image_grid_thw"],
            labels,
        )
    
class EvalCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        _, data, examples = zip(*batch)
        suffixes = [d["suffix"] for d in data]
        examples = [e[:2] for e in examples]
        texts = [self.processor.apply_chat_template(example, tokenize=False) for example in examples]
        image_inputs = [process_vision_info(example)[0] for example in examples]
        model_inputs = self.processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )
        return (
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["pixel_values"],
            model_inputs["image_grid_thw"],
            suffixes,
        )