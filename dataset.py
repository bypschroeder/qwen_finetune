import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset

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
