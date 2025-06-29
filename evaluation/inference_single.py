import torch
import pandas as pd
import os
import glob
import json
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info


MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
processor = Qwen2_5_VLProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS
)
ft_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "qwen2.5-3b-instruct-ft/latest",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
ft_processor = Qwen2_5_VLProcessor.from_pretrained(
    "qwen2.5-3b-instruct-ft/latest",
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS
)

def run_inference(model, processor, conversation, max_new_tokens=1024, device="cuda"):
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids
        in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]

image_path = "./test/001_1.png"

conversation=[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path, "resized_height": 1080, "resized_width": 480},
                {"type": "text", "text": "Where are the objects located in this image?"},
            ],
        }
    ]

output = run_inference(model, processor, conversation)
ft_output= run_inference(ft_model, ft_processor, conversation)
print(f"Original:\n{output}")
print(f"Trained:\n{ft_output}")