import torch
import os
import lightning as L
from dataset import JSONLDataset
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader
from nltk import edit_distance
from torch.optim import AdamW
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

train_dataset = JSONLDataset(
    jsonl_file_path=f"Data/train/annotations.jsonl",
    image_directory_path=f"Data/train",
)
valid_dataset = JSONLDataset(
    jsonl_file_path=f"DATA/val/annotations.jsonl",
    image_directory_path=f"DATA/val",
)


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_QLORA = True
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16,
    )
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config if USE_QLORA else None,
    torch_dtype=torch.bfloat16,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
processor = Qwen2_5_VLProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
)


def train_collate_fn(batch):
    _, _, examples = zip(*batch)
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]
    image_inputs = [process_vision_info(example)[0] for example in examples]
    model_inputs = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )
    labels = model_inputs["input_ids"].clone()
    # mask padding tokens in labels
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if isinstance(processor, Qwen2_5_VLProcessor):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [
            processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        ]
    # mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    image_grid_thw = model_inputs["image_grid_thw"]
    return input_ids, attention_mask, pixel_values, image_grid_thw, labels


def evaluation_collate_fn(batch):
    _, data, examples = zip(*batch)
    suffixes = [d["suffix"] for d in data]
    # drop the assistant portion so the model must generate it
    examples = [e[:2] for e in examples]
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]
    image_inputs = [process_vision_info(example)[0] for example in examples]
    model_inputs = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    image_grid_thw = model_inputs["image_grid_thw"]
    return input_ids, attention_mask, pixel_values, image_grid_thw, suffixes


BATCH_SIZE = 1
NUM_WORKERS = 0
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=train_collate_fn,
    num_workers=NUM_WORKERS,
    shuffle=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=evaluation_collate_fn,
    num_workers=NUM_WORKERS,
)


class Qwen2_5_Trainer(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, image_grid_thw, suffixes = batch
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=1024,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        generated_suffixes = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        scores = []
        for generated_suffix, suffix in zip(generated_suffixes, suffixes):
            score = edit_distance(generated_suffix, suffix)
            score = score / max(len(generated_suffix), len(suffix))
            scores.append(score)
            print("generated_suffix", generated_suffix)
            print("suffix", suffix)
            print("score", score)
        score = sum(scores) / len(scores)
        self.log(
            "val_edit_distance",
            score,
            prog_bar=True,
            logger=True,
            batch_size=self.config.get("batch_size"),
        )
        return scores

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size"),
            collate_fn=train_collate_fn,
            shuffle=True,
            num_workers=10,
        )

    def val_dataloader(self):
        return DataLoader(
            valid_dataset,
            batch_size=self.config.get("batch_size"),
            collate_fn=evaluation_collate_fn,
            num_workers=10,
        )


config = {
    "max_epochs": 10,
    "batch_size": 1,
    "lr": 2e-4,
    "check_val_every_n_epoch": 2,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 8,
    "num_nodes": 1,
    "warmup_steps": 50,
    "result_path": "qwen2.5-3b-instruct-ft",
}


early_stopping_callback = EarlyStopping(
    monitor="val_edit_distance", patience=3, verbose=False, mode="min"
)


class SaveCheckpoint(Callback):
    def __init__(self, result_path):
        self.result_path = result_path
        self.epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        checkpoint_path = f"{self.result_path}/{self.epoch}"
        os.makedirs(checkpoint_path, exist_ok=True)
        pl_module.processor.save_pretrained(checkpoint_path)
        pl_module.model.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        self.epoch += 1

    def on_train_end(self, trainer, pl_module):
        checkpoint_path = f"{self.result_path}/latest"
        os.makedirs(checkpoint_path, exist_ok=True)
        pl_module.processor.save_pretrained(checkpoint_path)
        pl_module.model.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


trainer = L.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=config.get("max_epochs"),
    accumulate_grad_batches=config.get("accumulate_grad_batches"),
    check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
    gradient_clip_val=config.get("gradient_clip_val"),
    limit_val_batches=1,
    num_sanity_val_steps=0,
    log_every_n_steps=10,
    callbacks=[
        SaveCheckpoint(result_path=config["result_path"]),
        early_stopping_callback,
    ],
)

model_module = Qwen2_5_Trainer(config, processor, model)
trainer.fit(model_module)
