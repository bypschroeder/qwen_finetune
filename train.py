import torch
import lightning as L
from dataset import JSONLDataset
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dataset import TrainCollator, EvalCollator
from model import Qwen2_5_Trainer, SaveCheckpoint

def main():
    train_dataset = JSONLDataset(
        jsonl_file_path=f"Data/train/annotations.jsonl",
        image_directory_path=f"Data/train",
    )
    val_dataset = JSONLDataset(
        jsonl_file_path=f"Data/val/annotations.jsonl",
        image_directory_path=f"Data/val",
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

    train_collate_fn = TrainCollator(processor)
    eval_collate_fn = EvalCollator(processor)

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

    model_module = Qwen2_5_Trainer(config, processor, model, train_dataset, val_dataset, train_collate_fn, eval_collate_fn)
    trainer.fit(model_module, train_dataloaders=model_module.train_dataloader(), val_dataloaders=model_module.val_dataloader())

if __name__ == '__main__':
    main()
