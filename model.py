import os
import lightning as L
from torch.utils.data import DataLoader
from nltk import edit_distance
from torch.optim import AdamW
from lightning.pytorch.callbacks import Callback

class Qwen2_5_Trainer(L.LightningModule):
    def __init__(self, config, processor, model, train_dataset, val_dataset, train_collate_fn, evaluation_collate_fn):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_collate_fn = train_collate_fn
        self.evaluation_collate_fn = evaluation_collate_fn

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
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        generated_suffixes = self.processor.batch_decode(
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
        return AdamW(self.model.parameters(), lr=self.config.get("lr"))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.get("batch_size"),
            collate_fn=self.train_collate_fn,
            shuffle=True,
            num_workers=10,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.get("batch_size"),
            collate_fn=self.evaluation_collate_fn,
            num_workers=10,
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