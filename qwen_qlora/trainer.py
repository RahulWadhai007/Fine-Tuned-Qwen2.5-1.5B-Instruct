import torch
import logging
from transformers import TrainingArguments
from trl import SFTTrainer
from .config import (
    OUTPUT_DIR, BATCH_SIZE, GRAD_ACCUM, NUM_EPOCHS, LEARNING_RATE,
    MAX_SEQ_LEN, SEED
)

logger = logging.getLogger(__name__)

def build_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="adamw_8bit",
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        seed=SEED,
        report_to="none",
        load_best_model_at_end=False,
    )

def build_trainer(model, tokenizer, train_data, eval_data):
    # TODO: Try Flash Attention 2 if we move off the 4GB VRAM constraint
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=1,
        packing=False,
        args=build_training_args(),
    )

def run_training(trainer):
    logger.info("Starting QLoRA training...")
    
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        used_mb = torch.cuda.memory_reserved(0) / 1024 / 1024
        total_mb = gpu_stats.total_memory / 1024 / 1024
        logger.info(f"GPU: {gpu_stats.name} | VRAM: {used_mb:.0f}MB / {total_mb:.0f}MB")

    try:
        stats = trainer.train()
        runtime_min = stats.metrics.get("train_runtime", 0) / 60
        logger.info(f"Training complete in {runtime_min:.1f} mins. Final loss: {stats.training_loss:.4f}")
        return stats
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def save_model(model, tokenizer):
    logger.info("Saving LoRA adapter...")
    try:
        adapter_path = f"{OUTPUT_DIR}/lora-adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        logger.info(f"Saved to {adapter_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise
