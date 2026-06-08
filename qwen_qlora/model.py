import torch
import logging
from unsloth import FastLanguageModel
from .config import MODEL_ID, MAX_SEQ_LEN, LORA_RANK, LORA_ALPHA, SEED

logger = logging.getLogger(__name__)

def load_model():
    logger.info(f"Loading {MODEL_ID} in 4-bit...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    return model, tokenizer

def attach_lora(model):
    logger.info("Attaching LoRA adapters...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
    
    return model
