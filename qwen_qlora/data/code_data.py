import logging
from datasets import load_dataset
from ..config import MAX_CODE, SEED
from . import make_chat

logger = logging.getLogger(__name__)

def format_code(example, tokenizer):
    instruction = example["instruction"].strip()
    inp = example.get("input", "").strip()
    output = example.get("output", "").strip()

    user_msg = f"{instruction}\n\n{inp}".strip() if inp else instruction
    cot_code = f"I'll solve this step by step.\n\n**Approach:** Let me think about what's needed here.\n\n{output}"
    
    return {"text": make_chat(tokenizer, user_msg, cot_code)}

def load_code_dataset(tokenizer):
    logger.info("Loading code dataset...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split=f"train[:{MAX_CODE}]")
    except Exception as e:
        logger.warning(f"CodeAlpaca failed ({e}), falling back to Python code instructions")
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split=f"train[:{MAX_CODE}]")
        
    ds = ds.map(lambda x: format_code(x, tokenizer), remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["text"]) > 50)
    
    logger.info(f"Loaded {len(ds)} code examples")
    return ds
