import logging
from datasets import load_dataset
from ..config import MAX_REASONING, SEED
from . import make_chat

logger = logging.getLogger(__name__)

def format_reasoning(example, tokenizer):
    convs = example.get("conversations", [])
    user_msg, bot_msg = "", ""
    
    for turn in convs:
        role = turn.get("from", "")
        val = turn.get("value", "").strip()
        if role in ("human", "user") and not user_msg:
            user_msg = val
        elif role in ("gpt", "assistant") and not bot_msg:
            bot_msg = val
            
    if not user_msg or not bot_msg:
        return {"text": ""}
        
    return {"text": make_chat(tokenizer, user_msg, bot_msg)}

def format_alpaca(ex, tokenizer):
    instruction = ex["instruction"].strip()
    inp = ex.get("input", "").strip()
    output = ex.get("output", "").strip()
    
    user_msg = f"{instruction}\n{inp}".strip() if inp else instruction
    return {"text": make_chat(tokenizer, user_msg, output)}

def load_reasoning_dataset(tokenizer):
    logger.info("Loading reasoning dataset...")
    try:
        ds = load_dataset("teknium/OpenHermes-2.5", split=f"train[:{MAX_REASONING}]")
        ds = ds.map(lambda x: format_reasoning(x, tokenizer), remove_columns=ds.column_names)
    except Exception as e:
        logger.warning(f"OpenHermes failed ({e}), falling back to Alpaca-cleaned")
        ds = load_dataset("yahma/alpaca-cleaned", split=f"train[:{MAX_REASONING}]")
        ds = ds.map(lambda x: format_alpaca(x, tokenizer), remove_columns=ds.column_names)
        
    ds = ds.filter(lambda x: len(x["text"]) > 50)
    logger.info(f"Loaded {len(ds)} reasoning examples")
    return ds
