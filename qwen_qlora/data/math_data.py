import logging
from datasets import load_dataset
from ..config import MAX_MATH, SEED
from . import make_chat

logger = logging.getLogger(__name__)

def format_math(example, tokenizer):
    question = example["question"].strip()
    raw_ans = example["answer"].strip()

    if "####" in raw_ans:
        steps, final = raw_ans.split("####", 1)
        steps, final = steps.strip(), final.strip()
    else:
        steps, final = raw_ans, ""

    cot_answer = f"Let me solve this step by step.\n\n{steps}\n\n**Final answer: {final}**"
    return {"text": make_chat(tokenizer, question, cot_answer)}

def load_math_dataset(tokenizer):
    logger.info("Loading GSM8K...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.shuffle(seed=SEED).select(range(min(MAX_MATH, len(ds))))
        ds = ds.map(lambda x: format_math(x, tokenizer), remove_columns=ds.column_names)
        logger.info(f"Loaded {len(ds)} math examples")
        return ds
    except Exception as e:
        logger.error(f"Failed to load GSM8K: {e}")
        raise
