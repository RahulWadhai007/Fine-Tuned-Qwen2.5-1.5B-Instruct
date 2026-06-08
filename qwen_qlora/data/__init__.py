import logging
from datasets import concatenate_datasets
from ..config import SEED, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def make_chat(tokenizer, user_msg: str, assistant_msg: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

from .math_data import load_math_dataset
from .reasoning_data import load_reasoning_dataset
from .code_data import load_code_dataset

def prepare_datasets(tokenizer):
    logger.info("Loading and formatting datasets...")
    
    math_ds = load_math_dataset(tokenizer)
    reasoning_ds = load_reasoning_dataset(tokenizer)
    code_ds = load_code_dataset(tokenizer)

    # FIXME: might OOM if datasets get too big, need to switch to streaming
    combined = concatenate_datasets([math_ds, reasoning_ds, code_ds]).shuffle(seed=SEED)
    split = combined.train_test_split(test_size=0.1, seed=SEED)

    logger.info(f"Combined dataset: {len(combined)} examples")
    logger.info(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")
    
    return split["train"], split["test"]
