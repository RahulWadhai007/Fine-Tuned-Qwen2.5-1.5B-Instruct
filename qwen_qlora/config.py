import random

# TODO: pull these from a yaml config or CLI args later
MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
OUTPUT_DIR = "./qwen-math-code-reasoning"

MAX_SEQ_LEN = 1024
LORA_RANK = 8
LORA_ALPHA = 8
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2

MAX_MATH = 3000
MAX_REASONING = 2000
MAX_CODE = 2000

SEED = 42
random.seed(SEED)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant that solves problems step by step. "
    "Always show your reasoning clearly before giving the final answer."
)
