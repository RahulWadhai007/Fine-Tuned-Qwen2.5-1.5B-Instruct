import re
import torch
import logging
from datasets import load_dataset
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_number(text):
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None

def evaluate():
    logger.info("Starting GSM8K evaluation")
    adapter_path = "./qwen-math-code-reasoning/lora-adapter"
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(adapter_path)
        FastLanguageModel.for_inference(model)
    except Exception as e:
        logger.error(f"Failed to load adapter from {adapter_path}: {e}")
        return

    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        logger.error(f"Failed to load GSM8K test set: {e}")
        return

    correct = 0
    sample_size = min(200, len(ds))
    logger.info(f"Evaluating {sample_size} samples...")
    
    for ex in ds.select(range(sample_size)):
        messages = [
            {"role": "system", "content": "Solve step by step."},
            {"role": "user", "content": ex["question"]},
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        outputs = model.generate(inputs, max_new_tokens=256, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
        pred = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        gold = extract_number(ex["answer"].split("####")[-1])
        got = extract_number(pred)
        
        if gold and got and gold == got:
            correct += 1

    accuracy = (correct / sample_size) * 100
    logger.info(f"GSM8K accuracy: {correct}/{sample_size} ({accuracy:.1f}%)")

if __name__ == "__main__":
    evaluate()
