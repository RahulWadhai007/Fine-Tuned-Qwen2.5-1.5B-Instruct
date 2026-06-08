import torch
import logging
from unsloth import FastLanguageModel
from .config import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def run_inference_test(model, tokenizer):
    logger.info("Running quick inference sanity check...")
    FastLanguageModel.for_inference(model)

    prompts = [
        "If a train travels 60 km/h for 2.5 hours, how far does it travel? Show your working.",
        "Write a Python function that checks if a number is prime.",
        "Explain why gradient descent can get stuck in local minima.",
    ]

    for idx, prompt in enumerate(prompts, 1):
        logger.info(f"Test {idx}: {prompt}")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"\n{response}\n{'-'*40}")
