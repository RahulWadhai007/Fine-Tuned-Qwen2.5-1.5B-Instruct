import torch
import warnings
from unsloth import FastLanguageModel
from qwen_qlora.config import MAX_SEQ_LEN, SYSTEM_PROMPT

# Suppress annoying HuggingFace warnings
warnings.filterwarnings("ignore")

print("Loading fine-tuned model... This might take a minute...")
# 1. Load your newly trained LoRA adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./lora-adapter", # Make sure you moved your adapter folder here!
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

# 2. Tell Unsloth we want to do inference (makes it 2x faster!)
FastLanguageModel.for_inference(model)

print("\n" + "="*50)
print("🤖 Qwen 1.5B Reasoning Engine is Online!")
print("Type 'quit' to exit.")
print("="*50 + "\n")

# 3. Start an interactive terminal chat
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "exit"]:
        break
        
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    print("\nThinking...")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=500,
            temperature=0.3, # Low temp for better logical reasoning
            top_p=0.9,
        )
        
    # Decode and print just the AI's response
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"AI: {response}")
