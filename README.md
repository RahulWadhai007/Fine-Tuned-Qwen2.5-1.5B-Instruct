# Qwen2.5-1.5B QLoRA Fine-Tuning
## Task: Math Reasoning + Chain-of-Thought + Code Generation
### Hardware: Google Colab T4 GPU (15GB VRAM)

---

## What this project does

Fine-tunes `Qwen2.5-1.5B-Instruct` using QLoRA (4-bit quantization + LoRA adapters)
on a combined dataset of math problems, reasoning tasks, and code generation.
The model learns to think step-by-step using Chain-of-Thought (CoT) formatting.

---

## Setup (do this once)

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install datasets trl transformers accelerate bitsandbytes evaluate
```

---

## Run training

```bash
python train.py
```

Expected time: **~2 hours** on a Google Colab T4 GPU.

---

## Datasets used

| Pillar | Dataset | HuggingFace ID | Examples used |
|--------|---------|----------------|---------------|
| Math | GSM8K | `openai/gsm8k` | 3,000 |
| Reasoning | OpenHermes-2.5 | `teknium/OpenHermes-2.5` | 2,000 |
| Code | CodeAlpaca-20k | `sahil2801/CodeAlpaca-20k` | 2,000 |
| **Total** | | | **~7,000** |

---

## If you get CUDA Out of Memory

Edit these values in `qwen_qlora/config.py`:

```python
MAX_SEQ_LEN = 384    # reduce from 1024
BATCH_SIZE  = 1      
```

---

## Output files

After training completes:

```
Fine tune/
  lora-adapter/               ← LoRA weights (~34 MB) — upload this to HuggingFace
    adapter_config.json
    adapter_model.safetensors
    tokenizer files...
```

---

## How it works (for your portfolio explanation)

```
Base model (frozen, 4-bit)
    + LoRA adapters (trainable, ~0.5% of params)
    + Chain-of-Thought formatted data
    + Gradient offloading to RAM (use_gradient_checkpointing="unsloth")
    = Fine-tuned model that reasons step-by-step
```

### Key techniques used:
- **QLoRA**: 4-bit NF4 quantization of base weights via bitsandbytes
- **LoRA rank r=8**: Only ~12M of 1.5B parameters are trained
- **Gradient checkpointing**: Offloads activations from VRAM to RAM
- **8-bit Adam optimizer**: Saves ~1GB vs standard Adam
- **Chain-of-Thought format**: Every answer includes step-by-step reasoning

---

## Evaluation (portfolio metric)

After training, run the evaluation block at the bottom of the script.
Report accuracy on GSM8K test set (1,319 math problems):

- Base Qwen2.5-1.5B:  ~35–45% accuracy
- After fine-tuning:   ~50–65% accuracy (expected improvement)

This gives you a **concrete before/after number** to show in your portfolio.

---

## Load and use the fine-tuned model later

You can interact with your newly trained model using the provided chat script:

```bash
python chat.py
```

Or you can use it in your own Python code like this:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "./lora-adapter"
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system",    "content": "Solve step by step."},
    {"role": "user",      "content": "What is 23% of 450?"},
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to("cuda")

outputs = model.generate(inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```
