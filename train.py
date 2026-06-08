import logging
from qwen_qlora.model import load_model, attach_lora
from qwen_qlora.data import prepare_datasets
from qwen_qlora.trainer import build_trainer, run_training, save_model
from qwen_qlora.inference import run_inference_test

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing QLoRA fine-tuning pipeline...")

    model, tokenizer = load_model()
    model = attach_lora(model)

    train_data, eval_data = prepare_datasets(tokenizer)

    trainer = build_trainer(model, tokenizer, train_data, eval_data)
    run_training(trainer)

    save_model(model, tokenizer)
    run_inference_test(model, tokenizer)

    logger.info("Pipeline complete. Run `python -m qwen_qlora.evaluate` for GSM8K metrics.")

if __name__ == "__main__":
    main()
