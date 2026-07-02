import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LoraConfig


def split_dataset(raw_dataset):
    if len(raw_dataset) < 2 or LoraConfig.TRAIN_TEST_SPLIT <= 0:
        return {"train": raw_dataset, "test": None}
    return raw_dataset.train_test_split(
        test_size=LoraConfig.TRAIN_TEST_SPLIT,
        seed=LoraConfig.SEED,
    )


def train():
    set_seed(LoraConfig.SEED)
    print(f"Loading data from {LoraConfig.DATASET_PATH}...")
    raw_dataset = load_dataset(
        "json",
        data_files=LoraConfig.DATASET_PATH,
        split="train",
        cache_dir=LoraConfig.DATASET_CACHE_DIR,
    )
    dataset_split = split_dataset(raw_dataset)
    has_eval_dataset = dataset_split["test"] is not None
    
    bnb_config = None
    if LoraConfig.USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=LoraConfig.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=LoraConfig.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=LoraConfig.USE_NESTED_QUANT
        )
        
    print(f"Loading base model: {LoraConfig.BASE_MODEL_NAME}...")
    # Using 'auto' or passing to specific device map for MPS/CPU stability on Mac
    model = AutoModelForCausalLM.from_pretrained(
        LoraConfig.BASE_MODEL_NAME, 
        quantization_config=bnb_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(LoraConfig.BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if LoraConfig.USE_4BIT:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=LoraConfig.GRADIENT_CHECKPOINTING)
    if LoraConfig.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    peft_config = PeftLoraConfig(
        r=LoraConfig.LORA_R, 
        lora_alpha=LoraConfig.LORA_ALPHA, 
        target_modules=LoraConfig.TARGET_MODULES, 
        lora_dropout=LoraConfig.LORA_DROPOUT, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    training_args = SFTConfig(
        output_dir=LoraConfig.OUTPUT_DIR, 
        num_train_epochs=LoraConfig.NUM_TRAIN_EPOCHS, 
        per_device_train_batch_size=LoraConfig.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=LoraConfig.PER_DEVICE_EVAL_BATCH_SIZE, 
        gradient_accumulation_steps=LoraConfig.GRADIENT_ACCUMULATION_STEPS,
        optim=LoraConfig.OPTIMIZER, 
        learning_rate=LoraConfig.LEARNING_RATE, 
        weight_decay=LoraConfig.WEIGHT_DECAY, 
        fp16=LoraConfig.FP16, 
        bf16=LoraConfig.BF16,
        max_grad_norm=LoraConfig.MAX_GRAD_NORM, 
        warmup_ratio=LoraConfig.WARMUP_RATIO, 
        lr_scheduler_type=LoraConfig.LR_SCHEDULER_TYPE, 
        logging_steps=10,
        eval_strategy="epoch" if has_eval_dataset else "no",
        save_strategy="epoch", 
        save_total_limit=2, 
        load_best_model_at_end=has_eval_dataset, 
        report_to="none", 
        gradient_checkpointing=LoraConfig.GRADIENT_CHECKPOINTING,
        max_length=LoraConfig.MAX_SEQ_LENGTH,
        seed=LoraConfig.SEED,
        data_seed=LoraConfig.SEED,
        dataset_kwargs={"skip_prepare_dataset": False},
    )
    
    trainer = SFTTrainer(
        model=model, 
        train_dataset=dataset_split["train"], 
        eval_dataset=dataset_split["test"], 
        processing_class=tokenizer,
        args=training_args
    )
    
    print("Starting training session matrix...")
    trainer.train()
    
    print(f"Saving fine-tuned adapter configuration to {LoraConfig.OUTPUT_DIR}...")
    trainer.model.save_pretrained(LoraConfig.OUTPUT_DIR)
    tokenizer.save_pretrained(LoraConfig.OUTPUT_DIR)
    print("Fine-tuning session successfully completed.")

if __name__ == "__main__":
    train()
