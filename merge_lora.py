import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LoraConfig
def merge():
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(LoraConfig.BASE_MODEL_NAME, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(LoraConfig.BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base_model, LoraConfig.OUTPUT_DIR)
    merged_model = model.merge_and_unload()
    os.makedirs(LoraConfig.MERGED_MODEL_DIR, exist_ok=True)
    merged_model.save_pretrained(LoraConfig.MERGED_MODEL_DIR)
    tokenizer.save_pretrained(LoraConfig.MERGED_MODEL_DIR)
if __name__ == "__main__": merge()
