import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import LoraConfig
class LoraTerminalChat:
    def __init__(self):
        self.model_path = LoraConfig.MERGED_MODEL_DIR if os.path.exists(LoraConfig.MERGED_MODEL_DIR) else LoraConfig.BASE_MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto", trust_remote_code=True)
        self.history = [{"role": "system", "content": "You are Lora, a brilliant, friendly, and highly capable programming assistant."}]
    def chat_loop(self):
        print("\n=== Chat initialized with Lora! ===\n")
        while True:
            try:
                user_input = input("You: ")
                if user_input.strip().lower() in ["exit", "quit"]: break
                self.history.append({"role": "user", "content": user_input})
                text = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(f"\nLora: {response}\n")
                self.history.append({"role": "assistant", "content": response})
            except KeyboardInterrupt: break
if __name__ == "__main__": LoraTerminalChat().chat_loop()
