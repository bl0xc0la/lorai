import torch
from pathlib import Path

class LoraConfig:
    # 0. Project Paths
    PROJECT_ROOT = Path(__file__).resolve().parent

    # 1. Optimized Base Model for 16GB Mac Hardware
    BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 2. Directory Paths
    OUTPUT_DIR = str(PROJECT_ROOT / "models" / "lora_adapter")
    MERGED_MODEL_DIR = str(PROJECT_ROOT / "models" / "lora_merged")
    DATASET_PATH = str(PROJECT_ROOT / "datasets" / "train_data.jsonl")
    DATASET_CACHE_DIR = str(PROJECT_ROOT / ".cache" / "datasets")
    TRAIN_TEST_SPLIT = 0.1
    
    # 3. LoRA Target Architecture
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 4. Apple Silicon (MPS) Compatability Settings
    USE_4BIT = False  # Deactivated: bitsandbytes 4-bit is native to CUDA/NVIDIA
    BNB_4BIT_QUANT_TYPE = "nf4"
    USE_NESTED_QUANT = False
    
    # Fallback to float16 or float32 depending on current MPS context
    BNB_4BIT_COMPUTE_DTYPE = torch.float16
    FP16 = False      # Deactivated: Set to False for local Mac training loops
    BF16 = False      # Deactivated: Set to False to avoid MPS compilation discrepancies
    
    # 5. Hyperparameters Tuning Loops
    NUM_TRAIN_EPOCHS = 3
    PER_DEVICE_TRAIN_BATCH_SIZE = 1  # Kept light to prevent memory pressure spikes
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 4  # Simulates a higher batch size cleanly
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 0.3
    WARMUP_RATIO = 0.03
    LR_SCHEDULER_TYPE = "cosine"
    OPTIMIZER = "adamw_torch"       # Swapped away from paged_adamw_32bit (CUDA only)
    MAX_SEQ_LENGTH = 1024            # Balanced context length for memory health
    GRADIENT_CHECKPOINTING = True
    SEED = 42
    
    # 6. Core Server Engine Network Matrix
    API_HOST = "0.0.0.0"
    API_PORT = 8000
