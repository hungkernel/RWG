import os
import torch
# Model path - can be either:
# 1. Hugging Face model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct") - will auto-download
# 2. Local path to downloaded model
# You can override this with environment variable: export RWG_MODEL_PATH="your/path"
MODEL_CONFIG = {
    "model_path": os.getenv(
        "RWG_MODEL_PATH",
        # "meta-llama/Llama-3.2-3B-Instruct"  # Use HF model ID for auto-download
        # "Qwen/Qwen2.5-3B"  # Use HF model ID for auto-download
        # "meta-llama/Llama-3.2-1B"
        "Qwen/Qwen2-1.5B"
    ),
    "dtype": torch.float16,
    "device_map": "auto",
    "temperature": 0.7,
    "max_new_tokens": 1400,
    "top_p": 0.9,
    "do_sample": True
}

AGENT_CONFIG = {
    "default_token_budget": 8000,
    "context_capacity": 32000,
    "cost_per_1k_tokens": 0.0,
    "alpha": 1.0,
    "beta": 1.0,
    "epsilon": 0.05,
    "confidence_threshold": 0.6,
    "gen_rate_hint": 180,
    "toolset": ["search", "summarize"]
}

NUM_AGENTS = 5
MAX_ROUNDS = 3

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# Number of possible actions for agents
NUM_ACTIONS = 5  # PLAN, SEARCH, WRITE, REVIEW, WAIT







