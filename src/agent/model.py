
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.config import MODEL_CONFIG
import os

class WorkerLLM:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        path = MODEL_CONFIG["model_path"]
        # dtype = getattr(torch, MODEL_CONFIG.get("dtype", "float16"))
        dtype = torch.float32
        device_map = MODEL_CONFIG.get("device_map", "auto")

        print(f"Loading model from: {path}")
        
        # Try loading with authentication token if available
        token = os.getenv("HUGGINGFACE_TOKEN", "")
        kwargs = {"token": token} if token else {}

        self.tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=dtype,
            device_map=device_map,
            **kwargs
        )
        print("Model loaded successfully!")


    def generate(self, prompt: str) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded.")
        # 1. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # 2. Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MODEL_CONFIG["max_new_tokens"],
            temperature=MODEL_CONFIG["temperature"],
            top_p=MODEL_CONFIG["top_p"],
            do_sample=MODEL_CONFIG["do_sample"]
        )
        # 3. Decode
        text_output = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return text_output