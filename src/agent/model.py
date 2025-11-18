
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.config import MODEL_CONFIG

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
        try:
            # Try loading with authentication token if available
            import os
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
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTroubleshooting:")
            print("1. If using Hugging Face model ID, ensure you have internet connection")
            print("2. For Llama models, you may need to:")
            print("   - Request access at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
            print("   - Set HUGGINGFACE_TOKEN environment variable")
            print("   - Or use a local model path in config.py")
            raise

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