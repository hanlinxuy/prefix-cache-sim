import os
from typing import List, Union

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from transformers import AutoTokenizer, BatchEncoding

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    BatchEncoding = None


class Qwen2Tokenizer:
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_chatml_message(self, role: str, content: str) -> List[int]:
        messages = [{"role": role, "content": content}]
        result = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        if isinstance(result, BatchEncoding):
            return result.get("input_ids", [])
        elif isinstance(result, dict):
            return result.get("input_ids", [])
        return result
