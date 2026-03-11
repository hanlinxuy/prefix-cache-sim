"""Minimal utilities for prefix cache simulation."""

from typing import Any, Dict, List

from prefix_cache_sim.tokenizer import Qwen2Tokenizer


def chatml_messages_to_prompt(
    messages: List[Dict[str, Any]], tokenizer: Qwen2Tokenizer
) -> List[int]:
    """Convert ChatML messages to token IDs using the tokenizer's chat template."""
    # Use the tokenizer's apply_chat_template for proper formatting
    result = tokenizer.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    # Handle different return types
    if hasattr(result, "get"):
        return result.get("input_ids", result)
    return result
