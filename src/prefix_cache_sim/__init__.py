"""SGLang-style Radix Tree Prefix Cache Simulation."""

from prefix_cache_sim.data_loader import load_chatml_jsonl, load_locomo_dataset
from prefix_cache_sim.radix_tree import RadixNode, RadixPrefixCache
from prefix_cache_sim.simulation import simulate_radix_cache
from prefix_cache_sim.tokenizer import Qwen2Tokenizer
from prefix_cache_sim.tool_parser import (
    ToolCall,
    convert_chatml_with_tools,
    format_tool_call_for_qwen,
    format_tool_response_for_qwen,
    parse_tool_calls,
)
from prefix_cache_sim.utils import chatml_messages_to_prompt, generate_fake_chatml_data

__version__ = "0.1.0"

__all__ = [
    "RadixNode",
    "RadixPrefixCache",
    "Qwen2Tokenizer",
    "load_locomo_dataset",
    "load_chatml_jsonl",
    "chatml_messages_to_prompt",
    "generate_fake_chatml_data",
    "simulate_radix_cache",
    "ToolCall",
    "convert_chatml_with_tools",
    "format_tool_call_for_qwen",
    "format_tool_response_for_qwen",
    "parse_tool_calls",
]
