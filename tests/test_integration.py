import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prefix_cache_sim import load_chatml_jsonl, chatml_messages_to_prompt
from prefix_cache_sim.tokenizer import Qwen2Tokenizer


def test_full_pipeline_with_tools():
    sessions = load_chatml_jsonl("tests/fixtures/dummy_chatml.jsonl")

    assert len(sessions) == 5

    tokenizer = Qwen2Tokenizer()

    for session in sessions:
        tokens = chatml_messages_to_prompt(session, tokenizer)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, int) for token in tokens)

    print(f"✓ Successfully processed {len(sessions)} sessions")
    print(
        f"✓ First session tokens: {len(chatml_messages_to_prompt(sessions[0], tokenizer))} tokens"
    )
    print(
        f"✓ Second session tokens: {len(chatml_messages_to_prompt(sessions[1], tokenizer))} tokens"
    )


def test_tool_call_tokenization():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Beijing"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "15°C"},
        {"role": "assistant", "content": "It's 15°C."},
    ]

    tokenizer = Qwen2Tokenizer()
    tokens = chatml_messages_to_prompt(messages, tokenizer)

    assert len(tokens) > 0
    assert all(isinstance(token, int) for token in tokens)

    print(f"✓ Tool call session tokenized to {len(tokens)} tokens")


if __name__ == "__main__":
    print("Running integration tests...")
    test_full_pipeline_with_tools()
    test_tool_call_tokenization()
    print("\nAll integration tests passed!")
