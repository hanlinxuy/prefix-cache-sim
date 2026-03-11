"""Test core property: cache miss equals current user message tokens.

In a sequential chat session, when processing turn N:
- Query = full history up to current user message
- Cache contains all previous turns (user + assistant)
- Therefore: miss should equal exactly the current user message's tokens

This holds regardless of whether the user message is:
- Real user input
- Tool/observation results (also marked as user role in tool-use scenarios)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prefix_cache_sim import Qwen2Tokenizer, chatml_messages_to_prompt
from prefix_cache_sim.radix_tree import RadixPrefixCache


def test_miss_equals_user_tokens():
    """Verify cache miss equals current user message token count."""
    # Load test data
    data_path = Path(__file__).parent.parent / "data" / "open_wiki_traj_test_5.jsonl"

    if not data_path.exists():
        # Skip if data not available
        return True

    with open(data_path, "r") as f:
        data = json.loads(f.readline())
        messages = data["conversations"]

    tokenizer = Qwen2Tokenizer()
    cache = RadixPrefixCache(bos_token_id=tokenizer.bos_token_id)

    passed = 0
    failed = 0

    for i in range(len(messages)):
        role = messages[i]["role"]

        # Build history up to current message
        history = messages[: i + 1]
        tokens = chatml_messages_to_prompt(history, tokenizer)

        # Query cache
        result = cache.query(tokens)
        miss_tokens = result["miss_len"]

        if role == "user":
            # Calculate this message's token count
            if i == 0:
                expected_miss = len(tokens)
            else:
                prev_history = messages[:i]
                prev_tokens = chatml_messages_to_prompt(prev_history, tokenizer)
                expected_miss = len(tokens) - len(prev_tokens)

            if miss_tokens == expected_miss:
                passed += 1
            else:
                failed += 1
                print(f"FAIL Turn {i}: expected {expected_miss}, got {miss_tokens}")

        # Add to cache
        cache.add(tokens)

    total_user_turns = passed + failed
    if total_user_turns == 0:
        return True

    print(f"Test miss_equals_user_tokens: {passed}/{total_user_turns} passed")
    return failed == 0


if __name__ == "__main__":
    success = test_miss_equals_user_tokens()
    sys.exit(0 if success else 1)
