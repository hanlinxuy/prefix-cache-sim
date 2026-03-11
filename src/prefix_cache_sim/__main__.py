"""Minimal CLI for prefix cache simulation."""

import argparse
import json
from typing import Any, Dict, List

from prefix_cache_sim import Qwen2Tokenizer, chatml_messages_to_prompt
from prefix_cache_sim.radix_tree import RadixPrefixCache


def simulate_single_session_incremental(
    session: List[Dict[str, Any]],
    tokenizer: Qwen2Tokenizer,
    cache: RadixPrefixCache,
) -> List[Dict[str, Any]]:
    """Simulate incremental requests for a single session.

    Request 1: [msg1]
    Request 2: [msg1, msg2]
    Request 3: [msg1, msg2, msg3]
    ...
    """
    results = []

    for i in range(1, len(session) + 1):
        messages = session[:i]
        tokens = chatml_messages_to_prompt(messages, tokenizer)
        result = cache.query(tokens)

        results.append(
            {
                "request_id": i,
                "num_messages": len(messages),
                "total_tokens": len(tokens),
                "hit_tokens": result["prefix_hit_len"],
                "miss_tokens": result["miss_len"],
                "exact_hit": result["exact_hit"],
            }
        )

        # Add to cache for next request
        cache.add(tokens)

    return results


def main():
    parser = argparse.ArgumentParser(description="Radix Tree Prefix Cache Simulation")
    parser.add_argument(
        "--cache-max-size",
        type=int,
        default=None,
        help="Maximum cache size in tokens (default: unlimited)",
    )
    parser.add_argument(
        "--eviction-policy",
        type=str,
        default="lru",
        choices=["lru", "fifo"],
        help="Cache eviction policy (default: lru)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Tokenizer model name (default: Qwen/Qwen2-0.5B)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Radix Tree Prefix Cache - Single Session Test")
    print("=" * 60)

    # Simple test session
    session = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well!"},
    ]

    print(f"\nSession: {len(session)} messages")
    for i, msg in enumerate(session):
        print(f"  [{i}] {msg['role']:10s}: {msg['content']}")

    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = Qwen2Tokenizer(model_name=args.model_name)
    print(f"BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}")

    cache = RadixPrefixCache(
        bos_token_id=tokenizer.bos_token_id,
        max_size=args.cache_max_size,
        eviction_policy=args.eviction_policy,
    )

    print(f"\nCache: max_size={args.cache_max_size}, policy={args.eviction_policy}")
    print("\n" + "-" * 60)
    print("Incremental Requests (Request N = Messages[0:N])")
    print("-" * 60)

    results = simulate_single_session_incremental(session, tokenizer, cache)

    for r in results:
        print(
            f"\nRequest {r['request_id']} ({r['num_messages']} msgs, {r['total_tokens']} tokens):"
        )
        print(f"  Hit:   {r['hit_tokens']:3d} tokens")
        print(f"  Miss:  {r['miss_tokens']:3d} tokens")
        print(f"  Exact: {r['exact_hit']}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    stats = cache.get_statistics()
    print(f"Total requests: {len(results)}")
    print(f"Total tokens processed: {stats['total_tokens_processed']}")
    print(f"Prefix hit tokens: {stats['prefix_hits_tokens']}")
    print(f"Cache nodes created: {stats['cache_nodes']}")
    print(f"Cache size: {stats['current_size']}")

    if stats["max_size"]:
        print(f"Cache utilization: {stats['current_size'] / stats['max_size'] * 100:.1f}%")

    print(f"Token-level hit rate: {stats['token_hit_rate']}")


if __name__ == "__main__":
    main()
