"""Minimal CLI for prefix cache simulation."""

import argparse
from typing import Any, Dict, List

from prefix_cache_sim import Qwen2Tokenizer
from prefix_cache_sim.radix_tree import RadixPrefixCache
from prefix_cache_sim.simulator import CacheSimulator, print_simulation_summary


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
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input jsonl file with conversations (default: use built-in test)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-session breakdown",
    )

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = Qwen2Tokenizer(model_name=args.model_name)
    print(f"BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}")

    cache = RadixPrefixCache(
        bos_token_id=tokenizer.bos_token_id,
        max_size=args.cache_max_size,
        eviction_policy=args.eviction_policy,
    )

    # Create simulator
    simulator = CacheSimulator(tokenizer=tokenizer, cache=cache)

    if args.input:
        # Process jsonl file
        print("=" * 70)
        print(f"Radix Tree Prefix Cache - Processing {args.input}")
        print("=" * 70)

        result = simulator.process_jsonl(args.input)

        print(f"\nTotal sessions: {len(set(r.session_id for r in result.requests))}")
        print(f"Total requests: {len(result.requests)}")

        print(f"\nCache: max_size={args.cache_max_size}, policy={args.eviction_policy}")

        # Print per-session summary
        print("\n" + "-" * 70)
        print("Per-Session Summary")
        print("-" * 70)

        session_ids = sorted(set(r.session_id for r in result.requests))
        for sid in session_ids:
            session_requests = [r for r in result.requests if r.session_id == sid]
            print(f"\nSession {sid + 1} ({len(session_requests)} requests):")

            # First 2 requests
            for r in session_requests[:2]:
                print(f"  Req {r.request_id:3d}: hit={r.hit_tokens:5d}, miss={r.miss_tokens:4d}")

            if len(session_requests) > 4:
                print(f"  ... ({len(session_requests) - 4} requests)")

            # Last 2 requests
            for r in session_requests[-2:]:
                print(f"  Req {r.request_id:3d}: hit={r.hit_tokens:5d}, miss={r.miss_tokens:4d}")
    else:
        # Simple test session
        print("=" * 60)
        print("Radix Tree Prefix Cache - Single Session Test")
        print("=" * 60)

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

        print(f"\nCache: max_size={args.cache_max_size}, policy={args.eviction_policy}")
        print("\n" + "-" * 60)
        print("Incremental Requests (Request N = Messages[0:N])")
        print("-" * 60)

        result = simulator.process_sessions([session])

        for r in result.requests:
            print(f"\nRequest {r.request_id} ({r.num_messages} msgs, {r.total_tokens} tokens):")
            print(f"  Hit:   {r.hit_tokens:3d} tokens")
            print(f"  Miss:  {r.miss_tokens:3d} tokens")
            print(f"  Exact: {r.exact_hit}")

    # Print global summary
    print_simulation_summary(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
