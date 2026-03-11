import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

from prefix_cache_sim import chatml_messages_to_prompt
from prefix_cache_sim.radix_tree import RadixPrefixCache
from prefix_cache_sim.tokenizer import Qwen2Tokenizer


def load_chatml_from_toolmind(jsonl_file: str, max_samples: int = None):
    sessions = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            if not line.strip():
                continue

            try:
                data = json.loads(line)

                if "conversations" in data:
                    conversations = data["conversations"]
                    sessions.append(conversations)
                elif "messages" in data:
                    messages = data["messages"]
                    sessions.append(messages)

                if (i + 1) % 100 == 0:
                    print(f"  Loaded {i + 1} sessions")

            except json.JSONDecodeError as e:
                print(f"  Error parsing line {i}: {e}")
                continue

    print(f"  Loaded {len(sessions)} total sessions")
    return sessions


def extract_user_requests_with_history(session):
    requests = []
    history = []

    for msg in session:
        role = msg.get("role", "unknown")

        if role == "user":
            requests.append({"history": history.copy(), "current_message": msg})
        elif role == "assistant":
            history.append(msg)
        elif role == "system":
            history.append(msg)

    return requests


def run_cache_experiment(
    sessions, tokenizer, cache_warmup=100, cache_max_size=100000, eviction_policy="lru"
):
    cache = RadixPrefixCache(
        bos_token_id=tokenizer.bos_token_id,
        max_size=cache_max_size,
        eviction_policy=eviction_policy,
    )

    print(f"\n{'=' * 60}")
    print("CACHE EXPERIMENT")
    print(f"{'=' * 60}")
    print(f"Total sessions: {len(sessions)}")
    print(f"Cache warmup requests: {cache_warmup}")
    print(f"Cache max size: {cache_max_size}")
    print(f"Eviction policy: {eviction_policy}")
    print()

    print("Extracting user requests with history...")
    all_requests = []
    for session in sessions:
        requests = extract_user_requests_with_history(session)
        all_requests.extend(requests)

    print(f"Total user requests extracted: {len(all_requests)}")
    print()

    print("Warming up cache...")
    warmup_count = 0
    for i, req in enumerate(all_requests[:cache_warmup]):
        full_context = req["history"] + [req["current_message"]]
        tokens = chatml_messages_to_prompt(full_context, tokenizer)
        cache.add(tokens)
        warmup_count += 1
        if warmup_count % 20 == 0:
            print(f"  Warmed up {warmup_count}/{cache_warmup} requests")

    print(f"Cache warmed up. Tree nodes: {cache.total_nodes}")
    print()

    print("Processing test requests...")
    test_requests = all_requests[cache_warmup:]
    results = []

    for i, req in enumerate(test_requests):
        full_context = req["history"] + [req["current_message"]]
        tokens = chatml_messages_to_prompt(full_context, tokenizer)
        result = cache.query(tokens)
        result["request_id"] = i + cache_warmup + 1
        results.append(result)
        cache.add(tokens)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_requests)} requests")

    print(f"Processed {len(test_requests)} test requests")
    print()

    print(f"{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    stats = cache.get_statistics()
    print(f"Total sequences processed: {stats['total_sequences']}")
    print(f"Exact hits: {stats['exact_hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Exact hit rate: {stats['exact_hit_rate']}")
    print(f"Token-level hit rate: {stats['token_hit_rate']}")
    print(f"Total tokens processed: {stats['total_tokens_processed']}")
    print(f"Prefix hit tokens: {stats['prefix_hits_tokens']}")
    print(f"Cache tree nodes: {stats['cache_nodes']}")
    print(f"Current cache size: {stats['current_size']}")
    print(f"Max cache size: {stats['max_size']}")
    print(f"Eviction count: {stats['eviction_count']}")
    print(f"Total tokens written: {stats['total_tokens_written']}")
    print(f"Total tokens evicted: {stats['total_tokens_evicted']}")
    print(f"Total write volume: {stats['total_write_volume']}")

    hit_rates = [r["prefix_hit_len"] / r["total"] * 100 for r in results if r["total"] > 0]
    prefix_hit_lengths = [r["prefix_hit_len"] for r in results]

    if hit_rates:
        print(f"\nAverage prefix hit rate: {sum(hit_rates) / len(hit_rates):.2f}%")
        print(f"MinMin prefix hit rate: {min(hit_rates):.2f}%")
        print(f"Max prefix hit: {max(hit_rates):.2f}%")

    if prefix_hit_lengths:
        import statistics

        avg_prefix_len = sum(prefix_hit_lengths) / len(prefix_hit_lengths)
        min_prefix_len = min(prefix_hit_lengths)
        max_prefix_len = max(prefix_hit_lengths)
        median_prefix_len = statistics.median(prefix_hit_lengths)

        print(f"\nPrefix Hit Length Statistics:")
        print(f"  Average: {avg_prefix_len:.2f} tokens")
        print(f"  Min: {min_prefix_len} tokens")
        print(f"  Max: {max_prefix_len} tokens")
        print(f"  Median: {median_prefix_len} tokens")
        print(f"  Std Dev: {statistics.stdev(prefix_hit_lengths):.2f} tokens")

    return stats, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run cache experiment on ToolMind open-wiki-traj dataset with user request granularity"
    )
    parser.add_argument(
        "--input-file",
        default="data/open-wiki-traj.jsonl",
        help="Input JSONL file (default: data/open-wiki-traj.jsonl)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process (default: 1000)",
    )
    parser.add_argument(
        "--cache-warmup",
        type=int,
        default=100,
        help="Number of user requests for cache warmup (default: 100)",
    )
    parser.add_argument(
        "--cache-max-size",
        type=int,
        default=100000,
        help="Maximum cache size in tokens (default: 100000)",
    )
    parser.add_argument(
        "--eviction-policy",
        default="lru",
        choices=["lru", "fifo"],
        help="Cache eviction policy (default: lru)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ToolMind open-wiki-traj Cache Experiment (User Request Granularity)")
    print("=" * 60)

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"\nError: File not found: {input_file}")
        print(f"Download dataset first:")
        print(
            f"  KMP_DUPLICATE_LIB_OK=TRUE uv run python download_toolmind.py --files {input_file.name}"
        )
        sys.exit(1)

    print(f"\n1. Loading ChatML format from {input_file}...")
    sessions = load_chatml_from_toolmind(str(input_file), max_samples=args.max_samples)

    print(f"\n2. Loading Qwen2 tokenizer...")
    tokenizer = Qwen2Tokenizer()
    print(f"   Tokenizer loaded. BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}")

    print(f"\n3. Running cache experiment...")
    stats, results = run_cache_experiment(
        sessions,
        tokenizer,
        cache_warmup=args.cache_warmup,
        cache_max_size=args.cache_max_size,
        eviction_policy=args.eviction_policy,
    )

    output_file = "data/open_wiki_traj_user_request_cache_results.json"
    output = {
        "config": {
            "dataset": "ToolMind-Web-QA/open-wiki-traj",
            "source_file": str(input_file),
            "num_sessions": len(sessions),
            "cache_warmup_requests": args.cache_warmup,
            "cache_max_size": args.cache_max_size,
            "eviction_policy": args.eviction_policy,
            "granularity": "user_request_with_history",
        },
        "stats": stats,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n4. Results saved to {output_file}")

    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("=" * 60)
