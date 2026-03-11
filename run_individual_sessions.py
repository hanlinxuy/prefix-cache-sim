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


def run_single_session_experiment(
    session, session_id, tokenizer, cache_warmup=0, cache_max_size=100000, eviction_policy="lru"
):
    cache = RadixPrefixCache(
        bos_token_id=tokenizer.bos_token_id,
        max_size=cache_max_size,
        eviction_policy=eviction_policy,
    )

    print(f"\n{'=' * 60}")
    print(f"SESSION {session_id} ANALYSIS")
    print(f"{'=' * 60}")

    requests = extract_user_requests_with_history(session)
    print(f"Total user requests in session: {len(requests)}")

    if cache_warmup > 0:
        print(f"\nWarming up cache with first {cache_warmup} requests...")
        for i, req in enumerate(requests[:cache_warmup]):
            full_context = req["history"] + [req["current_message"]]
            tokens = chatml_messages_to_prompt(full_context, tokenizer)
            cache.add(tokens)
        print(f"Cache warmed up. Tree nodes: {cache.total_nodes}")

    test_requests = requests[cache_warmup:]
    print(f"\nProcessing {len(test_requests)} test requests...")

    results = []
    for i, req in enumerate(test_requests):
        full_context = req["history"] + [req["current_message"]]
        tokens = chatml_messages_to_prompt(full_context, tokenizer)
        result = cache.query(tokens)
        result["request_id"] = i + 1
        result["history_length"] = len(req["history"])
        result["total_tokens"] = len(tokens)
        results.append(result)
        cache.add(tokens)

    print(f"Processed {len(test_requests)} test requests")
    print()

    print(f"{'=' * 60}")
    print("SESSION RESULTS")
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

    hit_rates = [r["prefix_hit_len"] / r["total"] * 100 for r in results if r["total"] > 0]
    prefix_hit_lengths = [r["prefix_hit_len"] for r in results]

    if hit_rates:
        import statistics

        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        min_hit_rate = min(hit_rates)
        max_hit_rate = max(hit_rates)

        print(f"\nHit Rate Statistics:")
        print(f"  Average: {avg_hit_rate:.2f}%")
        print(f"  Min: {min_hit_rate:.2f}%")
        print(f"  Max: {max_hit_rate:.2f}%")

    if prefix_hit_lengths:
        import statistics

        avg_prefix_len = sum(prefix_hit_lengths) / len(prefix_hit_lengths)
        min_prefix_len = min(prefix_hit_lengths)
        max_prefix_len = max(prefix_hit_lengths)
        median_prefix_len = statistics.median(prefix_hit_lengths)
        std_prefix_len = statistics.stdev(prefix_hit_lengths)

        print(f"\nPrefix Hit Length Statistics:")
        print(f"  Average: {avg_prefix_len:.2f} tokens")
        print(f"  Min: {min_prefix_len} tokens")
        print(f"  Max: {max_prefix_len} tokens")
        print(f"  Median: {median_prefix_len:.2f} tokens")
        print(f"  Std Dev: {std_prefix_len:.2f} tokens")

    print(f"\n{'=' * 60}")

    return stats, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run cache experiment on individual sessions")
    parser.add_argument(
        "--input-file", default="data/open-wiki-traj-subset-1000.jsonl", help="Input JSONL file"
    )
    parser.add_argument(
        "--session-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Session indices to analyze (default: first 5)",
    )
    parser.add_argument(
        "--cache-warmup",
        type=int,
        default=0,
        help="Number of requests for cache warmup per session (default: 0)",
    )
    parser.add_argument(
        "--cache-max-size",
        type=int,
        default=100000,
        help="Maximum cache size in tokens (default: 100000)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Individual Session Cache Experiment")
    print("=" * 60)

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"\nError: File not found: {input_file}")
        sys.exit(1)

    print(f"\nLoading ChatML format from {input_file}...")
    sessions = load_chatml_from_toolmind(str(input_file))

    print(f"\nLoading Qwen2 tokenizer...")
    tokenizer = Qwen2Tokenizer()
    print(f"   Tokenizer loaded. BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}")

    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")

    all_stats = []
    all_results = []

    for idx in args.session_indices:
        if idx >= len(sessions):
            print(f"\n⚠ Session {idx} not found (max: {len(sessions) - 1})")
            continue

        print(f"\n{'#' * 60}")
        print(f"# SESSION {idx + 1}")
        print(f"{'#' * 60}")

        stats, results = run_single_session_experiment(
            sessions[idx],
            idx + 1,
            tokenizer,
            cache_warmup=args.cache_warmup,
            cache_max_size=args.cache_max_size,
        )

        all_stats.append(stats)
        all_results.extend(results)

    print(f"\n{'=' * 60}")
    print("AGGREGATED STATISTICS")
    print(f"{'=' * 60}")

    total_tokens_processed = sum(s["total_tokens_processed"] for s in all_stats)
    total_prefix_hits = sum(s["prefix_hits_tokens"] for s in all_stats)
    total_exact_hits = sum(s["exact_hits"] for s in all_stats)
    total_misses = sum(s["misses"] for s in all_stats)

    overall_token_hit_rate = (
        total_prefix_hits / total_tokens_processed * 100 if total_tokens_processed > 0 else 0
    )
    overall_exact_hit_rate = (
        total_exact_hits / (total_exact_hits + total_misses) * 100
        if (total_exact_hits + total_misses) > 0
        else 0
    )

    print(f"Total sessions analyzed: {len(args.session_indices)}")
    print(f"Total requests processed: {len(all_results)}")
    print(f"Total tokens processed: {total_tokens_processed}")
    print(f"Total prefix hit tokens: {total_prefix_hits}")
    print(f"Overall Token-level hit rate: {overall_token_hit_rate:.2f}%")
    print(f"Overall Exact hit rate: {overall_exact_hit_rate:.2f}%")

    all_hit_rates = [r["prefix_hit_len"] / r["total"] * 100 for r in all_results if r["total"] > 0]
    if all_hit_rates:
        import statistics

        avg_hit_rate = sum(all_hit_rates) / len(all_hit_rates)
        min_hit_rate = min(all_hit_rates)
        max_hit_rate = max(all_hit_rates)
        median_hit_rate = statistics.median(all_hit_rates)

        print(f"\nRequest Hit Rate Distribution:")
        print(f"  Average: {avg_hit_rate:.2f}%")
        print(f"  Median: {median_hit_rate:.2f}%")
        print(f"  Min: {min_hit_rate:.2f}%")
        print(f"  Max: {max_hit_rate:.2f}%")
        print(f"  Std Dev: {statistics.stdev(all_hit_rates):.2f}%")

    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)
