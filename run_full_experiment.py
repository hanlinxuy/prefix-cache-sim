import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

from prefix_cache_sim import load_chatml_jsonl, chatml_messages_to_prompt
from prefix_cache_sim.radix_tree import RadixPrefixCache
from prefix_cache_sim.tokenizer import Qwen2Tokenizer


def convert_toolmind_to_chatml(jsonl_file: str, output_file: str, max_samples: int = None):
    sessions = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            if not line.strip():
                continue

            try:
                data = json.loads(line)

                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant with access to web search tools.",
                    },
                    {"role": "user", "content": data.get("question", "")},
                ]

                sessions.append(messages)

                if (i + 1) % 1000 == 0:
                    print(f"  Converted {i + 1} sessions")

            except json.JSONDecodeError:
                continue

    with open(output_file, "w", encoding="utf-8") as f:
        for session in sessions:
            f.write(json.dumps({"messages": session}, ensure_ascii=False) + "\n")

    print(f"  Converted {len(sessions)} total sessions")
    return sessions


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
    print(f"Cache warmup sessions: {cache_warmup}")
    print(f"Cache max size: {cache_max_size}")
    print(f"Eviction policy: {eviction_policy}")
    print()

    print("Warming up cache...")
    for i, session in enumerate(sessions[:cache_warmup]):
        tokens = chatml_messages_to_prompt(session, tokenizer)
        cache.add(tokens)
        if (i + 1) % 20 == 0:
            print(f"  Warmed up {i + 1}/{cache_warmup} sessions")

    print(f"Cache warmed up. Tree nodes: {cache.total_nodes}")
    print()

    print("Processing test sessions...")
    test_sessions = sessions[cache_warmup:]
    results = []

    for i, session in enumerate(test_sessions):
        tokens = chatml_messages_to_prompt(session, tokenizer)
        result = cache.query(tokens)
        result["session_id"] = i + cache_warmup + 1
        results.append(result)
        cache.add(tokens)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_sessions)} sessions")

    print(f"Processed {len(test_sessions)} test sessions")
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
    if hit_rates:
        print(f"\nAverage prefix hit rate: {sum(hit_rates) / len(hit_rates):.2f}%")
        print(f"Min prefix hit rate: {min(hit_rates):.2f}%")
        print(f"Max prefix hit: {max(hit_rates):.2f}%")

    return stats, results


def process_dataset(
    dataset_name, jsonl_file, max_samples=None, cache_warmup=100, cache_max_size=100000
):
    print(f"\n{'#' * 60}")
    print(f"# Processing Dataset: {dataset_name}")
    print(f"{'#' * 60}")

    chatml_file = f"data/{dataset_name}_chatml.jsonl"

    print(f"\n1. Converting {jsonl_file} to ChatML format...")
    sessions = convert_toolmind_to_chatml(jsonl_file, chatml_file, max_samples=max_samples)

    print(f"\n2. Loading Qwen2 tokenizer...")
    tokenizer = Qwen2Tokenizer()
    print(f"   Tokenizer loaded. BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}")

    print(f"\n3. Running cache experiment...")
    stats, results = run_cache_experiment(
        sessions, tokenizer, cache_warmup=cache_warmup, cache_max_size=cache_max_size
    )

    output_file = f"data/{dataset_name}_cache_results.json"
    output = {
        "config": {
            "dataset": dataset_name,
            "source_file": jsonl_file,
            "num_sessions": len(sessions),
            "cache_warmup_sessions": cache_warmup,
            "cache_max_size": cache_max_size,
            "eviction_policy": "lru",
        },
        "stats": stats,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n4. Results saved to {output_file}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full cache experiment on ToolMind datasets")
    parser.add_argument(
        "--dataset",
        choices=["syn_wikiqa", "open_wiki_traj", "all"],
        default="all",
        help="Dataset to process (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--cache-warmup",
        type=int,
        default=100,
        help="Number of sessions for cache warmup (default: 100)",
    )
    parser.add_argument(
        "--cache-max-size",
        type=int,
        default=100000,
        help="Maximum cache size in tokens (default: 100000)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ToolMind Full Dataset Cache Experiment")
    print("=" * 60)

    datasets = {
        "syn_wikiqa": "data/syn_wikiqa.jsonl",
        "open_wiki_traj": "data/open-wiki-traj.jsonl",
    }

    if args.dataset == "all":
        for name, file in datasets.items():
            if Path(file).exists():
                process_dataset(
                    name,
                    file,
                    max_samples=args.max_samples,
                    cache_warmup=args.cache_warmup,
                    cache_max_size=args.cache_max_size,
                )
            else:
                print(f"\n⚠ File not found: {file}")
                print(f"  Run: python download_toolmind.py --files {Path(file).name}")
    else:
        file = datasets[args.dataset]
        if Path(file).exists():
            process_dataset(
                args.dataset,
                file,
                max_samples=args.max_samples,
                cache_warmup=args.cache_warmup,
                cache_max_size=args.cache_max_size,
            )
        else:
            print(f"\n⚠ File not found: {file}")
            print(f"  Run: python download_toolmind.py --files {Path(file).name}")

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)
