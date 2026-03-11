import json
import os
import sys

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

            except json.JSONDecodeError:
                continue

    with open(output_file, "w", encoding="utf-8") as f:
        for session in sessions:
            f.write(json.dumps({"messages": session}, ensure_ascii=False) + "\n")

    print(f"Converted {len(sessions)} sessions to {output_file}")
    return sessions


def run_cache_experiment(sessions, tokenizer, cache_warmup=10, cache_max_size=100000):
    cache = RadixPrefixCache(
        bos_token_id=tokenizer.bos_token_id,
        max_size=cache_max_size,
        eviction_policy="lru",
    )

    print(f"\n{'=' * 60}")
    print("CACHE EXPERIMENT")
    print(f"{'=' * 60}")
    print(f"Total sessions: {len(sessions)}")
    print(f"Cache warmup sessions: {cache_warmup}")
    print(f"Cache max size: {cache_max_size}")
    print(f"Eviction policy: LRU")
    print()

    print("Warming up cache...")
    for i, session in enumerate(sessions[:cache_warmup]):
        tokens = chatml_messages_to_prompt(session, tokenizer)
        cache.add(tokens)

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


if __name__ == "__main__":
    print("ToolMind Dataset Cache Experiment")
    print("=" * 60)

    jsonl_file = "data/syn_wikiqa.jsonl"
    chatml_file = "data/toolmind_chatml.jsonl"

    print(f"Converting {jsonl_file} to ChatML format...")
    sessions = convert_toolmind_to_chatml(jsonl_file, chatml_file, max_samples=1000)

    print(f"\nLoading Qwen2 tokenizer...")
    tokenizer = Qwen2Tokenizer()
    print(f"Tokenizer loaded. BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}")

    stats, results = run_cache_experiment(
        sessions, tokenizer, cache_warmup=50, cache_max_size=100000
    )

    output_file = "data/toolmind_cache_results.json"
    output = {
        "config": {
            "dataset": "ToolMind-Web-QA/syn_wikiqa",
            "num_sessions": len(sessions),
            "cache_warmup_sessions": 50,
            "cache_max_size": 100000,
            "eviction_policy": "lru",
        },
        "stats": stats,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_file}")
