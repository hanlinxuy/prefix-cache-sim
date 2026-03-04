import json

from prefix_cache_sim import (
    Qwen2Tokenizer,
    load_locomo_dataset,
    simulate_radix_cache,
)


def main():
    print("=" * 60)
    print("SGLang-style Radix Tree Prefix Cache Simulation")
    print("=" * 60)
    print()

    NUM_SESSIONS = None
    CACHE_WARMUP = 10
    CACHE_MAX_SIZE = 100000
    EVICTION_POLICY = "lru"
    MODEL_NAME = "Qwen/Qwen2-0.5B"
    DATASET_NAME = "Percena/locomo-mc10"

    print(f"Loading Qwen2 tokenizer: {MODEL_NAME}")
    tokenizer = Qwen2Tokenizer(model_name=MODEL_NAME)
    print(f"Tokenizer loaded. BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}\n")

    print(f"Loading dataset: {DATASET_NAME}")
    sessions = load_locomo_dataset(max_sessions=NUM_SESSIONS)
    print(f"Loaded {len(sessions)} sessions\n")

    print("Sample session (first one):")
    print(json.dumps(sessions[0], indent=2)[:500])
    print("\n" + "=" * 60 + "\n")

    results, cache = simulate_radix_cache(
        sessions=sessions,
        tokenizer=tokenizer,
        cache_warmup_sessions=CACHE_WARMUP,
        cache_max_size=CACHE_MAX_SIZE,
        eviction_policy=EVICTION_POLICY,
    )

    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

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
    print(f"Eviction policy: {stats['eviction_policy']}")

    hit_rates = [r["prefix_hit_len"] / r["total"] * 100 for r in results if r["total"] > 0]
    if hit_rates:
        print(f"\nAverage prefix hit rate: {sum(hit_rates) / len(hit_rates):.2f}%")
        print(f"Min prefix hit rate: {min(hit_rates):.2f}%")
        print(f"Max prefix hit rate: {max(hit_rates):.2f}%")

    output = {
        "config": {
            "num_sessions": NUM_SESSIONS,
            "cache_warmup_sessions": CACHE_WARMUP,
            "cache_type": "radix_tree",
            "cache_max_size": CACHE_MAX_SIZE,
            "eviction_policy": EVICTION_POLICY,
            "tokenizer_model": MODEL_NAME,
            "dataset": DATASET_NAME,
        },
        "overall_stats": stats,
        "per_session_results": results,
    }

    with open("prefix_cache_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to prefix_cache_results.json")


if __name__ == "__main__":
    main()
