from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from prefix_cache_sim.radix_tree import RadixPrefixCache
from prefix_cache_sim.tokenizer import Qwen2Tokenizer
from prefix_cache_sim.utils import chatml_messages_to_prompt


def simulate_radix_cache(
    sessions: List[List[Dict[str, str]]],
    tokenizer: Qwen2Tokenizer,
    cache_warmup_sessions: int = 10,
    cache_max_size: Optional[int] = None,
    eviction_policy: str = "lru",
) -> Tuple[List[Dict[str, Any]], RadixPrefixCache]:
    cache = RadixPrefixCache(
        bos_token_id=tokenizer.bos_token_id,
        max_size=cache_max_size,
        eviction_policy=eviction_policy,
    )
    results = []

    print(f"Simulating Radix Tree Prefix Cache with {len(sessions)} sessions")
    if cache_max_size:
        print(f"Cache max size: {cache_max_size}, eviction: {eviction_policy}")
    print(f"Warming up cache with first {cache_warmup_sessions} sessions...\n")

    for i, session in enumerate(sessions[:cache_warmup_sessions]):
        tokens = chatml_messages_to_prompt(session, tokenizer)
        cache.add(tokens)

    print(f"Cache warmed up. Tree nodes: {cache.total_nodes}\n")

    cum_hit = 0
    cum_miss = 0
    test_sessions = sessions[cache_warmup_sessions:]

    for i, session in enumerate(tqdm(test_sessions, desc="Processing sessions")):
        tokens = chatml_messages_to_prompt(session, tokenizer)
        result = cache.query(tokens)

        session_num = i + cache_warmup_sessions + 1

        this_hit = result["prefix_hit_len"]
        this_miss = result["miss_len"]
        cum_hit += this_hit
        cum_miss += this_miss

        result["session_id"] = session_num
        result["cum_hit"] = cum_hit
        result["cum_miss"] = cum_miss
        results.append(result)

        cache.add(tokens)

    return results, cache
