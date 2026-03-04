#!/usr/bin/env python3
"""
SGLang-style Radix Tree Prefix Cache Simulation.
"""

import json
import os
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

# Set HF mirror endpoint before importing transformers
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Try to import optional dependencies
try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class RadixNode:
    token_id: int
    children: Dict[int, "RadixNode"] = field(default_factory=dict)
    is_end: bool = False
    ref_count: int = 1
    size: int = 1


class RadixPrefixCache:
    def __init__(
        self,
        bos_token_id: int = 1,
        max_size: Optional[int] = None,
        eviction_policy: str = "lru",
    ):
        self.bos_token_id = bos_token_id
        self.root = RadixNode(token_id=-1)
        self.total_nodes = 0
        self.current_size = 0
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.eviction_count = 0
        self.longest_prefix_hits = 0
        self.exact_hits = 0
        self.misses = 0
        self.total_tokens_processed = 0

    def _should_evict(self, tokens_to_add: int) -> bool:
        if self.max_size is None:
            return False
        return self.current_size + tokens_to_add > self.max_size

    def _evict_lru(self) -> None:
        if self.max_size is None:
            return

        candidates = []

        def find_leaf_nodes(node: RadixNode, path: List[int]) -> None:
            if not node.children:
                candidates.append((path.copy(), node.ref_count, node.size))
            else:
                for child_token, child_node in node.children.items():
                    find_leaf_nodes(child_node, path + [child_token])

        find_leaf_nodes(self.root, [])

        if not candidates:
            return

        candidates.sort(key=lambda x: (x[1], x[2]))
        path, _, _ = candidates[0]

        current = self.root
        for token_id in path[:-1]:
            current = current.children[token_id]

        removed_token = path[-1]
        removed_node = current.children.pop(removed_token)

        def count_nodes_and_size(node: RadixNode) -> int:
            size = node.size
            for child in node.children.values():
                size += count_nodes_and_size(child)
            return size

        removed_size = count_nodes_and_size(removed_node)
        self.current_size -= removed_size
        self.total_nodes -= 1
        self.eviction_count += 1

    def _evict_fifo(self) -> None:
        if self.max_size is None:
            return

        candidates = []

        def find_leaf_nodes(node: RadixNode, path: List[int]) -> None:
            if not node.children:
                candidates.append((path.copy(), node.ref_count, node.size))
            else:
                for child_token, child_node in node.children.items():
                    find_leaf_nodes(child_node, path + [child_token])

        find_leaf_nodes(self.root, [])

        if not candidates:
            return

        candidates.sort(key=lambda x: x[1])
        path, _, _ = candidates[0]

        current = self.root
        for token_id in path[:-1]:
            current = current.children[token_id]

        removed_token = path[-1]
        removed_node = current.children.pop(removed_token)

        def count_nodes_and_size(node: RadixNode) -> int:
            size = node.size
            for child in node.children.values():
                size += count_nodes_and_size(child)
            return size

        removed_size = count_nodes_and_size(removed_node)
        self.current_size -= removed_size
        self.total_nodes -= 1
        self.eviction_count += 1

    def evict(self) -> None:
        if self.eviction_policy == "lru":
            self._evict_lru()
        else:
            self._evict_fifo()

    def add(self, token_ids: List[int]) -> None:
        if not token_ids:
            return

        tokens_to_add = len(token_ids)

        if self.max_size and tokens_to_add > self.max_size:
            return

        max_evictions = 100
        eviction_iter = 0
        while self._should_evict(tokens_to_add) and eviction_iter < max_evictions:
            self.evict()
            eviction_iter += 1

        if eviction_iter >= max_evictions:
            return

        current = self.root

        for token_id in token_ids:
            if token_id not in current.children:
                new_node = RadixNode(token_id=token_id)
                current.children[token_id] = new_node
                self.total_nodes += 1
                self.current_size += 1
            else:
                current.children[token_id].ref_count += 1

            current = current.children[token_id]

        current.is_end = True

    def find_longest_prefix(
        self, token_ids: List[int]
    ) -> Tuple[int, Optional[RadixNode]]:
        if not token_ids:
            return 0, None

        current = self.root
        matched_len = 0
        last_node = None

        for i, token_id in enumerate(token_ids):
            if token_id in current.children:
                current = current.children[token_id]
                matched_len += 1
                if current.is_end:
                    last_node = current
            else:
                break

        return matched_len, last_node

    def query(self, token_ids: List[int]) -> Dict[str, Any]:
        self.total_tokens_processed += len(token_ids)

        matched_len, _ = self.find_longest_prefix(token_ids)

        if matched_len == len(token_ids):
            self.exact_hits += 1
            self.longest_prefix_hits += matched_len
            return {
                "exact_hit": True,
                "prefix_hit_len": matched_len,
                "miss_len": 0,
                "total": len(token_ids),
            }
        elif matched_len > 0:
            self.longest_prefix_hits += matched_len
            self.misses += 1
            return {
                "exact_hit": False,
                "prefix_hit_len": matched_len,
                "miss_len": len(token_ids) - matched_len,
                "total": len(token_ids),
            }
        else:
            self.misses += 1
            return {
                "exact_hit": False,
                "prefix_hit_len": 0,
                "miss_len": len(token_ids),
                "total": len(token_ids),
            }

    def get_statistics(self) -> Dict[str, Any]:
        total = self.exact_hits + self.misses
        exact_hit_rate = self.exact_hits / total if total > 0 else 0

        token_hit_rate = (
            self.longest_prefix_hits / self.total_tokens_processed
            if self.total_tokens_processed > 0
            else 0
        )

        return {
            "total_sequences": total,
            "exact_hits": self.exact_hits,
            "misses": self.misses,
            "exact_hit_rate": f"{exact_hit_rate * 100:.2f}%",
            "token_hit_rate": f"{token_hit_rate * 100:.2f}%",
            "total_tokens_processed": self.total_tokens_processed,
            "prefix_hits_tokens": self.longest_prefix_hits,
            "cache_nodes": self.total_nodes,
            "current_size": self.current_size,
            "max_size": self.max_size,
            "eviction_count": self.eviction_count,
            "eviction_policy": self.eviction_policy,
        }

    def reset(self):
        self.root = RadixNode(token_id=-1)
        self.total_nodes = 0
        self.current_size = 0
        self.eviction_count = 0
        self.longest_prefix_hits = 0
        self.exact_hits = 0
        self.misses = 0
        self.total_tokens_processed = 0


class Qwen2Tokenizer:
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_chatml_message(self, role: str, content: str) -> List[int]:
        messages = [{"role": role, "content": content}]
        return self.tokenizer.apply_chat_template(messages, tokenize=True)


def load_locomo_dataset(
    split: str = "train", max_sessions: Optional[int] = None
) -> List[List[Dict[str, str]]]:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    import json
    from huggingface_hub import hf_hub_download

    file_path = hf_hub_download(
        repo_id="Percena/locomo-mc10",
        filename="transformed/locomo_mc10_with_name.json",
        repo_type="dataset",
    )

    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    sessions = []
    for item in data:
        if max_sessions and len(sessions) >= max_sessions:
            break

        haystack_sessions = item.get("haystack_sessions", [])
        if not haystack_sessions:
            continue

        for session_msgs in haystack_sessions:
            if max_sessions and len(sessions) >= max_sessions:
                break

            session = []
            for msg in session_msgs:
                role = msg.get("role", "user")
                name = msg.get("name", "")
                content = msg.get("content", "")
                if name:
                    content = f"{name}: {content}"
                session.append({"role": role, "content": content})

            if session:
                sessions.append(session)

    return sessions


def chatml_messages_to_prompt(
    messages: List[Dict[str, str]], tokenizer: Qwen2Tokenizer
) -> List[int]:
    tokens = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_tokens = tokenizer.encode_chatml_message(role, content)
        tokens.extend(msg_tokens)
    return tokens


def generate_fake_chatml_data(
    num_sessions: int = 100, messages_per_session: int = 5
) -> List[List[Dict[str, str]]]:
    system_prompts = [
        "You are a helpful assistant.",
        "You are a coding assistant.",
        "You are a friendly chatbot.",
    ]

    user_messages = [
        "Hello how are you",
        "What is Python",
        "Explain machine learning",
        "Write a function to add numbers",
        "What is the weather today",
        "Tell me a joke",
        "How do I learn programming",
        "What is AI",
        "Can you help me with math",
        "What are your capabilities",
    ]

    assistant_messages = [
        "I am doing well thank you",
        "Python is a high level programming language",
        "Machine learning is a subset of AI that enables systems to learn from data",
        "Here is a function: def add a b return a plus b",
        "I am sorry I do not have access to weather data",
        "Why did the developer go broke because he used up all his cache",
        "Start with basics and practice daily",
        "Artificial Intelligence is the simulation of human intelligence by machines",
        "I can help with various math problems",
        "I can answer questions write code and assist with many tasks",
    ]

    sessions = []

    base_system = system_prompts[0]

    session_variations = []
    for i in range(50):
        session_variations.append(
            {
                "system": system_prompts[i % len(system_prompts)],
                "user": user_messages[i % len(user_messages)],
                "assistant": assistant_messages[i % len(assistant_messages)],
            }
        )

    for session_idx in range(num_sessions):
        session = []

        session.append(
            {
                "role": "system",
                "content": base_system,
            }
        )

        for msg_idx in range(random.randint(2, messages_per_session)):
            variation = session_variations[
                (session_idx + msg_idx) % len(session_variations)
            ]

            session.append(
                {
                    "role": "user",
                    "content": variation["user"],
                }
            )
            session.append(
                {
                    "role": "assistant",
                    "content": variation["assistant"],
                }
            )

        sessions.append(session)

    return sessions


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
    print(
        f"Tokenizer loaded. BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}\n"
    )

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

    hit_rates = [
        r["prefix_hit_len"] / r["total"] * 100 for r in results if r["total"] > 0
    ]
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
