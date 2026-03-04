#!/usr/bin/env python3
"""
SGLang-style Radix Tree Prefix Cache Simulation.
"""

import json
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class RadixNode:
    token_id: int
    children: Dict[int, "RadixNode"] = field(default_factory=dict)
    is_end: bool = False
    ref_count: int = 1


class RadixPrefixCache:
    def __init__(self, bos_token_id: int = 1):
        self.bos_token_id = bos_token_id
        self.root = RadixNode(token_id=-1)
        self.total_nodes = 0
        self.longest_prefix_hits = 0
        self.exact_hits = 0
        self.misses = 0
        self.total_tokens_processed = 0

    def add(self, token_ids: List[int]) -> None:
        if not token_ids:
            return

        current = self.root

        for token_id in token_ids:
            if token_id not in current.children:
                new_node = RadixNode(token_id=token_id)
                current.children[token_id] = new_node
                self.total_nodes += 1
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
        }

    def reset(self):
        self.root = RadixNode(token_id=-1)
        self.total_nodes = 0
        self.longest_prefix_hits = 0
        self.exact_hits = 0
        self.misses = 0
        self.total_tokens_processed = 0


class MockTokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.role_tokens = {
            "system": 100,
            "user": 101,
            "assistant": 102,
        }

    def encode(self, text: str) -> List[int]:
        tokens = [self.bos_token_id]
        words = text.split()
        for word in words:
            token_id = (hash(word) % (self.vocab_size - 200)) + 200
            tokens.append(token_id)
        tokens.append(self.eos_token_id)
        return tokens

    def encode_chatml_message(self, role: str, content: str) -> List[int]:
        role_id = self.role_tokens.get(role, 103)
        content_tokens = self.encode(content)
        return [role_id] + content_tokens


def chatml_messages_to_prompt(
    messages: List[Dict[str, str]], tokenizer: MockTokenizer
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
    base_user = user_messages[0]
    base_assistant = assistant_messages[0]

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
    tokenizer: MockTokenizer,
    cache_warmup_sessions: int = 10,
) -> Tuple[List[Dict[str, Any]], RadixPrefixCache]:
    cache = RadixPrefixCache(bos_token_id=tokenizer.bos_token_id)
    results = []

    print(f"Simulating Radix Tree Prefix Cache with {len(sessions)} sessions")
    print(f"Warming up cache with first {cache_warmup_sessions} sessions...\n")

    for i, session in enumerate(sessions[:cache_warmup_sessions]):
        tokens = chatml_messages_to_prompt(session, tokenizer)
        cache.add(tokens)

    print(f"Cache warmed up. Tree nodes: {cache.total_nodes}\n")
    print("=" * 80)
    print(
        f"{'Session':<10} {'Exact':<8} {'This Hit':<12} {'This Miss':<12} {'Cum Hit':<12} {'Cum Miss':<12} {'Total':<10}"
    )
    print("=" * 80)

    cum_hit = 0
    cum_miss = 0

    for i, session in enumerate(sessions[cache_warmup_sessions:]):
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

        exact_str = "YES" if result["exact_hit"] else "NO"
        print(
            f"{session_num:<10} {exact_str:<8} "
            f"{this_hit:<12} {this_miss:<12} "
            f"{cum_hit:<12} {cum_miss:<12} "
            f"{result['total']:<10}"
        )

        cache.add(tokens)

    print("=" * 80)

    return results, cache


def main():
    print("=" * 60)
    print("SGLang-style Radix Tree Prefix Cache Simulation")
    print("=" * 60)
    print()

    NUM_SESSIONS = 100
    MESSAGES_PER_SESSION = 5
    CACHE_WARMUP = 10

    tokenizer = MockTokenizer(vocab_size=50000)

    print("Generating fake ChatML dataset...")
    sessions = generate_fake_chatml_data(
        num_sessions=NUM_SESSIONS,
        messages_per_session=MESSAGES_PER_SESSION,
    )
    print(f"Generated {len(sessions)} sessions\n")

    print("Sample session (first one):")
    print(json.dumps(sessions[0], indent=2)[:500])
    print("\n" + "=" * 60 + "\n")

    results, cache = simulate_radix_cache(
        sessions=sessions,
        tokenizer=tokenizer,
        cache_warmup_sessions=CACHE_WARMUP,
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
            "messages_per_session": MESSAGES_PER_SESSION,
            "cache_warmup_sessions": CACHE_WARMUP,
            "cache_type": "radix_tree",
        },
        "overall_stats": stats,
        "per_session_results": results,
    }

    with open("prefix_cache_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to prefix_cache_results.json")


if __name__ == "__main__":
    main()
