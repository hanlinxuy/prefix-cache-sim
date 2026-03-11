import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.setrecursionlimit(10000)


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
        self.total_tokens_written = 0  # Total tokens written to cache (new nodes)
        self.total_tokens_evicted = 0  # Total tokens evicted from cache
        self.total_write_volume = 0  # Total write volume (tokens written + evicted)
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

        def find_leaf_nodes(node: RadixNode, path: List[int], depth: int = 0) -> None:
            if depth > 1000:
                return
            if not node.children:
                candidates.append((path.copy(), node.ref_count, node.size))
            else:
                for child_token, child_node in node.children.items():
                    find_leaf_nodes(child_node, path + [child_token], depth + 1)

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
        self.total_tokens_evicted += removed_size
        self.total_write_volume += removed_size

    def _evict_fifo(self) -> None:
        if self.max_size is None:
            return

        candidates = []

        def find_leaf_nodes(node: RadixNode, path: List[int], depth: int = 0) -> None:
            if depth > 1000:
                return
            if not node.children:
                candidates.append((path.copy(), node.ref_count, node.size))
            else:
                for child_token, child_node in node.children.items():
                    find_leaf_nodes(child_node, path + [child_token], depth + 1)

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
        self.total_tokens_evicted += removed_size
        self.total_write_volume += removed_size

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

        new_nodes_created = 0
        for token_id in token_ids:
            if token_id not in current.children:
                new_node = RadixNode(token_id=token_id)
                current.children[token_id] = new_node
                self.total_nodes += 1
                self.current_size += 1
                new_nodes_created += 1
            else:
                current.children[token_id].ref_count += 1

            current = current.children[token_id]

        current.is_end = True

        if new_nodes_created > 0:
            self.total_tokens_written += new_nodes_created
            self.total_write_volume += new_nodes_created

    def find_longest_prefix(self, token_ids: List[int]) -> Tuple[int, Optional[RadixNode]]:
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
            "total_tokens_written": self.total_tokens_written,
            "total_tokens_evicted": self.total_tokens_evicted,
            "total_write_volume": self.total_write_volume,
        }

    def reset(self):
        self.root = RadixNode(token_id=-1)
        self.total_nodes = 0
        self.current_size = 0
        self.eviction_count = 0
        self.total_tokens_written = 0
        self.total_tokens_evicted = 0
        self.total_write_volume = 0
        self.longest_prefix_hits = 0
        self.exact_hits = 0
        self.misses = 0
        self.total_tokens_processed = 0
