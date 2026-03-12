import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.setrecursionlimit(10000)


@dataclass
class RadixNode:
    """Radix Tree node storing a sequence of tokens on the edge."""

    tokens: List[int] = field(default_factory=list)
    children: Dict[int, "RadixNode"] = field(default_factory=dict)
    parent: Optional["RadixNode"] = field(default=None)
    is_end: bool = False
    ref_count: int = 0
    last_access: int = 0  # For LRU eviction

    def __post_init__(self):
        if not self.tokens and self.parent is not None:
            raise ValueError("Non-root node must have tokens")


class RadixPrefixCache:
    """
    Radix Tree based prefix cache with LRU eviction.

    Unlike a simple trie (one token per node), this uses edge compression
    where each node stores a sequence of tokens, splitting edges when
    partial matches occur.
    """

    def __init__(
        self,
        bos_token_id: int = 1,
        max_size: Optional[int] = None,
        eviction_policy: str = "lru",
    ):
        self.bos_token_id = bos_token_id
        self.root = RadixNode(tokens=[], parent=None)
        self.max_size = max_size
        self.eviction_policy = eviction_policy

        # Statistics
        self.current_size = 0  # 当前使用的cache size (tokens)
        self.total_nodes = 0
        self.eviction_count = 0
        self.total_tokens_written = 0
        self.total_tokens_evicted = 0
        self.total_write_volume = 0
        self.longest_prefix_hits = 0  # 累计hit tokens
        self.exact_hits = 0
        self.misses = 0
        self.total_tokens_processed = 0

        # Core metrics for prefill cost calculation
        self.total_requests = 0  # 总请求数
        self.total_miss_tokens = 0  # 累计miss tokens
        self.total_hit_tokens = 0  # 累计hit tokens

        # Concurrent mode stats
        self.concurrent_mode = False
        self.num_concurrent_sessions = 0
        self.total_concurrent_requests = 0
        self.avg_miss_tokens_per_request = 0.0

        # LRU counter
        self._access_counter = 0

    def _update_access_time(self, node: RadixNode) -> None:
        """Update access timestamp for LRU ordering."""
        self._access_counter += 1
        node.last_access = self._access_counter

    def _get_node_size(self, node: RadixNode) -> int:
        """Get total token count in a node and its descendants."""
        size = len(node.tokens)
        for child in node.children.values():
            size += self._get_node_size(child)
        return size

    def _get_all_leaf_paths(self) -> List[Tuple[List[int], RadixNode]]:
        """Get all leaf nodes with their paths for eviction."""
        leaves = []

        def traverse(node: RadixNode, path: List[int]) -> None:
            if not node.children:
                leaves.append((path.copy(), node))
            else:
                for first_token, child in node.children.items():
                    traverse(child, path + [first_token])

        traverse(self.root, [])
        return leaves

    def _evict_lru(self, required_tokens: int) -> bool:
        """
        Evict least recently used nodes until we have space.
        Returns True if eviction succeeded, False if cache is full and in use.
        """
        if self.max_size is None:
            return True

        max_iterations = 100
        iteration = 0

        while self.current_size + required_tokens > self.max_size and iteration < max_iterations:
            leaves = self._get_all_leaf_paths()

            if not leaves:
                return False  # Cache full with no evictable nodes

            # Find LRU leaf (lowest last_access)
            lru_path, lru_node = min(leaves, key=lambda x: x[1].last_access)

            if not lru_path:
                return False  # Root is only node

            # Remove from parent
            parent = lru_node.parent
            if parent is None:
                return False

            first_token = lru_node.tokens[0] if lru_node.tokens else lru_path[-1]

            # Find the correct key in parent's children
            key_to_remove = None
            for key, child in parent.children.items():
                if child is lru_node:
                    key_to_remove = key
                    break

            if key_to_remove is not None:
                removed_size = self._get_node_size(lru_node)
                del parent.children[key_to_remove]

                self.current_size -= removed_size
                self.total_nodes -= 1
                self.eviction_count += 1
                self.total_tokens_evicted += removed_size
                self.total_write_volume += removed_size

            iteration += 1

        return self.current_size + required_tokens <= self.max_size

    def _find_best_match(
        self, tokens: List[int]
    ) -> Tuple[RadixNode, Optional[RadixNode], int, int]:
        """
        Find the best matching path in the tree.

        Returns:
            (parent_node, child_node, match_len_in_child, total_matched_tokens)
            - parent_node: last fully matched node
            - child_node: partially matched child (or None if complete match/no match)
            - match_len_in_child: how many tokens matched in child node (0 if no partial match)
            - total_matched_tokens: total tokens matched along the path
        """
        node = self.root
        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token not in node.children:
                # No matching child, return current position
                return node, None, 0, i

            child = node.children[token]
            child_tokens = child.tokens

            # Compare tokens along this edge
            match_len = 0
            while (
                match_len < len(child_tokens)
                and i + match_len < len(tokens)
                and child_tokens[match_len] == tokens[i + match_len]
            ):
                match_len += 1

            if match_len == len(child_tokens):
                # Full match of this edge
                node = child
                i += match_len
                self._update_access_time(node)
            else:
                # Partial match - need to split this edge
                self._update_access_time(child)
                return node, child, match_len, i + match_len

        # All tokens matched
        return node, None, 0, i

    def _split_node(self, parent: RadixNode, child: RadixNode, split_pos: int) -> RadixNode:
        """
        Split a child node at split_pos.

        Before: parent -> child[tokens[:N]]
        After:  parent -> new_child[tokens[:split_pos]] -> child[tokens[split_pos:]]

        Returns the new middle node.
        """
        original_tokens = child.tokens

        # Create new middle node with first part of tokens
        new_node = RadixNode(
            tokens=original_tokens[:split_pos],
            parent=parent,
            children={},
            is_end=False,
            ref_count=child.ref_count,
            last_access=child.last_access,
        )

        # Update child with remaining tokens
        child.tokens = original_tokens[split_pos:]
        child.parent = new_node

        # Transfer child's children to new node
        new_node.children = child.children
        for gc in new_node.children.values():
            gc.parent = new_node

        # Clear child's children (they moved to new_node)
        child.children = {}

        # Add child as child of new_node
        if child.tokens:
            new_node.children[child.tokens[0]] = child

        # Update parent's reference
        first_token = new_node.tokens[0]
        parent.children[first_token] = new_node

        self.total_nodes += 1

        return new_node

    def add(self, token_ids: List[int]) -> None:
        """
        Add a sequence of tokens to the cache.
        Handles edge splitting for partial matches.
        """
        if not token_ids:
            return

        tokens_to_add = len(token_ids)

        # Reject if single request exceeds max capacity
        if self.max_size and tokens_to_add > self.max_size:
            return

        # Evict if necessary
        if self.max_size:
            success = self._evict_lru(tokens_to_add)
            if not success:
                return

        # Find matching path
        parent, child, match_len, total_matched = self._find_best_match(token_ids)
        remaining = token_ids[total_matched:]

        if not remaining and child is None:
            # Complete match, just mark as end
            parent.is_end = True
            self._update_access_time(parent)
            return

        # Split node if partial match
        if child is not None and match_len > 0:
            parent = self._split_node(parent, child, match_len)
            # After split, remaining tokens start from original child's position
            remaining = token_ids[total_matched:]

        # Insert remaining tokens as new edge
        if remaining:
            new_node = RadixNode(
                tokens=remaining,
                parent=parent,
                children={},
                is_end=True,
                ref_count=1,
                last_access=self._access_counter + 1,
            )
            parent.children[remaining[0]] = new_node

            self.total_nodes += 1
            self.current_size += len(remaining)
            self.total_tokens_written += len(remaining)
            self.total_write_volume += len(remaining)
            self._access_counter += 1

    def find_longest_prefix(self, token_ids: List[int]) -> Tuple[int, Optional[RadixNode]]:
        """
        Find the longest prefix match in the cache.

        Returns:
            (matched_length, end_node) where end_node is the node at the end
            of the match (if it's marked as end), or None.
        """
        if not token_ids:
            return 0, None

        node = self.root
        matched_len = 0
        last_end_node = None

        i = 0
        while i < len(token_ids):
            token = token_ids[i]

            if token not in node.children:
                break

            child = node.children[token]
            child_tokens = child.tokens

            # Check how many tokens match along this edge
            match_len = 0
            while (
                match_len < len(child_tokens)
                and i + match_len < len(token_ids)
                and child_tokens[match_len] == token_ids[i + match_len]
            ):
                match_len += 1

            matched_len += match_len
            i += match_len
            node = child

            if node.is_end and match_len == len(child_tokens):
                last_end_node = node

            # If partial match, stop here
            if match_len < len(child_tokens):
                break

        return matched_len, last_end_node

    def query(self, token_ids: List[int]) -> Dict[str, Any]:
        """
        Query the cache for a token sequence.
        Updates statistics but does NOT add to cache.
        """
        self.total_tokens_processed += len(token_ids)
        self.total_requests += 1

        matched_len, end_node = self.find_longest_prefix(token_ids)
        miss_len = len(token_ids) - matched_len

        # Core metrics accumulation
        self.total_hit_tokens += matched_len
        self.total_miss_tokens += miss_len

        # Exact hit only if we matched all tokens AND ended at an end node
        # (not in the middle of an edge)
        if matched_len == len(token_ids) and end_node is not None:
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
                "miss_len": miss_len,
                "total": len(token_ids),
            }
        else:
            self.misses += 1
            return {
                "exact_hit": False,
                "prefix_hit_len": 0,
                "miss_len": miss_len,
                "total": len(token_ids),
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics focused on prefill cost calculation."""
        total = self.exact_hits + self.misses
        exact_hit_rate = self.exact_hits / total if total > 0 else 0

        token_hit_rate = (
            self.longest_prefix_hits / self.total_tokens_processed
            if self.total_tokens_processed > 0
            else 0
        )

        # Core metrics for prefill cost
        avg_hit_tokens = (
            self.total_hit_tokens / self.total_requests if self.total_requests > 0 else 0
        )
        avg_miss_tokens = (
            self.total_miss_tokens / self.total_requests if self.total_requests > 0 else 0
        )

        return {
            # Core prefill metrics
            "total_requests": self.total_requests,
            "avg_hit_tokens_per_request": round(avg_hit_tokens, 2),
            "avg_miss_tokens_per_request": round(avg_miss_tokens, 2),
            "total_hit_tokens": self.total_hit_tokens,
            "total_miss_tokens": self.total_miss_tokens,
            # Cache capacity metrics
            "cache_max_size": self.max_size,
            "cache_used_size": self.current_size,
            "cache_utilization": (
                f"{self.current_size / self.max_size * 100:.1f}%"
                if self.max_size
                else "N/A (unlimited)"
            ),
            # Hit rates
            "exact_hit_rate": f"{exact_hit_rate * 100:.2f}%",
            "token_hit_rate": f"{token_hit_rate * 100:.2f}%",
            "total_tokens_processed": self.total_tokens_processed,
            # Additional info
            "cache_nodes": self.total_nodes,
            "eviction_count": self.eviction_count,
        }

    def reset(self) -> None:
        """Reset the cache to empty state."""
        self.root = RadixNode(tokens=[], parent=None)
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
        self._access_counter = 0

        # Core metrics
        self.total_requests = 0
        self.total_miss_tokens = 0
        self.total_hit_tokens = 0
