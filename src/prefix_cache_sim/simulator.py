"""Cache simulator for batch processing conversations."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from prefix_cache_sim.tokenizer import Qwen2Tokenizer
from prefix_cache_sim.radix_tree import RadixPrefixCache
from prefix_cache_sim.utils import chatml_messages_to_prompt


@dataclass
class RequestResult:
    """Result of a single request simulation."""

    session_id: int
    request_id: int
    num_messages: int
    total_tokens: int
    hit_tokens: int
    miss_tokens: int
    exact_hit: bool


@dataclass
class SimulationResult:
    """Results from a full simulation run."""

    requests: List[RequestResult] = field(default_factory=list)
    cache_stats: Dict[str, Any] = field(default_factory=dict)

    def get_global_avg_miss(self) -> float:
        """Get global average miss tokens per request."""
        if not self.requests:
            return 0.0
        total_miss = sum(r.miss_tokens for r in self.requests)
        return total_miss / len(self.requests)

    def get_global_avg_hit(self) -> float:
        """Get global average hit tokens per request."""
        if not self.requests:
            return 0.0
        total_hit = sum(r.hit_tokens for r in self.requests)
        return total_hit / len(self.requests)

    def get_session_summary(self, session_id: int) -> Dict[str, Any]:
        """Get summary for a specific session."""
        session_requests = [r for r in self.requests if r.session_id == session_id]
        if not session_requests:
            return {}

        total_miss = sum(r.miss_tokens for r in session_requests)
        total_hit = sum(r.hit_tokens for r in session_requests)

        return {
            "session_id": session_id,
            "total_requests": len(session_requests),
            "avg_miss_tokens": total_miss / len(session_requests),
            "avg_hit_tokens": total_hit / len(session_requests),
            "total_tokens_processed": sum(r.total_tokens for r in session_requests),
        }


class CacheSimulator:
    """Simulator for prefix cache behavior across multiple sessions."""

    def __init__(
        self,
        tokenizer: Qwen2Tokenizer,
        cache: RadixPrefixCache,
    ):
        self.tokenizer = tokenizer
        self.cache = cache
        self.results: List[RequestResult] = []

    def process_session(
        self,
        session: List[Dict[str, str]],
        session_id: int = 0,
        skip_on_overflow: bool = True,
    ) -> List[RequestResult]:
        """Process a single session incrementally.

        Request 1: [msg1]
        Request 2: [msg1, msg2]
        Request 3: [msg1, msg2, msg3]
        ...

        Args:
            session: List of messages in the conversation
            session_id: ID for this session
            skip_on_overflow: If True, skip requests that exceed cache max_size

        Returns list of RequestResult for each incremental request.
        """
        session_results = []

        for i in range(1, len(session) + 1):
            messages = session[:i]
            tokens = chatml_messages_to_prompt(messages, self.tokenizer)

            # Check if request exceeds cache capacity
            if skip_on_overflow and self.cache.max_size and len(tokens) > self.cache.max_size:
                print(
                    f"  [Session {session_id + 1}] Request {i} skipped: "
                    f"{len(tokens)} tokens > max_size {self.cache.max_size}"
                )
                continue

            result = self.cache.query(tokens)

            request_result = RequestResult(
                session_id=session_id,
                request_id=i,
                num_messages=len(messages),
                total_tokens=len(tokens),
                hit_tokens=result["prefix_hit_len"],
                miss_tokens=result["miss_len"],
                exact_hit=result["exact_hit"],
            )
            session_results.append(request_result)
            self.results.append(request_result)

            # Add to cache for next request
            self.cache.add(tokens)

        return session_results

    def process_sessions(
        self,
        sessions: List[List[Dict[str, str]]],
    ) -> SimulationResult:
        """Process multiple sessions and return aggregated results."""
        for session_id, session in enumerate(sessions):
            self.process_session(session, session_id=session_id)

        return SimulationResult(
            requests=self.results,
            cache_stats=self.cache.get_statistics(),
        )

    def process_jsonl(self, filepath: str) -> SimulationResult:
        """Load sessions from jsonl file and process them."""
        sessions = []
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line)
                sessions.append(data["conversations"])

        return self.process_sessions(sessions)

    def reset(self) -> None:
        """Reset simulator state."""
        self.results = []
        self.cache.reset()


def print_simulation_summary(result: SimulationResult, verbose: bool = False) -> None:
    """Print formatted simulation summary."""
    stats = result.cache_stats

    print("\n" + "=" * 70)
    print("Prefill Cost Summary (Core Metrics)")
    print("=" * 70)

    print(f"\n[Cache Configuration]")
    max_size_str = f"{stats['cache_max_size']:,}" if stats["cache_max_size"] else "unlimited"
    print(f"  Max cache size:  {max_size_str} tokens")
    print(f"  Used size:       {stats['cache_used_size']:,} tokens")
    print(f"  Utilization:     {stats['cache_utilization']}")

    print(f"\n[Prefill Cost per Request]")
    print(f"  Total requests:           {stats['total_requests']}")
    print(
        f"  Avg miss tokens/request:  {stats['avg_miss_tokens_per_request']:.1f}  <-- prefill cost"
    )
    print(f"  Avg hit tokens/request:   {stats['avg_hit_tokens_per_request']:.1f}")
    print(f"  Total miss tokens:        {stats['total_miss_tokens']:,}")
    print(f"  Total hit tokens:         {stats['total_hit_tokens']:,}")

    print(f"\n[Hit Rates]")
    print(f"  Token hit rate:  {stats['token_hit_rate']}")
    print(f"  Exact hit rate:  {stats['exact_hit_rate']}")

    if verbose:
        # Per-session breakdown
        print(f"\n[Per-Session Breakdown]")
        session_ids = sorted(set(r.session_id for r in result.requests))
        for sid in session_ids:
            summary = result.get_session_summary(sid)
            print(f"  Session {sid + 1}:")
            print(f"    Requests: {summary['total_requests']}")
            print(f"    Avg miss: {summary['avg_miss_tokens']:.1f} tokens")
            print(f"    Avg hit:  {summary['avg_hit_tokens']:.1f} tokens")

    print(f"\n{'=' * 70}")
