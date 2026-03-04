# Prefix Cache Simulation

SGLang-style Radix Tree Prefix Cache for analyzing token-level cache hit/miss patterns in ChatML conversations.

## Quick Start

```bash
python3 prefix_cache_sim.py
```

## Output

The simulation outputs per-session statistics:

| Column | Description |
|--------|-------------|
| Session | Session number |
| Exact | Whether full sequence was cached |
| This Hit | Prefix hit tokens for this session |
| This Miss | Tokens that need computation |
| Cum Hit | Cumulative prefix hit tokens |
| Cum Miss | Cumulative tokens needing computation |

## Configuration

Edit these variables in `main()`:

```python
NUM_SESSIONS = 100        # Total sessions to simulate
MESSAGES_PER_SESSION = 5  # Messages per session
CACHE_WARMUP = 10        # Sessions to preload into cache
```

## Use with Real Data

Replace `generate_fake_chatml_data()` with your data loader. Expected format:

```python
sessions: List[List[Dict[str, str]]]
# Each session: [{"role": "user/assistant/system", "content": "..."}, ...]
```

## Results

Output saved to `prefix_cache_results.json`.

## Requirements

- Python 3.9+
- (Optional) transformers for real tokenizer
