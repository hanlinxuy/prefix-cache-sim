# Prefix Cache Simulation

SGLang-style Radix Tree Prefix Cache for analyzing token-level cache hit/miss patterns in ChatML conversations.

## Quick Start

```bash
python -m prefix_cache_sim
```

## Installation

```bash
uv sync
```

## Project Structure

```
prefix_cache_sim/
├── src/prefix_cache_sim/
│   ├── __init__.py        # Package exports
│   ├── __main__.py        # CLI entry point
│   ├── radix_tree.py      # RadixNode & RadixPrefixCache
│   ├── tokenizer.py       # Qwen2Tokenizer wrapper
│   ├── data_loader.py     # Dataset loading
│   ├── utils.py           # Utilities
│   └── simulation.py      # Core simulation logic
├── pyproject.toml
└── README.md
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

Edit these variables in `src/prefix_cache_sim/__main__.py`:

```python
NUM_SESSIONS = None           # Total sessions (None = all)
CACHE_WARMUP = 10            # Sessions to preload into cache
CACHE_MAX_SIZE = 100000      # Max cache size in tokens
EVICTION_POLICY = "lru"      # "lru" or "fifo"
MODEL_NAME = "Qwen/Qwen2-0.5B"
```

## Use with Real Data

Replace `generate_fake_chatml_data()` with your data loader. Expected format:

```python
sessions: List[List[Dict[str, str]]]
# Each session: [{"role": "user/assistant/system", "content": "..."}, ...]
```

## Results

Output saved to `prefix_cache_results.json`.

## Testing

```bash
uv run ruff check src/prefix_cache_sim/
```

All checks pass.

## Requirements

- Python 3.9+
- uv (package manager)
