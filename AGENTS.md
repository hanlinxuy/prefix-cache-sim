# AGENTS.md

## Project Overview

SGLang-style Radix Tree Prefix Cache simulation for LLM inference. Simulates token-level cache hit/miss patterns in ChatML conversations.

## Build, Lint, Test Commands

```bash
# Run simulation
PYTHONPATH=src uv run python -m prefix_cache_sim

# Install dependencies
uv sync

# Lint
uv run ruff check .
uv run ruff check --fix .

# Type check
uv run mypy src/prefix_cache_sim/

# Run tests
uv run pytest tests/ -v
uv run pytest tests/test_cache_property.py -v
uv run pytest tests/test_cache_property.py::test_miss_equals_user_tokens -v
```

## Code Style

### Imports

```python
# Standard library
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Third party
from transformers import AutoTokenizer

# Local
from prefix_cache_sim.radix_tree import RadixNode
from prefix_cache_sim.utils import chatml_messages_to_prompt
```

### Naming

```python
# snake_case: modules, functions, variables
radix_tree.py
def find_longest_prefix():
token_ids = []

# PascalCase: classes
class RadixPrefixCache:
class RadixNode:

# UPPER_SNAKE_CASE: constants
DEFAULT_MAX_SIZE = 10000
```

### Types

```python
@dataclass
class RadixNode:
    tokens: List[int] = field(default_factory=list)
    children: Dict[int, "RadixNode"] = field(default_factory=dict)

def process(
    sessions: List[List[Dict[str, str]]],
    tokenizer: Qwen2Tokenizer,
) -> Tuple[int, RadixPrefixCache]:
    """Process sessions and return hit count."""
    ...
```

### Error Handling

```python
# Early returns for validation
if not token_ids:
    return

# Explicit exceptions
try:
    result = process(data)
except ValueError as e:
    raise ValueError(f"Failed: {e}") from e
```

## File Structure

```
src/prefix_cache_sim/
├── __init__.py       # Exports
├── __main__.py       # CLI
├── radix_tree.py     # RadixPrefixCache, RadixNode
├── tokenizer.py      # Qwen2Tokenizer
└── utils.py          # chatml_messages_to_prompt

tests/
└── test_cache_property.py

data/
└── open_wiki_traj_test_5.jsonl
```

## Testing

Test file: `test_*.py`
Function: `test_*`

```python
def test_miss_equals_user_tokens():
    """Verify cache miss equals current user message tokens."""
    cache = RadixPrefixCache()
    # ... test logic
    assert miss_tokens == expected_miss
```

## Key Implementation Details

1. **Radix Tree**: Edge-compressed (stores token sequences), not one-token-per-node
2. **Cache Flow**: Query(history) → Generate → Cache.add(full_conversation)
3. **Property**: Miss tokens == current user message tokens (history is cached)
4. **Eviction**: LRU based on access counter (not timestamps)

## Git Workflow

```bash
# Branches
feature/radix-optimization
fix/cache-miss-calculation

# Commits
feat(cache): add LRU eviction
fix(tree): correct partial match handling
test: add cache property verification
```

## Principles

1. Single responsibility functions (< 50 lines)
2. Explicit types for all public functions
3. Fail fast with early validation
4. No placeholder comments or TODOs without context
5. Use `uv` for package management
6. Write code a senior engineer would write
