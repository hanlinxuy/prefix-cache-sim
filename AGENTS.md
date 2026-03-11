# AGENTS.md

## Project Overview

Python package implementing SGLang-style Radix Tree prefix cache simulation for LLM inference optimization. Analyzes token-level cache hit/miss patterns in ChatML-formatted conversations with tool call support.

## Build, Lint, Test Commands

### Running the Simulation

```bash
# Run the prefix cache simulation
python -m prefix_cache_sim

# Or using uv
uv run python -m prefix_cache_sim
```

### Installation

```bash
# Install dependencies with uv
uv sync

# Install dev dependencies
uv sync --group dev
```

### Linting

```bash
# Lint all Python files
uv run ruff check .

# Lint specific directory
uv run ruff check src/prefix_cache_sim/

# Auto-fix linting issues
uv run ruff check --fix .
```

### Type Checking

```bash
# Type check the project
uv run mypy src/prefix_cache_sim/

# Type check specific file
uv run mypy src/prefix_cache_sim/radix_tree.py
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_integration.py -v

# Run specific test function
uv run pytest tests/test_integration.py::test_full_pipeline_with_tools -v

# Run with coverage
uv run pytest --cov=prefix_cache_sim tests/
```

## Code Style Guidelines

### Imports

Order imports by type with blank lines between groups:

```python
# Standard library
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Third party
from tqdm import tqdm
from transformers import AutoTokenizer

# Local application
from prefix_cache_sim.radix_tree import RadixNode
from prefix_cache_sim.utils import chatml_messages_to_prompt
```

### Naming Conventions

```python
# Modules: snake_case
prefix_cache_sim/
radix_tree.py
simulation.py

# Classes: PascalCase
class RadixNode:
class RadixPrefixCache:
class Qwen2Tokenizer:

# Functions/variables: snake_case
def simulate_radix_cache():
def find_longest_prefix():
token_ids = []
bos_token_id = 1

# Constants: UPPER_SNAKE_CASE
DEFAULT_WARMUP_SESSIONS = 10
MAX_TOKEN_LENGTH = 4096
```

### Type Checking and Annotations

```python
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class RadixNode:
    token_id: int
    children: Dict[int, "RadixNode"] = field(default_factory=dict)
    is_end: bool = False
    ref_count: int = 1

def simulate_radix_cache(
    sessions: List[List[Dict[str, str]]],
    tokenizer: Qwen2Tokenizer,
    cache_warmup_sessions: int = 10,
) -> Tuple[List[Dict[str, Any]], RadixPrefixCache]:
    ...
```

### Error Handling

```python
# Use early returns for validation
def add(self, token_ids: List[int]) -> None:
    if not token_ids:
        return
    
    if self.max_size and tokens_to_add > self.max_size:
        return

# Avoid empty except blocks
try:
    result = process(data)
except ValueError as e:
    raise ValueError(f"Failed to process: {e}") from e
```

### Function Design

- Keep functions focused and small (< 50 lines when possible)
- Use clear parameter names
- Add type hints for all public functions
- Prefer explicit returns over implicit None
- Use progress bars for long-running operations (tqdm)

### File Organization

```
prefix-cache-sim/
├── src/prefix_cache_sim/
│   ├── __init__.py        # Package exports
│   ├── __main__.py        # CLI entry point
│   ├── radix_tree.py      # RadixNode & RadixPrefixCache
│   ├── tokenizer.py       # Qwen2Tokenizer wrapper
│   ├── data_loader.py     # Dataset loading
│   ├── tool_parser.py     # Tool call parsing
│   ├── utils.py           # Utility functions
│   └── simulation.py      # Core simulation logic
├── tests/
│   ├── test_integration.py
│   └── fixtures/
├── pyproject.toml
└── AGENTS.md
```

### Documentation

- Use docstrings for module-level and class-level documentation
- Keep docstrings concise - describe what/why, not how
- Example format:

```python
def simulate_radix_cache(
    sessions: List[List[Dict[str, str]]],
    tokenizer: Qwen2Tokenizer,
    cache_warmup_sessions: int = 10,
) -> Tuple[List[Dict[str, Any]], RadixPrefixCache]:
    """Simulate radix tree prefix cache on chat sessions."""
    ...
```

### Testing Guidelines

Test file naming: `test_*.py`
Test function naming: `test_*`

```python
def test_full_pipeline_with_tools():
    sessions = load_chatml_jsonl("tests/fixtures/dummy_chatml.jsonl")
    assert len(sessions) == 5
    ...
```

### Git Workflow

```bash
# Branch naming
feature/radix-cache-optimization
bugfix/fix-cache-miss-handling

# Commit message format
<type>: <description>

# Types: feat, fix, docs, style, refactor, test, chore
feat(cache): add LRU eviction policy
fix(tree): resolve prefix match boundary case
```

### General Principles

1. **Single responsibility**: Each function does one thing well
2. **Explicit over implicit**: Clear naming and types
3. **Immutability**: Prefer immutable data where practical
4. **Fail fast**: Validate inputs early
5. **No AI slop**: Write code that a senior engineer would write - no placeholder comments, no "TODO" without explanation
6. **Use uv**: Package manager for dependency management
7. **Progress feedback**: Use tqdm for long-running operations
