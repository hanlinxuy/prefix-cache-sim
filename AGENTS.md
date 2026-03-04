# AGENTS.md

## Project Overview

This is a Python project containing a prefix cache simulation for LLM inference optimization. It implements a SGLang-style Radix Tree to analyze token-level cache hit/miss patterns in ChatML-formatted conversations.

## Build, Lint, Test Commands

### Running the Simulation

```bash
# Run the prefix cache simulation
python3 prefix_cache_sim.py

# Run with specific parameters (edit the NUM_SESSIONS, CACHE_WARMUP variables in main())
python3 prefix_cache_sim.py
```

### Linting

```bash
# Install ruff if needed
pip install ruff

# Lint all Python files
ruff check .

# Auto-fix linting issues
ruff check --fix .
```

### Type Checking

```bash
# Install mypy if needed
pip install mypy

# Type check the project
mypy prefix_cache_sim.py
```

## Code Style Guidelines

### Imports

Order imports by type:

```python
# Standard library
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

# Third party (if any)
# from transformers import AutoTokenizer

# Local application
# from mymodule import MyClass
```

### Naming Conventions

```python
# Modules: snake_case
prefix_cache_sim.py
radix_tree.py

# Classes: PascalCase
class RadixNode:
class RadixPrefixCache:

# Functions/variables: snake_case
def chatml_messages_to_prompt():
def simulate_radix_cache():
token_ids = []
bos_token_id = 1

# Constants: UPPER_SNAKE_CASE
DEFAULT_WARMUP_SESSIONS = 10
MAX_TOKEN_LENGTH = 4096
```

### Type Annotations

```python
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class RadixNode:
    token_id: int
    children: Dict[int, "RadixNode"] = field(default_factory=dict)
    is_end: bool = False
    ref_count: int = 1

def find_longest_prefix(
    self, token_ids: List[int]
) -> Tuple[int, Optional[RadixNode]]:
    ...
```

### Error Handling

```python
# Use early returns for validation
def add(self, token_ids: List[int]) -> None:
    if not token_ids:
        return
    
    # Main logic...

# Avoid empty except blocks
try:
    result = process(data)
except ValueError as e:
    raise ValueError(f"Failed to process: {e}") from e
```

### Dataclass Usage

Use `@dataclass` for simple data containers:

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class RadixNode:
    token_id: int
    children: Dict[int, "RadixNode"] = field(default_factory=dict)
    is_end: bool = False
```

### Function Design

- Keep functions focused and small (< 50 lines when possible)
- Use clear parameter names
- Add type hints for all public functions
- Prefer explicit returns over implicit None

### File Organization

```
project/
├── prefix_cache_sim.py    # Main entry point
├── prefix_cache_results.json  # Output (generated)
├── data/                  # Data directory
│   └── (dataset files)
├── tests/                 # Test files (if added)
└── AGENTS.md             # This file
```

### Documentation

- Use docstrings for module-level and class-level documentation
- Keep docstrings concise - describe what/why, not how
- Example docstring format:

```python
def chatml_messages_to_prompt(
    messages: List[Dict[str, str]], tokenizer: MockTokenizer
) -> List[int]:
    """Convert ChatML messages to token list."""
    ...
```

### Testing Guidelines

When adding tests:

```bash
# Run a single test file
python3 -m pytest tests/test_cache.py -v

# Run a specific test
python3 -m pytest tests/test_cache.py::test_radix_tree_insert -v
```

Test file naming: `test_*.py`
Test class naming: `Test*`
Test function naming: `test_*`

### Git Workflow (if applicable)

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
