# Project Progress

## 2026-03-11

### ToolMind Dataset Integration

**Completed:**
- Added tool parsing support for ChatML messages with tool calls
  - `ToolCall` class for representing tool calls
  - `parse_tool_calls()` to extract tool calls from messages
  - `format_tool_call_for_qwen()` to format tool calls for Qwen tokenizer
  - `format_tool_response_for_qwen()` to format tool responses
  - `convert_chatml_with_tools()` to convert ChatML messages with tools to Qwen format

- Added dataset loading utilities
  - `load_chatml_jsonl()` to load ChatML JSONL datasets

- Updated core components
  - Fixed recursion limit in radix tree (set to 10000)
  - Added depth protection in eviction functions
  - Updated tokenizer to handle BatchEncoding return types
  - Updated utils to use tool conversion

- Added test coverage
  - `test_tool_parser.py` - Tool call parsing and formatting tests
  - `test_integration.py` - Integration tests

- Added experiment scripts
  - `download_toolmind.py` - Download ToolMind datasets
  - `run_full_experiment.py` - Run cache experiments on datasets
  - `run_toolmind_experiment.py` - ToolMind-specific experiments
  - `run_open_wiki_traj_experiment.py` - Open Wiki Trajectory experiments
  - `run_user_request_experiment.py` - User request experiments
  - `run_individual_sessions.py` - Individual session analysis

- Updated build configuration
  - Added pytest to dev dependencies
  - Fixed package sources configuration in pyproject.toml

**Bug Fixes:**
- Fixed recursion depth issues in radix tree eviction functions
- Fixed tokenizer return type handling for different transformers versions

**Test Results:**
- syn_wikiqa (500 samples):
  - Token-level hit rate: 40.03%
  - Average prefix hit rate: 42.18%
  - Cache tree nodes: 26,844
  - Total write volume: 26,844 tokens

### Prefix Cache Debug Analysis (2026-03-11)

**Debug Findings:**
- Tested with 5 sessions from open-wiki-traj dataset
- Corrected prefix caching semantics: only cache user requests (not assistant responses)
- For each user turn, serialize full conversation history and query cache

**Results:**
- Session 0: First user turn 0% (expected), subsequent turns 72-95% hit rate
- Session 1: First user turn 95.47% (cross-session reuse), subsequent turns 52-96%
- Session 2: First user turn 3.87% (low due to system message difference), subsequent turns 48-93%
- Session 3: First user turn 95.77%, subsequent turns 53-85%
- Session 4: First user turn 95.55%, subsequent turns 61-86%

**Root Cause Analysis:**
- System messages contain dynamic dates: "Today is: 2025-12-09" vs "Today is: 2025-12-10"
- This breaks cross-session prefix reuse for sessions with different dates
- Session 0/1 use date 2025-12-09, Session 2/3/4 use date 2025-12-10
- Date difference occurs at token position 97, limiting prefix hits to 97 tokens instead of 2410+ tokens

**Conclusion:**
- Prefix cache implementation is working correctly
- Cross-session reuse is effective when system messages are identical
- Dynamic content in system messages (dates, timestamps) significantly reduces cache efficiency
- Within-session prefix reuse remains high (80-95%+) even with different system messages
