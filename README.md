# 前缀缓存模拟器

SGLang 风格的 Radix Tree 前缀缓存，用于分析 ChatML 对话中的 token 级别缓存命中/未命中模式。

## 快速开始

```bash
python -m prefix_cache_sim
```

## 安装

```bash
uv sync
```

## 项目结构

```
prefix_cache_sim/
├── src/prefix_cache_sim/
│   ├── __init__.py        # 包导出
│   ├── __main__.py        # CLI 入口
│   ├── radix_tree.py      # RadixNode & RadixPrefixCache
│   ├── tokenizer.py       # Qwen2Tokenizer 封装
│   ├── data_loader.py     # 数据集加载
│   ├── utils.py           # 工具函数
│   └── simulation.py      # 核心模拟逻辑
├── pyproject.toml
└── README.md
```

## 配置

在 `src/prefix_cache_sim/__main__.py` 中修改以下配置：

```python
NUM_SESSIONS = None           # 总会话数 (None = 全部)
CACHE_WARMUP = 10            # 预热缓存的会话数
CACHE_MAX_SIZE = 100000      # 最大缓存大小 (token 数)
EVICTION_POLICY = "lru"      # 驱逐策略: "lru" 或 "fifo"
MODEL_NAME = "Qwen/Qwen2-0.5B"
```

## 输出指标

| 指标 | 说明 |
|------|------|
| total_sequences | 处理的总会话数 |
| exact_hits | 完全命中的会话数 |
| misses | 未命中的会话数 |
| exact_hit_rate | 完全命中率 |
| token_hit_rate | token 级别命中率 |
| total_tokens_processed | 处理的总 token 数 |
| prefix_hits_tokens | 前缀命中的 token 数 |
| cache_nodes | 缓存树节点数 |
| current_size | 当前缓存大小 |
| max_size | 最大缓存大小 |
| eviction_count | 驱逐次数 |
| total_tokens_written | 写入缓存的新 token 数 |
| total_tokens_evicted | 被驱逐的 token 数 |
| total_write_volume | 总写入量 (written + evicted) |

**SSD 寿命相关**: `total_write_volume` 可用于估算 SSD 写入寿命。

## 使用自定义数据

替换数据加载函数，预期格式：

```python
sessions: List[List[Dict[str, str]]]
# 每个会话: [{"role": "user/assistant/system", "content": "..."}, ...]
```

## 结果输出

结果保存到 `prefix_cache_results.json`。

## 测试

```bash
uv run ruff check src/prefix_cache_sim/
```

## 依赖

- Python 3.9+
- uv (包管理器)
