# ToolMind Dataset Download and Experiment Scripts

## Download Datasets

### List available files
```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run python download_toolmind.py --list
```

### Download specific file
```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run python download_toolmind.py --files syn_wikiqa.jsonl
KMP_DUPLICATE_LIB_OK=TRUE uv run python download_toolmind.py --filesfiles open-wiki-traj.jsonl
```

### Download all JSONL files
```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run python download_toolmind.py
```

### Force re-download
```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run python download_toolmind.py --force
```

## Run Experiments

### Run on specific dataset
```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run python run_full_experiment.py --dataset syn_wikiqa
KMP_DUPLICATE_LIB_OK=TRUE uv run python run_full_experiment.py --dataset open_wiki_traj
```

### Run on all datasets
```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run python run_full_experiment.py --dataset all
```

### With custom parameters
```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run python run_full_experiment.py \
  --dataset syn_wikiqa \
  --max-samples 1000 \
  --cache-warmup 100 \
  --cache-max-size 200000
```

## Dataset Files

- **syn_wikiqa.jsonl** (7.06 MB): 6,801 synthetic QA pairs
- **open-wiki-traj.jsonl** (1.5 GB): Large trajectory dataset

## Output Files

- `{dataset}_chatml.jsonl`: Converted ChatML format
- `{dataset}_cache_results.json`: Cache experiment results

## Example Results

### syn_wikiqa (500 samples)
- Token-level hit rate: 40.03%
- Average prefix hit rate: 42.18%
- Cache tree nodes: 26,844
- Total write volume: 26,844 tokens
