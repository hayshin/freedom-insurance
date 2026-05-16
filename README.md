# OGPO insurance scoring

Full training pipeline:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_pipeline.py
```

The pipeline aggregates `dataset/train.csv` and `dataset/test.csv` to `contract_number`
level, trains frequency and severity models, calibrates repricing to a 70% target
loss ratio, and writes:

- `submissions/submission.csv`
- `artifacts/models.pkl`
- `artifacts/metrics.json`

Generated artifacts are ignored by git.
