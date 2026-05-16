# OGPO insurance scoring

## Назначение

Проект строит скоринг убыточности ОГПО и формирует новую стоимость полисов с целевым коэффициентом выплат около 70%. Используется подход frequency-severity, leakage-safe risk encoding и OOF калибровка pricing.

## Быстрый запуск (финальная генерация)

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_pipeline.py \
	--final-mode both \
	--compare-risk-encoding \
	--production-splits 5
```

## Выходные артефакты

- submissions/submission.csv
- artifacts/models.pkl
- artifacts/metrics.json
- artifacts/production_metrics.json
- artifacts/final_metrics.json

## Ключевые метрики

Holdout (untouched, Nov-Dec 2022):

- ROC-AUC: 0.6787, GINI: 0.3575
- Post-pricing loss ratio: 0.6655
- Keep/Decrease share: 0.7276

Production OOF (калибровка на train):

- ROC-AUC: 0.6835, GINI: 0.3670
- Post-pricing loss ratio: 0.6985
- Min/Max new-to-old ratio: 0.90 / 3.00

## Структура репозитория

- dataset/ - train/test CSV от организатора
- features/ - генерация признаков
- pipeline/ - сборка фичей, обучение, калибровка, pricing
- scripts/ - CLI для запуска пайплайна
- artifacts/ - метрики и модели
- submissions/ - итоговый файл сабмита
- reports/ - отчеты по проекту

## Отчеты

- reports/01_overview_results.md
- reports/02_data_analysis.md
- reports/03_modeling_pricing.md

## Примечание

Артефакты и сабмиты могут быть исключены из git и пересоздаются при запуске пайплайна.
