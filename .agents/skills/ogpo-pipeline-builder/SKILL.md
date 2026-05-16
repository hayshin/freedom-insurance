---
name: ogpo-pipeline-builder
description: "Собирает и чинит полный runnable ML pipeline для OGPO hackathon: чтение train/test, contract-level aggregation, frequency-severity models, leakage-safe preprocessing, pricing calibration до loss_ratio около 70%, метрики, artifacts/models.pkl и submissions/submission.csv. Использовать, когда нужно сделать end-to-end обучение, валидацию, submission или улучшить scripts/train_pipeline.py."
---

# OGPO Pipeline Builder

## Роль

Ты отвечаешь за полностью работающий end-to-end пайплайн для кейса ОГПО: от `dataset/train.csv` и `dataset/test.csv` до обученных моделей, метрик, pricing calibration и файла submission.

Перед работой прочитай:
- `problem.md` - постановка, constraints и критерии сдачи.
- `scripts/train_pipeline.py` - текущий пайплайн.
- `features/` - подключенные feature modules.
- `pyproject.toml` - доступные библиотеки.

## Целевой результат

Пайплайн должен:
- агрегировать driver-level строки до `contract_number`;
- обучать frequency model для `is_claim`;
- обучать severity model по `claim_amount > 0`;
- считать `expected_claim = claim_probability * predicted_severity`;
- калибровать `new_premium` под target portfolio `loss_ratio ~= 0.70`;
- соблюдать ограничения: снижение не ниже 0, рост не выше `3 * premium`;
- считать метрики качества и business metrics;
- сохранять `artifacts/models.pkl`, `artifacts/metrics.json` и `submissions/submission.csv`;
- выводить воспроизводимый JSON с метриками.

## Инварианты

- Не изменять файлы в `dataset/`.
- Не использовать target/leakage columns в feature matrix.
- `premium_wo_term` и `loss_ratio` использовать для оценки/калибровки, но не как обычные model features, если это недоступно для test-time pricing.
- Train/test feature columns должны быть выровнены.
- Категориальные значения должны обрабатываться стабильно: missing, rare, unknown.
- Любая ошибка должна падать явно, а не молча портить submission.

## Рабочий порядок

1. Сначала добейся надежного запуска базового пайплайна.
2. Затем добавляй признаки или модели маленькими изменениями.
3. После каждого существенного изменения запускай пайплайн и проверяй:
   - ROC-AUC/GINI для frequency;
   - positive-subset MAE/RMSE/R2 для severity;
   - baseline и post-pricing loss_ratio;
   - долю группы keep/decrease;
   - loss_ratio отдельно для increased и keep/decrease групп, если доступно.
4. Если улучшение метрики ухудшает бизнес-ограничения, приоритет за business objective из `problem.md`.

## Модельный дизайн

Базовый подход:
- frequency: LightGBM classifier, fallback sklearn HistGradientBoostingClassifier;
- severity: LightGBM regressor на `log1p(claim_amount)` только для положительных выплат, fallback sklearn HistGradientBoostingRegressor;
- final expected claim: произведение probability и severity;
- pricing: grid/calibration strategy, оптимизирующая близость portfolio/group loss_ratio к 0.70 и сохраняющая максимум полисов без повышения.

Не усложняй раньше времени:
- не добавляй нейросети;
- не добавляй external data;
- не добавляй target encoding без out-of-fold;
- не смешивай exploratory notebooks с production pipeline.

## Submission contract

Файл результата должен содержать:
- `contract_number`;
- `claim_probability`;
- `pred_loss_ratio`;
- `new_premium`.

Если организатор требует точное имя `Contract_number`, добавь совместимость осознанно и зафиксируй в ответе.

## Валидация

Основная команда:

```bash
uv run python scripts/train_pipeline.py --artifacts-dir artifacts --submission submissions/submission.csv
```

Быстрый fallback без LightGBM:

```bash
uv run python scripts/train_pipeline.py --force-sklearn --artifacts-dir /tmp/ogpo-artifacts --submission /tmp/ogpo-submission.csv
```

Если команда падает из-за sandbox/network/dependencies, не скрывай это. Исправь кодовые ошибки, а в финальном ответе укажи точную причину, почему полный запуск не был подтвержден.

## Финальный ответ

Коротко перечисли:
- измененные файлы;
- что теперь делает пайплайн;
- какие проверки запускались;
- ключевые метрики, если пайплайн был выполнен.
