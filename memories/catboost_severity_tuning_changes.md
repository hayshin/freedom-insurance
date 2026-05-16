# CatBoost severity tuning changes

Дата: 2026-05-16

## Контекст

До изменений команда:

```bash
uv run python scripts/train_pipeline.py \
  --model-backend catboost \
  --tune-severity \
  --severity-objective rmse_r2 \
  --severity-r2-weight 100000 \
  --severity-trials 30 \
  --severity-time-budget 1800
```

давала слабую severity-модель и недостаточно большую группу 1:

- Gini: `0.3272`
- ROC-AUC: `0.6636`
- severity R2 positive: `-0.0480`
- group 1 keep/decrease: `15,392` полиса, `46.88%`
- post-pricing loss ratio: `0.6960`

Главная проблема: severity обучалась на `log1p(claim_amount)`, но метрики считались в деньгах. На редких крупных выплатах такая модель легко становится хуже константы по R2.

## Что изменили

1. Frequency CatBoost:
   - заменили `auto_class_weights="Balanced"` на `auto_class_weights="SqrtBalanced"`.
   - Зачем: `Balanced` слишком сильно раздувал вероятности редкого класса. `SqrtBalanced` дал лучший ranking claims на validation.

2. Severity target:
   - добавили режим `--severity-target claim_per_premium`, он теперь default.
   - Вместо прямого `log1p(claim_amount)` модель учится на `log1p(claim_amount / premium)`, а прогноз обратно переводится в деньги через `premium`.
   - Зачем: тяжесть выплаты становится нормализованной относительно масштаба договора. Это лучше соответствует страховой задаче и pricing.

3. Severity calibration:
   - добавили affine-калибровку на positive validation: `calibrated = intercept + slope * raw_pred`.
   - Калибровка сохраняется внутри `SeverityModel` и применяется на test.
   - Добавлен флаг `--disable-severity-calibration`, если нужно сравнить raw severity.
   - Зачем: raw CatBoost severity всё ещё имела отрицательный R2, а простая линейная калибровка переводит R2 positive в положительную область.

4. Severity tuning:
   - random-search теперь оптимизирует уже выбранный severity target и calibrated predictions.
   - В `metrics.json` сохраняются both calibrated и raw severity metrics:
     - `best_r2_positive`
     - `best_raw_r2_positive`
     - `severity_calibration`
     - `severity_target`
   - Зачем: можно видеть, сколько качества дала модель сама, а сколько post-calibration.

5. Pricing calibration:
   - расширили grid по threshold: добавили quantiles `0.98`, `0.99`, `0.995`.
   - усилили бонус за `keep_or_decrease_share`.
   - Зачем: увеличить количество полисов в группе 1, сохраняя общий loss ratio около `0.70` и адекватный loss ratio по обеим группам.

6. Metrics:
   - добавили `keep_or_decrease_count` и `increase_count`.
   - Зачем: group 1 теперь удобно смотреть не только как долю, но и как абсолютное количество полисов.

## Проверки

Синтаксис и diff:

```bash
git diff --check
uv run python -m py_compile scripts/train_pipeline.py pipeline/models.py pipeline/runner.py pipeline/pricing.py pipeline/config.py pipeline/evaluation.py
```

Оба check прошли.

## Финальные validation-метрики

Команда с tuning на 30 trials:

```bash
uv run python scripts/train_pipeline.py \
  --model-backend catboost \
  --tune-severity \
  --severity-objective rmse_r2 \
  --severity-r2-weight 100000 \
  --severity-trials 30 \
  --severity-time-budget 1800
```

Результат:

- Gini: `0.3467`
- ROC-AUC: `0.6734`
- PR-AUC: `0.0437`
- F1: `0.0913`
- severity MAE positive: `473,629`
- severity RMSE positive: `822,963`
- severity R2 positive: `0.0173`
- raw severity R2 positive до calibration: `-0.0391`
- group 1 keep/decrease: `20,699` полисов, `63.05%`
- group 2 increase: `12,131` полисов, `36.95%`
- post-pricing loss ratio: `0.7007`
- group 1 loss ratio: `0.6972`
- group 2 loss ratio: `0.7042`
- group loss ratio gap: `0.0070`

## Важное сравнение

Без `--tune-severity` новый default CatBoost дал:

- Gini: `0.3421`
- severity R2 positive: `0.0157`
- group 1 keep/decrease: `22,822` полиса, `69.52%`
- post-pricing loss ratio: `0.7033`
- group loss ratio gap: `0.0057`

Вывод: tuned severity чуть лучше по model metrics, но non-tuned режим лучше по количеству полисов в группе 1. Для сабмита стоит сравнивать оба режима по leaderboard/business objective.

