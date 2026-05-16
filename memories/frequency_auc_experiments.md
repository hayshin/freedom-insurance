# Frequency AUC experiments

Дата: 2026-05-16

## Цель

Поднять ROC-AUC/Gini frequency-модели `is_claim`. Это отдельная задача от severity/pricing:

- AUC/Gini зависят от ranking `claim_probability`.
- Severity target, severity calibration и pricing threshold не поднимают AUC напрямую.

Исходный сильный baseline после предыдущих изменений:

- CatBoost frequency с `auto_class_weights="SqrtBalanced"`
- ROC-AUC около `0.673`
- Gini около `0.347`

## Что попробовали

### 1. Leakage-safe OOF risk encoding

Добавлен модуль `pipeline/risk_encoding.py`.

Он строит out-of-fold claim-rate признаки на train split и train-only mapping для valid/test:

- smoothed claim rate
- log count
- claim lift относительно portfolio prior

Категории и бины:

- `region_macro_mode`
- `region_climate_mode`
- `mark_clean_mode`
- `model_clean_mode`
- `mark_model_pair`
- `vehicle_type_name_mode`
- `car_age`
- `operation_month`
- `operation_quarter`
- `ownerkato_short_mode`
- binned `bonus_malus`, driver experience, engine, score, premium features
- несколько interaction колонок типа `region x vehicle_type`, `vehicle_type x bonus_malus`

Зачем: CatBoost умеет работать с категориями, но явные сглаженные risk stats иногда дают лучший ranking для редкого binary target.

Результат без ID risk encoding:

- ROC-AUC: `0.6821-0.6824`
- Gini: `0.3643-0.3647`
- PR-AUC: `0.0469-0.0473`
- F1: `0.0974-0.0990`

Вывод: non-ID risk encoding полезен, но прирост умеренный. Включается флагом:

```bash
--enable-risk-encoding
```

### 2. ID risk encoding

Для эксперимента добавлены helper-поля:

- `driver_iin_mode`
- `insurer_iin_mode`
- `car_number_mode`

Они внесены в `LEAKAGE_COLUMNS`, чтобы сырые ID не попадали в feature matrix. Используются только как ключи для OOF статистик при флаге:

```bash
--enable-id-risk-encoding
```

Результат:

- ROC-AUC: `0.6705`
- Gini: `0.3411`

Вывод: ID risk encoding на текущем split ухудшает ranking, вероятно из-за шума/переобучения/малой повторяемости ID. Не включать по умолчанию.

### 3. CatBoost classifier tuning

Быстрый sweep показал, что дефолтный classifier ещё не оптимален. Проверялись варианты class weights, depth, l2, learning rate, bootstrap и early stopping по validation AUC.

Лучшие одиночные варианты:

- без risk encoding: `sqrt_bayes_d6`, ROC-AUC около `0.6831` в isolated sweep.
- с risk encoding: `depth=5`, `learning_rate=0.035`, `l2_leaf_reg=10`, `random_strength=0.5`, `SqrtBalanced`, ROC-AUC `0.6836` в isolated sweep.

Текущий default в `make_catboost_classifier` переключен на:

```text
iterations=1400
learning_rate=0.035
depth=5
l2_leaf_reg=10.0
auto_class_weights="SqrtBalanced"
random_strength=0.5
od_type="Iter"
od_wait=80
use_best_model=True
eval_metric="AUC"
```

Зачем: это дало лучший AUC среди быстрых одиночных конфигураций и быстрее/проще ensemble.

### 4. Frequency ensemble

Проверили average/rank-average нескольких CatBoost профилей:

- `sqrt_bayes_d6`
- `sqrt_d6_l2_12`
- `sqrt_d5_l2_10`
- `scale30_d5`

Результат:

- лучший single model: ROC-AUC `0.6836`
- best average ensemble: ROC-AUC около `0.6829`

Вывод: ensemble не дал прироста выше лучшей одиночной модели. Не добавляли в production pipeline.

## Что изменено в коде

- `pipeline/risk_encoding.py`: новый модуль OOF/stat risk encoding.
- `pipeline/runner.py`: подключение risk encoding после train/valid split и до `build_feature_lists`.
- `scripts/train_pipeline.py`: новые флаги:
  - `--enable-risk-encoding`
  - `--enable-id-risk-encoding`
  - `--risk-encoding-splits`
  - `--risk-encoding-smoothing`
- `pipeline/models.py`: CatBoost classifier tuned params и `eval_set` для early stopping.
- `pipeline/contracts.py`: helper ID mode columns для optional ID risk encoding.
- `pipeline/config.py`: ID helper columns добавлены в leakage exclusions.

## Текущий вывод

До `0.70` пока не дошли. Реалистичный стабильный уровень после быстрых экспериментов: `0.68-0.684` ROC-AUC.

Самый полезный режим для дальнейшего сравнения:

```bash
uv run python scripts/train_pipeline.py \
  --model-backend catboost \
  --enable-risk-encoding
```

Последний полный прогон этого режима:

- ROC-AUC: `0.6821`
- Gini: `0.3643`
- PR-AUC: `0.0469`
- F1: `0.0974`
- severity R2 positive: `0.0192`
- group 1: `21,230` полисов, `64.67%`
- post-pricing loss ratio: `0.6997`
- group loss ratio gap: `0.0006`

Не использовать `--enable-id-risk-encoding` как default: он ухудшил validation AUC.

## Следующие кандидаты для AUC 0.70+

1. Более широкий Optuna/random search именно для frequency CatBoost.
2. Проверка CV вместо одного split: возможно текущий validation split шумный.
3. Новые hand-crafted frequency features по driver/vehicle/score extremes.
4. Отдельная модель на score-only и rank blending, если score groups несут независимый сигнал.
5. Проверка temporal split: если `operation_month` даёт нестабильность, стоит валидировать стратегию split.
