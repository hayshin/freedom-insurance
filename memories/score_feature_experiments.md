# SCORE feature experiments

Дата: 2026-05-16

## Контекст

Цель эксперимента - проверить гипотезу, что `features/score.py` неправильно хендлит `SCORE_*` признаки и из-за этого frequency ROC-AUC не доходит до `0.70+`.

Текущий baseline перед изменениями:

- default CatBoost без risk encoding: ROC-AUC `0.679615`, Gini `0.359231`, PR-AUC `0.047396`, F1 `0.101389`.
- старый лучший режим из `memories/frequency_auc_experiments.md` с `--enable-risk-encoding`: ROC-AUC примерно `0.6821-0.6824`.

Валидация считается на holdout split внутри `dataset/train.csv`, потому что в `dataset/test.csv` нет истинного `is_claim`/`claim_amount`. `dataset/test.csv` используется пайплайном для генерации submission, но AUC на нем локально посчитать нельзя.

## Что показал анализ SCORE

В `dataset/train.csv` найдено `128` колонок `SCORE_*`.

Сырые score-сигналы есть, но они слабые:

- лучший одиночный contract-level агрегат: `SCORE_2_1_max`, AUC около `0.5474`;
- сильные группы по одиночным агрегатам: `SCORE_8`, `SCORE_2`, `SCORE_10`, частично `SCORE_6`, `SCORE_7`, `SCORE_4`;
- направления риска разные: например, часть `SCORE_2`, `SCORE_6`, `SCORE_7` инвертирована относительно `is_claim`;
- текущие compact group-level фичи из `features/score.py` сами по себе дают score-only CatBoost AUC около `0.5948`;
- полный набор детальных `SCORE_* x mean/min/max` дает score-only AUC около `0.5991`, но `current + all detail` падает до `0.5917`, то есть широкий набор добавляет много шума.

Вывод: score-колонки не бесполезны, но это слабые массовые сигналы. Они не выглядят как источник скачка до `0.70+` AUC сами по себе.

## Что изменено

В `features/score.py` оставлены прежние compact score features и добавлен только узкий детальный набор для группы `SCORE_8`:

- `score_8_1_mean`, `score_8_1_min`, `score_8_1_max`;
- `score_8_2_mean`, `score_8_2_min`, `score_8_2_max`;
- `score_8_3_mean`, `score_8_3_min`, `score_8_3_max`.

Более широкий curated-набор по группам `2/4/6/7/8/10/11` проверялся, но не оставлен в финальном коде: он поднял размер feature matrix до `401` признака и не улучшил ROC-AUC.

`pipeline/risk_encoding.py` в финальном коде не расширен новыми score bins. Попытка добавить bins по `score_g8`, `score_g2`, `score_g6`, `score_g10`, `score_g11_missing_rate` ухудшила результат risk-encoding режима.

## Результаты полных прогонов

Все прогоны обучали full pipeline и генерировали submission по `dataset/test.csv` в `/tmp`.

### Широкий detail SCORE набор

Команда:

```bash
devenv shell uv run python scripts/train_pipeline.py \
  --model-backend catboost \
  --artifacts-dir /tmp/ogpo-score-artifacts \
  --submission /tmp/ogpo-score-submission.csv \
  --quiet
```

Результат:

- ROC-AUC `0.679562`
- Gini `0.359125`
- PR-AUC `0.047799`
- F1 `0.103435`
- features `401`

Вывод: AUC практически не изменился против baseline, широкий detail-набор не стоит оставлять.

### Широкий detail SCORE + расширенный score risk encoding

Команда:

```bash
devenv shell uv run python scripts/train_pipeline.py \
  --model-backend catboost \
  --enable-risk-encoding \
  --artifacts-dir /tmp/ogpo-score-risk-artifacts \
  --submission /tmp/ogpo-score-risk-submission.csv \
  --quiet
```

Результат:

- ROC-AUC `0.682029`
- Gini `0.364057`
- PR-AUC `0.048075`
- F1 `0.104269`
- features `519`
- risk encoding features `96`

Вывод: не лучше старого risk-encoding ориентира; расширенные score bins добавляют шум.

### Финальный узкий SCORE_8 detail без risk encoding

Команда:

```bash
devenv shell uv run python scripts/train_pipeline.py \
  --model-backend catboost \
  --artifacts-dir /tmp/ogpo-score8-artifacts \
  --submission /tmp/ogpo-score8-submission.csv \
  --quiet
```

Результат:

- ROC-AUC `0.680608`
- Gini `0.361217`
- PR-AUC `0.046763`
- F1 `0.096894`
- features `207`
- keep/decrease share `0.711331`
- post-pricing loss ratio `0.705847`

Вывод: это лучший default/no-risk вариант в этой серии. Он дает небольшой ROC-AUC прирост против baseline `0.679615 -> 0.680608`, но не решает задачу `0.70+`.

### Финальный узкий SCORE_8 detail + старый risk encoding

Команда:

```bash
devenv shell uv run python scripts/train_pipeline.py \
  --model-backend catboost \
  --enable-risk-encoding \
  --artifacts-dir /tmp/ogpo-score8-orig-risk-artifacts \
  --submission /tmp/ogpo-score8-orig-risk-submission.csv \
  --quiet
```

Результат:

- ROC-AUC `0.681883`
- Gini `0.363767`
- PR-AUC `0.046013`
- F1 `0.099749`
- features `301`
- risk encoding features `78`

Вывод: лучше default baseline, но хуже старого лучшего risk-encoding режима. Для максимального validation AUC на текущем split старый `--enable-risk-encoding` без новых score detail может оставаться предпочтительнее.

## Итог

Ответ на вопрос "вообще никакие score колонки не дают пользы?": дают, но мало.

Полезнее всего выглядит `SCORE_8`: узкие детальные агрегаты по `SCORE_8_1..3` дали небольшой прирост default CatBoost ROC-AUC. Остальные score-группы несут слабый и разнонаправленный сигнал; широкий набор детальных score-фичей ухудшает или не улучшает ranking из-за шума.

Практический вывод:

- оставить `SCORE_8` detail как небольшой default improvement;
- не расширять risk encoding score bins без отдельного отбора;
- не ждать от `SCORE_*` одних скачка до `0.70+`;
- следующий реалистичный путь к `0.70+` - не score feature engineering, а более широкий frequency tuning/CV, rank blending отдельных моделей или новая семья признаков вне `SCORE_*`.
