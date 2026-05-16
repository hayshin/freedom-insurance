# Итоговый отчет: результаты и сабмит

## Краткое резюме

Проект решает задачу скоринга убыточности ОГПО и формирования справедливой цены полиса. Используется двухступенчатая модель frequency-severity, риск-энкодинг без ID, калибровка тяжести выплат и калибровка pricing под целевой коэффициент выплат около 70%.

Ключевые результаты:

- Holdout (ноябрь-декабрь 2022, untouched): ROC-AUC 0.6787, GINI 0.3575, post-pricing loss ratio 0.6655.
- Production OOF (калибровка на всех train): ROC-AUC 0.6835, GINI 0.3670, post-pricing loss ratio 0.6985.
- Ограничения по премии соблюдены: new_premium в диапазоне 0.90x .. 3.00x от исходной premium.

## Использованная конфигурация

- Модель: CatBoost (frequency) + CatBoost (severity).
- Target severity: claim_per_premium.
- Risk encoding: включен, без ID признаков.
- Количество признаков: 301 (из них 17 категориальных).
- Разбиение для holdout: time_month_1_8__9_10__11_12.
- Production калибровка: OOF 5 folds.

## Метрики на holdout (untouched)

Frequency:

- ROC-AUC: 0.6787
- GINI: 0.3575
- PR-AUC: 0.0502
- F1: 0.1001 (threshold 0.2402)

Severity (positive):

- MAE: 480,862
- RMSE: 689,695
- R2: 0.0163

Business:

- Baseline LR (premium): 0.9053
- Baseline LR (premium_wo_term): 1.1352
- Post-pricing LR: 0.6655
- Keep/Decrease share: 0.7276
- Increase share: 0.2724
- Group LR gap: 0.1967
- Min/Max new-to-old ratio: 1.00 / 3.00

## Метрики production OOF (калибровка на train)

Frequency:

- ROC-AUC: 0.6835
- GINI: 0.3670
- PR-AUC: 0.0411
- F1: 0.0849 (threshold 0.2044)

Severity (positive):

- MAE: 528,440
- RMSE: 973,618
- R2: 0.0098

Business:

- Baseline LR (premium): 0.9957
- Baseline LR (premium_wo_term): 1.2334
- Post-pricing LR: 0.6985
- Keep/Decrease share: 0.5012
- Increase share: 0.4988
- Group LR gap: 0.0079
- Min/Max new-to-old ratio: 0.90 / 3.00

## Формат сабмита

Файл: submissions/submission.csv

Колонки и порядок:

- contract_number
- claim_probability
- pred_loss_ratio
- new_premium

Проверки:

- Нет NaN/inf в вероятностях и loss ratio.
- new_premium >= 0.
- new_premium <= 3 * premium.
- Количество строк равно числу контрактов test.csv.
