# Premium и premium_wo_term: выводы по loss ratio

## Summary
На уровне полиса премии дают сильный сигнал для frequency (особенно `premium_wo_term`), а сегментация по премии показывает резко падающий loss_ratio при росте премии, но `premium_wo_term` имеет высокий leakage-риск для loss_ratio.

## Correlations
- `is_claim`: `premium` Pearson $0.063$, Spearman $0.064$; `premium_wo_term` Pearson $0.087$, Spearman $0.087$ — слабая положительная связь сохраняется на contract-level.
- `claim_amount`: `premium` Pearson $-0.032$, Spearman $-0.046$; `premium_wo_term` Pearson $-0.024$, Spearman $-0.028$ — слабая отрицательная связь, эффект небольшой.
- `loss_ratio`: `premium` Spearman $-0.420$, `premium_wo_term` Spearman $-0.479$ — сильная отрицательная связь, но частично механическая из-за знаменателя.

## Feature quality
- IV (is_claim): `premium` $0.227$ (средний сигнал), `premium_wo_term` $0.532$ (сильный сигнал).
- Missingness: 0% по обоим признакам.
- Stability: сегменты показывают устойчиво убывающий loss_ratio при росте премии; монотонность Spearman: `premium` $-0.915$, `premium_wo_term` $-1.000$.
- Leakage risk: `premium_wo_term` — **high** для loss_ratio (знаменатель), `premium` — **possible** (через тариф и связь с loss_ratio), для frequency допустим.

## Business impact
- Рост премии связан с более низким loss_ratio и более низкой долей убыточных полисов; сильный эффект в нижних бинах премии, где loss_ratio экстремален.
- Для repricing логично использовать премию как экспозицию и контролировать риск, но избегать прямого таргетинга на loss_ratio через `premium_wo_term`.
- В портфеле общий loss_ratio $\approx 1.233$ — высоко убыточный уровень, что подчеркивает необходимость корректировок цены, но с учетом ограничения leakage.

## Engineering ideas
- `premium_wo_term_ratio = premium_wo_term / premium` как индикатор расторжений (в данных много единиц и нулей).
- Лог-преобразования `log1p_premium`, `log1p_premium_wo_term` для стабильного биннинга и регуляризации.
- Нормализация риска: `claim_rate` и `claim_amount_mean` по бинам премии как признаки для pricing tiers.

## Caveats
- Экстремальные loss_ratio в низких бинах `premium_wo_term` могут быть обусловлены почти нулевым знаменателем (механический эффект), нужна фильтрация по минимальной премии.
- Сильный IV для `premium_wo_term` может быть завышен из-за включенности в расчет loss_ratio и косвенной связи с таргетами.
- В нижних бинах премии малый объем полисов усиливает нестабильность метрик.

## Next steps
1. Проверить устойчивость сигналов по временным срезам и по subset без нулевых/минимальных `premium_wo_term`.
2. Оценить влияние `premium_wo_term_ratio` на frequency отдельно от исходных премий.
3. Пересчитать сегментацию с winsorization и минимальным порогом премии, чтобы убрать эффект нулевого знаменателя.
