# Отчет: моделирование и pricing

## Архитектура модели

Используется frequency-severity подход:

1. Frequency: вероятность выплаты (is_claim).
2. Severity: ожидаемая величина выплаты при наличии claims.
3. Expected claim = P(claim) * Severity.
4. Pricing калибруется на целевой loss ratio около 0.70.

## Frequency модель

- CatBoostClassifier с авто-весами SqrtBalanced.
- Калибровка не требуется, используется probability ranking.
- Risk encoding (non-ID) включен для улучшения ROC-AUC.

## Severity модель

- CatBoostRegressor по target claim_per_premium.
- Калибровка через линейное преобразование на positive-валидации.
- Это стабилизирует R2 positive и уменьшает смещение на редких крупных выплатах.

## Pricing калибровка

- Калибровка loss ratio проводится на калибровочном сплите.
- Для production используется OOF-оценка на всех train контрактах, чтобы избежать in-sample bias.
- Целевая метрика: общий loss ratio около 0.70 и минимальный gap между группами.

## OOF production протокол

- 5-fold StratifiedKFold по is_claim.
- На каждом fold строятся OOF вероятности и severity.
- После калибровки pricing финальные модели обучаются на всем train.
- Сабмит строится на test.csv.

## Итоги выбора модели

Ключевые решения:

- Включен non-ID risk encoding, ID risk encoding отключен.
- Использован CatBoost для обеих задач как лучший баланс по качеству и стабильности.
- Оценка по времени (operation_month) дала честные holdout метрики.
- Итоговая конфигурация выбрана по бизнес-цели: loss ratio и доля keep/decrease.
