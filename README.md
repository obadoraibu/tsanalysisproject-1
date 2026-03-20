# Scaling Impact Study

**Автор**: Завьялов Егор Сергеевич

## Гипотеза
Нормализация временных рядов особенно важна для
глобальных нейросетевых моделей, тогда как для моделей других классов её влияние может быть
слабее и неоднозначнее

## Данные
- M4 Competition monthly subset (300 series)
- Input: 24 steps, Horizon: 18 steps

## Модели
- **Baselines**: Naive, Seasonal Naive, Theta, ETS
- **Main**: CatBoost, LSTM
- **Scaling**: none, standard, robust, quantile

## Запуск
```bash
pip install -r requirements.txt
python run_experiment.py
```
