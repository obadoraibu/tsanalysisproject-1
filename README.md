# Scaling Impact Study

**Автор**: Завьялов Егор Сергеевич

## Гипотеза
Нормализация временных рядов особенно важна для
глобальных нейросетевых моделей, тогда как для моделей других классов её влияние может быть
слабее и неоднозначнее

## Data
- M4 Competition monthly subset (300 series)
- Input: 24 steps, Horizon: 18 steps

## Models
- **Baselines**: Naive, Seasonal Naive, Theta, ETS
- **Main**: CatBoost, LSTM
- **Scaling**: none, standard, robust, quantile

## Usage
```bash
pip install -r requirements.txt
python run_experiment.py
```

## Key Results
| Model | sMAPE (%) |
|-------|-----------|
| auto_ets | 11.62 |
| catboost_standard | 12.51 |
| lstm_robust | 13.77 |
| lstm_none | 154.93 |
