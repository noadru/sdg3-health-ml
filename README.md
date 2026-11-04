# sdg3-health-mlSDG 3: Which countries will reach the child-survival target by 2030?

Analysis of under-5 mortality (SDG indicator 3.2.1) across ~200 countries, 2015-2024,
combining UN SDG API data with World Bank economic indicators.

## Methods

- Data pipeline: UN SDG API + World Bank API, merged on ISO3 country code and year
- Panel regression with country and year fixed effects (linearmodels), compared against a pooled model
- XGBoost vs. linear baseline, evaluated with country-grouped cross-validation; SHAP for interpretation
- 2030 projection: country-level log-linear trend vs. the target of 25 deaths per 1,000 live births

## Key findings

- 133 countries have already met the target, 5 are on track, and 61 are off track,
  within one country per category of the UN's own classification (SDG Report 2026)
- Across countries, GDP per capita is strongly associated with under-5 mortality;
  within countries over 2015-2024, effects are small and imprecisely estimated
- A linear baseline matched XGBoost (R² 0.79 vs 0.78), so SHAP adds interpretation rather than accuracy

## Run it

pip install -r requirements.txt
python sdg3_pipeline.py
python sdg3_analysis.py

Mac users: XGBoost needs `brew install libomp`.

## Limitations

Associations, not causal effects. Ten years of data. Forecasts assume past trends continue.
