import pandas as pd
import numpy as np
from linearmodels import PanelOLS
import shap 
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

df = pd.read_csv("sdg3_panel_model.csv")

print(f"Panel shape: {df.shape}")
print(df.isna().mean().round(2).sort_values(ascending=False))
df['log_under5'] = np.log(df['under5_mortality'].where(df['under5_mortality'] > 0))

outcome = 'log_under5'
predictors = ['log_gdp_per_capita', 'female_secondary_enrol', 'urban_pop_pct', 'health_expenditure_pct']

d = df.dropna(subset=[outcome] + predictors).copy()
print(len(d), d['iso3'].nunique(), 'countries')

d = d.set_index(['iso3', 'year'])

y = d[outcome]
X = d[predictors]

model = PanelOLS(y, X, entity_effects=True, time_effects=True)
res = model.fit(cov_type='clustered', cluster_entity=True)
#print(res.summary)

pooled = PanelOLS(y, X, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(pooled.summary)

feats = ["log_gdp_per_capita", "health_expenditure_pct", "urban_pop_pct", "female_secondary_enrol"]

d2 = df.dropna(subset=['log_under5'] + feats).reset_index(drop=True)
X2 = d2[feats]
y2 = d2['log_under5']

model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)

pred = cross_val_predict(model, X2, y2, groups=d2["iso3"], cv=GroupKFold(n_splits=5))
print("R2:", round(r2_score(y2, pred), 3))
print("MAE (log points):", round(mean_absolute_error(y2, pred), 3))

lin_pred = cross_val_predict(LinearRegression(), X2, y2, groups=d2["iso3"], cv=GroupKFold(n_splits=5))
print("Linear Model R2:", round(r2_score(y2, lin_pred), 3))
print("Linear Model MAE (log points):", round(mean_absolute_error(y2, lin_pred), 3))

model.fit(X2, y2)
sv = shap.TreeExplainer(model).shap_values(X2)
shap.summary_plot(sv, X2, show=False)
plt.savefig("shap_summary.png", dpi=200, bbox_inches='tight')
plt.close()

shap.summary_plot(sv, X2, plot_type="bar", show=False)
plt.savefig("shap_summary_bar.png", dpi=200, bbox_inches='tight')
plt.close()

