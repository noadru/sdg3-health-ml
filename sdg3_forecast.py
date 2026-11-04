import numpy as np
import pandas as pd

df = pd.read_csv("sdg3_panel_model.csv")

TARGET = 25
TARGET_YEAR = 2030

out = []
for iso, g in df.dropna(subset=['under5_mortality']).groupby('iso3'):
    if g['year'].nunique() < 5:
        continue

    slope, intercept = np.polyfit(g['year'], np.log(g['under5_mortality']), 1)
    last = g.sort_values('year').iloc[-1]
    proj = np.exp(slope * TARGET_YEAR + intercept)

    if last['under5_mortality'] <= TARGET:
        status = "Target already met"
    elif proj <= TARGET:
        status = "On track"
    else:
        status = "Off track"

    out.append({
        "country": g["country"].iloc[0],
        "iso3": iso,
        "latest_year": int(last["year"]),
        "latest_value": round(last["under5_mortality"], 1),
        "annual_decline_pct": round((1 - np.exp(slope)) * 100, 2),
        "projected_2030": round(proj, 1),
        "status": status,
    })

ontrack = pd.DataFrame(out).sort_values("projected_2030", ascending=False)
print(ontrack['status'].value_counts())
ontrack.to_csv("sdg3_on_track.csv", index=False)