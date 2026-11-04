import json
import os
import time

import numpy as np 
import pandas as pd
import pycountry
import requests

UN_URL = "https://unstats.un.org/sdgs/UNSDGAPIV5/v1/sdg/Series/Data"
WB_URL = "https://api.worldbank.org/v2/country/all/indicator/{code}"
START_YEAR = 2015
END_YEAR = 2024
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

UN_SERIES = {
    "under5_mortality":      ("SH_DYN_MORT", "BOTHSEX", None),
    "neonatal_mortality":    ("SH_DYN_NMRT", "BOTHSEX", None),
    "maternal_mortality":    ("SH_STA_MORT", None, None),
    "skilled_birth_attend":  ("SH_STA_BRTC", None, None),
    "dtp3_immunization":     ("SH_ACS_DTP3", None, None),
}

WB_SERIES = {
    "gdp_per_capita_ppp":     "NY.GDP.PCAP.PP.KD",
    "health_expenditure_pct": "SH.XPD.CHEX.GD.ZS",
    "urban_pop_pct":          "SP.URB.TOTL.IN.ZS",
    "female_secondary_enrol": "SE.SEC.ENRR.FE",
    "population":             "SP.POP.TOTL",
}

KEYS = ['country', 'country_code', 'year']

TARGETS_2030 = {
    "under5_mortality": 25,
    "neonatal_mortality": 12,
    "maternal_mortality": 70,
}

def get_with_retry(url, params, tries=3):
    for attempt in range(tries):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f" attemps {attempt + 1} failed: {e}")
            if attempt < tries - 1:
                time.sleep(5)
    raise RuntimeError(f"Failed to fetch {url} after {tries} attempts.")

def probe_series(series_code):
    params = {
        "seriesCode": series_code,
        "timePeriodStart": START_YEAR,
        "timePeriodEnd": END_YEAR,
        "pageSize": 5,
    }
    payload = get_with_retry(UN_URL, params)
    return payload.get("totalElements", 0)

def fetch_series(series_code, page_size=1000):
    all_rows, page = [], 1
    while True:
        params = {
            "seriesCode": series_code,
            "timePeriodStart": START_YEAR,
            "timePeriodEnd": END_YEAR,
            "pageSize": page_size,
            "page": page,
        }
        payload = get_with_retry(UN_URL, params)
        all_rows.extend(payload['data'])
        if page >= payload["totalPages"]:
            break
        page += 1
    return all_rows

def fetch_series_cached(series_code):
    path = os.path.join(CACHE_DIR, f"{series_code}_raw.json")
    if os.path.exists(path):
        with open (path) as f:
            return json.load(f)
    rows = fetch_series(series_code)
    with open(path, "w") as f:
        json.dump(rows, f)
    return rows 

def flatten(rows):
    records = []
    for d in rows:
        dims = d.get("dimensions") or {}
        attrs = d.get("attributes") or {}
        records.append({
            "series": d["series"],
            "country": d["geoAreaName"],
            "country_code": d["geoAreaCode"],
            "year": d["timePeriodStart"],
            "value": d["value"],
            "sex": dims.get("Sex"),
            "age": dims.get("Age"),
            "reporting_type": dims.get("Reporting Type"),
            "nature": attrs.get("Nature"),
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df["vlaue"] = pd.to_numeric(df["value"], errors="coerce")
        df["year"] = pd.to_numeric(df['year'], errors="coerce").astype("Int64")
    return df

def pick(df, name, sex=None, age=None, aggregate=False):
    out = df[df['nature'] == 'N'] if aggregate else df[df['nature'] != "N"]
    if sex:
        out = out[out['sex'] == sex]
    if age:
        out = out[out['age'] == age]
    out = out[KEYS + ['value']].rename(columns={"value": name})
    n_dup = out.duplicated(subset=KEYS).sum()
    if n_dup:
        print(f" not: {name}: {n_dup} duplicate country-year rows, keeping first")
        out = out.drop_duplicates(subset=KEYS)
    return out

def to_iso3(code):
    try:
        match = pycountry.countries.get(numeric=str(code).zfill(3))
    except (KeyError, LookupError):
        match = None 
    return match.alpha_3 if match else None

def build_un_panel(aggregate=False):
    parts = []
    for name, (code, sex, age) in UN_SERIES.items():
        raw = fetch_series_cached(code)
        if not raw:
            continue
        df = flatten(raw)
        part = pick(df, name, sex, age, aggregate=aggregate)
        print(f" {name}: {len(part)} rows, {part['country'].nunique()} areas")
        parts.append(part)

        wide = parts[0]
        for p in parts[1:]:
            wide = wide.merge(p, on=KEYS, how="outer")
        return wide


def fetch_wb(code, name):
    path = os.path.join(CACHE_DIR, f"WB_{code}_raw.json")
    if os.path.exists(path):
        with open(path) as f:
            rows = json.load(f)
    else:
        params = {'format': "json", "per_page": 20000, "date": f"{START_YEAR}:{END_YEAR}"}
        payload = get_with_retry(WB_URL.format(code=code), params)
        rows = payload[1] if len(payload) > 1 and payload[1] else []
        with open(path, "w") as f:
            json.dump(rows, f)
    df = pd.DataFrame([
            {'iso3': r['countryiso3code'], 'year': int(r['date']), name: r['value']}
            for r in rows if r.get("countryiso3code")
            ])
    print(f". {name}: {df[name].notna().sum()} non-null values")
    return df
    
def build_wb_panel():
    wide = None
    for name, code in WB_SERIES.items():
        part = fetch_wb(code, name)
        wide = part if wide is None else wide.merge(part, on=['iso3', 'year'], how='outer')
    return wide

def main():
    print("Probing UN series codes...")
    for name, (code, _, _) in UN_SERIES.items():
        n = probe_series(code)
        flag = 'OK ' if n else "EMPTY -- check this series code"
        print(f" {code:<14} {name:<22} {n:>7} rows {flag}")

    print("\nBuilding UN country panel...")
    un = build_un_panel(aggregate=False)
    un['iso3'] = un['country_code'].apply(to_iso3)

    print(f"countries without an ISO3 match: {un[un['iso3'].isna()]['country'].nunique()}")
    print(f"sample: {sorted(un[un['iso3'].isna()]['country'].unique())[:15]}")
    un = un.dropna(subset=['iso3'])
    
    print("\nFetching World Bank covariates...")
    wb = build_wb_panel()

    panel = un.merge(wb, on=['iso3', 'year'], how='left')

    panel = panel.sort_values(['iso3', 'year'])
    panel['female_secondary_enrol'] = (
        panel.groupby("iso3")["female_secondary_enrol"]
        .transform(lambda s: s.interpolate(limit_area='inside'))
    )

    gdp = panel['gdp_per_capita_ppp']
    panel['log_gdp_per_capita'] = np.log(gdp.where(gdp > 0))


    print("\nPanel summary")
    print(f"  rows: {len(panel)}  countries: {panel['iso3'].nunique()}  years: {panel['year'].min()}-{panel['year'].max()}")
    print("non-null counts:")
    print(panel.drop(columns=['country', 'country_code', 'iso3']).count().to_string())

    panel.to_csv("sdg3_panel_model.csv", index=False)

    labels = {
        "country": "Country or territory",
        "country_code": "UN Country code",
        "iso3": "ISO country code (3-letter)",
        "year": "Year",
        "under5_mortality": "Under-5 mortality rate (per 1,000 live births)",
        "neonatal_mortality": "Neonatal mortality rate (per 1,000 live births)",
        "maternal_mortality": "Maternal mortality ratio (per 100,000 live births)",
        "skilled_birth_attend": "Births attended by skilled health personnel (%)",
        "dtp3_immunization": "DTP3 immunization coverage (% of 1-year-olds)",
        "gdp_per_capita_ppp": "GDP per capita, PPP (constant intl $)",
        "health_expenditure_pct": "Health expenditure (% of GDP)",
        "urban_pop_pct": "Urban population (% of total)",
        "female_secondary_enrol": "Female secondary school enrollment (gross %)",
        "population": "Population",
        "log_gdp_per_capita": "Log GDP per capita, PPP",
    }

    panel.rename(columns=labels).to_csv("sdg3_countries_for_tableau.csv", index=False)

    print("\nBuilding regional aggregates for Tableau...")
    regions = build_un_panel(aggregate=True)
    regions.rename(columns={**labels, 'country':  "Region or group"}).to_csv(
        "sdg3_regions_for_tableau.csv", index = False
    )


if __name__ == "__main__":
    main()