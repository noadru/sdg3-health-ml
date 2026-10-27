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
            response = requests = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json
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
    out = df[df['nature'] == 'N'] if aggregate else df[df['nature' != "N"]]
    if sex:
        out = out[out['sex'] == sex]
    if age:
        out = out[out['age'] == age]
    out = out[KEYS + ['value']].rename(cloumns={"value": name})
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
