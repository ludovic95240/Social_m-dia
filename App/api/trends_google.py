# trends_google.py — Robust: chemins absolus, création des dossiers, guard clauses, warning pytrends
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import time, datetime as dt, json, warnings

import pandas as pd
from pytrends.request import TrendReq
from pytrends import exceptions as ptexc

# --- Optionnel: masquer le FutureWarning interne à pytrends (fillna downcasting)
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*pytrends.*")

# === Helpers chemins ===
HERE = Path(__file__).resolve()
# App/
APP_ROOT = HERE.parents[1] if HERE.name.lower().endswith(".py") else HERE.parent
# API root (dossier où tu veux écrire les csv)
API_DIR = APP_ROOT / "api"
DATA_DIR = API_DIR / "data" / "processed"
OUT_PREFIX = DATA_DIR / "google_trends"  # on écrira _time.csv, _cities.csv, etc.

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def six_months_range_yyyy_mm_dd() -> str:
    end = dt.date.today()
    start = end - dt.timedelta(days=182)
    return f"{start:%Y-%m-%d} {end:%Y-%m-%d}"

@dataclass
class TrendsConfig:
    hl: str = "fr-FR"
    tz: int = 120       # Europe/Paris (UTC+2 l’été)
    retries: int = 4
    backoff_s: float = 2.0
    sleep_between_calls: float = 1.5

class GoogleTrendsClient:
    def __init__(self, cfg: TrendsConfig):
        self.cfg = cfg
        self._new_session()

    def _new_session(self):
        self.tr = TrendReq(
            hl=self.cfg.hl, tz=self.cfg.tz,
            timeout=(10, 25), retries=0, backoff_factor=0.0,
            requests_args={"headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            }}
        )

    def _safe_build(self, kw_list: List[str], timeframe: str, geo: str, cat: int = 0, gprop: str = ""):
        last = None
        for i in range(self.cfg.retries):
            try:
                self.tr.build_payload(kw_list, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
                return
            except ptexc.ResponseError as e:
                last = e
                self._new_session()
                time.sleep(self.cfg.backoff_s * (2 ** i))
        # fallback si pytrends devient capricieux avec > 2 mots-clés
        if len(kw_list) > 2:
            mid = len(kw_list) // 2
            self._safe_build(kw_list[:mid], timeframe, geo, cat, gprop)
            self._safe_build(kw_list[mid:], timeframe, geo, cat, gprop)
        else:
            raise last or RuntimeError("build_payload failed")

    def interest_over_time(self, kw_list: List[str], timeframe: str, geo: str = "FR") -> pd.DataFrame:
        self._safe_build(kw_list, timeframe=timeframe, geo=geo)
        time.sleep(self.cfg.sleep_between_calls)
        df = self.tr.interest_over_time()
        if df is None or df.empty:
            return pd.DataFrame(columns=["date"] + kw_list)
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        return df.reset_index(names="date")

    def interest_by_city(self, kw_list: List[str], timeframe: str, geo: str = "FR") -> pd.DataFrame:
        self._safe_build(kw_list, timeframe=timeframe, geo=geo)
        time.sleep(self.cfg.sleep_between_calls)
        df = self.tr.interest_by_region(resolution="CITY", inc_low_vol=True)
        return df if df is not None else pd.DataFrame()

    def related(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Return {kw: {'top':df, 'rising':df}} pour le dernier payload construit"""
        rel = self.tr.related_queries() or {}
        out = {}
        for kw, d in rel.items():
            out[kw] = {}
            if d and d.get("top") is not None:
                out[kw]["top"] = d["top"]
            if d and d.get("rising") is not None:
                out[kw]["rising"] = d["rising"]
        return out

class GoogleTrendsPipeline:
    def __init__(self, client: GoogleTrendsClient, out_prefix: Path, cities_filter: List[str]):
        self.client = client
        self.out_prefix = out_prefix  # Path sans extension
        self.cities_filter = cities_filter
        # créer le dossier une fois pour toutes
        ensure_parent(self.out_prefix.with_suffix(".csv"))

    def run(self, brand_kw: List[str], comp_kw: List[str], timeframe: str):
        # 1) Time series (brand + competitors)
        all_kw = list(dict.fromkeys(brand_kw + comp_kw))
        df_time = self.client.interest_over_time(all_kw, timeframe=timeframe, geo="FR")

        # Si vide, on sort proprement
        time_csv = self.out_prefix.parent / f"{self.out_prefix.name}_time.csv"
        time_smooth_csv = self.out_prefix.parent / f"{self.out_prefix.name}_time_smooth.csv"
        ensure_parent(time_csv)

        if df_time.empty:
            # écrire des CSV vides mais structurés (utile pour la suite du pipeline)
            pd.DataFrame(columns=["date"] + all_kw).to_csv(time_csv, index=False)
            pd.DataFrame(columns=["date"] + all_kw).to_csv(time_smooth_csv, index=False)
            print("[WARN] Série temporelle vide (pytrends). CSVs vides écrits →", time_csv)
        else:
            # smoothing 7 jours
            time_cols = [c for c in df_time.columns if c != "date"]
            df_time_smooth = df_time.copy()
            if time_cols:
                df_time_smooth[time_cols] = df_time_smooth[time_cols].rolling(7, min_periods=3).mean()
            df_time.to_csv(time_csv, index=False)
            df_time_smooth.to_csv(time_smooth_csv, index=False)
            print("[OK] Série temporelle →", time_csv)

        # 2) By city (brand only)
        df_city = self.client.interest_by_city(brand_kw, timeframe=timeframe, geo="FR")
        cities_csv = self.out_prefix.parent / f"{self.out_prefix.name}_cities.csv"
        ensure_parent(cities_csv)
        if df_city is None or df_city.empty:
            pd.DataFrame().to_csv(cities_csv)
            print("[WARN] Villes: dataframe vide. CSV vide écrit →", cities_csv)
        else:
            # filtrage sur liste fournie
            df_city = df_city.loc[df_city.index.isin(self.cities_filter)]
            df_city.to_csv(cities_csv)
            print("[OK] Villes →", cities_csv)

        # 3) Related queries/topics (use the last built payload context)
        related = self.client.related()
        rel_dir = self.out_prefix.parent / f"{self.out_prefix.name}_related"
        rel_dir.mkdir(parents=True, exist_ok=True)

        # JSON global
        (rel_dir / "related.json").write_text(
            json.dumps(
                {k: {kk: v.to_dict(orient="records") for kk, v in d.items()} for k, d in related.items()},
                ensure_ascii=False, indent=2
            ),
            encoding="utf-8"
        )
        # CSV par mot-clé
        for kw, d in related.items():
            if "top" in d and isinstance(d["top"], pd.DataFrame):
                d["top"].to_csv(rel_dir / f"{kw}_top.csv", index=False)
            if "rising" in d and isinstance(d["rising"], pd.DataFrame):
                d["rising"].to_csv(rel_dir / f"{kw}_rising.csv", index=False)
        print("[OK] Related queries/topics →", rel_dir)

if __name__ == "__main__":
    # --- Paramètres par défaut (à adapter) ---
    brand_kw = ["Honda Moto France", "Honda Transalp", "Honda Africa Twin", "Honda CB500X"]
    competitors_kw = ["Yamaha", "BMW Motorrad", "KTM", "Triumph", "Ducati"]
    timeframe = six_months_range_yyyy_mm_dd()
    cities = ["Paris","Lyon","Marseille","Toulouse","Bordeaux","Lille","Nantes","Rennes","Nice","Strasbourg"]

    cfg = TrendsConfig()
    client = GoogleTrendsClient(cfg)
    pipeline = GoogleTrendsPipeline(client, out_prefix=OUT_PREFIX, cities_filter=cities)
    pipeline.run(brand_kw, competitors_kw, timeframe=timeframe)
