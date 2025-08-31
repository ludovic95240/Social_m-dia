# x_collect_only.py
from __future__ import annotations
import os, time, csv, sys
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Optional
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv


# ============== Utils ==============
def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def iso_6months() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=180))\
        .isoformat(timespec="seconds").replace("+00:00", "Z")


# ============== Config ==============
@dataclass
class XConfig:
    bearer: str
    base: str = "https://api.x.com/2"
    page_size: int = 100            # 10..100
    sleep_s: float = 0.2            # entre pages
    max_pages: int = 60             # sécurité
    hard_limit_rows: int = 20000    # sécurité
    out_csv: str = "data/raw/x_tweets.csv"

    @staticmethod
    def from_env() -> "XConfig":
        load_dotenv()
        b = os.getenv("X_BEARER_TOKEN", "")
        assert b, "X_BEARER_TOKEN manquant (.env)"
        return XConfig(bearer=b)


# ============== Client API ==============
class XClient:
    """
    Client minimal API v2 /tweets/search/recent (ou /search/all si tu as l'historique).
    Gère 429 (rate-limit) avec backoff exponentiel.
    """
    def __init__(self, cfg: XConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {cfg.bearer}"})

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.cfg.base}/{path.lstrip('/')}"
        backoff = 1.0
        for attempt in range(7):
            r = self.session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                # rate limit → on lit reset si dispo, sinon backoff exponentiel
                reset = r.headers.get("x-rate-limit-reset")
                if reset:
                    wait = max(0, int(reset) - int(time.time())) + 1
                else:
                    wait = backoff
                    backoff = min(backoff * 2, 64)
                time.sleep(wait)
                continue
            try:
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                # 4xx/5xx non 429: on retente un peu, sinon on remonte
                if 500 <= r.status_code < 600 and attempt < 3:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 32)
                    continue
                print("HTTP ERROR:", e, r.text, file=sys.stderr)
                raise
        raise RuntimeError("Too many retries on X API")

    def search(self,
               query: str,
               start_time_iso: Optional[str] = None,
               end_time_iso: Optional[str] = None,
               next_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Utilise /tweets/search/recent (fenêtre courte) par défaut.
        Si ton offre inclut l'historique, remplace le chemin par 'tweets/search/all' et fournis start_time.
        """
        path = "tweets/search/recent"
        params = {
            "query": query,
            "max_results": self.cfg.page_size,
            "tweet.fields": "id,text,lang,created_at,public_metrics,entities,referenced_tweets,source,author_id,conversation_id",
            "expansions": "author_id",
            "user.fields": "id,username,name,public_metrics,verified"
        }
        if start_time_iso: params["start_time"] = start_time_iso
        if end_time_iso: params["end_time"] = end_time_iso
        if next_token: params["next_token"] = next_token
        return self._get(path, params)


# ============== Normalizer ==============
class XNormalizer:
    """
    Transforme la réponse API en lignes CSV 'projet' (collecte uniquement).
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def _user_index(resp: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        users = resp.get("includes", {}).get("users", []) or []
        return {u["id"]: u for u in users}

    def rows_from_response(self, resp: Dict[str, Any], source_query: str) -> List[Dict[str, Any]]:
        idx = self._user_index(resp)
        rows: List[Dict[str, Any]] = []
        for t in resp.get("data", []) or []:
            u = idx.get(t.get("author_id") or "", {})
            pm = t.get("public_metrics", {}) or {}
            rows.append({
                "platform": "x",
                "post_id": t.get("id"),
                "created_at": t.get("created_at"),
                "text": (t.get("text") or "").replace("\r", " ").replace("\n", " ").strip(),
                "lang": t.get("lang"),
                "author_id": t.get("author_id"),
                "author_username": u.get("username"),
                "author_name": u.get("name"),
                "likes": pm.get("like_count", 0),
                "retweets": pm.get("retweet_count", 0),
                "replies": pm.get("reply_count", 0),
                "quotes": pm.get("quote_count", 0),
                "conversation_id": t.get("conversation_id"),
                "source_query": source_query
            })
        return rows


# ============== Export CSV ==============
class CSVExporter:
    def __init__(self, out_csv: str) -> None:
        self.out_csv = out_csv
        ensure_dir(out_csv)
        self.header = [
            "platform","post_id","created_at","text","lang",
            "author_id","author_username","author_name",
            "likes","retweets","replies","quotes",
            "conversation_id","source_query"
        ]
        if not Path(out_csv).exists():
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.header).writeheader()

    def append(self, rows: List[Dict[str, Any]]) -> None:
        if not rows: return
        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.header)
            for r in rows:
                w.writerow(r)


# ============== Pipeline ==============
class XCollectPipeline:
    def __init__(self, client: XClient, normalizer: XNormalizer, exporter: CSVExporter) -> None:
        self.client = client
        self.norm = normalizer
        self.export = exporter

    def run_query(self, q: str, start_time_iso: Optional[str], end_time_iso: Optional[str]) -> int:
        total = 0
        token = None
        pages = 0
        while pages < self.client.cfg.max_pages and total < self.client.cfg.hard_limit_rows:
            resp = self.client.search(q, start_time_iso, end_time_iso, token)
            rows = self.norm.rows_from_response(resp, source_query=q)
            self.export.append(rows)
            total += len(rows)
            pages += 1
            token = resp.get("meta", {}).get("next_token")
            if not token:
                break
            time.sleep(self.client.cfg.sleep_s)
        print(f"[{q}] {total} tweets exportés ({pages} pages)")
        return total

    def run(self, queries: List[str], start_time_iso: Optional[str], end_time_iso: Optional[str]) -> None:
        grand_total = 0
        for q in queries:
            grand_total += self.run_query(q, start_time_iso, end_time_iso)
        print(f"Terminé → {grand_total} lignes → {self.export.out_csv}")


# ============== main ==============
if __name__ == "__main__":
    cfg = XConfig.from_env()
    client = XClient(cfg)
    normalizer = XNormalizer()
    exporter = CSVExporter(cfg.out_csv)

    # ⚠️ /search/recent ignore start_time si ta fenêtre dépasse la limite du palier.
    # Laisse end_time vide pour "maintenant". start_time_iso = iso_6months() si tu as /search/all.
    start_time = None        # ex: iso_6months() si tu as l'historique
    end_time = None

    # Requêtes FR propres pour Honda moto (modifie/ajoute si besoin)
    queries = [
        '(Honda OR "Honda Moto" OR "Honda France" OR "Honda Moto France" OR Transalp OR "Africa Twin" OR CB500X) lang:fr -is:retweet',
    ]

    XCollectPipeline(client, normalizer, exporter).run(queries, start_time, end_time)
