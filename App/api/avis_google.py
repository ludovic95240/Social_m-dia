from __future__ import annotations
import os, time, csv, hashlib, math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve()
APP_ROOT = HERE.parents[1]   # le dossier "App"
RAW_DIR = APP_ROOT / "api" / "data" / "raw"

def must_exist_dir(p: Path) -> None:
    if not p.exists() or not p.is_dir():
        raise OSError(f"[EXPORT] Dossier requis manquant : {p}\n"
                      f"Crée-le manuellement dans App/ et relance.")


def epoch_to_iso(ts: int | float | None) -> str | None:
    try:
        if ts is None:
            return None
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None

def md5_id(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(str(p or "").encode("utf-8"))
    return h.hexdigest()

def try_import_translator():
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        return GoogleTranslator
    except Exception:
        return None

# ==================== Modèles ====================
@dataclass
class GPlacesConfig:
    base: str = "https://maps.googleapis.com/maps/api/place"
    sleep: float = 0.25            # délai standard entre appels
    backoff_max: float = 32.0
    page_wait_s: float = 2.0       # Google impose ~2s avant d'utiliser next_page_token
    lang: str = "fr"               # langue des résultats
    region: str = "fr"             # biais régional
    out_csv: str = "api/data/raw/google_reviews.csv"
    max_places_per_city: int = 30  # limite douce pour éviter l'explosion
    # champs demandés en details:
    fields: str = "name,formatted_address,geometry,place_id,url,rating,user_ratings_total,reviews"

class PlacesClient:
    def __init__(self, api_key: Optional[str] = None, cfg: Optional[GPlacesConfig] = None):
        load_dotenv()
        self.key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        assert self.key, "GOOGLE_MAPS_API_KEY manquant dans .env"
        self.cfg = cfg or GPlacesConfig()
        self.s = requests.Session()

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.cfg.base}/{path}"
        p = dict(params); p["key"] = self.key
        backoff = 1.0
        last_text = ""
        for attempt in range(8):
            r = self.s.get(url, params=p, timeout=30)
            last_text = r.text
            # Erreurs quota/abuse → on patiente
            if r.status_code in (429, 403, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, self.cfg.backoff_max)
                continue
            try:
                r.raise_for_status()
                data = r.json()
                status = (data or {}).get("status", "OK")
                # L'API Places utilise un champ "status" côté payload
                if status not in ("OK", "ZERO_RESULTS"):
                    # Possible: OVER_QUERY_LIMIT, REQUEST_DENIED, INVALID_REQUEST
                    # On backoff si blocage quota/abuse
                    if status in ("OVER_QUERY_LIMIT", "UNKNOWN_ERROR"):
                        time.sleep(backoff)
                        backoff = min(backoff * 2, self.cfg.backoff_max)
                        continue
                    raise RuntimeError(f"Places API status={status}: {data}")
                return data
            except requests.HTTPError:
                if 500 <= r.status_code < 600:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.cfg.backoff_max)
                    continue
                raise RuntimeError(f"HTTP {r.status_code}: {last_text[:500]}")
            except ValueError:
                raise RuntimeError(f"Invalid JSON response: {last_text[:500]}")
        raise RuntimeError(f"API retry exhausted: {path} :: {last_text[:500]}")

    # ------------ Recherche de lieux (Text Search) ------------
    def text_search(self, query: str, language: Optional[str] = None, region: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        params = {
            "query": query,
            "language": language or self.cfg.lang,
            "region": region or self.cfg.region,
        }
        data = self._get("textsearch/json", params)
        yield from (data.get("results") or [])
        next_token = data.get("next_page_token")
        pages = 0
        while next_token and pages < 2:  # max 3 pages autorisées
            time.sleep(self.cfg.page_wait_s)
            data = self._get("textsearch/json", {"pagetoken": next_token})
            yield from (data.get("results") or [])
            next_token = data.get("next_page_token")
            pages += 1

    # ------------ Détails d'un lieu (reviews inclus) ------------
    def details(self, place_id: str, language: Optional[str] = None) -> Dict[str, Any]:
        params = {
            "place_id": place_id,
            "fields": self.cfg.fields,
            "language": language or self.cfg.lang,
            # <— clé importante : on demande explicitement des reviews traduites si l’API le permet
            "reviews_no_translations": "false",
        }
        return self._get("details/json", params)


# ==================== Normalisation & Export ====================
class CSVExporter:
    def __init__(self, out_csv: str | Path):
        self.out_csv = Path(out_csv)
        # NE PAS créer: on exige que le dossier existe déjà
        must_exist_dir(self.out_csv.parent)
        # >>> En-têtes corrects pour Google Reviews <<<
        self.header = [
            "platform","post_id","created_at","text","text_fr","author_name","rating",
            "place_id","place_name","formatted_address","city","source_query",
            "user_ratings_total","place_rating","review_lang","profile_photo_url","review_url"
        ]
        if not self.out_csv.exists():
            with self.out_csv.open("w", newline="", encoding="utf-8") as f:
                # extrasaction="ignore" => robustesse si champs inattendus
                csv.DictWriter(f, fieldnames=self.header, extrasaction="ignore").writeheader()

    def append(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        count = 0
        with self.out_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.header, extrasaction="ignore")
            for r in rows:
                # On n'écrit que les colonnes prévues ; valeurs manquantes -> ""
                w.writerow({k: r.get(k, "") for k in self.header})
                count += 1
        return count


class ReviewNormalizer:
    def __init__(self):
        GT = try_import_translator()
        self.translator = GT(source="auto", target="fr") if GT else None

    def _translate_fr(self, text: str) -> str:
        if not text: return ""
        if self.translator is None:
            return text  # fallback: on garde le texte original
        try:
            return self.translator.translate(text)
        except Exception:
            return text

    def rows_from_details(self, details: Dict[str, Any], source_query: str, city_hint: str) -> List[Dict[str, Any]]:
        res = (details.get("result") or {})
        reviews = res.get("reviews") or []
        if not reviews:
            return []
        place_id = res.get("place_id")
        place_name = res.get("name")
        address = res.get("formatted_address")
        place_rating = res.get("rating")
        user_ratings_total = res.get("user_ratings_total")

        rows: List[Dict[str, Any]] = []
        for rv in reviews:
            # Champs review (5 max)
            text = (rv.get("text") or "").strip()
            text_fr = self._translate_fr(text)
            created_at = epoch_to_iso(rv.get("time"))
            author = rv.get("author_name")
            rating = rv.get("rating")
            lang = rv.get("language")
            purl = rv.get("profile_photo_url")
            # ID stable pour éviter les doublons
            rid = md5_id(place_id, author or "", text, str(rv.get("time")))
            rows.append({
                "platform": "google_reviews",
                "post_id": rid,
                "created_at": created_at,
                "text": text,
                "text_fr": text_fr,
                "author_name": author,
                "rating": rating,
                "place_id": place_id,
                "place_name": place_name,
                "formatted_address": address,
                "city": city_hint,
                "source_query": source_query,
                "user_ratings_total": user_ratings_total,
                "place_rating": place_rating,
                "review_lang": lang,
                "profile_photo_url": purl,
                # Google n'expose pas d'URL directe de review; on met l'URL du lieu si dispo
                "review_url": res.get("url"),
            })
        return rows

# ==================== Pipeline ====================
class GoogleReviewsPipeline:
    def __init__(self, api: PlacesClient, norm: ReviewNormalizer, exp: CSVExporter):
        self.api = api; self.norm = norm; self.exp = exp

    def run(self, search_patterns: List[str], cities: List[str]) -> None:
        total_places = 0
        total_reviews = 0
        for city in cities:
            for patt in search_patterns:
                q = patt.format(city=city)
                print(f"[City: {city}] TextSearch → {q}")
                places: List[Dict[str, Any]] = []
                for it in self.api.text_search(q):
                    places.append(it)
                    if len(places) >= self.api.cfg.max_places_per_city:
                        break

                print(f"  Lieux trouvés: {len(places)} (limite: {self.api.cfg.max_places_per_city})")
                total_places += len(places)

                for p in places:
                    pid = p.get("place_id")
                    if not pid:
                        continue
                    try:
                        det = self.api.details(pid)
                    except RuntimeError as e:
                        msg = str(e).lower()
                        # quota/abuse → on skippe juste ce lieu
                        if "over_query_limit" in msg or "429" in msg or "quota" in msg:
                            print(f"    [WARN] Quota / LIMIT pour place_id={pid} → skip")
                            continue
                        raise
                    rows = self.norm.rows_from_details(det, source_query=q, city_hint=city)
                    self.exp.append(rows)
                    total_reviews += len(rows)
                    # petit délai entre lieux
                    time.sleep(0.05)
        print(f"Terminé. Lieux: {total_places} | Avis exportés: {total_reviews} → {self.exp.out_csv}")

# ==================== Exécution directe ====================
if __name__ == "__main__":
    # Exemples de patterns de recherche
    # Astuce: mixer "Honda moto {city}" / "Concession Honda {city}" pour couvrir large
    search_patterns = [
        "Concession Honda moto {city}",
        "Honda moto {city}",
        "Honda Motorcycle {city}",
    ]
    # Grandes villes FR (adaptable)
    cities = ["Paris","Lyon","Marseille","Toulouse","Bordeaux","Lille","Nantes","Rennes","Nice","Strasbourg"]

    client = PlacesClient()
    norm = ReviewNormalizer()
    exp = CSVExporter(RAW_DIR / "google_reviews.csv")

    GoogleReviewsPipeline(client, norm, exp).run(search_patterns, cities)
