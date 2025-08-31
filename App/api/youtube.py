from __future__ import annotations
import os, csv, time, math
from typing import List, Dict, Any, Iterable, Optional, TypedDict
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from pathlib import Path
import requests
from dotenv import load_dotenv
import csv

# --- Racine App (…/App/api/youtube.py -> APP_ROOT = …/App)
HERE = Path(__file__).resolve()
APP_ROOT = HERE.parents[1]   # le dossier "App"
RAW_DIR = APP_ROOT / "api" / "data" / "raw"   # dossier des CSV bruts

def must_exist_dir(p: Path) -> None:
    if not p.exists() or not p.is_dir():
        raise OSError(f"[EXPORT] Dossier requis manquant : {p}\n"
                      f"Crée-le manuellement dans App/ et relance.")

# ================== Utils ==================
def iso_6months() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=180))\
        .isoformat(timespec="seconds").replace("+00:00", "Z")

def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

# ================== Data Model ==================
@dataclass(frozen=True)
class CommentRow:
    platform: str
    post_id: str
    created_at: str
    text: str
    author_name: Optional[str] = None
    likes: int = 0
    parent_id: Optional[str] = None
    parent_author_name: Optional[str] = None
    video_id: Optional[str] = None
    video_title: Optional[str] = None
    channel_title: Optional[str] = None
    source_query: Optional[str] = None
    video_views: Optional[int] = None
    video_likes: Optional[int] = None
    video_comment_count: Optional[int] = None

class VideoHit(TypedDict, total=False):
    video_id: str
    title: str
    channel_title: str
    published_at: str
    source_query: str

# ================== API Client ==================
@dataclass
class YTConfig:
    base: str = "https://www.googleapis.com/youtube/v3"
    sleep: float = 0.25
    max_pages: int = 3
    relevance_language: str = "fr"
    region_code: str = "FR"
    page_size: int = 50

class YouTubeClient:
    def __init__(self, api_key: Optional[str] = None, cfg: Optional[YTConfig] = None) -> None:
        load_dotenv()
        self.key = api_key or os.getenv("YOUTUBE_API_KEY")
        assert self.key, "YOUTUBE_API_KEY manquant (mets-le dans .env)"
        self.cfg = cfg or YTConfig()
        self.session = requests.Session()

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Appel GET robuste:
        - backoff exponentiel sur 429/403/5xx
        - remonte une erreur explicite si on abandonne
        - ne renvoie jamais None silencieusement
        """
        p = dict(params);
        p["key"] = self.key
        url = f"{self.cfg.base}/{path}"
        backoff = 1.0
        last_text = ""
        last_status = None

        for attempt in range(8):
            r = self.session.get(url, params=p, timeout=30)
            # Si 403 avec "commentsDisabled", retry inutile → on remonte direct
            if r.status_code == 403:
                try:
                    payload = r.json()
                except Exception:
                    payload = {}
                reason = ""
                if isinstance(payload, dict):
                    err = (payload.get("error") or {})
                    errs = err.get("errors") or []
                    if errs and isinstance(errs, list):
                        reason = (errs[0].get("reason") or "").lower()
                    # parfois le message suffit
                    if not reason:
                        reason = (err.get("message") or "").lower()
                if "commentsdisabled" in reason:
                    raise RuntimeError("HTTP 403: commentsDisabled")  # capté plus haut et skip

            last_status = r.status_code
            try:
                last_text = r.text
            except Exception:
                last_text = "<no body>"

            # Quotas / accès / abuse: 403/429
            if r.status_code in (429, 403):
                # petit backoff exponentiel
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)
                continue

            # Autres cas
            try:
                r.raise_for_status()
                # si pas de JSON, lance une ValueError
                return r.json()
            except requests.HTTPError:
                # 5xx: retente quelques fois
                if 500 <= r.status_code < 600 and attempt < 3:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 32)
                    continue
                # 4xx non gérable: on sort proprement avec message clair
                raise RuntimeError(f"HTTP {r.status_code} on {path}: {last_text[:500]}") from None
            except ValueError:
                # corps non JSON
                raise RuntimeError(f"Invalid JSON response from {path}: {last_text[:500]}") from None

        # si on sort de la boucle, c'est qu'on a 403/429 répétés
        raise RuntimeError(f"API retry exhausted ({last_status}) on {path}: {last_text[:500]}")

    def search_videos(self, query: str, published_after_iso: str,
                      max_results: int = 25) -> List[VideoHit]:
        assert 5 <= max_results <= 50, "max_results ∈ [5,50]"
        hits: List[VideoHit] = []
        params = {
            "part": "snippet",
            "type": "video",
            "q": query,
            "publishedAfter": published_after_iso,
            "maxResults": min(max_results, self.cfg.page_size),
            "relevanceLanguage": self.cfg.relevance_language,
            "regionCode": self.cfg.region_code,
            "safeSearch": "none",
        }
        next_token = None
        pages = 0
        while pages < self.cfg.max_pages:
            if next_token: params["pageToken"] = next_token
            data = self._get("search", params)
            for it in data.get("items", []):
                sn = it.get("snippet", {}) or {}
                vid = it.get("id", {}).get("videoId")
                if vid:
                    hits.append(VideoHit(
                        video_id=vid,
                        title=sn.get("title",""),
                        channel_title=sn.get("channelTitle",""),
                        published_at=sn.get("publishedAt",""),
                        source_query=query
                    ))
            next_token = data.get("nextPageToken")
            pages += 1
            if not next_token: break
            time.sleep(self.cfg.sleep)
        return hits

    def videos_stats(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return {video_id: {'views':int,'likes':int,'commentCount':int}}"""
        out: Dict[str, Dict[str, Any]] = {}
        if not ids: return out
        # YouTube batch allows up to 50 ids
        for i in range(0, len(ids), 50):
            chunk = ids[i:i+50]
            data = self._get("videos", {
                "part": "statistics,snippet",
                "id": ",".join(chunk)
            })
            for it in data.get("items", []):
                vid = it.get("id")
                st = it.get("statistics", {}) or {}
                out[vid] = {
                    "views": int(st.get("viewCount", 0) or 0),
                    "likes": int(st.get("likeCount", 0) or 0),
                    "commentCount": int(st.get("commentCount", 0) or 0),
                }
            time.sleep(self.cfg.sleep)
        return out

    def list_comment_threads(self, video_id: str, page_size: int = 100, max_pages: int = 10):
        """
        Itère sur les threads de commentaires d'une vidéo.
        - Si les commentaires sont désactivés / vidéo privée / quota → on LOG puis on SKIP la vidéo.
        - Ne lève pas d'erreur dans ces cas.
        """
        assert 1 <= page_size <= 100, "page_size ∈ [1,100]"
        params = {
            "part": "snippet,replies",  # ← replies pour récupérer les réponses
            "videoId": video_id,
            "maxResults": page_size,
            "textFormat": "plainText",
        }
        next_token = None
        pages = 0

        while pages < max_pages:
            if next_token:
                params["pageToken"] = next_token
            try:
                data = self._get("commentThreads", params)
            except Exception as e:
                msg = str(e).lower()
                # cas fréquents → on skip sans planter
                if ("commentsdisabled" in msg or "disabled comments" in msg
                        or "has disabled comments" in msg):
                    print(f"[YT] Commentaires désactivés pour {video_id} → skip.")
                    return
                if "http 403" in msg or "forbidden" in msg or "private" in msg:
                    print(f"[YT] 403/forbidden pour {video_id} (privée/droits) → skip.")
                    return
                if "http 429" in msg or "quota" in msg or "rate limit" in msg:
                    print(f"[YT] 429/quota pour {video_id} → skip cette vidéo.")
                    return
                # autre erreur inconnue → on remonte pour investigation
                raise

            for it in data.get("items", []) or []:
                yield it

            next_token = data.get("nextPageToken")
            pages += 1
            if not next_token:
                break
            time.sleep(self.cfg.sleep)  # ← FIX: utiliser self.cfg.sleep


# ================== Normalizer ==================
class YouTubeCommentNormalizer:
    def from_thread(self, thread: Dict[str, Any], video_id: str,
                    video_title: str, channel_title: str, source_query: str,
                    vstats: Optional[Dict[str, Any]] = None) -> List[CommentRow]:
        rows: List[CommentRow] = []
        # top level
        tl = (thread.get("snippet", {}) or {}).get("topLevelComment", {}) or {}
        tl_sn = tl.get("snippet", {}) or {}
        tl_id = tl.get("id") or thread.get("id")
        if tl_id and tl_sn:
            rows.append(CommentRow(
                platform="youtube",
                post_id=str(tl_id),
                created_at=tl_sn.get("publishedAt"),
                text=(tl_sn.get("textDisplay") or "").strip(),
                author_name=tl_sn.get("authorDisplayName"),
                likes=int(tl_sn.get("likeCount", 0) or 0),
                parent_id=None,
                parent_author_name=None,
                video_id=video_id, video_title=video_title, channel_title=channel_title,
                source_query=source_query,
                video_views=(vstats or {}).get("views"),
                video_likes=(vstats or {}).get("likes"),
                video_comment_count=(vstats or {}).get("commentCount"),
            ))
        # replies (if any)
        rep = (thread.get("replies", {}) or {}).get("comments", []) or []
        for c in rep:
            sn = c.get("snippet", {}) or {}
            rows.append(CommentRow(
                platform="youtube",
                post_id=str(c.get("id")),
                created_at=sn.get("publishedAt"),
                text=(sn.get("textDisplay") or "").strip(),
                author_name=sn.get("authorDisplayName"),
                likes=int(sn.get("likeCount", 0) or 0),
                parent_id=tl_id,
                parent_author_name=tl_sn.get("authorDisplayName") if tl_sn else None,
                video_id=video_id, video_title=video_title, channel_title=channel_title,
                source_query=source_query,
                video_views=(vstats or {}).get("views"),
                video_likes=(vstats or {}).get("likes"),
                video_comment_count=(vstats or {}).get("commentCount"),
            ))
        return rows

# ================== CSV Exporter ==================
class CSVExporter:
    def __init__(self, out_csv: str | Path):
        self.out_csv = Path(out_csv)
        # NE PAS créer: on exige que le dossier existe
        must_exist_dir(self.out_csv.parent)
        self.header = [
            "platform","post_id","created_at","text","author_name","likes",
            "parent_id","parent_author_name",
            "video_id","video_title","channel_title","source_query",
            "video_views","video_likes","video_comment_count"
        ]
        if not self.out_csv.exists():
            with self.out_csv.open("w", newline="", encoding="utf-8") as f:
                import csv
                csv.DictWriter(f, fieldnames=self.header).writeheader()

    def append(self, rows: List[CommentRow]) -> None:
        if not rows: return
        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.header)
            for r in rows:
                w.writerow({
                    "platform": r.platform,
                    "post_id": r.post_id,
                    "created_at": r.created_at,
                    "text": r.text,
                    "author_name": r.author_name,
                    "likes": r.likes,
                    "parent_id": r.parent_id,
                    "parent_author_name": r.parent_author_name,
                    "video_id": r.video_id, "video_title": r.video_title, "channel_title": r.channel_title,
                    "source_query": r.source_query,
                    "video_views": r.video_views,
                    "video_likes": r.video_likes,
                    "video_comment_count": r.video_comment_count
                })

# ================== Pipeline ==================
class YouTubeCollectPipeline:
    def __init__(self, api: YouTubeClient, norm: YouTubeCommentNormalizer, exp: CSVExporter) -> None:
        self.api = api; self.norm = norm; self.exp = exp
    def run(self, queries: List[str], since_iso: str, max_videos_per_query: int = 25) -> None:
        total = 0
        for q in queries:
            hits = self.api.search_videos(q, since_iso, max_results=max_videos_per_query)
            stats = self.api.videos_stats([h["video_id"] for h in hits])
            print(f"[{q}] vidéos trouvées: {len(hits)}")
            for h in hits:
                vid = h["video_id"]; vstats = stats.get(vid, {})
                rows: List[CommentRow] = []
                for th in self.api.list_comment_threads(vid, page_size=100, max_pages=10):
                    rows.extend(self.norm.from_thread(
                        th, video_id=vid, video_title=h.get("title",""),
                        channel_title=h.get("channel_title",""), source_query=q,
                        vstats=vstats
                    ))
                self.exp.append(rows)
                total += len(rows)
                time.sleep(0.05)
            print(f"  [{q}] cumul commentaires exportés: {total}")
        print(f"Terminé. Total commentaires: {total} → {self.exp.out_csv}")

if __name__ == "__main__":
    client = YouTubeClient()
    norm = YouTubeCommentNormalizer()
    exp = CSVExporter(RAW_DIR / "yt_comments_enriched.csv")
    queries = ["Honda Moto France", "Honda Transalp", "Honda Africa Twin", "Honda CB500X"]
    YouTubeCollectPipeline(client, norm, exp).run(queries, since_iso=iso_6months(), max_videos_per_query=25)
