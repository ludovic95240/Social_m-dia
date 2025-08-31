# App/api/ig.py
from __future__ import annotations
import os, csv, time, re, math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv, dotenv_values
from apify_client import ApifyClient

# ---------- Chemins ----------
HERE = Path(__file__).resolve()
APP_ROOT = HERE.parents[1]
RAW_DIR  = APP_ROOT / "api" / "data" / "raw"
ENV_FILE = APP_ROOT / ".env"
RAW_DIR.mkdir(parents=True, exist_ok=True)

POSTS_CSV    = RAW_DIR / "instagram_posts.csv"
COMMENTS_CSV = RAW_DIR / "instagram_comments.csv"

# ---------- .env ----------
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=False)
else:
    load_dotenv(override=False)

def _debug_env_token() -> None:
    tok = os.getenv("APIFY_TOKEN")
    if tok:
        masked = f"{tok[:4]}…{tok[-4:]}" if len(tok) > 8 else tok
        print(f"[ENV] APIFY_TOKEN trouvé (len={len(tok)}): {masked}")
    else:
        print(f"[ENV] APIFY_TOKEN introuvable | cwd={Path.cwd()}")
        if ENV_FILE.exists():
            vals = dotenv_values(ENV_FILE)
            t2 = vals.get("APIFY_TOKEN")
            if t2:
                masked = f"{t2[:4]}…{t2[-4:]}" if len(t2) > 8 else t2
                print(f"[ENV] .env lu, token détecté (len={len(t2)}): {masked}")
            else:
                print("[ENV] .env présent mais sans APIFY_TOKEN")

# ---------- Config ----------
@dataclass
class IGConfig:
    actor_posts: str    = "apify/instagram-scraper"
    actor_comments: str = "apify/instagram-comment-scraper"
    results_limit: int  = 100
    newer_than: str     = "6 months"
    comment_chunk: int  = 20

# ---------- Utils ----------
RE_POST_URL = re.compile(r"^https?://(?:www\.)?instagram\.com/(?:p|reel)/[^/?#&\s]+/?$", re.IGNORECASE)
RE_SC_FROM_URL = re.compile(r"instagram\.com/(?:p|reel)/([^/?#\s]+)/?", re.IGNORECASE)

def iso_utc(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    # int/float epoch seconds
    if isinstance(ts, (int, float)) and not math.isnan(ts):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
        except Exception:
            return None
    # numeric-in-string
    if isinstance(ts, str) and ts.isdigit():
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
        except Exception:
            pass
    # ISO-ish strings
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        except Exception:
            return ts
    return None

def open_csv(path: Path, header: List[str]) -> Tuple[csv.DictWriter, Any]:
    f = open(path, "w", newline="", encoding="utf-8")  # overwrite à chaque run
    w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
    w.writeheader()
    return w, f

# ---------- Normalisation / Flatten ----------
POST_KEYS = (
    "id","shortCode","shortcode","code","caption","ownerUsername","owner",
    "timestamp","takenAtTimestamp","taken_at_timestamp","takenAt",
    "likesCount","commentsCount","isVideo","videoViewCount","url","postUrl","link","permalink"
)

def _as_username(d: Dict[str,Any]) -> Optional[str]:
    return (
        d.get("ownerUsername")
        or (d.get("owner") or {}).get("username")
        or (d.get("user") or {}).get("username")
        or (d.get("author") or {}).get("username")
    )

def _as_caption(d: Dict[str,Any]) -> str:
    return (
        d.get("caption")
        or d.get("text")
        or (d.get("edge_media_to_caption",{}).get("edges",[{"node":{"text":""}}])[0].get("node",{}).get("text",""))
        or ""
    )

def _as_ts(d: Dict[str,Any]) -> Any:
    return (
        d.get("timestamp")
        or d.get("takenAtTimestamp")
        or d.get("taken_at_timestamp")
        or d.get("takenAt")
        or d.get("date")
        or d.get("created_at")
    )

def _as_likes(d: Dict[str,Any]) -> int:
    return int(
        d.get("likesCount")
        or (d.get("edge_liked_by") or {}).get("count", 0)
        or 0
    )

def _as_comments(d: Dict[str,Any]) -> int:
    return int(
        d.get("commentsCount")
        or (d.get("edge_media_to_comment") or {}).get("count", 0)
        or 0
    )

def _as_video_flag(d: Dict[str,Any]) -> bool:
    v = d.get("isVideo")
    if v is None: v = d.get("is_video")
    return bool(v)

def _as_video_views(d: Dict[str,Any]) -> int:
    return int(d.get("videoViewCount") or d.get("video_view_count") or 0)

def _candidate_url(d: Dict[str,Any]) -> str:
    return d.get("url") or d.get("postUrl") or d.get("link") or d.get("permalink") or ""

def _candidate_sc(d: Dict[str,Any]) -> Optional[str]:
    for k in ("shortCode","shortcode","code"):
        if d.get(k): return str(d[k])
    u = _candidate_url(d)
    m = RE_SC_FROM_URL.search(u) if u else None
    return m.group(1) if m else None

def _looks_like_post(d: Dict[str,Any]) -> bool:
    if not isinstance(d, dict):
        return False
    # heuristique : présence d'un shortcode OU d'une url de post
    if _candidate_sc(d): return True
    u = _candidate_url(d)
    return bool(u and RE_POST_URL.match(u))

def _normalize_post_dict(d: Dict[str,Any]) -> Dict[str,Any]:
    sc = _candidate_sc(d)
    u = _candidate_url(d)
    if not (u and RE_POST_URL.match(u)) and sc:
        u = f"https://www.instagram.com/p/{sc}/"
    pid = d.get("id") or sc or ""
    return {
        "id": pid,
        "shortCode": sc or "",
        "caption": _as_caption(d),
        "ownerUsername": _as_username(d),
        "timestamp": _as_ts(d),
        "likesCount": _as_likes(d),
        "commentsCount": _as_comments(d),
        "isVideo": _as_video_flag(d),
        "videoViewCount": _as_video_views(d),
        "url": u or "",
    }

def _walk(obj: Any) -> Iterable[Dict[str,Any]]:
    """Parcourt récursivement et yield les dicts post-like."""
    if isinstance(obj, dict):
        if _looks_like_post(obj):
            yield obj
        # explorer enfants
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _walk(it)

def flatten_posts(raw_items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    posts: List[Dict[str,Any]] = []
    seen = set()
    # certains actors renvoient déjà des posts “plats”
    for it in raw_items:
        # si c'est un post direct, on prend tel quel
        if _looks_like_post(it):
            nd = _normalize_post_dict(it)
            key = nd["url"] or nd["shortCode"] or nd["id"]
            if key and key not in seen:
                seen.add(key); posts.append(nd)
        # sinon, on fouille les sous-objets
        for sub in _walk(it):
            nd = _normalize_post_dict(sub)
            key = nd["url"] or nd["shortCode"] or nd["id"]
            if key and key not in seen:
                seen.add(key); posts.append(nd)
    print(f"[IG] Flatten: found {len(posts)} post-like objects in {len(raw_items)} items")
    # nettoyage minimal
    posts = [p for p in posts if p["url"] and RE_POST_URL.match(p["url"])]
    print(f"[IG] Flatten: valid post URLs after cleaning → {len(posts)}")
    return posts

# ---------- Apify client ----------
def make_client() -> ApifyClient:
    token = os.getenv("APIFY_TOKEN") or dotenv_values(ENV_FILE).get("APIFY_TOKEN", "")
    _debug_env_token()
    if not token:
        raise RuntimeError("APIFY_TOKEN manquant (ajoute-le dans .env ou en variable d’environnement).")
    return ApifyClient(token)

# ---------- Collecte POSTS ----------
def _run_posts_actor(client: ApifyClient, run_input: Dict[str, Any], label: str) -> List[Dict[str, Any]]:
    run = client.actor(IGConfig.actor_posts).call(run_input=run_input)
    dataset = client.dataset(run["defaultDatasetId"])
    items = list(dataset.iterate_items())
    print(f"[IG] {label} → {len(items)} items")
    return items

def fetch_posts(client: ApifyClient,
                usernames: List[str] | None = None,
                hashtags: List[str] | None = None,
                cfg: IGConfig = IGConfig()) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []

    if usernames:
        direct_urls = [u if u.startswith("http") else f"https://www.instagram.com/{u.strip('@').strip('/')}/" for u in usernames]
        run_input = {
            "directUrls": direct_urls,
            "resultsType": "posts",
            "resultsLimit": cfg.results_limit,
            "onlyPostsNewerThan": cfg.newer_than,
            "addParentData": True,
        }
        all_items += _run_posts_actor(client, run_input, f"users:{len(direct_urls)}")

    if hashtags:
        for tag in hashtags:
            tag_clean = tag.lstrip("#").strip()
            # 1) recherche hashtag
            run_input = {
                "search": tag_clean,
                "searchType": "hashtag",
                "resultsType": "posts",
                "resultsLimit": cfg.results_limit,
                "onlyPostsNewerThan": cfg.newer_than,
                "addParentData": True,
            }
            items = _run_posts_actor(client, run_input, f"hashtag(search):#{tag_clean}")
            # 2) fallback direct
            if not items:
                url = f"https://www.instagram.com/explore/tags/{tag_clean}/"
                run_input2 = {
                    "directUrls": [url],
                    "resultsType": "posts",
                    "resultsLimit": cfg.results_limit,
                    "onlyPostsNewerThan": cfg.newer_than,
                    "addParentData": True,
                }
                items = _run_posts_actor(client, run_input2, f"hashtag(direct):{url}")
            all_items += items

    print(f"[IG] Total items collectés (avant flatten): {len(all_items)}")
    posts = flatten_posts(all_items)
    return posts

# ---------- Collecte COMMENTAIRES ----------
def fetch_comments_for_posts(client: ApifyClient, post_urls: List[str],
                             cfg: IGConfig = IGConfig()) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not post_urls:
        return out
    for i in range(0, len(post_urls), cfg.comment_chunk):
        chunk = post_urls[i:i+cfg.comment_chunk]
        run_input = {"directUrls": chunk, "resultsLimit": cfg.results_limit}
        run = client.actor(cfg.actor_comments).call(run_input=run_input)
        dataset = client.dataset(run["defaultDatasetId"])
        out.extend(dataset.iterate_items())
        time.sleep(0.3)
    return out

# ---------- Exports ----------
POST_HEADER = [
    "platform","post_id","created_at","text","author_name","likes","comments","shares","views","post_url"
]
CMT_HEADER = [
    "platform","post_id","created_at","text","author_name","likes","comments","shares","views","parent_post_url"
]

def export_posts(posts: List[Dict[str, Any]]) -> int:
    w, f = open_csv(POSTS_CSV, POST_HEADER)
    n = 0
    for p in posts:
        row = {
            "platform": "instagram",
            "post_id": p.get("id") or p.get("shortCode") or "",
            "created_at": iso_utc(p.get("timestamp")),
            "text": (p.get("caption") or "").strip(),
            "author_name": p.get("ownerUsername") or "",
            "likes": p.get("likesCount", 0),
            "comments": p.get("commentsCount", 0),
            "shares": 0,
            "views": p.get("videoViewCount", 0) if p.get("isVideo") else 0,
            "post_url": p.get("url") or "",
        }
        w.writerow(row); n += 1
    f.close()
    return n

def export_comments(items: List[Dict[str, Any]]) -> int:
    w, f = open_csv(COMMENTS_CSV, CMT_HEADER)
    n = 0
    for it in items:
        row = {
            "platform": "instagram",
            "post_id": it.get("id") or "",
            "created_at": iso_utc(it.get("timestamp")),
            "text": (it.get("text") or "").strip(),
            "author_name": it.get("ownerUsername") or (it.get("owner") or {}).get("username") or "",
            "likes": it.get("likesCount", 0),
            "comments": 0, "shares": 0, "views": 0,
            "parent_post_url": it.get("postUrl") or it.get("url") or "",
        }
        w.writerow(row); n += 1
    f.close()
    return n

# ---------- Main ----------
if __name__ == "__main__":
    cfg = IGConfig(results_limit=100, newer_than="6 months", comment_chunk=20)
    usernames = ["hondamotofrance"]   # peut être restreint par Instagram
    hashtags  = ["Honda", "AfricaTwin", "Transalp", "CB500X"]

    client = make_client()

    # 1) Posts (déjà flatten & normalisés)
    posts = fetch_posts(client, usernames=usernames, hashtags=hashtags, cfg=cfg)
    n_posts = export_posts(posts)
    print(f"[IG] Posts exportés: {n_posts} → {POSTS_CSV}")

    # 2) URLs de posts valides pour commentaires
    post_urls = [p["url"] for p in posts if p.get("url") and RE_POST_URL.match(p["url"])]
    print(f"[IG] URLs valides (post/reel) pour commentaires: {len(post_urls)}")

    # 3) Commentaires
    if post_urls:
        comments = fetch_comments_for_posts(client, post_urls, cfg=cfg)
        n_cmts = export_comments(comments)
        print(f"[IG] Commentaires exportés: {n_cmts} → {COMMENTS_CSV}")
    else:
        export_comments([])
        print("[IG] Aucun lien de post/reel valide → étape commentaires ignorée.")

    print("[OK] Instagram collect terminé.")
