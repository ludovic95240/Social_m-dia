# App/Sentiments/sentiment.py
# Partie 4 — Sentiment (YT + IG + optional Google Reviews) — autonome, robuste, FR
# - Lit les sources depuis App/api/data/raw/
# - Exporte vers App/api/data/processed/social/ et App/api/reports/
# - N'installe rien et ne crée aucun dossier automatiquement

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import numpy as np
import re
import sys

# ===================== Chemins (STRICT: ne crée rien) =====================
HERE = Path(__file__).resolve()
APP_ROOT = HERE.parents[1]  # .../App

RAW_DIR          = APP_ROOT / "api" / "data" / "raw"
PROC_DIR_SOCIAL  = APP_ROOT / "api" / "data" / "processed" / "social"
REPORTS_DIR      = APP_ROOT / "api" / "reports"

def must_exist_dir(p: Path) -> None:
    if not p.exists() or not p.is_dir():
        raise OSError(f"[Sentiment] Dossier requis manquant : {p}\n"
                      f"Crée-le manuellement sous App/ puis relance.")

for d in (RAW_DIR, PROC_DIR_SOCIAL, REPORTS_DIR):
    must_exist_dir(d)

# ===================== Config =====================
@dataclass
class Paths:
    yt_comments: str = str(RAW_DIR / "yt_comments_enriched.csv")
    instagram_posts: str = str(RAW_DIR / "instagram_posts.csv")
    greviews: str   = str(RAW_DIR / "google_reviews.csv")   # optionnel
    out_dir: str    = str(PROC_DIR_SOCIAL)
    report_md: str  = str(REPORTS_DIR / "sentiment.md")

# ===================== Loaders (sources = data/raw) =====================
def _pick(df: pd.DataFrame, *names: str) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([pd.NA] * len(df))

def load_youtube_csv(p: str | Path) -> pd.DataFrame:
    p = Path(p)
    if not p.exists():
        print(f"[INFO][YT] Fichier introuvable: {p} → YouTube ignoré.")
        return pd.DataFrame(columns=["platform","post_id","created_at","author_name","text"])
    df = pd.read_csv(p)
    out = pd.DataFrame({
        "platform": "youtube",
        "post_id": df.get("post_id"),
        "created_at": pd.to_datetime(df.get("created_at"), utc=True, errors="coerce"),
        "author_name": df.get("author_name"),
        "text": df.get("text", "").fillna(""),
    })
    return out.dropna(subset=["created_at"]).reset_index(drop=True)

def load_instagram_csv(p: str | Path) -> pd.DataFrame:
    """
    Charge 'instagram_posts.csv' de manière tolérante aux schémas:
    - post_id/id/shortcode
    - created_at/taken_at/timestamp/date
    - text/caption/description
    - author_name/username/owner_username
    """
    p = Path(p)
    if not p.exists():
        print(f"[INFO][IG] Fichier introuvable: {p} → Instagram ignoré.")
        return pd.DataFrame(columns=["platform","post_id","created_at","author_name","text"])
    df = pd.read_csv(p)
    out = pd.DataFrame({
        "platform": "instagram",
        "post_id": _pick(df, "post_id","id","shortcode"),
        "created_at": pd.to_datetime(_pick(df, "created_at","taken_at","timestamp","date"), utc=True, errors="coerce"),
        "author_name": _pick(df, "author_name","username","owner_username"),
        "text": _pick(df, "text","caption","description").fillna(""),
    })
    return out.dropna(subset=["created_at"]).reset_index(drop=True)

def load_greviews_csv(p: str | Path) -> Optional[pd.DataFrame]:
    pp = Path(p)
    if not pp.exists():
        print("[INFO][GReviews] Fichier introuvable → ignoré.")
        return None
    df = pd.read_csv(pp)
    for c in ("post_id","created_at","author_name","text","text_fr","rating"):
        if c not in df.columns: df[c] = pd.NA
    out = pd.DataFrame({
        "platform": "google_reviews",
        "post_id": df["post_id"],
        "created_at": pd.to_datetime(df["created_at"], utc=True, errors="coerce"),
        "author_name": df["author_name"],
        "text": df["text_fr"].fillna(df["text"]).fillna(""),
        "rating": pd.to_numeric(df["rating"], errors="coerce"),
    })
    return out.dropna(subset=["created_at"]).reset_index(drop=True)

# ===================== Moteur de sentiment =====================
def _torch_ok() -> bool:
    """
    True si torch >= 2.6 est dispo (sinon on évite l'erreur CVE-2025-32434 signalée par HF).
    """
    try:
        import torch
        from packaging.version import parse as vparse
        return vparse(torch.__version__) >= vparse("2.6")
    except Exception:
        return False

def _try_hf_pipeline():
    """
    Essaie d'activer un pipeline HF multilingue si:
    - transformers est installé
    - torch >= 2.6 (ou pas de torch si backend safetensors seulement)
    En cas d'échec -> None (on utilisera l'heuristique).
    """
    try:
        if not _torch_ok():
            raise RuntimeError("torch<2.6 ou indisponible")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True, truncation=True)
        id2label = mdl.config.id2label
        _ = {v.lower(): k for k, v in id2label.items()}  # vérif
        print("[OK] Pipeline HF activé.")
        return pipe
    except Exception as e:
        print(f"[INFO] Transformers indisponible ou modèle non utilisable ({e}) → fallback heuristique.")
        return None

HF_PIPELINE = _try_hf_pipeline()

NEGATIONS = r"(?:pas|plus|jamais|aucun|aucune|ni|not|never|no|none|without)"
POS_WORDS = [
    "super","excellent","parfait","top","génial","incroyable","magnifique","bravo","merci",
    "love","amazing","awesome","great","good","nice","perfect","wow","best","cool",
    "🔥","💯","😍","😘","👏","💪","🤩","😊"
]
NEG_WORDS = [
    "nul","mauvais","horrible","pire","déçu","décevant","bug","arnaque","scam","shit",
    "bad","hate","angry","furious","🤮","💩","😡","😠","👎","😤","☹️"
]

POS_RGX = re.compile("|".join(map(re.escape, POS_WORDS)), re.IGNORECASE)
NEG_RGX = re.compile("|".join(map(re.escape, NEG_WORDS)), re.IGNORECASE)
NEGATION_RGX = re.compile(rf"\b{NEGATIONS}\b", re.IGNORECASE)

def heuristic_sentiment(text: str) -> Tuple[int, float]:
    """
    (label, score) avec label ∈ {-1,0,1}, score ∈ [0,1].
    Gère négations simples: "pas mal" => positif léger ; "not great" => négatif léger.
    """
    if not isinstance(text, str):
        return 0, 0.0
    t = text.strip()
    if not t:
        return 0, 0.0

    pos_matches = list(POS_RGX.finditer(t))
    neg_matches = list(NEG_RGX.finditer(t))

    inv_pos = 0
    for m in neg_matches:
        start = max(0, m.start() - 12)
        window = t[start:m.start()]
        if NEGATION_RGX.search(window):
            inv_pos += 1  # négation + mot négatif => positif léger

    inv_neg = 0
    for m in pos_matches:
        start = max(0, m.start() - 12)
        window = t[start:m.start()]
        if NEGATION_RGX.search(window):
            inv_neg += 1  # négation + mot positif => négatif léger

    pos = len(pos_matches) + inv_pos
    neg = len(neg_matches) + inv_neg

    if pos == neg == 0:
        return 0, 0.0
    if pos > neg:
        score = min(1.0, (pos - neg) / max(1, pos + neg))
        return 1, float(score)
    if neg > pos:
        score = min(1.0, (neg - pos) / max(1, pos + neg))
        return -1, float(score)
    return 0, 0.2

def hf_sentiment(text: str) -> Tuple[int, float]:
    """
    Mappe la sortie HF en (-1,0,1). Si indispo → heuristique.
    """
    if HF_PIPELINE is None:
        return heuristic_sentiment(text)
    try:
        out = HF_PIPELINE(text[:512])
        if not out or not isinstance(out, list) or not out[0]:
            return heuristic_sentiment(text)
        scores = {d["label"].lower(): d["score"] for d in out[0]}
        pos = scores.get("positive", 0.0)
        neg = scores.get("negative", 0.0)
        neu = scores.get("neutral", 0.0)
        if pos >= neg and pos >= neu:
            return 1, float(pos)
        if neg >= pos and neg >= neu:
            return -1, float(neg)
        return 0, float(neu)
    except Exception:
        return heuristic_sentiment(text)

# ===================== Pipeline KPI Sentiment =====================
def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute:
      - sentiment_label ∈ {-1,0,1}
      - sentiment_score ∈ [0,1]
      - week (lundi 00:00 UTC) pour agrégations
    """
    if df.empty:
        return df.assign(sentiment_label=pd.Series(dtype="Int8"),
                         sentiment_score=pd.Series(dtype=float),
                         week=pd.NaT)

    res = df["text"].astype(str).apply(hf_sentiment)
    df["sentiment_label"] = res.apply(lambda x: x[0]).astype("Int8")
    df["sentiment_score"] = res.apply(lambda x: x[1]).astype(float)

    ts = df["created_at"].dt.tz_convert("UTC")
    midnight = ts.dt.floor("D")
    df["week"] = (midnight - pd.to_timedelta(midnight.dt.weekday, unit="D")).dt.tz_localize(None)
    return df

def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    KPIs hebdo par plateforme :
      - mentions
      - pos_share, neg_share
      - sent_mean (moyenne label -1/0/1)
      - score_mean (moyenne des scores)
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "week","platform","mentions","pos_share","neg_share","sent_mean","score_mean"
        ])
    g = (df.groupby(["week","platform"], observed=True)
            .agg(mentions=("post_id","size"),
                 pos_share=("sentiment_label", lambda s: (s==1).mean() if len(s) else np.nan),
                 neg_share=("sentiment_label", lambda s: (s==-1).mean() if len(s) else np.nan),
                 sent_mean=("sentiment_label", "mean"),
                 score_mean=("sentiment_score","mean"))
            .reset_index())
    return g

def write_report(report_path: Path, wk: pd.DataFrame, overall: Dict[str, float]) -> None:
    lines = []
    lines.append("# Rapport Sentiment — Partie 4\n")
    lines.append("## KPIs globaux (période)")
    lines.append(f"- Mentions totales (YT+IG): **{int(overall['mentions_total']):,}**".replace(",", " "))
    lines.append(f"- Part positive: **{overall['pos_share']:.0%}** | négative: **{overall['neg_share']:.0%}** | score moyen: **{overall['score_mean']:.2f}**\n")

    if not wk.empty:
        lines.append("## Snapshot — 4 dernières semaines (par plateforme)")
        snap = wk.sort_values("week").groupby("platform", observed=True).tail(4)
        for plat in snap["platform"].unique():
            sub = snap[snap["platform"].eq(plat)]
            lines.append(f"### {plat}")
            for _, r in sub.iterrows():
                lines.append(f"- {r['week'].date()} • mentions **{int(r['mentions'])}** • "
                             f"pos **{r['pos_share']:.0%}** • neg **{r['neg_share']:.0%}** • "
                             f"sent_moy **{r['sent_mean']:.2f}** • score_moy **{r['score_mean']:.2f}**")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("[OK] Rapport Sentiment →", report_path)

# ===================== Main =====================
if __name__ == "__main__":
    paths = Paths()

    # 1) Load depuis data/raw
    yt = load_youtube_csv(paths.yt_comments)
    ig = load_instagram_csv(paths.instagram_posts)
    # Laisse None si tu ne veux pas les avis :
    gr = None  # load_greviews_csv(paths.greviews)

    frames = []
    for df in (yt, ig):
        if df is not None and not df.empty:
            frames.append(df[["platform","post_id","created_at","author_name","text"]])
    if gr is not None and not gr.empty:
        frames.append(gr[["platform","post_id","created_at","author_name","text"]])

    if not frames:
        raise ValueError("[Sentiment] Aucune source texte (YT/IG/GR) trouvée dans App/api/data/raw/.")

    all_posts = pd.concat(frames, ignore_index=True).dropna(subset=["created_at"])
    all_posts["text"] = all_posts["text"].fillna("").astype(str)

    # 2) Scoring
    scored = score_dataframe(all_posts)

    # 3) Agrégations
    weekly = aggregate_weekly(scored)

    # 4) Exports (CSV) vers processed/social
    outdir = Path(paths.out_dir); must_exist_dir(outdir)
    scored.to_csv(outdir / "sentiment_scored_posts.csv", index=False)
    weekly.to_csv(outdir / "sentiment_weekly.csv", index=False)

    # 5) KPIs globaux + rapport
    overall = {
        "mentions_total": int(scored.shape[0]),
        "pos_share": float((scored["sentiment_label"]==1).mean(skipna=True)),
        "neg_share": float((scored["sentiment_label"]==-1).mean(skipna=True)),
        "score_mean": float(scored["sentiment_score"].mean(skipna=True))
    }
    report_path = Path(paths.report_md); must_exist_dir(report_path.parent)
    write_report(report_path, weekly, overall)

    print("[OK] Exports Sentiment (partie 4) →", outdir)
