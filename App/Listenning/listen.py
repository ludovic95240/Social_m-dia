# App/Listenning/kpi_part3.py
# Partie 3 — KPIs Social Listening (YT + IG) — robuste, FR

from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np
import re

# ===================== Chemins (STRICT: ne crée rien) =====================
HERE = Path(__file__).resolve()
APP_ROOT = HERE.parents[1]  # .../App

RAW_DIR          = APP_ROOT / "api" / "data" / "raw"
PROC_DIR_SOCIAL  = APP_ROOT / "api" / "data" / "processed" / "social"
REPORTS_DIR      = APP_ROOT / "api" / "reports"

def must_exist_dir(p: Path) -> None:
    if not p.exists() or not p.is_dir():
        raise OSError(f"[KPI] Dossier requis manquant : {p}\n"
                      f"Crée-le manuellement sous App/ puis relance.")

for d in (RAW_DIR, PROC_DIR_SOCIAL, REPORTS_DIR):
    must_exist_dir(d)

# ===================== Réparation CSV YT (schéma 15 colonnes) =====================
YT_EXPECTED = [
    "platform","post_id","created_at","text","author_name","likes",
    "parent_id","parent_author_name",
    "video_id","video_title","channel_title","source_query",
    "video_views","video_likes","video_comment_count"
]

def repair_youtube_csv_inplace(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            old_header = next(reader)
        except StopIteration:
            with p.open("w", encoding="utf-8", newline="") as out:
                csv.writer(out).writerow(YT_EXPECTED)
            return

        if list(old_header) == YT_EXPECTED:
            return

        print(f"[FIX] CSV YT: header {len(old_header)} ≠ {len(YT_EXPECTED)} → réparation ligne à ligne…")
        idx_by_name = {str(n).strip().lower(): i for i, n in enumerate(old_header)}

        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as out:
            w = csv.writer(out)
            w.writerow(YT_EXPECTED)
            for row in reader:
                if not row:
                    continue
                out_row = [""] * len(YT_EXPECTED)
                for j, col in enumerate(YT_EXPECTED):
                    i = idx_by_name.get(col.lower())
                    if i is not None and i < len(row):
                        out_row[j] = row[i]
                w.writerow(out_row)

    p.unlink(); tmp.rename(p)
    print(f"[OK] Réparé → {p}")

# ===================== Config & Lexique =====================
@dataclass
class Paths:
    yt_comments: str = str(RAW_DIR / "yt_comments_enriched.csv")
    instagram_posts: str = str(RAW_DIR / "instagram_posts.csv")  # <- ajouté
    # Google Reviews facultatif (désactivé par défaut)
    greviews: str   = str(RAW_DIR / "google_reviews.csv")
    out_dir: str    = str(PROC_DIR_SOCIAL)
    report_md: str  = str(REPORTS_DIR / "Listen.md")

@dataclass
class CompetitorLexicon:
    brand: str = "honda"
    competitors: tuple[str, ...] = ("yamaha", "bmw", "ktm", "triumph", "ducati")
    brand_products: tuple[str, ...] = ("transalp","africa twin","cb500x","cb650r","cb750","nt1100","x-adv","goldwing")
    comp_products: tuple[str, ...] = ("ténéré","tenere","gs","890","tiger","scrambler")

# ===================== Chargement sources =====================
def load_youtube_csv(p: str | Path) -> pd.DataFrame:
    p = Path(p)
    if not p.exists():
        print(f"[INFO] Fichier YT introuvable: {p} → YT ignoré.")
        return pd.DataFrame(columns=[
            "platform","post_id","created_at","author_name","text","likes",
            "video_id","video_title","channel_title","video_views","video_likes","video_comment_count"
        ])
    repair_youtube_csv_inplace(p)
    df = pd.read_csv(p)
    df = pd.DataFrame({
        "platform": "youtube",
        "post_id": df.get("post_id"),
        "created_at": pd.to_datetime(df.get("created_at"), utc=True, errors="coerce"),
        "author_name": df.get("author_name"),
        "text": df.get("text", "").fillna(""),
        "likes": pd.to_numeric(df.get("likes", 0), errors="coerce").fillna(0).astype("Int64"),
        "video_id": df.get("video_id"),
        "video_title": df.get("video_title"),
        "channel_title": df.get("channel_title"),
        "video_views": pd.to_numeric(df.get("video_views"), errors="coerce"),
        "video_likes": pd.to_numeric(df.get("video_likes"), errors="coerce"),
        "video_comment_count": pd.to_numeric(df.get("video_comment_count"), errors="coerce"),
    })
    return df.dropna(subset=["created_at"])

def load_instagram_csv(p: str | Path) -> pd.DataFrame:
    """
    Charge un export 'instagram_posts.csv' flexible.
    Colonnes généralement présentes selon scrapers:
    - post_id, created_at, text/caption, author_name/username, likes, comments, views/plays, url
    """
    p = Path(p)
    if not p.exists():
        print(f"[INFO] Fichier IG introuvable: {p} → IG ignoré.")
        return pd.DataFrame(columns=[
            "platform","post_id","created_at","author_name","text","likes","comments","views","url"
        ])

    df = pd.read_csv(p)
    # Harmonisation noms de colonnes potentiels
    def pick(*names):
        for n in names:
            if n in df.columns:
                return df[n]
        return pd.Series([pd.NA] * len(df))

    out = pd.DataFrame({
        "platform": "instagram",
        "post_id": pick("post_id","id","shortcode"),
        "created_at": pd.to_datetime(pick("created_at","taken_at","timestamp","date"), utc=True, errors="coerce"),
        "author_name": pick("author_name","username","owner_username"),
        "text": pick("text","caption","description").fillna(""),
        "likes": pd.to_numeric(pick("likes","like_count"), errors="coerce").fillna(0).astype("Int64"),
        "comments": pd.to_numeric(pick("comments","comment_count"), errors="coerce").fillna(0).astype("Int64"),
        "views": pd.to_numeric(pick("views","play_count","video_view_count"), errors="coerce"),
        "url": pick("url","permalink","link"),
    })
    return out.dropna(subset=["created_at"])

def load_greviews_csv(p: str | Path) -> Optional[pd.DataFrame]:
    pp = Path(p)
    if not pp.exists():
        print("[INFO] google_reviews.csv introuvable → KPIs avis ignorés.")
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
        "city": df["city"] if "city" in df.columns else pd.NA,
    })
    return out.dropna(subset=["created_at"])

# ===================== Tags: brand / concurrents / produits =====================
def tag_entities(df: pd.DataFrame, lex: CompetitorLexicon) -> pd.DataFrame:
    t = df["text"].astype(str).str.lower()
    brand_re = re.compile(rf"\b{re.escape(lex.brand)}\b", re.IGNORECASE)
    df["is_brand"] = t.str.contains(brand_re, na=False)
    for comp in lex.competitors:
        rgx = re.compile(rf"\b{re.escape(comp)}\b", re.IGNORECASE)
        df[f"is_{comp}"] = t.str.contains(rgx, na=False)
    df["is_competitor"] = df[[f"is_{c}" for c in lex.competitors]].any(axis=1)
    prod_terms = tuple(w.lower() for w in (lex.brand_products + lex.comp_products))
    df["mentions_product"] = t.apply(lambda s: any(p in s for p in prod_terms))
    return df

# ===================== KPIs =====================
def compute_kpis(yt: pd.DataFrame, ig: pd.DataFrame, gr: Optional[pd.DataFrame], lex: CompetitorLexicon) -> Dict[str, pd.DataFrame]:
    # union basique (Google Reviews optionnel → désactivé par défaut)
    frames = []
    if not yt.empty: frames.append(yt[[
        "platform","post_id","created_at","author_name","text","likes",
        "video_id","video_title","channel_title","video_views","video_likes","video_comment_count"
    ]])
    if not ig.empty: frames.append(ig[[
        "platform","post_id","created_at","author_name","text","likes"
    ]].assign(video_id=pd.NA, video_title=pd.NA, channel_title=pd.NA,
              video_views=pd.NA, video_likes=pd.NA, video_comment_count=pd.NA))
    if gr is not None and not gr.empty:
        frames.append(pd.DataFrame({
            "platform": "google_reviews",
            "post_id": gr["post_id"],
            "created_at": gr["created_at"],
            "author_name": gr["author_name"],
            "text": gr["text"],
            "likes": pd.Series([pd.NA]*len(gr), dtype="Int64"),
            "video_id": pd.NA, "video_title": pd.NA, "channel_title": pd.NA,
            "video_views": pd.NA, "video_likes": pd.NA, "video_comment_count": pd.NA
        }))

    if not frames:
        raise ValueError("[KPI] Aucune source chargée (YT/IG vides).")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["created_at"]).copy()
    df["date"] = df["created_at"].dt.tz_convert("UTC").dt.date
    # semaine ISO (lundi 00:00 UTC)
    ts = df["created_at"].dt.tz_convert("UTC")
    midnight = ts.dt.floor("D")
    df["week"] = (midnight - pd.to_timedelta(midnight.dt.weekday, unit="D")).dt.tz_localize(None)

    # tags
    df = tag_entities(df, lex)

    # 1) Volume jour / plate-forme
    volume_day = (df.groupby(["platform","date"], observed=True, as_index=False)
                  .size().rename(columns={"size":"mentions"}))

    # 2) SOV hebdo brand vs competitors
    sov_weekly = (df.assign(bucket=df["week"],
                            who=np.where(df["is_brand"], "honda",
                                  np.where(df["is_competitor"], "competitors", "other"))))
    sov_weekly = (sov_weekly.groupby(["bucket","who"], observed=True, as_index=False)
                             .size())
    pv = sov_weekly.pivot(index="bucket", columns="who", values="size").fillna(0)
    for col in ("honda","competitors"):
        if col not in pv.columns: pv[col] = 0
    pv["sov_honda"] = pv["honda"] / (pv["honda"] + pv["competitors"]).replace(0, np.nan)
    sov_weekly = pv.reset_index()

    # 3) Engagement par plateforme
    # --- YouTube (comme avant)
    yt_only = df[df["platform"].eq("youtube")].dropna(subset=["video_id"]).copy()
    if yt_only.empty:
        top_videos = pd.DataFrame(columns=["video_id","video_title","channel_title","views","likes","comments","eng_score"])
        eng_summary_yt = pd.DataFrame([{"eng_top10_mean_yt": np.nan, "eng_median_yt": np.nan}])
        awareness_weekly = pd.DataFrame(columns=["week","views_sum"])
    else:
        vid = (yt_only.groupby(["video_id","video_title","channel_title"], observed=True, as_index=False)
                        .agg(views=("video_views","max"),
                             likes=("video_likes","max"),
                             comments=("video_comment_count","max")))
        vid["views"]    = pd.to_numeric(vid["views"], errors="coerce")
        vid["likes"]    = pd.to_numeric(vid["likes"], errors="coerce").fillna(0.0)
        vid["comments"] = pd.to_numeric(vid["comments"], errors="coerce").fillna(0.0)
        vid["eng_score"] = (vid["likes"] + vid["comments"] + 1.0) / vid["views"].replace(0, np.nan)
        top_videos = vid.sort_values("eng_score", ascending=False).head(10)
        eng_summary_yt = pd.DataFrame([{
            "eng_top10_mean_yt": float(top_videos["eng_score"].mean(skipna=True)) if not top_videos.empty else np.nan,
            "eng_median_yt": float(vid["eng_score"].median(skipna=True)) if not vid.empty else np.nan
        }])
        yt_only["week"] = df.loc[yt_only.index, "week"]
        awareness_weekly = (yt_only.groupby("week", observed=True)["video_views"]
                                     .max().groupby("week").sum().reset_index()
                                     .rename(columns={"video_views":"views_sum"}))

    # --- Instagram (likes, comments, views si dispo)
    ig_full = load_instagram_csv(Paths().instagram_posts)  # pour avoir comments/views
    if ig_full.empty:
        top_posts_ig = pd.DataFrame(columns=["post_id","author_name","text","likes","comments","views","eng_score"])
        eng_summary_ig = pd.DataFrame([{"eng_top10_mean_ig": np.nan, "eng_median_ig": np.nan}])
    else:
        ig_full["likes"] = pd.to_numeric(ig_full.get("likes"), errors="coerce").fillna(0)
        ig_full["comments"] = pd.to_numeric(ig_full.get("comments"), errors="coerce").fillna(0)
        ig_full["views"] = pd.to_numeric(ig_full.get("views"), errors="coerce")
        # Si pas de vues -> on normalise par un facteur constant pour éviter division par 0
        denom = ig_full["views"].replace(0, np.nan)
        # fallback: si views manquant partout, utilise un score brut likes+comments
        if denom.notna().sum() == 0:
            ig_full["eng_score"] = ig_full["likes"] + ig_full["comments"]
        else:
            ig_full["eng_score"] = (ig_full["likes"] + ig_full["comments"] + 1.0) / denom
            # Remplace NaN (posts sans vues) par score brut pour ne pas les exclure
            mask_nan = ig_full["eng_score"].isna()
            ig_full.loc[mask_nan, "eng_score"] = (ig_full.loc[mask_nan, "likes"] + ig_full.loc[mask_nan, "comments"]).astype(float)

        top_posts_ig = ig_full.sort_values("eng_score", ascending=False).head(10)[
            ["post_id","author_name","text","likes","comments","views","eng_score"]
        ]
        eng_summary_ig = pd.DataFrame([{
            "eng_top10_mean_ig": float(top_posts_ig["eng_score"].mean(skipna=True)) if not top_posts_ig.empty else np.nan,
            "eng_median_ig": float(ig_full["eng_score"].median(skipna=True)) if not ig_full.empty else np.nan
        }])

    # 4) (Optionnel) Google reviews KPIs — désactivé si gr is None
    if gr is None or gr.empty:
        reviews_weekly = pd.DataFrame(columns=["week","reviews","avg_rating","pos_share"])
        reviews_overall = pd.DataFrame([{"reviews_total":0,"avg_rating":np.nan,"pos_share":np.nan}])
    else:
        gr2 = gr.copy()
        gr2["week"] = (gr2["created_at"].dt.tz_convert("UTC").dt.floor("D") -
                       pd.to_timedelta(gr2["created_at"].dt.weekday, unit="D")).dt.tz_localize(None)
        gr2["is_pos"] = gr2["rating"] >= 4
        reviews_weekly = (gr2.groupby("week", observed=True)
                             .agg(reviews=("rating","size"),
                                  avg_rating=("rating","mean"),
                                  pos_share=("is_pos","mean"))
                             .reset_index())
        reviews_overall = pd.DataFrame([{
            "reviews_total": int(gr2["rating"].shape[0]),
            "avg_rating": float(gr2["rating"].mean(skipna=True)),
            "pos_share": float(gr2["is_pos"].mean(skipna=True))
        }])

    # 5) Couverture produit (dans mentions Honda)
    brand_df = df[df["is_brand"]]
    prod_cov_weekly = (brand_df.groupby("week", observed=True)["mentions_product"]
                       .mean().reset_index().rename(columns={"mentions_product":"product_cover_share"}))

    # 6) Pics (z-score sur volume total)
    daily = (df.groupby("date", observed=True, as_index=False).size()
               .rename(columns={"size":"mentions"}))
    roll = daily["mentions"].rolling(14, min_periods=7)
    daily["ma"] = roll.mean(); daily["sd"] = roll.std(ddof=0)
    daily["z"] = (daily["mentions"] - daily["ma"]) / daily["sd"]
    daily["is_peak"] = daily["z"] > 2.5

    # ——— Table KPI hebdo consolidée ———
    base_week = pd.DataFrame({"week": sorted(df["week"].dropna().unique())})
    kpi_week = base_week.merge(sov_weekly[["bucket","sov_honda"]], left_on="week", right_on="bucket", how="left")\
                        .drop(columns=["bucket"])
    # mentions / semaine
    wk_vol = (df.groupby("week", observed=True).size()
                .rename("mentions_total").reset_index())
    kpi_week = kpi_week.merge(wk_vol, on="week", how="left")
    # split plate-forme
    wk_ch = (df.groupby(["week","platform"], observed=True).size()
               .rename("mentions").reset_index()
               .pivot(index="week", columns="platform", values="mentions").fillna(0))
    wk_ch = wk_ch.add_prefix("mentions_").reset_index()
    kpi_week = kpi_week.merge(wk_ch, on="week", how="left")
    # awareness YT
    if not awareness_weekly.empty:
        kpi_week = kpi_week.merge(awareness_weekly, on="week", how="left")
    else:
        kpi_week["views_sum"] = np.nan
    # reviews (optionnel)
    if not reviews_weekly.empty:
        kpi_week = kpi_week.merge(reviews_weekly, on="week", how="left")
    else:
        kpi_week[["reviews","avg_rating","pos_share"]] = np.nan
    # couverture produit
    kpi_week = kpi_week.merge(prod_cov_weekly, on="week", how="left")

    # ——— KPI overall ———
    kpi_overall = pd.DataFrame([{
        "mentions_total": int(df.shape[0]),
        "sov_honda_mean": float(kpi_week["sov_honda"].mean(skipna=True)),
        # YT
        "eng_top10_mean_yt": float(eng_summary_yt["eng_top10_mean_yt"].iloc[0]) if not eng_summary_yt.empty else np.nan,
        "eng_median_yt": float(eng_summary_yt["eng_median_yt"].iloc[0]) if not eng_summary_yt.empty else np.nan,
        "views_sum_total": float(kpi_week["views_sum"].sum(skipna=True)) if "views_sum" in kpi_week else np.nan,
        # IG
        "eng_top10_mean_ig": float(eng_summary_ig["eng_top10_mean_ig"].iloc[0]) if not eng_summary_ig.empty else np.nan,
        "eng_median_ig": float(eng_summary_ig["eng_median_ig"].iloc[0]) if not eng_summary_ig.empty else np.nan,
        # Reviews optionnels
        "reviews_total": int(reviews_overall["reviews_total"].iloc[0]) if not reviews_overall.empty else 0,
        "avg_rating": float(reviews_overall["avg_rating"].iloc[0]) if not reviews_overall.empty else np.nan,
        "pos_share": float(reviews_overall["pos_share"].iloc[0]) if not reviews_overall.empty else np.nan,
        # Produit + pics
        "product_cover_mean": float(kpi_week["product_cover_share"].mean(skipna=True)),
        "peak_days": int(daily["is_peak"].sum())
    }])

    # tops
    top_videos_out = (top_videos.copy() if "top_videos" in locals() else
                      pd.DataFrame(columns=["video_id","video_title","channel_title","views","likes","comments","eng_score"]))
    top_instagram_out = (top_posts_ig.copy() if "top_posts_ig" in locals() else
                         pd.DataFrame(columns=["post_id","author_name","text","likes","comments","views","eng_score"]))

    return {
        "kpi_week": kpi_week,
        "kpi_overall": kpi_overall,
        "volume_day": volume_day,
        "sov_weekly": sov_weekly,
        "daily_peaks": daily,
        "top_videos": top_videos_out,
        "top_instagram": top_instagram_out
    }

# ===================== Rapport =====================
def write_report(paths: Paths, kpis: Dict[str, pd.DataFrame]) -> None:
    kpw = kpis["kpi_week"].copy()
    kpo = kpis["kpi_overall"].iloc[0].to_dict()
    peaks = kpis["daily_peaks"]
    topv = kpis["top_videos"]
    topig = kpis["top_instagram"]

    lines = []
    lines.append("# Rapport KPIs — Partie 3\n")
    # Overall
    lines.append("## KPIs Globaux (période)")
    lines.append(f"- Mentions totales : **{int(kpo['mentions_total']):,}**".replace(",", " "))
    if not np.isnan(kpo.get("sov_honda_mean", np.nan)):
        lines.append(f"- SOV Honda (moyenne hebdo) : **{kpo['sov_honda_mean']:.0%}**")
    if not np.isnan(kpo.get("eng_top10_mean_yt", np.nan)):
        lines.append(f"- Engagement YT (moy. top10) : **{kpo['eng_top10_mean_yt']:.2%}** (médiane : {kpo['eng_median_yt']:.2%})")
    if not np.isnan(kpo.get("eng_top10_mean_ig", np.nan)):
        lines.append(f"- Engagement IG (moy. top10) : **{kpo['eng_top10_mean_ig']:.2%}** (médiane : {kpo['eng_median_ig']:.2%})")
    if not np.isnan(kpo.get("views_sum_total", np.nan)):
        lines.append(f"- Awareness YT (somme vues) : **~{int(kpo['views_sum_total']):,}**".replace(",", " "))
    if not np.isnan(kpo.get("product_cover_mean", np.nan)):
        lines.append(f"- Couverture produits (dans mentions Honda) : **{kpo['product_cover_mean']:.0%}**")
    lines.append(f"- Jours “pics” (z>2.5) : **{int(kpo['peak_days'])}**\n")

    # Dernières semaines (snapshot)
    lines.append("## Snapshot — 4 dernières semaines")
    snap = kpw.sort_values("week").tail(4).copy()
    def _fmt(n):
        return "—" if pd.isna(n) else (f"{n:.0%}" if (isinstance(n, float) and n<=1.0) else f"{int(n):,}".replace(",", " "))
    for _, r in snap.iterrows():
        lines.append(f"- {r['week'].date()} : mentions **{_fmt(r['mentions_total'])}**, "
                     f"SOV **{_fmt(r.get('sov_honda'))}**, "
                     f"YT **{_fmt(r.get('mentions_youtube'))}**, "
                     f"IG **{_fmt(r.get('mentions_instagram'))}**")

    # Tops
    if not topv.empty:
        lines.append("\n## Top vidéos YouTube (engagement proxy)")
        for _, rv in topv.head(10).iterrows():
            rate = 0.0 if pd.isna(rv["eng_score"]) else float(rv["eng_score"])
            lines.append(f"- {rv['video_title']} — {rv['channel_title']} | score≈{rate:.2%}")
    if not topig.empty:
        lines.append("\n## Top posts Instagram (engagement proxy)")
        for _, rv in topig.head(10).iterrows():
            rate = 0.0 if pd.isna(rv["eng_score"]) else float(rv["eng_score"])
            lines.append(f"- @{rv['author_name']} — likes {int(rv['likes'])} / com {int(rv['comments'])} | score≈{rate:.2%}")

    out = Path(paths.report_md)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("[OK] Rapport KPIs →", out)

# ===================== Main =====================
if __name__ == "__main__":
    paths = Paths()
    lex = CompetitorLexicon()

    yt = load_youtube_csv(paths.yt_comments)
    ig = load_instagram_csv(paths.instagram_posts)

    # IMPORTANT : tu ne veux pas Google Reviews → on passe None.
    gr = None  # load_greviews_csv(paths.greviews) si un jour tu réactives

    kpis = compute_kpis(yt, ig, gr, lex)

    outdir = Path(paths.out_dir); must_exist_dir(outdir)
    kpis["kpi_week"].to_csv(outdir / "kpis_weekly.csv", index=False)
    kpis["kpi_overall"].to_csv(outdir / "kpis_overall.csv", index=False)
    # exports utiles pour debug/présentation
    kpis["volume_day"].to_csv(outdir / "volume_by_day_kpi.csv", index=False)
    kpis["sov_weekly"].to_csv(outdir / "sov_weekly_kpi.csv", index=False)
    kpis["daily_peaks"].to_csv(outdir / "daily_peaks_kpi.csv", index=False)
    kpis["top_videos"].to_csv(outdir / "top_videos_kpi.csv", index=False)
    kpis["top_instagram"].to_csv(outdir / "top_instagram_kpi.csv", index=False)

    write_report(paths, kpis)
    print("[OK] Exports KPIs (partie 3) →", outdir)
