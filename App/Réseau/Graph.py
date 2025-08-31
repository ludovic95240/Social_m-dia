# App/Network/network_ig_yt_gr.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import re
import pandas as pd
import networkx as nx

# Louvain (python-louvain). Si non installé, on taggue communauté = -1.
try:
    import community as community_louvain
except Exception:
    community_louvain = None

# --------------------- Paths ---------------------
HERE = Path(__file__).resolve()
BASE = HERE.parents[1]                                   # .../App
RAW_DIR  = BASE / "api" / "data" / "raw"
PROC_DIR = BASE / "api" / "processed" / "network"
PROC_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class Paths:
    ig_csv: str      = str(RAW_DIR / "instagram_posts.csv")
    yt_csv: str      = str(RAW_DIR / "yt_comments_enriched.csv")
    gr_csv: str      = str(RAW_DIR / "google_reviews.csv")
    out_nodes: str   = str(PROC_DIR / "nodes.csv")
    out_edges: str   = str(PROC_DIR / "edges.csv")
    out_top: str     = str(PROC_DIR / "top_influencers.csv")
    out_gexf: str    = str(PROC_DIR / "interaction_graph.gexf")

@dataclass
class BuildParams:
    min_edge_weight: int = 1        # filtrage des toutes petites arêtes à l’export

# --------------------- Repository: chargement & normalisation ---------------------
class Repo:
    """
    On standardise chaque source avec les colonnes utiles à la construction du graphe.
    """
    def __init__(self, p: Paths):
        self.p = p

    # -------- Instagram --------
    def load_instagram(self) -> Optional[pd.DataFrame]:
        f = Path(self.p.ig_csv)
        if not f.exists():
            print("[INFO] instagram_posts.csv introuvable → IG ignoré.")
            return None
        df = pd.read_csv(f)
        # Attendus (minimum) : text, author_name, created_at
        # On essaie de trouver des colonnes alias si besoin.
        if "author_name" not in df.columns:
            for alt in ("username", "user_name", "owner_username"):
                if alt in df.columns:
                    df["author_name"] = df[alt]; break
        if "text" not in df.columns:
            for alt in ("caption", "body", "content"):
                if alt in df.columns:
                    df["text"] = df[alt]; break
        for c in ("author_name", "text"):
            if c not in df.columns:
                df[c] = ""

        df["author_name"] = df["author_name"].fillna("").astype(str)
        df["text"] = df["text"].fillna("").astype(str)
        df["created_at"] = pd.to_datetime(df.get("created_at"), utc=True, errors="coerce")
        df = df.dropna(subset=["created_at"]).copy()

        # Extraire les @mentions depuis la légende/texte
        # Regex IG approx. : lettres/chiffres/._, 1-30, ne commence pas par point
        mention_re = re.compile(r"@([A-Za-z0-9](?:[A-Za-z0-9._]{0,29}))")
        df["mentions"] = df["text"].str.findall(mention_re).apply(lambda xs: "|".join(xs) if xs else "")
        return df[["author_name","text","mentions","created_at"]]

    # -------- YouTube --------
    def load_youtube(self) -> Optional[pd.DataFrame]:
        f = Path(self.p.yt_csv)
        if not f.exists():
            print("[INFO] yt_comments_enriched.csv introuvable → YouTube ignoré.")
            return None
        df = pd.read_csv(f)
        df["author_name"]   = df.get("author_name").fillna("").astype(str)
        df["channel_title"] = df.get("channel_title").fillna("").astype(str)
        df["created_at"]    = pd.to_datetime(df.get("created_at"), utc=True, errors="coerce")
        df = df.dropna(subset=["created_at"])
        return df[["author_name","channel_title","created_at"]]

    # -------- Google Reviews --------
    def load_greviews(self) -> Optional[pd.DataFrame]:
        f = Path(self.p.gr_csv)
        if not f.exists():
            print("[INFO] google_reviews.csv introuvable → Google Reviews ignoré.")
            return None
        df = pd.read_csv(f)

        # place_name robustifier
        place_col = None
        for c in ("place_name","place","business_name"):
            if c in df.columns: place_col = c; break
        if place_col is None:
            df["place_name"] = df.get("place_id","").fillna("").astype(str)
            place_col = "place_name"

        # ville (optionnel)
        city_col = None
        for c in ("city","locality","town"):
            if c in df.columns: city_col = c; break
        if city_col is None:
            df["city"] = ""
            city_col = "city"

        df["author_name"] = df.get("author_name").fillna("").astype(str)
        df["place_name"]  = df[place_col].fillna("").astype(str)
        df["city"]        = df[city_col].fillna("").astype(str)
        df["rating"]      = pd.to_numeric(df.get("rating"), errors="coerce")
        df["created_at"]  = pd.to_datetime(df.get("created_at"), utc=True, errors="coerce")
        df = df.dropna(subset=["created_at"])
        return df[["author_name","place_name","city","rating","created_at"]]

# --------------------- Graph Builder ---------------------
class GraphBuilder:
    """
    Types d’arêtes :
      - IG  : author -> @mentionné           (etype='ig_mention')
      - YT  : commentateur -> yt:{chaine}    (etype='yt_comment')
      - GR  : auteur_avis -> gr:{lieu}       (etype='gr_review', agrège rating moyen sur l’arête)
    """
    def __init__(self):
        self.G = nx.DiGraph()

    @staticmethod
    def _add_node(G: nx.DiGraph, node: str, **attrs):
        if not node: return
        if node not in G:
            G.add_node(node, **attrs)
        else:
            # merge souple
            for k, v in attrs.items():
                if k not in G.nodes[node] or not G.nodes[node][k]:
                    G.nodes[node][k] = v

    @staticmethod
    def _add_edge(G: nx.DiGraph, src: str, dst: str, etype: str, w: int = 1, extra: Optional[Dict[str,Any]] = None):
        if not src or not dst or src == dst: return
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += w
            G[src][dst]["types"].add(etype)
            if extra:
                for k, v in extra.items():
                    if k == "rating" and v is not None:
                        # moyenne incrémentale
                        if "rating_sum" not in G[src][dst]:
                            G[src][dst]["rating_sum"] = 0.0
                            G[src][dst]["rating_n"] = 0
                        G[src][dst]["rating_sum"] += float(v)
                        G[src][dst]["rating_n"] += 1
                    else:
                        G[src][dst][k] = v
        else:
            d = {"weight": w, "types": {etype}}
            if extra: d.update(extra)
            if extra and "rating" in extra and extra["rating"] is not None:
                d["rating_sum"] = float(extra["rating"])
                d["rating_n"]   = 1
            G.add_edge(src, dst, **d)

    # ---- IG ----
    def ingest_instagram(self, dfi: pd.DataFrame):
        for _, r in dfi.iterrows():
            src = (r.get("author_name") or "").strip()
            self._add_node(self.G, src, node_type="ig_user", platform="instagram", label=src)
            mentions = str(r.get("mentions") or "")
            # fallback si "mentions" vide → re-scan texte
            if not mentions:
                txt = str(r.get("text") or "")
                m = re.findall(r"@([A-Za-z0-9](?:[A-Za-z0-9._]{0,29}))", txt)
                mentions = "|".join(m)
            for tgt in [m.strip() for m in mentions.split("|") if m.strip()]:
                node_tgt = "@"+tgt
                self._add_node(self.G, node_tgt, node_type="ig_user", platform="instagram", label=node_tgt)
                self._add_edge(self.G, src, node_tgt, etype="ig_mention", w=1)

    # ---- YT ----
    def ingest_youtube(self, dfy: pd.DataFrame):
        for _, r in dfy.iterrows():
            src = (r.get("author_name") or "").strip()
            ch  = (r.get("channel_title") or "").strip()
            if not src or not ch:
                continue
            dst = "yt:" + ch
            self._add_node(self.G, src, node_type="yt_user", platform="youtube", label=src)
            self._add_node(self.G, dst, node_type="yt_channel", platform="youtube", label=ch)
            self._add_edge(self.G, src, dst, etype="yt_comment", w=1)

    # ---- GR ----
    def ingest_greviews(self, dfr: pd.DataFrame):
        for _, r in dfr.iterrows():
            src = (r.get("author_name") or "").strip()
            plc = (r.get("place_name")  or "").strip()
            if not src or not plc:
                continue
            dst = "gr:" + plc
            city = (r.get("city") or "").strip()
            rating = r.get("rating", None)
            self._add_node(self.G, src, node_type="gr_user", platform="google_reviews", label=src)
            self._add_node(self.G, dst, node_type="gr_place", platform="google_reviews", label=plc, city=city)
            self._add_edge(self.G, src, dst, etype="gr_review", w=1, extra={"rating": rating, "city": city})

# --------------------- Mesures & Communautés ---------------------
class MetricsComputer:
    def __init__(self, G: nx.DiGraph):
        self.G = G
    def compute(self) -> pd.DataFrame:
        G = self.G
        indeg  = dict(G.in_degree(weight="weight"))
        outdeg = dict(G.out_degree(weight="weight"))
        btw    = nx.betweenness_centrality(G, weight="weight", normalized=True)
        try:
            eig = nx.eigenvector_centrality_numpy(G.to_undirected(), weight="weight")
        except Exception:
            eig = {n: 0.0 for n in G.nodes()}
        rows = []
        for n, d in G.nodes(data=True):
            rows.append({
                "node": n,
                "label": d.get("label", n),
                "platform": d.get("platform",""),
                "node_type": d.get("node_type",""),
                "city": d.get("city",""),
                "in_degree": float(indeg.get(n, 0)),
                "out_degree": float(outdeg.get(n, 0)),
                "betweenness": float(btw.get(n, 0.0)),
                "eigenvector": float(eig.get(n, 0.0)),
            })
        df = pd.DataFrame(rows)
        return df.sort_values(["eigenvector","betweenness","in_degree"], ascending=False)

class CommunityDetector:
    def __init__(self, G: nx.DiGraph):
        self.G = G
    def louvain(self) -> pd.Series:
        if community_louvain is None or self.G.number_of_nodes() == 0:
            return pd.Series({n: -1 for n in self.G.nodes()}, name="community")
        part = community_louvain.best_partition(self.G.to_undirected(), weight="weight", random_state=42)
        return pd.Series(part, name="community")

# --------------------- Export ---------------------
class Exporter:
    def __init__(self, p: Paths, min_edge_weight: int = 1):
        self.p = p
        self.min_w = max(1, int(min_edge_weight))

    def _edges_df(self, G: nx.DiGraph) -> pd.DataFrame:
        rows: List[Dict[str,Any]] = []
        for u, v, d in G.edges(data=True):
            if d.get("weight", 1) < self.min_w:
                continue
            rating_avg = None
            if "rating_sum" in d and d.get("rating_n", 0) > 0:
                rating_avg = d["rating_sum"] / d["rating_n"]
            rows.append({
                "source": u,
                "target": v,
                "weight": d.get("weight", 1),
                "types": "|".join(sorted(d.get("types", set())) if isinstance(d.get("types"), set) else [d.get("types","")]),
                "rating_avg": rating_avg,
                "city": d.get("city","")
            })
        return pd.DataFrame(rows)

    def write_all(self, G: nx.DiGraph, nodes: pd.DataFrame) -> None:
        # CSV nodes/edges
        edges = self._edges_df(G)
        nodes.to_csv(self.p.out_nodes, index=False)
        edges.to_csv(self.p.out_edges, index=False)

        # GEXF (convertir types set->str)
        G2 = G.copy()
        for _, _, d in G2.edges(data=True):
            if isinstance(d.get("types"), set):
                d["types"] = "|".join(sorted(list(d["types"])))
        nx.write_gexf(G2, self.p.out_gexf)

        # Top influenceurs (score mixte simple)
        top = nodes.copy()
        def _nzmax(s: pd.Series) -> float:
            m = s.max()
            return float(m) if float(m) > 0 else 1.0
        top["score"] = (
            0.5 * (top["eigenvector"] / _nzmax(top["eigenvector"])) +
            0.3 * (top["betweenness"] / _nzmax(top["betweenness"])) +
            0.2 * (top["in_degree"] / _nzmax(top["in_degree"]))
        )
        top = top.sort_values("score", ascending=False).head(50)
        top.to_csv(self.p.out_top, index=False)

        print("[OK] exports :")
        print(" -", self.p.out_nodes)
        print(" -", self.p.out_edges)
        print(" -", self.p.out_gexf)
        print(" -", self.p.out_top)

# --------------------- Orchestrateur ---------------------
class NetworkAnalyzer:
    def __init__(self, p: Paths, params: BuildParams):
        self.p = p; self.params = params
    def run(self) -> None:
        repo = Repo(self.p)
        G = GraphBuilder()

        dfi = repo.load_instagram()
        dfy = repo.load_youtube()
        dfr = repo.load_greviews()

        if dfi is None and dfy is None and dfr is None:
            raise FileNotFoundError("Aucune source disponible (IG / YouTube / Google Reviews).")

        if dfi is not None: G.ingest_instagram(dfi)
        if dfy is not None: G.ingest_youtube(dfy)
        if dfr is not None: G.ingest_greviews(dfr)

        graph = G.G
        metrics = MetricsComputer(graph).compute()
        comm = CommunityDetector(graph).louvain()
        nodes = metrics.join(comm, on="node")

        Exporter(self.p, self.params.min_edge_weight).write_all(graph, nodes)

# --------------------- main ---------------------
if __name__ == "__main__":
    paths = Paths()
    params = BuildParams(min_edge_weight=2)   # ↑ augmente si tu veux un graphe plus “propre”
    NetworkAnalyzer(paths, params).run()
