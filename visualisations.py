# -*- coding: utf-8 -*-
"""
=============================================================================
CineGraph 2026 — Visualisation Script
=============================================================================
Generates six interactive/static visualisations from the movies CSV dataset.
No API calls required — all graphs are built directly from the CSV data.

Requirements:
    pip install pandas networkx pyvis matplotlib seaborn

Output files (saved to the same folder as the CSV):
    1. actor_movie_network.html       — Actor–Movie Network Graph
    2. collaboration_network.html     — Director–Actor Collaboration Network
    3. genre_evolution_timeline.png   — Genre Evolution Timeline
    4. recommendation_graph.html      — Movie Recommendation/Similarity Graph
    5. centrality_charts.png          — Centrality & Influence Charts
    6. pathfinding.html               — Pathfinding / Hidden Connections Graph

Usage:
    python visualisations.py

    When prompted, enter two movie titles for the pathfinding visualisation.
    All other graphs generate automatically.
=============================================================================
"""

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — renders to file, not screen
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
import json
from pathlib import Path
from pyvis.network import Network
from difflib import get_close_matches

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# All paths are relative to this script's folder, so the project runs from
# any location. Keep movies.csv in the same folder as this file.
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "movies.csv"

# Graphs and charts are saved next to the script
OUTPUT_DIR = BASE_DIR

# Styling constants
BACKGROUND_COLOUR = "#0d0d0d"
ACCENT_GOLD       = "#f5c518"       # IMDB-style gold
ACCENT_TEAL       = "#00c9b1"
ACCENT_CORAL      = "#ff6b6b"
ACCENT_GREY       = "#444444"
TEXT_COLOUR       = "#e8e8e8"

# Graph size limits — keep visualisations readable
MAX_ACTOR_NODES        = 100    # top N actors by film count for actor-movie graph
MAX_COLLAB_DIRECTORS   = 30     # top N directors for collaboration network
MAX_RECO_MOVIES        = 80     # top N movies for recommendation graph
MIN_VOTE_COUNT         = 100    # minimum votes for a film to appear in graphs
MIN_CENTRALITY_NODES   = 20     # minimum nodes to compute centrality meaningfully

# ---------------------------------------------------------------------------
# DATA LOADING & SHARED UTILITIES
# ---------------------------------------------------------------------------

def load_data(csv_path: Path | str) -> pd.DataFrame:
    """Load the movies CSV and apply basic quality filters."""
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} movies from CSV")

    # Drop rows missing the fields most graphs depend on
    df = df.dropna(subset=["title", "release_year"])
    df["release_year"] = df["release_year"].astype(int)
    df["vote_count"]   = pd.to_numeric(df["vote_count"],   errors="coerce").fillna(0)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)
    df["popularity"]   = pd.to_numeric(df["popularity"],   errors="coerce").fillna(0)

    print(f"  {len(df):,} movies after quality filter")
    return df


def split_field(value, delimiter=", ") -> list[str]:
    """Safely split a delimited string field into a list, handling nulls."""
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [v.strip() for v in str(value).split(delimiter) if v.strip()]


def save_html(net: Network, filename: str) -> None:
    """Save a PyVis network to the output directory with explicit UTF-8 encoding."""
    path = str(OUTPUT_DIR / filename)

    # PyVis's save_graph() inherits the Windows default encoding (cp1252),
    # which can't encode characters like → (\u2192). We instead generate
    # the HTML and write it ourselves with explicit UTF-8 encoding.
    html_content = net.generate_html()
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  ✓ Saved: {path}")


def pyvis_base(height="800px", width="100%", heading="") -> Network:
    """
    Returns a pre-configured PyVis Network with a dark cinema theme.
    All six HTML graphs share this base configuration for visual consistency.
    """
    net = Network(height=height, width=width, bgcolor=BACKGROUND_COLOUR,
                  font_color=TEXT_COLOUR, heading=heading)
    net.set_options(json.dumps({
        "nodes": {
            "borderWidth": 2,
            "shadow": {"enabled": True, "size": 10}
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "shadow": False
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -60,
                "centralGravity": 0.005,
                "springLength": 120,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "navigationButtons": True,
            "keyboard": True
        }
    }))
    return net


# ---------------------------------------------------------------------------
# GRAPH 1 — ACTOR–MOVIE NETWORK
# Nodes: actors (gold) and movies (teal). Edges connect each actor to every
# film they appeared in. Node size reflects film count (actors) or rating
# (movies). Filtered to the top N most-connected actors for readability.
# ---------------------------------------------------------------------------

def graph1_actor_movie_network(df: pd.DataFrame) -> None:
    print("\n[1/6] Building Actor–Movie Network...")

    # Count films per actor to identify the most connected actors
    actor_film_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        for actor in split_field(row.get("cast_names", ""))[:8]:
            actor_film_counts[actor] = actor_film_counts.get(actor, 0) + 1

    # Keep only the top N actors to keep the graph readable
    top_actors = set(
        sorted(actor_film_counts, key=actor_film_counts.get, reverse=True)[:MAX_ACTOR_NODES]
    )

    # Build the graph — only include movies that feature at least one top actor
    net = pyvis_base(heading="CineGraph — Actor–Movie Network")
    added_movies: set[str] = set()
    added_actors: set[str] = set()

    for _, row in df.iterrows():
        actors_in_film = [
            a for a in split_field(row.get("cast_names", ""))[:8]
            if a in top_actors
        ]
        if not actors_in_film:
            continue

        movie_id  = f"movie_{row['tmdb_id']}"
        movie_label = f"{row['title']} ({row['release_year']})"
        rating    = float(row["vote_average"]) if row["vote_average"] > 0 else 5.0
        # Scale movie node size by rating (5–20px)
        movie_size = 5 + (rating / 10) * 15

        if movie_id not in added_movies:
            net.add_node(
                movie_id,
                label=row["title"],
                title=f"{movie_label}\nRating: {rating:.1f}/10\nGenres: {row.get('genres', 'N/A')}",
                color=ACCENT_TEAL,
                size=movie_size,
                shape="dot",
            )
            added_movies.add(movie_id)

        for actor in actors_in_film:
            actor_id = f"actor_{actor.lower().replace(' ', '_')}"
            # Scale actor node size by number of films (8–25px)
            actor_size = 8 + min(actor_film_counts[actor], 20) * 0.85
            if actor_id not in added_actors:
                net.add_node(
                    actor_id,
                    label=actor,
                    title=f"{actor}\nFilms in dataset: {actor_film_counts[actor]}",
                    color=ACCENT_GOLD,
                    size=actor_size,
                    shape="ellipse",
                )
                added_actors.add(actor_id)
            net.add_edge(actor_id, movie_id, color="#333333", width=1)

    save_html(net, "actor_movie_network.html")
    print(f"  Nodes: {len(net.nodes):,}  |  Edges: {len(net.edges):,}")


# ---------------------------------------------------------------------------
# GRAPH 2 — DIRECTOR–ACTOR COLLABORATION NETWORK
# Nodes: directors (coral) and actors (gold). An edge is drawn between a
# director and actor if they worked together on at least one film. Edge
# weight reflects the number of collaborations. Filtered to top directors.
# ---------------------------------------------------------------------------

def graph2_collaboration_network(df: pd.DataFrame) -> None:
    print("\n[2/6] Building Director–Actor Collaboration Network...")

    # Count films per director to identify the most prolific directors
    director_film_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        for director in split_field(row.get("directors", "")):
            director_film_counts[director] = director_film_counts.get(director, 0) + 1

    top_directors = set(
        sorted(director_film_counts, key=director_film_counts.get, reverse=True)
        [:MAX_COLLAB_DIRECTORS]
    )

    # Build collaboration edges: director → actor, weighted by shared films
    collab_weights: dict[tuple[str, str], int] = {}
    actor_directors: dict[str, set[str]] = {}

    for _, row in df.iterrows():
        directors = [d for d in split_field(row.get("directors", "")) if d in top_directors]
        actors    = split_field(row.get("cast_names", ""))[:6]
        if not directors or not actors:
            continue
        for director in directors:
            for actor in actors:
                key = (director, actor)
                collab_weights[key] = collab_weights.get(key, 0) + 1
                actor_directors.setdefault(actor, set()).add(director)

    net = pyvis_base(heading="CineGraph — Director–Actor Collaboration Network")
    added_actors: set[str] = set()

    # Add director nodes
    for director, count in director_film_counts.items():
        if director not in top_directors:
            continue
        net.add_node(
            f"dir_{director}",
            label=director,
            title=f"Director: {director}\nFilms in dataset: {count}",
            color=ACCENT_CORAL,
            size=10 + min(count, 30) * 0.7,
            shape="diamond",
        )

    # Add actor nodes and weighted edges
    for (director, actor), weight in collab_weights.items():
        actor_id = f"act_{actor}"
        dirs_worked_with = len(actor_directors.get(actor, set()))
        if actor_id not in added_actors:
            net.add_node(
                actor_id,
                label=actor,
                title=f"Actor: {actor}\nDirectors worked with: {dirs_worked_with}",
                color=ACCENT_GOLD,
                size=7 + dirs_worked_with * 2,
                shape="ellipse",
            )
            added_actors.add(actor_id)
        # Edge width scales with number of shared films
        net.add_edge(
            f"dir_{director}", actor_id,
            value=weight,
            title=f"{weight} film(s) together",
            color="#555555",
        )

    save_html(net, "collaboration_network.html")
    print(f"  Nodes: {len(net.nodes):,}  |  Edges: {len(net.edges):,}")


# ---------------------------------------------------------------------------
# GRAPH 3 — GENRE EVOLUTION TIMELINE
# Static matplotlib chart. Each genre gets a line showing how its average
# rating and film count evolved across decades. Two subplots: film count
# (top) and average rating (bottom).
# ---------------------------------------------------------------------------

def graph3_genre_evolution_timeline(df: pd.DataFrame) -> None:
    print("\n[3/6] Building Genre Evolution Timeline...")

    # Explode genres so each genre–year pair becomes its own row
    rows = []
    for _, row in df.iterrows():
        if row["vote_count"] < MIN_VOTE_COUNT:
            continue
        for genre in split_field(row.get("genres", "")):
            rows.append({
                "genre":        genre,
                "release_year": row["release_year"],
                "vote_average": row["vote_average"],
            })

    genre_df = pd.DataFrame(rows)

    # Snap to decade for cleaner grouping
    genre_df["decade"] = (genre_df["release_year"] // 10 * 10).astype(int)

    # Focus on the 8 most common genres
    top_genres = (
        genre_df["genre"].value_counts().head(8).index.tolist()
    )
    genre_df = genre_df[genre_df["genre"].isin(top_genres)]

    # Aggregate by genre + decade
    grouped = genre_df.groupby(["genre", "decade"]).agg(
        film_count  = ("vote_average", "count"),
        avg_rating  = ("vote_average", "mean"),
    ).reset_index()

    # Only show decades with enough data
    grouped = grouped[grouped["decade"] >= 1960]

    # Colour palette — one distinct colour per genre
    palette = sns.color_palette("tab10", len(top_genres))
    genre_colours = {g: palette[i] for i, g in enumerate(top_genres)}

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor=BACKGROUND_COLOUR)
    fig.suptitle("Genre Evolution Timeline", fontsize=18, color=TEXT_COLOUR,
                 fontweight="bold", y=0.98)

    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors=TEXT_COLOUR)
        ax.spines["bottom"].set_color(ACCENT_GREY)
        ax.spines["left"].set_color(ACCENT_GREY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(TEXT_COLOUR)

    # Top subplot — film count per decade
    ax_count = axes[0]
    ax_count.set_title("Films Released per Decade", color=TEXT_COLOUR, fontsize=13, pad=10)
    ax_count.set_ylabel("Number of Films", color=TEXT_COLOUR)

    for genre in top_genres:
        data = grouped[grouped["genre"] == genre].sort_values("decade")
        ax_count.plot(
            data["decade"], data["film_count"],
            marker="o", linewidth=2.5, markersize=5,
            label=genre, color=genre_colours[genre],
        )

    ax_count.legend(facecolor="#1a1a1a", edgecolor=ACCENT_GREY,
                    labelcolor=TEXT_COLOUR, fontsize=9, loc="upper left")
    ax_count.set_xticks(grouped["decade"].unique())
    ax_count.set_xticklabels(
        [f"{d}s" for d in sorted(grouped["decade"].unique())], rotation=45
    )
    ax_count.grid(axis="y", color=ACCENT_GREY, alpha=0.3, linestyle="--")

    # Bottom subplot — average rating per decade
    ax_rating = axes[1]
    ax_rating.set_title("Average Rating per Decade", color=TEXT_COLOUR, fontsize=13, pad=10)
    ax_rating.set_ylabel("Average Rating (out of 10)", color=TEXT_COLOUR)
    ax_rating.set_xlabel("Decade", color=TEXT_COLOUR)

    for genre in top_genres:
        data = grouped[grouped["genre"] == genre].sort_values("decade")
        ax_rating.plot(
            data["decade"], data["avg_rating"],
            marker="s", linewidth=2.5, markersize=5,
            label=genre, color=genre_colours[genre],
        )

    ax_rating.set_xticks(grouped["decade"].unique())
    ax_rating.set_xticklabels(
        [f"{d}s" for d in sorted(grouped["decade"].unique())], rotation=45
    )
    ax_rating.set_ylim(4, 9)
    ax_rating.axhline(y=7.0, color=ACCENT_GOLD, linestyle="--",
                      alpha=0.5, linewidth=1, label="7.0 threshold")
    ax_rating.legend(facecolor="#1a1a1a", edgecolor=ACCENT_GREY,
                     labelcolor=TEXT_COLOUR, fontsize=9, loc="lower right")
    ax_rating.grid(axis="y", color=ACCENT_GREY, alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = str(OUTPUT_DIR / "genre_evolution_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND_COLOUR)
    plt.close()
    print(f"  ✓ Saved: {path}")


# ---------------------------------------------------------------------------
# GRAPH 4 — RECOMMENDATION / SIMILARITY NETWORK
# Nodes: movies. Edges drawn from the dataset's pre-computed similar_ids
# column. Node colour reflects genre; size reflects vote_average. Filtered
# to well-rated, well-known films for a clean, meaningful graph.
# ---------------------------------------------------------------------------

def graph4_recommendation_graph(df: pd.DataFrame) -> None:
    print("\n[4/6] Building Recommendation / Similarity Network...")

    # Filter to well-rated films with enough votes
    qualified = df[
        (df["vote_count"] >= MIN_VOTE_COUNT) &
        (df["vote_average"] >= 6.5) &
        df["similar_ids"].notna()
    ].copy()

    # Take top N by popularity so the graph stays navigable
    qualified = qualified.nlargest(MAX_RECO_MOVIES, "popularity")
    qualified_ids = set(qualified["tmdb_id"].astype(str))

    # Assign a colour per primary genre for visual clustering
    genre_colour_map = {
        "Action":      "#e74c3c",
        "Adventure":   "#e67e22",
        "Animation":   "#f1c40f",
        "Comedy":      "#2ecc71",
        "Crime":       "#9b59b6",
        "Documentary": "#1abc9c",
        "Drama":       "#3498db",
        "Family":      "#fd79a8",
        "Fantasy":     "#a29bfe",
        "Horror":      "#d63031",
        "Romance":     "#e84393",
        "Science Fiction": "#00cec9",
        "Thriller":    "#636e72",
        "Western":     "#b8860b",
    }
    default_colour = "#888888"

    net = pyvis_base(heading="CineGraph — Movie Recommendation Network")

    # Add movie nodes
    id_to_row = {str(row["tmdb_id"]): row for _, row in qualified.iterrows()}

    for tmdb_id, row in id_to_row.items():
        primary_genre = split_field(row.get("genres", ""))
        primary_genre = primary_genre[0] if primary_genre else "Unknown"
        colour  = genre_colour_map.get(primary_genre, default_colour)
        rating  = float(row["vote_average"])
        size    = 8 + (rating - 5) * 3   # 8–23px range

        net.add_node(
            tmdb_id,
            label=row["title"],
            title=(
                f"{row['title']} ({int(row['release_year'])})\n"
                f"Rating: {rating:.1f}/10  |  Votes: {int(row['vote_count']):,}\n"
                f"Genres: {row.get('genres', 'N/A')}"
            ),
            color=colour,
            size=size,
            shape="dot",
        )

    # Add similarity edges (only between nodes that exist in our filtered set)
    edges_added = set()
    for tmdb_id, row in id_to_row.items():
        for sim_id in split_field(str(row.get("similar_ids", "")), delimiter=", "):
            if sim_id in qualified_ids and sim_id != tmdb_id:
                edge_key = tuple(sorted([tmdb_id, sim_id]))
                if edge_key not in edges_added:
                    net.add_edge(tmdb_id, sim_id, color="#333333", width=1)
                    edges_added.add(edge_key)

    save_html(net, "recommendation_graph.html")
    print(f"  Nodes: {len(net.nodes):,}  |  Edges: {len(net.edges):,}")


# ---------------------------------------------------------------------------
# GRAPH 5 — CENTRALITY & INFLUENCE CHARTS
# Static matplotlib chart. Computes three NetworkX centrality metrics on
# the actor-movie graph and visualises the top 20 most influential actors
# for each metric: degree, betweenness, and eigenvector centrality.
# ---------------------------------------------------------------------------

def graph5_centrality_charts(df: pd.DataFrame) -> None:
    print("\n[5/6] Building Centrality & Influence Charts...")
    print("  Computing centrality metrics (this may take 30–60 seconds)...")

    # Build a lightweight undirected graph for centrality computation
    G = nx.Graph()

    actor_film_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        if row["vote_count"] < MIN_VOTE_COUNT:
            continue
        actors = split_field(row.get("cast_names", ""))[:8]
        movie_id = f"movie_{row['tmdb_id']}"
        G.add_node(movie_id, node_type="movie", label=row["title"])
        for actor in actors:
            actor_id = f"actor_{actor.lower().replace(' ', '_')}"
            G.add_node(actor_id, node_type="actor", label=actor)
            G.add_edge(actor_id, movie_id)
            actor_film_counts[actor] = actor_film_counts.get(actor, 0) + 1

    if len(G.nodes) < MIN_CENTRALITY_NODES:
        print("  ⚠ Not enough nodes to compute centrality — skipping.")
        return

    # Extract only actor nodes for the metrics
    actor_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "actor"]

    # Degree centrality — number of films an actor connects to (normalised)
    degree_cent = nx.degree_centrality(G)
    actor_degree = {G.nodes[n]["label"]: degree_cent[n] for n in actor_nodes}

    # Betweenness centrality — how often an actor sits on the shortest path
    # between two other nodes; indicates "bridge" actors who connect communities.
    # Computed on a sample for performance on large graphs.
    print("  Computing betweenness centrality (sampled)...")
    between_cent = nx.betweenness_centrality(G, k=min(200, len(G.nodes)), seed=42)
    actor_between = {G.nodes[n]["label"]: between_cent[n] for n in actor_nodes}

    # Eigenvector centrality — an actor is influential if connected to other
    # influential nodes (movies with large, well-connected casts).
    print("  Computing eigenvector centrality...")
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=500, tol=1e-4)
        actor_eigen = {G.nodes[n]["label"]: eigen_cent[n] for n in actor_nodes}
    except nx.PowerIterationFailedConvergence:
        # Fall back to degree centrality if eigenvector fails to converge
        print("  ⚠ Eigenvector centrality did not converge — using degree as fallback.")
        actor_eigen = actor_degree

    def top20(metric_dict: dict) -> tuple[list, list]:
        """Return the top 20 actors and their scores from a centrality dict."""
        sorted_items = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        names  = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        return names, scores

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), facecolor=BACKGROUND_COLOUR)
    fig.suptitle("Actor Centrality & Influence", fontsize=18,
                 color=TEXT_COLOUR, fontweight="bold", y=1.01)

    chart_configs = [
        (actor_degree,  "Degree Centrality",      ACCENT_GOLD,  "Film connections (normalised)"),
        (actor_between, "Betweenness Centrality",  ACCENT_TEAL,  "Bridge score (path frequency)"),
        (actor_eigen,   "Eigenvector Centrality",  ACCENT_CORAL, "Influence score"),
    ]

    for ax, (metric, title, colour, xlabel) in zip(axes, chart_configs):
        names, scores = top20(metric)
        # Reverse so highest is at top
        names  = names[::-1]
        scores = scores[::-1]

        ax.set_facecolor("#1a1a1a")
        ax.barh(names, scores, color=colour, alpha=0.85, edgecolor="none")
        ax.set_title(title, color=TEXT_COLOUR, fontsize=13, pad=10)
        ax.set_xlabel(xlabel, color=TEXT_COLOUR, fontsize=9)
        ax.tick_params(colors=TEXT_COLOUR, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(ACCENT_GREY)
        ax.spines["left"].set_color(ACCENT_GREY)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(TEXT_COLOUR)
        ax.grid(axis="x", color=ACCENT_GREY, alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = str(OUTPUT_DIR / "centrality_charts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND_COLOUR)
    plt.close()
    print(f"  ✓ Saved: {path}")


# ---------------------------------------------------------------------------
# GRAPH 6 — PATHFINDING / HIDDEN CONNECTIONS
# Interactive PyVis graph showing the shortest relationship path between
# two user-specified movies, traversing shared actors and directors.
# Highlights the path nodes and edges in gold; surrounding context in grey.
# ---------------------------------------------------------------------------

def graph6_pathfinding(df: pd.DataFrame) -> None:
    print("\n[6/6] Building Pathfinding / Hidden Connections Graph...")

    # Build a full undirected graph for path traversal
    G = nx.Graph()
    movie_meta: dict[str, dict] = {}
    actor_meta: dict[str, dict] = {}

    for _, row in df.iterrows():
        movie_id    = f"movie_{row['tmdb_id']}"
        movie_label = f"{row['title']} ({int(row['release_year'])})"
        G.add_node(movie_id, node_type="movie", label=movie_label, title=row["title"])
        movie_meta[movie_id] = {"label": movie_label, "title": row["title"],
                                 "rating": row["vote_average"], "genres": row.get("genres", "")}

        for actor in split_field(row.get("cast_names", ""))[:8]:
            actor_id = f"actor_{actor.lower().replace(' ', '_')}"
            G.add_node(actor_id, node_type="actor", label=actor, title=actor)
            actor_meta[actor_id] = {"label": actor}
            G.add_edge(actor_id, movie_id)

        for director in split_field(row.get("directors", "")):
            dir_id = f"director_{director.lower().replace(' ', '_')}"
            G.add_node(dir_id, node_type="director", label=director, title=director)
            G.add_edge(dir_id, movie_id)

    # Build a lookup from clean title → node ID for user input matching
    title_to_id = {
        data["title"].lower(): node_id
        for node_id, data in G.nodes(data=True)
        if data.get("node_type") == "movie"
    }

    print("\n  Enter two movie titles to find the hidden connection between them.")
    print("  Example: The Dark Knight / Inception\n")

    def find_movie_id(prompt: str) -> str | None:
        """Prompt the user for a movie title and return its graph node ID.
        Supports case-insensitive exact match, substring search, and fuzzy
        matching so users don't need to spell titles perfectly.
        """
        all_titles = list(title_to_id.keys())  # all lowercased titles
    
        while True:
            title = input(f"  {prompt}: ").strip().lower()
    
            # 1. Exact match (case-insensitive)
            if title in title_to_id:
                return title_to_id[title]
    
            # 2. Substring match — user typed part of the title
            substring_matches = [t for t in all_titles if title in t]
    
            # 3. Fuzzy match — catches typos (e.g. "Apoclypse Now")
            fuzzy_matches = get_close_matches(title, all_titles, n=5, cutoff=0.6)
    
            # Combine, deduplicate, preserve order
            candidates = list(dict.fromkeys(substring_matches + fuzzy_matches))
    
            if not candidates:
                print(f"  '{title}' not found. Please try a different title.\n")
                continue
    
            # Present numbered options and let the user pick
            print(f"  Did you mean:")
            for i, match in enumerate(candidates[:5], 1):
                print(f"    {i}. {match.title()}")
            print(f"    0. Try again\n")
    
            choice = input("  Enter number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(candidates[:5]):
                return title_to_id[candidates[int(choice) - 1]]
            # If they enter 0 or anything invalid, loop back

    source_id = find_movie_id("Movie A")
    target_id = find_movie_id("Movie B")

    source_title = G.nodes[source_id]["title"]
    target_title = G.nodes[target_id]["title"]
    print(f"\n  Finding path: '{source_title}' → '{target_title}'...")

    try:
        path = nx.shortest_path(G, source=source_id, target=target_id)
    except nx.NetworkXNoPath:
        print(f"  ⚠ No path found between '{source_title}' and '{target_title}'.")
        print("  They may share no common actors or directors in this dataset.")
        return
    except nx.NodeNotFound as e:
        print(f"  ⚠ Node not found: {e}")
        return

    print(f"  Path found! Length: {len(path) - 1} step(s)")
    print("  Route:")
    for node_id in path:
        node_data = G.nodes[node_id]
        node_type = node_data.get("node_type", "?").capitalize()
        print(f"    [{node_type}] {node_data.get('label', node_id)}")

    # Build the visualisation — show path nodes + their immediate neighbours
    path_set  = set(path)
    net = pyvis_base(
        heading=f"Hidden Connection: {source_title} → {target_title}"
    )

    # Collect context nodes (neighbours of path nodes, for surrounding context)
    context_nodes = set()
    for node_id in path:
        for neighbour in list(G.neighbors(node_id))[:8]:
            if neighbour not in path_set:
                context_nodes.add(neighbour)

    def add_node_to_net(node_id: str, on_path: bool) -> None:
        data      = G.nodes[node_id]
        node_type = data.get("node_type", "unknown")
        label     = data.get("label", node_id)

        if on_path:
            # Path nodes are highlighted in gold, larger
            if node_type == "movie":
                colour, size, shape = ACCENT_GOLD, 25, "star"
            elif node_type == "actor":
                colour, size, shape = ACCENT_TEAL, 18, "ellipse"
            else:
                colour, size, shape = ACCENT_CORAL, 18, "diamond"
        else:
            # Context nodes are dimmed grey
            colour, size, shape = "#3a3a3a", 8, "dot"
            label = ""   # Hide labels on context nodes to reduce clutter

        net.add_node(node_id, label=label, title=data.get("label", node_id),
                     color=colour, size=size, shape=shape)

    # Add all nodes
    all_nodes = path_set | context_nodes
    for node_id in all_nodes:
        add_node_to_net(node_id, on_path=(node_id in path_set))

    # Add all edges — path edges in gold, context edges in dark grey
    for i in range(len(path) - 1):
        net.add_edge(path[i], path[i + 1], color=ACCENT_GOLD, width=3)

    for node_id in path_set:
        for neighbour in G.neighbors(node_id):
            if neighbour in context_nodes:
                net.add_edge(node_id, neighbour, color="#2a2a2a", width=1)

    save_html(net, "pathfinding.html")


# ---------------------------------------------------------------------------
# MAIN — Run all six visualisations in sequence
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("CineGraph 2026 — Visualisation Pipeline")
    print("="*60)

    df = load_data(CSV_PATH)

    graph1_actor_movie_network(df)
    graph2_collaboration_network(df)
    graph3_genre_evolution_timeline(df)
    graph4_recommendation_graph(df)
    graph5_centrality_charts(df)
    graph6_pathfinding(df)

    print("\n" + "="*60)
    print("All visualisations complete.")
    print(f"Output folder: {OUTPUT_DIR}")
    print("="*60)
    print("\nOpen the .html files in any browser for interactive graphs.")
    print("Open the .png files for the static charts.")