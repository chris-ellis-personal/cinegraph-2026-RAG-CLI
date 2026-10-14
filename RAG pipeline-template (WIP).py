# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:32:12 2026

@author: Chris
"""

"""
=============================================================================
CineGraph 2026 — Movies RAG Pipeline
=============================================================================
  - OpenAI text-embedding-3-large  (embeddings)
  - ChromaDB                        (local vector store)
  - OpenAI GPT-4o                   (generation)
  - Neo4j knowledge graph           (relational fact enrichment)

Requirements:
    pip install openai chromadb pandas tqdm neo4j python-dotenv

Usage:
    1. Credentials are loaded from .env — do not paste keys into this file.
    2. Run knowledge_graph.py once first to populate Neo4j.
    3. Run once with BUILD_INDEX = True to embed and persist the vector index.
    4. Set BUILD_INDEX = False on subsequent runs to load from disk.
    5. Call query_pipeline("your question here") to ask questions.
=============================================================================
"""

import os
import re
import json
import math
import time
import textwrap
from pathlib import Path

import pandas as pd
import chromadb
from openai import OpenAI
from tqdm import tqdm

from dotenv import load_dotenv
from knowledge_graph import get_driver, get_movies_by_actor, get_collaborators

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

CSV_PATH = (
    r"C:/Users/Chris/cinegraph-2026/movies.csv"
)

# Directory where ChromaDB will persist the vector index
CHROMA_PERSIST_DIR = str(
    Path(CSV_PATH).parent / "chroma_index"
)

COLLECTION_NAME = "movies"

# Set to True the first time (or after changing CSV_PATH / document design).
# Set to False to skip re-embedding and load the existing index.
BUILD_INDEX = False

# OpenAI settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-large"   # 3072-dim
GENERATION_MODEL = "gpt-4o"

# Retrieval settings
RETRIEVE_TOP_K = 20   # candidates fetched from vector store
RERANK_TOP_N = 5      # candidates passed to LLM after reranking

# Embedding batch size — stay under OpenAI's token-rate limits
EMBED_BATCH_SIZE = 100

# Minimum document length (chars) before we skip embedding
MIN_DOC_LENGTH = 50

# Minimum vote count to treat a vote_average as meaningful
MIN_VOTE_COUNT_FOR_RATING = 50

# ---------------------------------------------------------------------------
# OPENAI CLIENT
# ---------------------------------------------------------------------------

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# STEP 1 — DOCUMENT DESIGN
# Construct one rich text blob per movie from the most semantically useful
# fields. Structured metadata is kept separate for filter-based retrieval.
# ---------------------------------------------------------------------------

def build_document_text(row: pd.Series) -> str:
    """
    Returns a natural-language text representation of a single movie row.
    Only non-null fields are included. Cast is capped at 5 to prevent the
    embedding from over-weighting actor identity vs. plot/theme signals.
    """
    parts = []

    parts.append(f"Title: {row['title']}")

    # Include original title when it differs (e.g., non-English films)
    if pd.notna(row["original_title"]) and row["original_title"] != row["title"]:
        parts.append(f"Original title: {row['original_title']}")

    # Alternative titles support multilingual / alias queries
    if pd.notna(row["alternative_titles"]):
        # Strip locale prefixes like "KR:아리엘 | JP:..." and join as a clean list
        raw = row["alternative_titles"]
        clean_titles = [
            re.sub(r"^[A-Z]{2}:", "", t).strip()
            for t in raw.split(" | ")
        ]
        parts.append(f"Also known as: {', '.join(clean_titles)}")

    if pd.notna(row["tagline"]):
        parts.append(f"Tagline: {row['tagline']}")

    if pd.notna(row["overview"]):
        parts.append(f"Overview: {row['overview']}")

    if pd.notna(row["genres"]):
        parts.append(f"Genres: {row['genres']}")

    if pd.notna(row["keywords"]):
        parts.append(f"Keywords: {row['keywords']}")

    if pd.notna(row["directors"]):
        parts.append(f"Director(s): {row['directors']}")

    if pd.notna(row["cast_names"]):
        # Limit to top 5 cast members to keep plot/theme signals dominant
        top_cast = ", ".join(row["cast_names"].split(", ")[:5])
        parts.append(f"Cast: {top_cast}")

    if pd.notna(row["writers"]):
        parts.append(f"Writer(s): {row['writers']}")

    if pd.notna(row["release_year"]):
        parts.append(f"Year: {int(row['release_year'])}")

    if pd.notna(row["runtime_min"]):
        parts.append(f"Runtime: {int(row['runtime_min'])} min")

    if pd.notna(row["us_certification"]):
        parts.append(f"US Rating: {row['us_certification']}")

    if pd.notna(row["original_language"]):
        parts.append(f"Language: {row['original_language']}")

    if pd.notna(row["production_countries"]):
        parts.append(f"Country: {row['production_countries']}")

    if pd.notna(row["collection_name"]):
        parts.append(f"Part of collection: {row['collection_name']}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# STEP 2 — METADATA SCHEMA
# Stored alongside each embedding for filter-based retrieval.
# ChromaDB metadata values must be str | int | float | bool — no lists.
# Lists are serialised to JSON strings and parsed back on retrieval.
# ---------------------------------------------------------------------------

def build_metadata(row: pd.Series) -> dict:
    """
    Returns a flat dict of filterable metadata for a single movie.
    List-valued fields are JSON-serialised so ChromaDB can store them.
    """
    def safe_int(val):
        try:
            return int(val) if pd.notna(val) else None
        except (ValueError, TypeError):
            return None

    def safe_float(val):
        try:
            return float(val) if pd.notna(val) else None
        except (ValueError, TypeError):
            return None

    def split_csv(val):
        """Split a comma-separated string into a JSON list string."""
        if pd.notna(val) and str(val).strip():
            return json.dumps([v.strip() for v in str(val).split(",")])
        return json.dumps([])

    # Only store vote_average when vote_count meets our minimum threshold
    vote_count = safe_int(row.get("vote_count"))
    vote_average = (
        safe_float(row.get("vote_average"))
        if vote_count and vote_count >= MIN_VOTE_COUNT_FOR_RATING
        else None
    )

    meta = {
        # --- Identifiers ---
        "tmdb_id":          int(row["tmdb_id"]),             # always present; used as vector store primary key
        "imdb_id":          str(row["imdb_id"]) if pd.notna(row.get("imdb_id")) else "",
        "title":            str(row["title"]),

        # --- Core attributes (for display & filtering) ---
        "release_year":     safe_int(row.get("release_year")),
        "runtime_min":      safe_int(row.get("runtime_min")),
        "vote_average":     vote_average,
        "vote_count":       vote_count,
        "popularity":       safe_float(row.get("popularity")),
        "us_certification": str(row["us_certification"]) if pd.notna(row.get("us_certification")) else "",
        "original_language": str(row["original_language"]) if pd.notna(row.get("original_language")) else "",
        "status":           str(row["status"]) if pd.notna(row.get("status")) else "",

        # --- Lists (JSON-serialised for ChromaDB compatibility) ---
        "genres":                split_csv(row.get("genres")),
        "directors":             split_csv(row.get("directors")),
        "cast_names_top5":       json.dumps(
                                     row["cast_names"].split(", ")[:5]
                                     if pd.notna(row.get("cast_names")) else []
                                 ),
        "production_countries":  split_csv(row.get("production_countries")),
        "spoken_languages":      split_csv(row.get("spoken_languages")),
        "watch_us_flatrate":     split_csv(row.get("watch_us_flatrate")),

        # --- Financial (useful for ranking queries) ---
        "budget_usd":       safe_float(row.get("budget_usd")),
        "revenue_usd":      safe_float(row.get("revenue_usd")),
        "roi_pct":          safe_float(row.get("roi_pct")),

        # --- Flags ---
        "has_overview":     bool(pd.notna(row.get("overview")) and len(str(row.get("overview", ""))) > 10),
        "has_trailer":      bool(row.get("has_trailer")),
        "in_collection":    bool(pd.notna(row.get("collection_name"))),

        # --- URLs (for display in formatted output) ---
        "poster_url":       str(row["poster_url"]) if pd.notna(row.get("poster_url")) else "",
        "trailer_url":      str(row["trailer_url"]) if pd.notna(row.get("trailer_url")) else "",

        # --- Pre-computed similarity graph (for "more like X" queries) ---
        "similar_ids":      str(row["similar_ids"]) if pd.notna(row.get("similar_ids")) else "",
        "recommended_ids":  str(row["recommended_ids"]) if pd.notna(row.get("recommended_ids")) else "",
    }

    # ChromaDB rejects None values — replace with type-appropriate sentinels
    cleaned = {}
    for k, v in meta.items():
        if v is None:
            if k in ("release_year", "runtime_min", "vote_count"):
                cleaned[k] = -1
            elif k in ("vote_average", "popularity", "budget_usd", "revenue_usd", "roi_pct"):
                cleaned[k] = -1.0
            else:
                cleaned[k] = ""
        else:
            cleaned[k] = v

    return cleaned


# ---------------------------------------------------------------------------
# STEP 3 — EMBEDDING
# Converts document text into high-dimensional vectors using OpenAI's
# embedding model. Vectors capture semantic meaning, enabling similarity
# search. Includes exponential backoff to handle API rate limits gracefully.
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using OpenAI's embedding model.
    Includes simple exponential backoff for rate-limit handling.
    """
    for attempt in range(5):
        try:
            response = client.embeddings.create(
                input=texts,
                model=EMBEDDING_MODEL,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"  Embedding error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# STEP 4 — INDEX BUILD
# Loads the CSV, constructs documents + metadata, embeds, and upserts into
# ChromaDB. Skips near-empty documents (nulls in both overview and tagline).
# ---------------------------------------------------------------------------

def build_index(csv_path: str) -> chromadb.Collection:
    """
    Full indexing pipeline: CSV → documents → embeddings → ChromaDB.
    Returns the populated ChromaDB collection.
    """
    print(f"\n{'='*60}")
    print("PHASE 1 — Loading & Preparing Documents")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns from CSV")

    records = []   # list of (id, text, metadata) tuples
    skipped = 0

    for _, row in df.iterrows():
        doc_text = build_document_text(row)

        # Skip documents that are too sparse to embed meaningfully
        if len(doc_text.strip()) < MIN_DOC_LENGTH:
            skipped += 1
            continue

        doc_id = str(int(row["tmdb_id"]))
        meta = build_metadata(row)
        records.append((doc_id, doc_text, meta))

    print(f"  Documents prepared: {len(records):,}  |  Skipped (too sparse): {skipped}")

    # --- ChromaDB setup ---
    print(f"\n{'='*60}")
    print("PHASE 2 — Setting Up ChromaDB")
    print(f"{'='*60}")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Delete existing collection if rebuilding to avoid stale data
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    # Create a fresh collection using cosine similarity as the distance metric
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Created collection '{COLLECTION_NAME}' (cosine similarity)")

    # --- Embed and upsert in batches ---
    print(f"\n{'='*60}")
    print(f"PHASE 3 — Embedding ({len(records):,} documents in batches of {EMBED_BATCH_SIZE})")
    print(f"{'='*60}")

    n_batches = math.ceil(len(records) / EMBED_BATCH_SIZE)

    for i in tqdm(range(0, len(records), EMBED_BATCH_SIZE),
                  total=n_batches, desc="  Embedding batches"):
        batch = records[i : i + EMBED_BATCH_SIZE]
        ids   = [r[0] for r in batch]
        texts = [r[1] for r in batch]
        metas = [r[2] for r in batch]

        embeddings = embed_batch(texts)

        # Upsert writes new records and overwrites any existing ones with the same ID
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )

    print(f"\n  ✓ Index built. {collection.count():,} documents stored at:")
    print(f"    {CHROMA_PERSIST_DIR}")

    return collection


def load_index() -> chromadb.Collection:
    """Load an existing ChromaDB index from disk without re-embedding."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"  ✓ Loaded index: {collection.count():,} documents from {CHROMA_PERSIST_DIR}")
    return collection


# ---------------------------------------------------------------------------
# STEP 5 — QUERY PARSING
# A lightweight LLM call extracts structured intent from the user's natural-
# language question before retrieval. Returns a semantic query string and
# optional ChromaDB-compatible filter dict.
# ---------------------------------------------------------------------------

QUERY_PARSER_SYSTEM = textwrap.dedent("""
    You parse movie search queries into structured JSON for a RAG pipeline.
    Return ONLY valid JSON with this exact shape (omit keys you cannot infer):
    {
      "semantic_query": "<short rephrased query optimised for embedding search>",
      "filters": {
        "release_year_gte": <int or null>,
        "release_year_lte": <int or null>,
        "runtime_min_gte":  <int or null>,
        "runtime_min_lte":  <int or null>,
        "vote_average_gte": <float or null>,
        "original_language": "<ISO-639-1 code or null>",
        "us_certification": "<G|PG|PG-13|R|NC-17|NR or null>"
      },
      "genre_hint":    "<single genre string or null>",
      "director_hint": "<director name or null>",
      "actor_hint":    "<actor name or null>",
      "sort_hint":     "<highest_rated|lowest_rated|most_popular|newest|oldest|highest_revenue or null>",
      "more_like_title": "<exact movie title for similarity lookup or null>"
    }
    Rules:
    - semantic_query must NEVER be empty.
    - Omit any filter key whose value would be null (keep the JSON small).
    - "underrated" → vote_average_gte: 7.0 AND low popularity (no filter needed).
    - "family friendly" or "kids" → us_certification: "PG" or "G".
    - "more like X" → set more_like_title to X; set semantic_query to X's themes.
""")


def parse_query(user_query: str) -> dict:
    """Use GPT-4o to extract structured retrieval intent from a natural-language query."""
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": QUERY_PARSER_SYSTEM},
            {"role": "user",   "content": user_query},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if the model wraps the response in ```json
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: treat the whole query as the semantic query with no filters
        return {"semantic_query": user_query}


def build_chroma_filter(parsed: dict) -> dict | None:
    """
    Converts parsed filter hints into a ChromaDB-compatible $and filter dict.
    ChromaDB uses MongoDB-style operators: $eq, $gte, $lte.
    Returns None if no filterable constraints were parsed.

    Note: list-valued fields (genres, directors, cast) cannot be filtered
    natively here — they are handled as post-retrieval Python filters instead.
    """
    filters_raw = parsed.get("filters", {})
    clauses = []

    year_gte = filters_raw.get("release_year_gte")
    year_lte = filters_raw.get("release_year_lte")
    if year_gte:
        clauses.append({"release_year": {"$gte": year_gte}})
    if year_lte:
        clauses.append({"release_year": {"$lte": year_lte}})

    runtime_gte = filters_raw.get("runtime_min_gte")
    runtime_lte = filters_raw.get("runtime_min_lte")
    if runtime_gte:
        clauses.append({"runtime_min": {"$gte": runtime_gte}})
    if runtime_lte:
        clauses.append({"runtime_min": {"$lte": runtime_lte}})

    vote_gte = filters_raw.get("vote_average_gte")
    if vote_gte:
        clauses.append({"vote_average": {"$gte": float(vote_gte)}})

    lang = filters_raw.get("original_language")
    if lang:
        clauses.append({"original_language": {"$eq": lang}})

    cert = filters_raw.get("us_certification")
    if cert:
        clauses.append({"us_certification": {"$eq": cert}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ---------------------------------------------------------------------------
# STEP 6 — RETRIEVAL
# ---------------------------------------------------------------------------

def retrieve(
    collection: chromadb.Collection,
    parsed: dict,
    top_k: int = RETRIEVE_TOP_K,
) -> list[dict]:
    """
    Two-stage retrieval:
      1. Vector similarity search (with optional ChromaDB metadata filters).
      2. Python post-filters for list-valued fields (genre, director, actor).

    Returns a list of result dicts with keys: id, document, metadata, distance.
    """
    semantic_query = parsed.get("semantic_query", "")
    chroma_filter  = build_chroma_filter(parsed)

    # --- "More like X" shortcut ---
    # Bypasses vector search and uses the dataset's pre-computed similar_ids
    # column for more accurate film-to-film similarity lookup.
    more_like = parsed.get("more_like_title")
    if more_like:
        anchor = collection.get(
            where={"title": {"$eq": more_like}},
            include=["metadatas"],
        )
        if anchor["ids"]:
            similar_ids_str = anchor["metadatas"][0].get("similar_ids", "")
            if similar_ids_str:
                similar_ids = [s.strip() for s in similar_ids_str.split(",") if s.strip()]
                results = collection.get(
                    ids=similar_ids[:top_k],
                    include=["documents", "metadatas"],
                )
                return [
                    {"id": i, "document": d, "metadata": m, "distance": 0.0}
                    for i, d, m in zip(results["ids"], results["documents"], results["metadatas"])
                ]

    # --- Standard vector search ---
    # Embed the semantic query and find the top_k most similar documents
    query_embedding = embed_batch([semantic_query])[0]

    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if chroma_filter:
        query_kwargs["where"] = chroma_filter

    results = collection.query(**query_kwargs)

    candidates = [
        {
            "id":       results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]

    # --- Post-retrieval list-field filters ---
    # Applied in Python because ChromaDB cannot filter on JSON-serialised lists
    genre_hint    = (parsed.get("genre_hint") or "").strip().lower()
    director_hint = (parsed.get("director_hint") or "").strip().lower()
    actor_hint    = (parsed.get("actor_hint") or "").strip().lower()

    def matches_hint(candidate: dict, hint: str, meta_key: str) -> bool:
        if not hint:
            return True
        raw = candidate["metadata"].get(meta_key, "[]")
        try:
            items = [v.lower() for v in json.loads(raw)]
        except (json.JSONDecodeError, TypeError):
            items = [raw.lower()]
        return any(hint in item for item in items)

    filtered = [
        c for c in candidates
        if matches_hint(c, genre_hint, "genres")
        and matches_hint(c, director_hint, "directors")
        and matches_hint(c, actor_hint, "cast_names_top5")
    ]

    # Fall back to unfiltered candidates if post-filtering removed everything
    return filtered if filtered else candidates


# ---------------------------------------------------------------------------
# STEP 7 — RERANKING
# A lightweight LLM-based reranker scores each candidate 0–10 and returns
# the top-N. Avoids needing a separate cross-encoder model for this demo.
# ---------------------------------------------------------------------------

RERANK_SYSTEM = textwrap.dedent("""
    You are a relevance judge for a movie search engine.
    Given a user query and a list of movie summaries, score each movie's
    relevance to the query from 0 (completely irrelevant) to 10 (perfect match).

    Return ONLY a JSON array of objects in this format:
    [{"id": "<tmdb_id>", "score": <0-10>}, ...]

    Rules:
    - Order does not matter in your output.
    - Be strict: a score of 8+ means you'd confidently recommend this movie.
    - Consider plot, genre, era, tone, cast, and any specific constraints in the query.
""")


def rerank(candidates: list[dict], user_query: str, top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Ask GPT-4o to score each candidate against the user query.
    Returns the top_n candidates sorted by descending relevance score.
    """
    if len(candidates) <= top_n:
        return candidates

    # Build a compact candidate summary for the reranker to evaluate
    summaries = []
    for c in candidates:
        m = c["metadata"]
        year  = m.get("release_year", -1)
        year_str = str(year) if year != -1 else "Unknown"
        rating = m.get("vote_average", -1.0)
        rating_str = f"{rating:.1f}" if rating != -1.0 else "N/A"
        summaries.append(
            f'ID:{c["id"]} | {m.get("title", "?")} ({year_str}) | '
            f'Rating:{rating_str} | '
            + c["document"][:300]
        )

    prompt = (
        f'User query: "{user_query}"\n\n'
        + "\n---\n".join(summaries)
    )

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": RERANK_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        scores = {item["id"]: item["score"] for item in json.loads(raw)}
    except (json.JSONDecodeError, KeyError):
        # Reranker failed — fall back to original vector distance ordering
        return candidates[:top_n]

    # Attach rerank scores and sort descending
    for c in candidates:
        c["rerank_score"] = scores.get(c["id"], 0)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# STEP 8 — CONTEXT FORMATTING
# Formats retrieved movie records into a clean, labelled string block
# that is injected into the LLM generation prompt.
# ---------------------------------------------------------------------------

def format_context(candidates: list[dict]) -> str:
    """Build a structured context block from the top-N retrieved movies."""
    blocks = []
    for i, c in enumerate(candidates, start=1):
        m = c["metadata"]

        year     = m.get("release_year", -1)
        rating   = m.get("vote_average", -1.0)
        runtime  = m.get("runtime_min", -1)
        cert     = m.get("us_certification", "")
        flatrate = m.get("watch_us_flatrate", "[]")
        try:
            flatrate_list = json.loads(flatrate)
            flatrate_str  = ", ".join(flatrate_list) if flatrate_list else "Not available"
        except Exception:
            flatrate_str = flatrate or "Not available"

        lines = [
            f"[Movie {i}]",
            f"Title:     {m.get('title', '?')} ({year if year != -1 else '?'})",
        ]
        if m.get("directors"):
            try:
                dirs = json.loads(m["directors"])
                lines.append(f"Director:  {', '.join(dirs)}")
            except Exception:
                lines.append(f"Director:  {m['directors']}")
        if m.get("genres"):
            try:
                genres = json.loads(m["genres"])
                lines.append(f"Genres:    {', '.join(genres)}")
            except Exception:
                lines.append(f"Genres:    {m['genres']}")
        if rating != -1.0:
            votes = m.get("vote_count", -1)
            vote_str = f" ({votes:,} votes)" if votes != -1 else ""
            lines.append(f"Rating:    {rating:.1f}/10{vote_str}")
        if runtime != -1:
            lines.append(f"Runtime:   {runtime} min")
        if cert:
            lines.append(f"Cert:      {cert}")
        lines.append(f"Streaming: {flatrate_str}")
        if m.get("poster_url"):
            lines.append(f"Poster:    {m['poster_url']}")

        # Append the full document text (overview, keywords, cast, etc.)
        lines.append("")
        lines.append(c["document"])
        blocks.append("\n".join(lines))

    return "\n\n" + ("\n" + "-" * 60 + "\n").join(blocks) + "\n"


# ---------------------------------------------------------------------------
# STEP 9 — GENERATION
# Passes the formatted context and user query to GPT-4o to produce a
# grounded natural-language answer. Temperature is kept low to favour
# factual accuracy over creative variation.
# ---------------------------------------------------------------------------

GENERATION_SYSTEM = textwrap.dedent("""
    You are CineGraph, an expert movie assistant. You answer questions about
    films using ONLY the movie records provided in the context below.

    Guidelines:
    - Ground every claim in the provided records. Do not invent titles, facts,
      or details not present in the context.
    - For recommendations, explain briefly WHY each film matches the query,
      drawing on genres, overview, keywords, or cast.
    - Include the year, rating (if available), and where to stream (if available)
      for each recommendation.
    - If fewer than 2 strong matches exist in the context, say so honestly
      rather than padding with weak suggestions.
    - If the user asks about something outside the dataset (e.g., a very recent
      film not in the records), acknowledge the limitation.
    - Format recommendations as a numbered list. Keep explanations concise.
""")


def generate_answer(user_query: str, context: str) -> str:
    """Call GPT-4o with the retrieved context and return the generated answer."""
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User question: {user_query}\n\n"
                f"Retrieved movie records:\n{context}\n\n"
                "Please answer the user's question based solely on the records above."
            ),
        },
    ]
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0.3,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# STEP 10 — FULL PIPELINE ENTRY POINT
# Orchestrates all steps: parse → graph enrich → retrieve → rerank →
# format context → generate → return answer.
# ---------------------------------------------------------------------------

# Module-level handles initialised in the main block below
_collection: chromadb.Collection | None = None
neo4j_driver = None


def query_pipeline(user_query: str, verbose: bool = True) -> str:
    """
    End-to-end RAG pipeline:
      parse → graph enrich → retrieve → rerank → format context → generate.

    Args:
        user_query: Natural-language question from the user.
        verbose:    Print intermediate steps (useful for demo / inspection).

    Returns:
        Generated answer string.
    """
    global _collection, neo4j_driver
    if _collection is None:
        raise RuntimeError("Index not loaded. Run main block first.")

    if verbose:
        print(f"\n{'='*60}")
        print(f"QUERY: {user_query}")
        print(f"{'='*60}")

    # 1. Parse intent — extract semantic query and structured filters from the user's question
    parsed = parse_query(user_query)
    if verbose:
        print(f"\n[1] Parsed intent:\n{json.dumps(parsed, indent=2)}")

    # 2. Knowledge graph enrichment — prepend relational facts derived from
    #    the graph to the context before passing it to the LLM.
    #    This grounds actor/director queries in verified graph relationships
    #    rather than relying solely on vector similarity.
    graph_context = ""
    if neo4j_driver is not None:
        if parsed.get("actor_hint"):
            movie_ids = get_movies_by_actor(neo4j_driver, parsed["actor_hint"])
            if movie_ids:
                graph_context += (
                    f"\nGraph fact: {parsed['actor_hint']} appeared in "
                    f"{len(movie_ids)} films in the dataset.\n"
                )
        if parsed.get("director_hint"):
            collaborators = get_collaborators(neo4j_driver, parsed["director_hint"])
            if collaborators:
                graph_context += (
                    f"\nGraph fact: {parsed['director_hint']} has worked with: "
                    f"{', '.join(collaborators[:10])}.\n"
                )

    # 3. Retrieve candidates — vector similarity search with optional metadata filters
    candidates = retrieve(_collection, parsed, top_k=RETRIEVE_TOP_K)
    if verbose:
        print(f"\n[2] Retrieved {len(candidates)} candidates")

    # 4. Rerank — score each candidate 0-10 for relevance and keep top N
    top = rerank(candidates, user_query, top_n=RERANK_TOP_N)
    if verbose:
        print(f"\n[3] Reranked to top {len(top)}:")
        for c in top:
            title = c["metadata"].get("title", "?")
            score = c.get("rerank_score", "N/A")
            print(f"    {score:>4} — {title}")

    # 5. Format context — combine graph facts with the top-N retrieved movie records
    context = graph_context + format_context(top)

    # 6. Generate — pass the enriched context to GPT-4o and return the answer
    answer = generate_answer(user_query, context)
    if verbose:
        print(f"\n[4] Answer:\n{answer}")

    return answer


# ---------------------------------------------------------------------------
# MAIN — Build or load index, load knowledge graph, then run demo queries
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load or build the ChromaDB vector index
    if BUILD_INDEX:
        _collection = build_index(CSV_PATH)
    else:
        _collection = load_index()

    # Connect to Neo4j knowledge graph
    neo4j_driver = get_driver()

    print("\n" + "="*60)
    print("INDEX READY — CineGraph is ready. Type your question below.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")

    # --- Demonstration queries (commented out — uncomment to use) ---
    # demo_queries = [
    #     "Scary horror movies from the 1980s with a high rating",
    #     "Films directed by Aki Kaurismaki",
    #     "Good family films suitable for children under 12",
    #     "Movies similar to Parasite",
    #     "What is the highest-grossing film in the dataset?",
    #     "French language films with great reviews",
    #     "Thought-provoking sci-fi films that deal with artificial intelligence",
    #     "Which actors appeared in both a Horror film and a Comedy in the dataset?",
    #     "What are the best action movies of the 2000s?",
    #     "Recommend a short film under 90 minutes with a high rating",
    # ]
    # for q in demo_queries:
    #     answer = query_pipeline(q, verbose=True)
    #     print("\n" + "-"*60 + "\n")

    # --- Interactive query loop ---
    while True:
        user_input = input("Query: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting CineGraph.")
            break
        answer = query_pipeline(user_input, verbose=True)
        print("\n" + "-"*60 + "\n")

    # Close the Neo4j connection cleanly on exit
    neo4j_driver.close()