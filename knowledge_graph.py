# -*- coding: utf-8 -*-
"""
=============================================================================
CineGraph 2026 — Knowledge Graph
=============================================================================
Writes the movies CSV into Neo4j as a directed property graph, then provides
Cypher-backed query functions for the RAG pipeline.

Node types:  Movie | Actor | Director | Genre
Edge types:  APPEARED_IN | DIRECTED | HAS_GENRE | SIMILAR_TO

Run this script once to populate Neo4j (Phase 3), then leave it running
as a module imported by RAG_pipeline.py and visualisations.py.

Requirements:
    pip install neo4j python-dotenv pandas
=============================================================================
"""

import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

load_dotenv()

CSV_PATH = (
    r"C:/Users/Chris/cinegraph-2026/movies.csv"
)

NEO4J_URI  = os.environ.get("NEO4J_URI",  "neo4j://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DB   = os.environ.get("NEO4J_DATABASE", "cinegraph")


# ---------------------------------------------------------------------------
# CONNECTION
# ---------------------------------------------------------------------------

def get_driver():
    """Create and return a Neo4j driver using credentials from .env."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


# ---------------------------------------------------------------------------
# GRAPH CONSTRUCTION
# Reads the movies CSV and writes all nodes and relationships to Neo4j.
# Uses MERGE throughout so re-running is safe — no duplicate nodes created.
# ---------------------------------------------------------------------------

def build_knowledge_graph(csv_path: str) -> None:
    """
    Reads the movies CSV and populates Neo4j with Movie, Actor, Director,
    and Genre nodes connected by APPEARED_IN, DIRECTED, HAS_GENRE, and
    SIMILAR_TO relationships. Safe to re-run — MERGE prevents duplicates.
    """
    df     = pd.read_csv(csv_path)
    driver = get_driver()
    total  = len(df)

    with driver.session(database=NEO4J_DB) as session:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            tmdb_id = int(row["tmdb_id"])

            # --- Movie node ---
            session.run("""
                MERGE (m:Movie {tmdb_id: $id})
                SET m.title        = $title,
                    m.release_year = $year,
                    m.vote_average = $rating
            """,
                id     = tmdb_id,
                title  = str(row["title"]),
                year   = int(row["release_year"]) if pd.notna(row.get("release_year")) else None,
                rating = float(row["vote_average"]) if pd.notna(row.get("vote_average")) else None,
            )

            # --- Actor → Movie edges (capped at 10 per film) ---
            if pd.notna(row.get("cast_names")):
                for actor in row["cast_names"].split(", ")[:10]:
                    session.run("""
                        MERGE (a:Actor {name: $name})
                        WITH a
                        MATCH (m:Movie {tmdb_id: $id})
                        MERGE (a)-[:APPEARED_IN]->(m)
                    """, name=actor.strip(), id=tmdb_id)

            # --- Director → Movie edges ---
            if pd.notna(row.get("directors")):
                for director in row["directors"].split(", "):
                    session.run("""
                        MERGE (d:Director {name: $name})
                        WITH d
                        MATCH (m:Movie {tmdb_id: $id})
                        MERGE (d)-[:DIRECTED]->(m)
                    """, name=director.strip(), id=tmdb_id)

            # --- Movie → Genre edges ---
            if pd.notna(row.get("genres")):
                for genre in row["genres"].split(", "):
                    session.run("""
                        MERGE (g:Genre {name: $name})
                        WITH g
                        MATCH (m:Movie {tmdb_id: $id})
                        MERGE (m)-[:HAS_GENRE]->(g)
                    """, name=genre.strip(), id=tmdb_id)

            # --- Movie → Similar Movie edges (capped at 5) ---
            if pd.notna(row.get("similar_ids")):
                for sim_id in str(row["similar_ids"]).split(",")[:5]:
                    sim_id = sim_id.strip()
                    if sim_id:
                        session.run("""
                            MATCH (m:Movie {tmdb_id: $id})
                            MERGE (s:Movie {tmdb_id: $sim})
                            MERGE (m)-[:SIMILAR_TO]->(s)
                        """, id=tmdb_id, sim=int(sim_id))

            if i % 500 == 0:
                print(f"  {i:,} / {total:,} rows written...")

    driver.close()
    print(f"Graph ingestion complete — {total:,} movies written to Neo4j.")


# ---------------------------------------------------------------------------
# QUERY FUNCTIONS
# Called by RAG_pipeline.py and visualisations.py.
# Each opens its own short-lived session and closes it automatically.
# ---------------------------------------------------------------------------

def get_movies_by_actor(driver, actor_name: str) -> list[str]:
    """
    Returns a list of tmdb_id strings for all films an actor appeared in.
    Returns an empty list if the actor is not found.
    """
    with driver.session(database=NEO4J_DB) as session:
        result = session.run("""
            MATCH (a:Actor {name: $name})-[:APPEARED_IN]->(m:Movie)
            RETURN m.tmdb_id AS id
        """, name=actor_name)
        return [str(r["id"]) for r in result]


def get_collaborators(driver, director_name: str) -> list[str]:
    """
    Returns a list of actor names who appeared in any film directed by
    the given director. Returns an empty list if the director is not found.
    """
    with driver.session(database=NEO4J_DB) as session:
        result = session.run("""
            MATCH (d:Director {name: $name})-[:DIRECTED]->(m:Movie)
                  <-[:APPEARED_IN]-(a:Actor)
            RETURN DISTINCT a.name AS name
        """, name=director_name)
        return [r["name"] for r in result]


def find_path(driver, source_tmdb_id: int, target_tmdb_id: int) -> list[dict]:
    """
    Finds the shortest relationship path between two movies, traversing
    actors, directors, and genres in either direction.
    Returns a list of node dicts with keys: type, label, year.
    Returns an empty list if no path exists.
    """
    with driver.session(database=NEO4J_DB) as session:
        result = session.run("""
            MATCH path = shortestPath(
                (s:Movie {tmdb_id: $src})-[*]-(t:Movie {tmdb_id: $tgt})
            )
            RETURN [n IN nodes(path) | {
                type:  labels(n)[0],
                label: COALESCE(n.title, n.name),
                year:  n.release_year
            }] AS nodes
        """, src=source_tmdb_id, tgt=target_tmdb_id)
        row = result.single()
        return row["nodes"] if row else []


def resolve_title_to_id(driver, title_lower: str) -> int | None:
    """
    Looks up a movie's tmdb_id from a lowercased title string.
    Used by graph6_pathfinding in visualisations.py.
    Returns None if no match found.
    """
    with driver.session(database=NEO4J_DB) as session:
        result = session.run("""
            MATCH (m:Movie)
            WHERE toLower(m.title) = $title
            RETURN m.tmdb_id AS id
        """, title=title_lower)
        row = result.single()
        return row["id"] if row else None


# ---------------------------------------------------------------------------
# MAIN — Populate Neo4j from the CSV
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_knowledge_graph(CSV_PATH)