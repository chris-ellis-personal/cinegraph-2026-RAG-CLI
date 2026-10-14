# cinegraph-2026-RAG-CLI

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg?style=flat-square)](https://www.python.org/downloads/)

> A RAG pipeline, Neo4j knowledge graph and six visualisations over a TMDB/IMDB movies dataset.

CineGraph 2026 is an educational demonstration project, built as a Data Wrangling final project at North Metropolitan TAFE (Perth). It implements a full **Retrieval-Augmented Generation (RAG)** pipeline over a combined TMDB/IMDB dataset of roughly 22,393 films, augmented with a **knowledge graph** for relational queries and six **data visualisation** scripts.

It is not a production product. It is a demonstration of industry-standard data engineering and AI pipeline techniques: semantic embedding, vector search, LLM reranking, graph modelling and network analysis.

## Table of Contents

- [Security](#security)
- [Background](#background)
  - [Architecture](#architecture)
  - [Dataset](#dataset)
  - [Design Decisions](#design-decisions)
- [Install](#install)
  - [Dependencies](#dependencies)
  - [Neo4j Setup](#neo4j-setup)
  - [Environment Configuration](#environment-configuration)
- [Usage](#usage)
  - [Build the Knowledge Graph](#1-build-the-knowledge-graph-run-once)
  - [Build the Vector Index](#2-build-the-vector-index-run-once)
  - [Run the RAG Pipeline](#3-run-the-rag-pipeline)
  - [Generate Visualisations](#4-generate-visualisations)
  - [Every Subsequent Session](#every-subsequent-session)
  - [Saving Output to a File](#saving-output-to-a-file)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Knowledge Graph Schema](#knowledge-graph-schema)
- [Visualisations](#visualisations)
- [Cost](#cost)
- [Data Sources and Attribution](#data-sources-and-attribution)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [API](#api)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Security

All credentials (OpenAI API key, Neo4j password) are stored in a local `.env` file.

- **Never commit `.env` to version control** and never share it. Add it, `movies.csv` and `chroma_index/` to `.gitignore`.
- Each user needs their own OpenAI API key from [platform.openai.com](https://platform.openai.com). Do not share keys between users.
- Do not hard-code API keys in the Python scripts.
- Set a monthly spending limit in your OpenAI billing settings to cap accidental cost.

## Background

CineGraph combines three complementary approaches to answering questions about films:

1. **Vector search** (ChromaDB + OpenAI embeddings) for fuzzy, semantic questions such as "dark sci-fi about memory".
2. **A knowledge graph** (Neo4j) for relational facts that vector search cannot reliably answer, such as "which actors worked with Kubrick?" or multi-hop connections between films.
3. **LLM reranking and generation** (GPT-4o) to refine candidates and produce grounded natural-language answers.

### Architecture

```
User Query
    ↓
[1] Query Parser (GPT-4o)
    Extracts: semantic_query, filters, genre_hint, director_hint, actor_hint,
              sort_hint, more_like_title
    ↓
[2] Knowledge Graph Enrichment
    Prepends relational facts to context for actor/director queries
    ↓
[3] Retrieval (ChromaDB)
    - Vector similarity search (top 20 candidates)
    - Optional metadata filters: year, runtime, rating, language, certification
    - Post-retrieval Python filters for list fields (genre, director, actor)
    - "More like X" shortcut: uses the pre-computed similar_ids column
    ↓
[4] Reranking (GPT-4o)
    Scores each of the 20 candidates 0–10 for relevance, keeps the top 5
    ↓
[5] Context Formatting
    Combines graph facts + top-5 movie records into a structured prompt block
    ↓
[6] Generation (GPT-4o)
    Temperature 0.3; grounded system prompt instructs the model to use
    only the retrieved context
    ↓
Answer returned to user
```

**Document design.** Each movie row is converted to a text blob containing `title`, `original_title`, `alternative_titles`, `tagline`, `overview`, `genres`, `keywords`, `directors`, `cast_names` (top 5 only), `writers`, `release_year`, `runtime_min`, `us_certification`, `original_language`, `production_countries` and `collection_name`. Documents shorter than 50 characters (near-empty due to nulls) are skipped. Cast is truncated to the top 5 so actor identity does not dominate plot and theme signals in the embedding.

**Metadata schema.** Stored alongside each embedding in ChromaDB for filtering:

- List fields (genres, directors, cast, streaming, countries) are JSON-serialised strings, because ChromaDB only accepts scalar metadata values.
- `None` values are replaced with sentinels (`-1`, `-1.0`, `""`), because ChromaDB rejects Python `None`.
- `tmdb_id` is the vector store primary key.

**Vector store.** ChromaDB `PersistentClient` (no server required), saved to `chroma_index/`, using cosine distance (`hnsw:space: cosine`).

### Dataset

| Property | Value |
|---|---|
| File | `movies.csv` |
| Size | 22,393 rows × 55 columns |
| Source | [CineGraph — TMDB Movies, TV & People Dataset](https://www.kaggle.com/datasets/muhammetyorulmaz1/cinegraph-tmdb-movies-tv-and-people-dataset) on Kaggle (TMDB / IMDB combined data) |
| Primary key | `tmdb_id` (zero nulls; preferred over `imdb_id`, which has 3,845 nulls) |

Key columns used:

- **Text / semantic:** `title`, `overview`, `tagline`, `keywords`, `genres`, `cast_names`, `directors`, `alternative_titles`
- **Metadata / filters:** `release_year`, `runtime_min`, `vote_average`, `vote_count`, `us_certification`, `original_language`, `production_countries`
- **Relational:** `similar_ids`, `similar_titles`, `recommended_ids`, `cast_ids`, `collection_name`
- **Streaming:** `watch_us_flatrate`
- **Financial:** `budget_usd`, `revenue_usd`, `roi_pct`

Notable nulls: `tagline` (36%), `keywords` (24%), `budget_usd` / `revenue_usd` (~56%), `us_certification` (58%), `overview` (4%).

### Design Decisions

| Decision | Rationale |
|---|---|
| ChromaDB (local) over Pinecone/Weaviate | Zero config, no server; appropriate for a university demo |
| `text-embedding-3-large` over `text-embedding-3-small` | Higher accuracy for English-dominant text |
| Retrieve 20, rerank to 5 | Vector similarity alone doesn't capture nuanced intent |
| LLM-based reranker over cross-encoder | Avoids an additional model dependency |
| `tmdb_id` as primary key | Zero nulls vs 3,845 nulls in `imdb_id` |
| Cast truncated to top 5 in document | Prevents actor identity from drowning plot/theme signals |
| JSON-serialised list fields in metadata | ChromaDB only accepts scalar metadata values |
| Sentinel values instead of `None` | ChromaDB rejects Python `None` in metadata |
| "More like X" uses `similar_ids` | Pre-computed similarity is more accurate than vector search for direct comparisons |
| Neo4j for the knowledge graph | Enables a visual graph browser and Cypher queries |
| PyVis for interactive graphs | Self-contained HTML output, no server needed |

## Install

### Prerequisites

- **Python 3.11 or higher.** Download from [python.org](https://www.python.org/downloads/). On Windows, tick **"Add Python to PATH"** during installation.
- **Neo4j Desktop.** Download from [neo4j.com/download](https://neo4j.com/download/). A free account is required to obtain a licence key.
- **An OpenAI API key** from [platform.openai.com](https://platform.openai.com).
- **Project files**, all in the same folder. The scripts find `movies.csv` and `.env` relative to their own location, so no paths need editing:

```
cinegraph-2026-RAG-CLI/
├── knowledge_graph.py
├── RAG_pipeline.py
├── visualisations.py
├── .env
└── movies.csv
```

### Get the Code

```bash
git clone https://github.com/chris-ellis-personal/cinegraph-2026-RAG-CLI.git
cd cinegraph-2026-RAG-CLI
```

The dataset (`movies.csv`) and your `.env` file are **not** part of the repository. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/muhammetyorulmaz1/cinegraph-tmdb-movies-tv-and-people-dataset), place `movies.csv` in the cloned folder, and create your own `.env` file (see [Environment Configuration](#environment-configuration)). See [Data Sources and Attribution](#data-sources-and-attribution) for the terms that apply to the data.

### Dependencies

In a terminal in the project folder, run:

```bash
pip install openai chromadb pandas tqdm networkx neo4j python-dotenv pyvis matplotlib seaborn
```

If using **Anaconda**, install some packages via conda and the rest via pip:

```bash
conda install pandas matplotlib seaborn tqdm
pip install openai chromadb networkx neo4j python-dotenv pyvis
```

| Package | Purpose |
|---|---|
| `openai` | GPT-4o (generation, reranking, query parsing) and `text-embedding-3-large` (3072-dimensional embeddings) |
| `chromadb` | Local persistent vector store |
| `pandas` | CSV loading and data manipulation |
| `tqdm` | Progress bar during embedding |
| `networkx` | Graph analysis and pathfinding for visualisations |
| `neo4j` | Python driver for the Neo4j knowledge graph |
| `python-dotenv` | Loads credentials from `.env` |
| `pyvis` | Interactive HTML network graphs |
| `matplotlib` | Static chart visualisations |
| `seaborn` | Colour palettes and chart styling |

### Neo4j Setup

**1. Create the database**

1. Open **Neo4j Desktop** and create a new project.
2. Inside the project, click **Add** → **Local DBMS**.
3. Name it (e.g. `CineGraph`) and set a password. Note it down for the `.env` file.
4. Click **Start** and wait for the green **RUNNING** status.
5. Note the **Connection URI** shown on the database panel (e.g. `neo4j://127.0.0.1:7687`). This also goes in `.env`.

**2. Run the schema commands**

Click **Open** to launch Neo4j Browser, then run each command one at a time with **Ctrl+Enter**:

```cypher
CREATE CONSTRAINT movie_tmdb_id_unique FOR (m:Movie) REQUIRE m.tmdb_id IS UNIQUE;
CREATE CONSTRAINT actor_name_unique FOR (a:Actor) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT director_name_unique FOR (d:Director) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT genre_name_unique FOR (g:Genre) REQUIRE g.name IS UNIQUE;
CREATE INDEX movie_title_index FOR (m:Movie) ON (m.title);
CREATE INDEX movie_year_index FOR (m:Movie) ON (m.release_year);
CREATE INDEX movie_rating_index FOR (m:Movie) ON (m.vote_average);
```

**3. Verify the schema**

```cypher
SHOW CONSTRAINTS;
SHOW INDEXES;
```

You should see 4 constraints (one per node type) and 7 indexes in total.

### Environment Configuration

Create a `.env` file in the project folder (the same folder as the Python scripts):

```
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j
OPENAI_API_KEY=your_openai_api_key_here
```

> **`NEO4J_DATABASE`:** Neo4j Desktop creates a default database called `neo4j` when you add a Local DBMS. Use `neo4j` unless you explicitly created a separate database with another name.

**Creating `.env` on Windows**

1. Open **Notepad** and paste in the contents above with your values filled in.
2. Choose **File → Save As** and navigate to the project folder.
3. Set **Save as type** to **All Files (`*.*`)**.
4. Enter `.env` as the filename (leading dot, no extension) and click **Save**.

Verify it loaded:

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.environ.get('NEO4J_URI'))"
```

You should see the URI printed. If you see `None`, the file was saved as `.env.txt`. Enable "Show file extensions" in Windows Explorer and rename it.

## Usage

With Neo4j running and a terminal open in the project folder, run the steps below **in this exact order**.

### 1. Build the knowledge graph (run once)

```bash
python knowledge_graph.py
```

This writes all movies, actors, directors and genres into Neo4j. Progress is printed every 500 rows. Wait for:

```
Graph ingestion complete — X movies written to Neo4j.
```

Verify in Neo4j Browser:

```cypher
MATCH (n) RETURN labels(n), count(n)
```

You should see four rows: Movie, Actor, Director and Genre, each with a count.

### 2. Build the vector index (run once)

Open `RAG_pipeline.py` and set (around line 64):

```python
BUILD_INDEX = True
```

Then run:

```bash
python RAG_pipeline.py
```

This embeds all movies into ChromaDB via OpenAI, with a progress bar for each batch. Once complete, **immediately set it back to**:

```python
BUILD_INDEX = False
```

> **Important:** Leaving `BUILD_INDEX = True` re-embeds everything from scratch on every run, making unnecessary OpenAI API calls.

### 3. Run the RAG pipeline

With `BUILD_INDEX = False`:

```bash
python RAG_pipeline.py
```

An interactive prompt appears:

```
Query: [type here]
```

Try a few queries to confirm everything works:

- `Best sci-fi films from the 1990s`
- `Movies directed by Christopher Nolan`
- `Films similar to Parasite`

Type `quit` or `exit` to stop.

### 4. Generate visualisations

```bash
python visualisations.py
```

Graphs 1–5 generate automatically. Graph 6 (pathfinding) pauses to ask for two movie titles. Output is saved to the project folder: `.html` files open in any web browser; `.png` files are static charts. No API calls are made by this script.

### Every Subsequent Session

**Starting up**

1. Open **Neo4j Desktop**.
2. Click **Start** on the database and wait for green **RUNNING**.
3. Open a terminal and run whichever script you need.

**Shutting down**

1. Type `quit` or `exit` at any script prompt to close connections cleanly.
2. In Neo4j Desktop, click **Stop** on the database and wait for it to stop.
3. Close Neo4j Desktop.

### Saving Output to a File

```bash
python RAG_pipeline.py > output.txt 2>&1
```

## Configuration

Key constants in `RAG_pipeline.py`:

```python
BUILD_INDEX       = False          # True only for first run or after CSV changes
EMBEDDING_MODEL   = "text-embedding-3-large"
GENERATION_MODEL  = "gpt-4o"
RETRIEVE_TOP_K    = 20             # Candidates fetched from vector store
RERANK_TOP_N      = 5              # Candidates passed to LLM after reranking
EMBED_BATCH_SIZE  = 100            # OpenAI API batch size
MIN_DOC_LENGTH    = 50             # Skip near-empty documents
MIN_VOTE_COUNT_FOR_RATING = 50     # Minimum votes before rating is stored
```

## Project Structure

| File / Folder | Purpose |
|---|---|
| `movies.csv` | Source dataset, downloaded separately from Kaggle (not included in the repo). Must be placed in the project folder. |
| `knowledge_graph.py` | Builds the Neo4j graph from the CSV. Run once. |
| `RAG_pipeline.py` | Main query interface. Set `BUILD_INDEX = True` once, then `False`. |
| `visualisations.py` | Generates all six graphs and charts. |
| `.env` | Stores all credentials. Never share this file. |
| `chroma_index/` | ChromaDB vector index (auto-generated). Do not delete: rebuilding requires OpenAI API calls. |
| `actor_movie_network.html` | Output: Graph 1 |
| `collaboration_network.html` | Output: Graph 2 |
| `genre_evolution_timeline.png` | Output: Graph 3 |
| `recommendation_graph.html` | Output: Graph 4 |
| `centrality_charts.png` | Output: Graph 5 |
| `pathfinding.html` | Output: Graph 6 |

## Knowledge Graph Schema

```
Node types:
  Movie     (tmdb_id, title, release_year, vote_average)
  Actor     (name)
  Director  (name)
  Genre     (name)

Edge types:
  Actor    -[APPEARED_IN]->  Movie    (cast_names, capped at 10 per film)
  Director -[DIRECTED]->     Movie    (directors field)
  Movie    -[HAS_GENRE]->    Genre    (genres field)
  Movie    -[SIMILAR_TO]->   Movie    (similar_ids field, capped at 5)
```

Query helpers:

```python
get_movies_by_actor(G, actor_name)     # All films an actor appeared in
get_collaborators(G, director_name)    # All actors in a director's films
find_path(G, source_id, target_id)     # Shortest path between two nodes
```

Inside the RAG pipeline, graph facts are prepended to the LLM context after intent parsing:

- If `actor_hint` is present, the graph confirms how many films the actor appeared in.
- If `director_hint` is present, the graph lists all actors the director has worked with.

## Visualisations

| # | Output | Type | Description |
|---|---|---|---|
| 1 | `actor_movie_network.html` | Interactive (PyVis) | Top 100 actors by film count (gold ellipses) linked to their movies (teal circles). Cast capped at 8 per film. Actor size = film count; movie size = `vote_average`. |
| 2 | `collaboration_network.html` | Interactive (PyVis) | Top 30 directors (coral diamonds) linked to actors they worked with (gold ellipses), weighted by shared films. |
| 3 | `genre_evolution_timeline.png` | Static (matplotlib) | Two subplots by decade (1960s–present) for the top 8 genres: film count and average rating. Minimum 100 votes per film. |
| 4 | `recommendation_graph.html` | Interactive (PyVis) | Top 80 movies (`vote_average` ≥ 6.5, `vote_count` ≥ 100) linked via the pre-computed `similar_ids` column. Colour = primary genre; size = `vote_average`. |
| 5 | `centrality_charts.png` | Static (matplotlib) | Top 20 actors by degree, betweenness (sampled at k=200) and eigenvector centrality. Takes 30–60 seconds to compute. |
| 6 | `pathfinding.html` | Interactive (PyVis) | Shortest relationship path between two user-entered movie titles via shared actors/directors, using `nx.shortest_path()` on an undirected graph. Path nodes are highlighted; context nodes dimmed grey. |

## Cost

| Run | Approximate cost |
|---|---|
| First run (`BUILD_INDEX = True`) | ~$0.50–$2.00 |
| Subsequent runs (`BUILD_INDEX = False`) | ~$0.01–$0.05 per query |
| `visualisations.py` | No API cost |

## Data Sources and Attribution

The dataset is **not** covered by this project's licence and is not redistributed in this repository.

- **Dataset:** [CineGraph — TMDB Movies, TV & People Dataset](https://www.kaggle.com/datasets/muhammetyorulmaz1/cinegraph-tmdb-movies-tv-and-people-dataset) on Kaggle. The only licence documentation supplied with it is a generic MIT notice whose copyright lines credit Mark Otto (2013) and Andrew Fong (2017), apparently carried over from another project rather than written for this dataset. It does not override the TMDB and IMDb terms below, which still apply to the underlying data. Because of that ambiguity, `movies.csv` is not redistributed here.
- **TMDB:** This product uses the TMDB API but is not endorsed or certified by TMDB. Please review [TMDB's terms of use](https://www.themoviedb.org/api-terms-of-use) before reuse.
- **IMDb:** The dataset includes information originating from IMDb. This project uses it for personal, non-commercial, educational purposes only. Please review [IMDb's conditions of use](https://www.imdb.com/conditions).
- **OpenAI:** Movie text from the dataset is sent to the OpenAI API to generate embeddings and answers. Review [OpenAI's terms](https://openai.com/policies/) if you adapt this project.

This project is a personal, non-commercial demonstration.

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `AuthenticationError: Incorrect API key` | Placeholder text left in `OPENAI_API_KEY` | Replace with your actual `sk-proj-...` key in `.env` |
| `AuthError: The client is unauthorized` | `.env` password doesn't match the Neo4j database password | Reset the password in Neo4j Desktop (three-dot menu → Manage → Administration) and update `NEO4J_PASSWORD` |
| `.env` values return `None` | File was saved as `.env.txt` | Enable "Show file extensions" in Windows Explorer and rename it to `.env` |
| `RuntimeError: Index not loaded` | `BUILD_INDEX = False` but no `chroma_index/` folder exists | Set `BUILD_INDEX = True` for the first run |
| `BUILD_INDEX` left as `True` | Forgot to revert after first run | Set it to `False` and save. The existing `chroma_index/` is unaffected. |
| Neo4j Browser shows 0 nodes after ingestion | Querying the wrong database | Run `:use neo4j` (or `:use cinegraph`, depending on your database name), then re-run `MATCH (n) RETURN labels(n), count(n)` |
| `KeyError: 'actor_antonio_banderas'` | `pyvis.get_node()` raises `KeyError` instead of returning `None` | Track added nodes with a Python `set()` instead |
| `NameError: 'dirs_worked_with' is not defined` | Variable defined inside an `if` block but used outside it | Move the variable calculation above the `if` block |

## Roadmap

| Feature | Status | Notes |
|---|---|---|
| Sentiment heatmap | Discussed | Requires GPT-4o or a local NLP model (VADER/DistilBERT). The `movies.csv` used here has no review text, but the Kaggle dataset page lists 25k+ reviews, which may be available as a separate file |
| Knowledge graph explorer (interactive UI) | Discussed | Requires entity extraction from overviews via GPT-4o or spaCy |
| RAG-driven dynamic dashboard | Discussed | Requires the full RAG pipeline running; ties all components together |
| Multi-dimensional scatter (embedding space) | Discussed | Uses ChromaDB embeddings + UMAP/t-SNE; no new API calls if the index is already built |

## API

This project is a set of scripts rather than a library, so there is no public API. The graph query helpers are listed under [Knowledge Graph Schema](#knowledge-graph-schema) and the tunable constants under [Configuration](#configuration).

## Maintainers

Chris Ellis ([@chris-ellis-personal](https://github.com/chris-ellis-personal))

## Contributing

This is a university demonstration project, but feedback is welcome. [Open an issue](https://github.com/chris-ellis-personal/cinegraph-2026-RAG-CLI/issues) or submit a pull request.

Please keep this README compliant with the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.

## License

[MIT](LICENSE) © 2026 Chris Ellis

The MIT licence applies to the source code in this repository only. It does not apply to the dataset or to any third-party data or services (see [Data Sources and Attribution](#data-sources-and-attribution)).
