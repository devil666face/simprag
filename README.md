# Document Store Utilities

Two CLI helpers manage a local retrieval store built on Milvus Lite embeddings:

- `embed.py` converts supported documents into contextualized chunks, encodes them with SentenceTransformers, and loads them into a Milvus collection.
- `search.py` queries that collection once or exposes the same search logic as an MCP tool via FastMCP.

## `embed.py`

```
python embed.py [options]
```

Typical flow: point `--docs` at the source directory, confirm the Milvus URI, then run once to populate the `docs` collection.

Available flags:

| Flag                     | Default                                           | Purpose                                                     |
| ------------------------ | ------------------------------------------------- | ----------------------------------------------------------- |
| `--docs PATH`            | `docs/`                                           | Input directory scanned recursively for files.              |
| `--collection NAME`      | `docs`                                            | Target Milvus collection; created automatically if missing. |
| `--embed-model MODEL_ID` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | SentenceTransformer model used for chunk embeddings.        |
| `--milvus-uri URI`       | `data/docling.db`                                 | SQLite-backed Milvus Lite URI or remote connection string.  |
| `--batch-size N`         | `512`                                             | Number of chunks per insert (0 loads everything at once).   |
| `--chunk-chars N`        | `1500`                                            | Soft cap for merged chunk length; `0` disables merging.     |
| `--ext .EXT …`           | `(.pdf … .rst)`                                   | Case-insensitive whitelist of file extensions to ingest.    |
| `--log LEVEL`            | `INFO`                                            | Standard logging verbosity (`CRITICAL`…`DEBUG`).            |

The script reports skipped files, inserts chunks in batches, and flushes the collection before exiting. A non-zero exit means no files were found.

## `search.py`

```
python search.py --query "..." [options]
python search.py --serve [options]
```

Usage modes:

- `--query TEXT` runs a single search and prints JSON results.
- `--serve` starts a FastMCP server that exposes the `search_docs` tool.
- Running without either flag logs a reminder and exits.

Available flags:

| Flag                     | Default                                           | Purpose                                                            |
| ------------------------ | ------------------------------------------------- | ------------------------------------------------------------------ |
| `--collection NAME`      | `docs`                                            | Milvus collection to query.                                        |
| `--embed-model MODEL_ID` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | SentenceTransformer model used for query embeddings.               |
| `--milvus-uri URI`       | `data/docling.db`                                 | Milvus Lite SQLite file or remote URI.                             |
| `--top-k N`              | `5`                                               | Maximum results returned per query.                                |
| `--nprobe N`             | `10`                                              | Milvus IVF search parameter (`1…nlist`).                           |
| `--query TEXT`           | —                                                 | Immediate search term; bypasses MCP server mode.                   |
| `--source PATH`          | —                                                 | Optional exact source filter applied to Milvus search expressions. |
| `--serve`                | `false`                                           | Starts the FastMCP server when set.                                |
| `--log LEVEL`            | `INFO`                                            | Logging verbosity (`CRITICAL`…`DEBUG`).                            |

The MCP server registers a single `search_docs` tool that returns `text`, `source`, and cosine `score` fields for each hit.
