#!.venv/bin/python3
# from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, cast

from mcp.server.fastmcp import FastMCP
from mcp.types import Icon, ToolAnnotations
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

DEFAULT_COLLECTION = "docs"
DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
DEFAULT_MILVUS_URI = Path("data") / "docling.db"
DEFAULT_TOP_K = 5
DEFAULT_NPROBE = 10  # [1, nlist]
DEFAULT_SOURCE_NAME_CANDIDATES = 5

SEARCH_TOOL_ICON = Icon(
    src=(
        "data:image/svg+xml;utf8,"
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
        "<rect width='64' height='64' rx='12' fill='%23121d2f'/>"
        "<circle cx='28' cy='28' r='14' stroke='%23c4f1ff' stroke-width='4' fill='none'/>"
        "<line x1='40' y1='40' x2='56' y2='56' stroke='%23c4f1ff' stroke-width='5' stroke-linecap='round'/>"
        "</svg>"
    ),
    mimeType="image/svg+xml",
    sizes=["64x64"],
)

SEARCH_TOOL_ANNOTATIONS = ToolAnnotations(
    title="Semantic Document Search",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

SEARCH_TOOL_META = {
    "collectionDefault": DEFAULT_COLLECTION,
    "embeddingModel": DEFAULT_EMBED_MODEL_NAME,
    "maintainer": "RAG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query Milvus-backed document store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Milvus collection name",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL_NAME,
        help="Sentence-Transformers model for queries",
    )
    parser.add_argument(
        "--milvus-uri",
        type=Path,
        default=DEFAULT_MILVUS_URI,
        help="Milvus SQLite URI or connection string",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of documents to return",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=DEFAULT_NPROBE,
        help="Milvus nprobe search parameter",
    )
    parser.add_argument(
        "--query",
        nargs="?",
        default=None,
        const="",
        metavar="TEXT",
        help=(
            "Ad-hoc query to run instead of starting MCP server. "
            "Omit TEXT (i.e., pass --query without a value) to run a source-only search."
        ),
    )
    parser.add_argument(
        "--source-name",
        nargs="?",
        default=None,
        const="",
        metavar="NAME",
        help=(
            "File name to use for vector-based source search. "
            "Omit NAME (i.e., pass --source-name without a value) to disable it."
        ),
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print all distinct source values in the collection and exit",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run FastMCP server instead of one-off query",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class SearchContext:
    collection: Collection
    transformer: SentenceTransformer


def build_context(
    collection_name: str, milvus_uri: str, embed_model: str
) -> SearchContext:
    connections.connect("default", uri=milvus_uri)
    collection = Collection(collection_name)
    collection.load()
    transformer = SentenceTransformer(embed_model)
    return SearchContext(collection=collection, transformer=transformer)


def list_sources(ctx: SearchContext, limit: int = 16384) -> int:
    """Print all distinct source values from the collection.

    Returns process exit code (0 on success, non-zero on failure).
    """

    logging.info("Listing distinct sources from collection '%s'", ctx.collection.name)
    try:
        # Milvus требует указывать limit, если expr пустой.
        rows_any = ctx.collection.query(
            expr="",
            output_fields=["source"],
            limit=limit,
        )
        rows_result = getattr(rows_any, "result", None)
        rows_iter: Iterable[dict]
        if callable(rows_result):
            rows_iter = cast(Iterable[dict], rows_result())
        else:
            rows_iter = cast(Iterable[dict], rows_any)
    except Exception as exc:  # pragma: no cover - logging only
        logging.error("Failed to query sources: %s", exc)
        return 1

    sources = sorted({row.get("source", "") for row in rows_iter if row.get("source")})
    for src in sources:
        print(src)
    logging.info("Found %d distinct sources", len(sources))
    return 0


def register_tools(mcp: FastMCP, ctx: SearchContext) -> Callable[..., List[dict]]:
    @mcp.tool(
        name="search_docs",
        title="Search Stored Documents",
        description="Return the most similar document chunks from the RAG store.",
        annotations=SEARCH_TOOL_ANNOTATIONS,
        icons=[SEARCH_TOOL_ICON],
        meta=SEARCH_TOOL_META,
        structured_output=True,
    )
    def search_docs(
        query: str,
        source_name: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        nprobe: int = DEFAULT_NPROBE,
    ) -> List[dict]:
        """Search documents with optional file-name based filtering.

        - If ``source_name`` is provided, it is embedded and used to search the
          ``source_name_embedding`` field to find the closest file names.
          Then:
            * if ``query`` is non-empty, a semantic search is run over
              ``text_embedding`` restricted to those sources;
            * if ``query`` is empty, all chunks for those sources are returned
              without additional vector search.
        - Otherwise, a regular semantic search over ``text_embedding`` is
          performed across all documents.
        """

        normalized_query = (query or "").strip()
        has_query = bool(normalized_query)
        source_name_value = (source_name or "").strip()
        has_source_name = bool(source_name_value)

        # Helper: convert Milvus query() rows into output rows.
        def _rows_to_out(rows_any: object) -> List[dict]:
            rows_result = getattr(rows_any, "result", None)
            if callable(rows_result):
                rows_iter = cast(Iterable[dict], rows_result())
            else:
                rows_iter = cast(Iterable[dict], rows_any)
            out_rows: List[dict] = []
            for row in rows_iter:
                out_rows.append(
                    {
                        "text": row.get("text"),
                        "source": row.get("source"),
                        "score": 1.0,
                    }
                )
            return out_rows

        # 1) If a file-name vector search is requested.
        if has_source_name:
            name_limit = max(DEFAULT_SOURCE_NAME_CANDIDATES, top_k)
            name_vec = ctx.transformer.encode([source_name_value]).tolist()
            name_search = ctx.collection.search(
                data=name_vec,
                anns_field="source_name_embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": nprobe}},
                limit=max(name_limit, 1),
                output_fields=["source"],
            )
            result_fn = getattr(name_search, "result", None)
            if callable(result_fn):
                name_search = result_fn()
            candidate_sources: list[str] = []
            for hits in cast(Iterable, name_search):
                for hit in hits:
                    src_val = hit.entity.get("source")
                    if src_val:
                        src_str = str(src_val)
                        if src_str not in candidate_sources:
                            candidate_sources.append(src_str)
                    if len(candidate_sources) >= name_limit:
                        break
                if len(candidate_sources) >= name_limit:
                    break

            if not candidate_sources:
                logging.debug("No sources matched source_name '%s'", source_name_value)
                return []

            if not has_query:
                # Return all chunks for every candidate source.
                out_rows: List[dict] = []
                for src in candidate_sources:
                    rows_any = ctx.collection.query(
                        expr=f'source == "{src}"',
                        output_fields=["text", "source"],
                        limit=16384,
                    )
                    out_rows.extend(_rows_to_out(rows_any))
                logging.debug(
                    "Source-name-only search returned %s chunks for %s",
                    len(out_rows),
                    candidate_sources,
                )
                return out_rows

            # Vector search within the candidate sources.
            query_vec = ctx.transformer.encode([normalized_query]).tolist()
            sources_expr = json.dumps(candidate_sources, ensure_ascii=False)
            expr_clause = f"source in {sources_expr}"
            results = ctx.collection.search(
                data=query_vec,
                anns_field="text_embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": nprobe}},
                limit=top_k,
                expr=expr_clause,
                output_fields=["text", "source"],
            )
            result_fn = getattr(results, "result", None)
            if callable(result_fn):
                results = result_fn()
            iterable_results = cast(Iterable, results)
            out: List[dict] = []
            for hits in iterable_results:
                for hit in hits:
                    out.append(
                        {
                            "text": hit.entity.get("text"),
                            "source": hit.entity.get("source"),
                            "score": float(hit.distance),
                        }
                    )
            logging.debug(
                "Source-name + query search returned %s results across %s",
                len(out),
                candidate_sources,
            )
            return out

        # 2) Regular semantic search over text embeddings across all sources.
        query_vec = ctx.transformer.encode([normalized_query]).tolist()
        results = ctx.collection.search(
            data=query_vec,
            anns_field="text_embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": nprobe}},
            limit=top_k,
            expr=None,
            output_fields=["text", "source"],
        )
        result_fn = getattr(results, "result", None)
        if callable(result_fn):
            results = result_fn()
        iterable_results = cast(Iterable, results)
        out: List[dict] = []
        for hits in iterable_results:
            for hit in hits:
                out.append(
                    {
                        "text": hit.entity.get("text"),
                        "source": hit.entity.get("source"),
                        "score": float(hit.distance),
                    }
                )
        logging.debug("Search returned %s results", len(out))
        return out

    return search_docs


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(levelname)s: %(message)s",
    )

    ctx = build_context(
        collection_name=args.collection,
        milvus_uri=str(args.milvus_uri),
        embed_model=args.embed_model,
    )

    if args.list_sources:
        return list_sources(ctx)

    mcp = FastMCP("RAG", json_response=True)
    search_fn = register_tools(mcp, ctx)

    source_name_value = (args.source_name or "").strip() if args.source_name else None
    should_run_query = args.query is not None or bool(source_name_value)
    if should_run_query:
        query_value = args.query if args.query is not None else ""
        logging.info(
            "Running one-off query '%s' (top_k=%s, source_name=%s)",
            query_value,
            args.top_k,
            source_name_value or "",
        )
        results = search_fn(
            query=query_value,
            top_k=args.top_k,
            source_name=source_name_value,
            nprobe=args.nprobe,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    if args.serve:
        logging.info("Starting FastMCP server ...")
        mcp.run(transport="streamable-http")
        return 0

    logging.info("Nothing to do: use --query for ad-hoc search or --serve to run MCP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
