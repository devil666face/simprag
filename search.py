#!/usr/bin/env python3
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
    parser = argparse.ArgumentParser(description="Query Milvus-backed document store")
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
        help="Ad-hoc query to run instead of starting MCP server",
    )
    parser.add_argument(
        "--source",
        help="Optional path filter for search",
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
        top_k: int = DEFAULT_TOP_K,
        source_filter: str | None = None,
        nprobe: int = DEFAULT_NPROBE,
    ) -> List[dict]:
        query_vec = ctx.transformer.encode([query]).tolist()
        expr = None
        if source_filter:
            expr = f'source == "{source_filter}"'
        results = ctx.collection.search(
            data=query_vec,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": nprobe}},
            limit=top_k,
            expr=expr,
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

    mcp = FastMCP("RAG", json_response=True)
    search_fn = register_tools(mcp, ctx)

    if args.query:
        logging.info(
            "Running one-off query '%s' (top_k=%s, source=%s)",
            args.query,
            args.top_k,
            args.source or "*",
        )
        results = search_fn(
            query=args.query,
            top_k=args.top_k,
            source_filter=args.source,
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
