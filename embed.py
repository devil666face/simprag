#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Iterable

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer

DEFAULT_DOCS_DIR = Path("docs")
DEFAULT_COLLECTION = "docs"
DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
DEFAULT_MILVUS_URI = Path("data") / "docling.db"
DEFAULT_BATCH_SIZE = 512
DEFAULT_CHUNK_CHARS = 1500
DEFAULT_EXTENSIONS = (
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".txt",
    ".md",
    ".rst",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert documents to embeddings and store them in Milvus",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help="Directory with source documents",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Milvus collection name",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL_NAME,
        help="Sentence-Transformers model to use for embeddings",
    )
    parser.add_argument(
        "--milvus-uri",
        type=Path,
        default=DEFAULT_MILVUS_URI,
        help="Milvus SQLite URI or connection string",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks per Milvus insert (0 = all at once)",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help="Approximate max characters per chunk (0 disables merging)",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=list(DEFAULT_EXTENSIONS),
        metavar=".EXT",
        help="File extensions to ingest (case-insensitive)",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def iter_source_files(
    root: Path, include_ext: Iterable[str] | None = None
) -> Iterable[Path]:
    include_lower = {ext.lower() for ext in include_ext} if include_ext else None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if include_lower and path.suffix.lower() not in include_lower:
            logging.debug("Skipping %s (extension not allowed)", path)
            continue
        yield path


def convert_file(
    path: Path,
    converter: DocumentConverter,
    chunker: HybridChunker,
) -> list[str]:
    """Convert a file with Docling and return contextualized chunks."""
    doc = converter.convert(source=str(path)).document
    chunk_iter = chunker.chunk(dl_doc=doc)
    return [chunker.contextualize(chunk=chunk) for chunk in chunk_iter]


def merge_chunks(chunks: list[str], max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [chunk for chunk in chunks if chunk.strip()]
    merged: list[str] = []
    buffer: list[str] = []
    current_len = 0
    for raw_chunk in chunks:
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        chunk_len = len(chunk)
        if buffer and current_len + chunk_len + 2 > max_chars:
            merged.append("\n\n".join(buffer))
            buffer = [chunk]
            current_len = chunk_len
        else:
            buffer.append(chunk)
            current_len += chunk_len + (2 if buffer[:-1] else 0)
    if buffer:
        merged.append("\n\n".join(buffer))
    return merged


def get_or_create_collection(
    collection_name: str,
    milvus_uri: str,
    dim: int,
) -> Collection:
    connections.connect("default", uri=milvus_uri)
    if utility.has_collection(collection_name):
        return Collection(collection_name)
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=8192,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
        ),
    ]
    schema = CollectionSchema(fields, description="RAG documents")
    collection = Collection(name=collection_name, schema=schema)
    index_result = collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 1024},
        },
        _async=False,
    )
    if asyncio.iscoroutine(index_result):
        asyncio.run(index_result)
    return collection


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(levelname)s: %(message)s",
    )

    converter = DocumentConverter()
    transformer = SentenceTransformer(args.embed_model)
    dim = transformer.get_sentence_embedding_dimension()
    if dim is None:
        raise RuntimeError(
            "Embedding dimension is undefined for model %s" % args.embed_model
        )
    tokenizer = HuggingFaceTokenizer(tokenizer=transformer.tokenizer)
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    collection = get_or_create_collection(
        collection_name=args.collection,
        milvus_uri=str(args.milvus_uri),
        dim=dim,
    )

    files = list(iter_source_files(args.docs, args.ext))
    if not files:
        logging.warning("No files found under %s", args.docs)
        return 1

    sources_batch: list[str] = []
    texts_batch: list[str] = []
    vectors_batch: list[list[float]] = []
    ingested = 0
    skipped = 0

    for path in files:
        try:
            chunks = convert_file(path, converter, chunker)
        except Exception as exc:  # pragma: no cover - logging only
            skipped += 1
            logging.warning("Skipping %s: %s", path, exc)
            continue
        chunks = merge_chunks(chunks, args.chunk_chars)
        if not chunks:
            logging.debug("No chunks produced for %s", path)
            continue
        embs = transformer.encode(chunks).tolist()
        for chunk, vec in zip(chunks, embs):
            logging.info("Chunk from %s (%s chars)", path, len(chunk))
            logging.debug("%s", chunk)
            sources_batch.append(str(path))
            texts_batch.append(chunk)
            vectors_batch.append(vec)
            if args.batch_size > 0 and len(vectors_batch) >= args.batch_size:
                collection.insert([sources_batch, texts_batch, vectors_batch])
                ingested += len(vectors_batch)
                logging.info(
                    "Inserted %s chunks (total %s)", len(vectors_batch), ingested
                )
                sources_batch.clear()
                texts_batch.clear()
                vectors_batch.clear()

    if vectors_batch:
        collection.insert([sources_batch, texts_batch, vectors_batch])
        ingested += len(vectors_batch)
        logging.info("Inserted final %s chunks", len(vectors_batch))

    if ingested == 0:
        logging.info("No data to ingest")
        return 0

    collection.flush()
    logging.info(
        "Ingested %s chunks into '%s' at %s (skipped %s files)",
        ingested,
        args.collection,
        args.milvus_uri,
        skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
