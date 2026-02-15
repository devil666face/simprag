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
DEFAULT_CHUNK_CHARS = 1024
DEFAULT_MAX_TEXT_LEN = 8192
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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Approximate min characters per chunk (0 disables merging)",
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=DEFAULT_MAX_TEXT_LEN,
        help=(
            "Hard max characters per stored text field "
            "(must be within Milvus VARCHAR limits)"
        ),
    )
    parser.add_argument(
        "--by-source",
        action="store_true",
        help=(
            "Merge all chunks for a single source into one text and "
            "re-split it into pieces of up to --max-text-len characters"
        ),
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


def _split_with_max(text: str, max_len: int) -> list[str]:
    """Split ``text`` into pieces not exceeding ``max_len``."""
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]


def merge_chunks(chunks: list[str], min_chars: int, max_len: int) -> list[str]:
    """Merge smaller chunks until they reach at least ``min_chars``.

    Additionally enforces the hard upper limit ``TEXT_FIELD_MAX_LENGTH``
    so that no resulting chunk exceeds the Milvus ``text`` field size.

    The last chunk for a file may still be shorter than ``min_chars``
    if there is not enough text left to reach the threshold.
    """

    clean_chunks = [c.strip() for c in chunks if c.strip()]

    # No minimum: просто уважать максимальную длину поля текста.
    if min_chars <= 0:
        result: list[str] = []
        for chunk in clean_chunks:
            result.extend(_split_with_max(chunk, max_len))
        return result

    result: list[str] = []
    buffer_parts: list[str] = []
    buffer_len = 0

    for chunk in clean_chunks:
        chunk_len = len(chunk)

        # Если добавление чанка переполнит верхний предел — сначала сбросить буфер.
        if buffer_parts and buffer_len + 2 + chunk_len > max_len:
            result.extend(_split_with_max("\n\n".join(buffer_parts), max_len))
            buffer_parts = []
            buffer_len = 0

        # Если буфер пустой, просто стартуем новый.
        if not buffer_parts:
            buffer_parts = [chunk]
            buffer_len = chunk_len
            continue

        # Здесь буфер уже непустой и добавление не переполнит максимум.
        if buffer_len >= min_chars:
            # Набрали достаточно текста — сбрасываем текущий буфер
            # и начинаем следующий с нового чанка.
            result.extend(_split_with_max("\n\n".join(buffer_parts), max_len))
            buffer_parts = [chunk]
            buffer_len = chunk_len
        else:
            # Ещё не дотянули до минимума — продолжаем naкапливать.
            buffer_parts.append(chunk)
            buffer_len += 2 + chunk_len

    if buffer_parts:
        result.extend(_split_with_max("\n\n".join(buffer_parts), max_len))

    return result


def get_or_create_collection(
    collection_name: str,
    milvus_uri: str,
    dim: int,
    text_max_length: int,
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
            max_length=1024,
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=text_max_length,
        ),
        FieldSchema(
            name="text_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
        ),
        FieldSchema(
            name="source_name_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
        ),
    ]
    schema = CollectionSchema(fields, description="RAG documents")
    collection = Collection(name=collection_name, schema=schema)
    for field_name in ("text_embedding", "source_name_embedding"):
        index_result = collection.create_index(
            field_name=field_name,
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

    # Validate and normalize text and chunk size settings.
    if args.max_text_len < 1:
        logging.warning(
            "max-text-len (%d) is less than 1; clamping to %s", DEFAULT_MAX_TEXT_LEN
        )
        args.max_text_len = DEFAULT_MAX_TEXT_LEN
    elif args.max_text_len > 65535:
        logging.warning(
            "max-text-len (%d) is greater than 65535; clamping to 65535",
            args.max_text_len,
        )
        args.max_text_len = 65535

    if args.chunk_chars > 0 and args.chunk_chars > args.max_text_len:
        logging.warning(
            "chunk-chars (%d) is greater than max-text-len (%d); "
            "clamping chunk-chars to max-text-len",
            args.chunk_chars,
            args.max_text_len,
        )
        args.chunk_chars = args.max_text_len

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
        text_max_length=args.max_text_len,
    )

    files = list(iter_source_files(args.docs, args.ext))
    if not files:
        logging.warning("No files found under %s", args.docs)
        return 1

    sources_batch: list[str] = []
    texts_batch: list[str] = []
    vectors_batch: list[list[float]] = []
    source_name_vectors_batch: list[list[float]] = []
    ingested = 0
    skipped = 0

    for path in files:
        try:
            chunks = convert_file(path, converter, chunker)
        except Exception as exc:  # pragma: no cover - logging only
            skipped += 1
            logging.warning("Skipping %s: %s", path, exc)
            continue
        if args.by_source:
            # Concatenate all chunks for this source and then split into
            # pieces that fit into the Milvus text field.
            full_text = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
            if not full_text:
                logging.debug("No text produced for %s", path)
                continue
            chunks = _split_with_max(full_text, args.max_text_len)
        else:
            chunks = merge_chunks(chunks, args.chunk_chars, args.max_text_len)
        if not chunks:
            logging.debug("No chunks produced for %s", path)
            continue
        text_embs = transformer.encode(chunks).tolist()
        source_name_emb = transformer.encode([path.name]).tolist()[0]
        for chunk, vec in zip(chunks, text_embs):
            logging.info("Chunk from %s (%s chars)", path, len(chunk))
            logging.debug("%s", chunk)
            # Store only the file name (not the full path) in the `source` field.
            sources_batch.append(path.name)
            texts_batch.append(chunk)
            vectors_batch.append(vec)
            source_name_vectors_batch.append(source_name_emb)
            if args.batch_size > 0 and len(vectors_batch) >= args.batch_size:
                collection.insert(
                    [
                        sources_batch,
                        texts_batch,
                        vectors_batch,
                        source_name_vectors_batch,
                    ]
                )
                ingested += len(vectors_batch)
                logging.info(
                    "Inserted %s chunks (total %s)", len(vectors_batch), ingested
                )
                sources_batch.clear()
                texts_batch.clear()
                vectors_batch.clear()
                source_name_vectors_batch.clear()

    if vectors_batch:
        collection.insert(
            [
                sources_batch,
                texts_batch,
                vectors_batch,
                source_name_vectors_batch,
            ]
        )
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
