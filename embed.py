#!.venv/bin/python3
from pathlib import Path

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

DOCS_DIR = "docs"
COLLECTION = "docs"
EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
MILVUS_URI = str(Path("data") / "docling.db")

converter = DocumentConverter()
transformer = SentenceTransformer(EMBED_MODEL_NAME)
dimension = transformer.get_sentence_embedding_dimension()
tokenizer = HuggingFaceTokenizer(tokenizer=transformer.tokenizer)
chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)


def convert_file(path: str) -> list[str]:
    """Convert a file with Docling and return contextualized chunks."""
    result = converter.convert(path)
    dl_doc = result.document
    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    return [chunker.contextualize(chunk=c) for c in chunk_iter]


def get_or_create_collection() -> Collection:
    connections.connect("default", uri=MILVUS_URI)
    if utility.has_collection(COLLECTION):
        return Collection(COLLECTION)
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
            dim=dimension,
        ),
    ]
    schema = CollectionSchema(fields, description="RAG documents")
    collection = Collection(name=COLLECTION, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 1024},
        },
    )
    return collection


def main():
    collection = get_or_create_collection()
    # Traverse all files under DOCS_DIR; Docling will decide what it can parse.
    files = [p for p in Path(DOCS_DIR).rglob("*") if p.is_file()]
    sources, texts, vectors = [], [], []
    for path in files:
        try:
            chunks = convert_file(str(path))
        except Exception as exc:
            # Skip files Docling cannot handle.
            print(f"Skipping {path}: {exc}")
            continue
        if not chunks:
            continue
        embs = transformer.encode(chunks).tolist()
        for chunk, vec in zip(chunks, embs):
            sources.append(str(path))
            texts.append(chunk)
            vectors.append(vec)
    if not vectors:
        print("No data to ingest")
        return
    collection.insert([sources, texts, vectors])
    collection.flush()
    print(f"Ingested {len(vectors)} chunks into '{COLLECTION}' at {MILVUS_URI}")


if __name__ == "__main__":
    main()
