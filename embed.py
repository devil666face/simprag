import glob
from pathlib import Path
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

DOCS_DIR = "docs"
COLLECTION = "docs"
EMBED_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
MILVUS_URI = str(Path("data") / "docling.db")

converter = DocumentConverter()
emb_model = SentenceTransformer(EMBED_MODEL_NAME)
emb_dim = emb_model.get_sentence_embedding_dimension()


def chunk_text(text: str, max_chars: int = 800):
    text = text.strip()
    for i in range(0, len(text), max_chars):
        yield text[i : i + max_chars]


def convert_file(path: str) -> str:
    result = converter.convert(path)
    return result.document.export_to_markdown()


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
            dim=emb_dim,
        ),
    ]
    schema = CollectionSchema(fields, description="RAG documents")
    col = Collection(name=COLLECTION, schema=schema)
    col.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 1024},
        },
    )
    return col


def ingest():
    col = get_or_create_collection()
    files = []
    for ext in ("*.pdf", "*.docx", "*.txt"):
        files.extend(glob.glob(str(Path(DOCS_DIR) / ext)))
    sources, texts, vectors = [], [], []
    for path in files:
        full_text = convert_file(path)
        chunks = list(chunk_text(full_text))
        if not chunks:
            continue
        embs = emb_model.encode(chunks).tolist()
        for chunk, vec in zip(chunks, embs):
            sources.append(str(path))
            texts.append(chunk)
            vectors.append(vec)
    if not vectors:
        print("No data to ingest")
        return
    col.insert([sources, texts, vectors])
    col.flush()
    print(f"Ingested {len(vectors)} chunks into '{COLLECTION}' at {MILVUS_URI}")


if __name__ == "__main__":
    ingest()
