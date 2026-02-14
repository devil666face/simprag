#!.venv/bin/python3
from typing import List
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

COLLECTION = "docs"
EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
MILVUS_URI = str(Path("data") / "docling.db")

mcp = FastMCP("DocRAG-MilvusLite", json_response=True)

connections.connect("default", uri=MILVUS_URI)

collection = Collection(COLLECTION)
collection.load()

transformer = SentenceTransformer(EMBED_MODEL_NAME)


@mcp.tool()
def search_docs(
    query: str, top_k: int = 5, source_filter: str | None = None
) -> List[dict]:
    query_vec = transformer.encode([query]).tolist()
    expr = None
    if source_filter:
        expr = f'source == "{source_filter}"'
    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        expr=expr,
        output_fields=["text", "source"],
    )
    out: List[dict] = []
    for hits in results:
        for h in hits:
            out.append(
                {
                    "text": h.entity.get("text"),
                    "source": h.entity.get("source"),
                    "score": float(h.distance),
                }
            )
    return out


if __name__ == "__main__":
    out = search_docs("параллельные циклы")
    print(out)
    # mcp.run()
