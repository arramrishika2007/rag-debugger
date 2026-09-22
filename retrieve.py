import os
from dotenv import load_dotenv
import voyageai

from qdrant_client import QdrantClient


load_dotenv()

vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "rag_debugger_docs"


def retrieve(query: str, top_k=3) -> list[dict]:

    query_embedding = vo.embed(
        [query],
        model="voyage-3-lite",
        input_type="query"
    ).embeddings[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    )

    return [
        {
            "chunk_id": p.payload["chunk_id"],
            "text": p.payload["text"],
            "source": p.payload["source"],
            "score": p.score,
        }
        for p in results.points
    ]


if __name__ == "__main__":

    query = "What is the refund policy for enterprise customers?"

    results = retrieve(query)

    for r in results:
        print(
            f"[{r['score']:.3f}] "
            f"{r['chunk_id']} "
            f"({r['source']}): "
            f"{r['text'][:100]}"
        )
        