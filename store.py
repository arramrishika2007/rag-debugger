import os
from dotenv import load_dotenv
import voyageai

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ingest import load_documents, chunk_documents


load_dotenv()

vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "rag_debugger_docs"


def embed_and_store(chunks: list[dict]):
    texts = [c["text"] for c in chunks]

    result = vo.embed(
        texts,
        model="voyage-3-lite",
        input_type="document"
    )

    embeddings = result.embeddings

    dim = len(embeddings[0])

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE
        ),
    )

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "chunk_id": chunks[i]["chunk_id"],
                "text": chunks[i]["text"],
                "source": chunks[i]["source"],
            },
        )
        for i in range(len(chunks))
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Stored {len(points)} chunks in Qdrant")


if __name__ == "__main__":

    docs = load_documents("data/raw")

    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)

    print(f"Created {len(chunks)} chunks")

    embed_and_store(chunks)