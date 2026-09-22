import os
from dotenv import load_dotenv
import voyageai

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# Load API key
load_dotenv()

# Create Voyage AI client
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# Connect to local Qdrant
client = QdrantClient(url="http://localhost:6333")


# Documents
sentences = [
    "The refund policy allows returns within 30 days.",
    "Enterprise customers get a 60-day refund window.",
    "Our office is located in downtown Seattle."
]


# Create embeddings
result = vo.embed(
    sentences,
    model="voyage-3-lite",
    input_type="document"
)

embeddings = result.embeddings

# Get vector dimension
dim = len(embeddings[0])


# Create Qdrant collection
client.recreate_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(
        size=dim,
        distance=Distance.COSINE
    ),
)


# Create points
points = [
    PointStruct(
        id=i,
        vector=embeddings[i],
        payload={"text": sentences[i]}
    )
    for i in range(len(sentences))
]


# Insert vectors into Qdrant
client.upsert(
    collection_name="test_collection",
    points=points
)


print("Inserted", len(points), "points")

# User query
query = "What is the refund window for enterprise clients?"

# Convert query into an embedding
query_embedding = vo.embed(
    [query],
    model="voyage-3-lite",
    input_type="query"
).embeddings[0]

# Search Qdrant
search_result = client.query_points(
    collection_name="test_collection",
    query=query_embedding,
    limit=2,
)

# Display results
for point in search_result.points:
    print(f"Score: {point.score:.4f} | Text: {point.payload['text']}")