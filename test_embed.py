import os
from dotenv import load_dotenv
import voyageai

# Load API key from .env
load_dotenv()
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# Create Voyage AI client
vo = voyageai.Client(api_key=VOYAGE_API_KEY)

# Our documents
sentences = [
    "The refund policy allows returns within 30 days.",
    "Enterprise customers get a 60-day refund window.",
    "Our office is located in downtown Seattle."
]

# Convert text into embeddings
result = vo.embed(
    sentences,
    model="voyage-3-lite",
    input_type="document"
)

embeddings = result.embeddings

print(f"Number of embeddings: {len(embeddings)}")
print(f"Dimension of each embedding: {len(embeddings[0])}")
print(f"First 5 values of embedding 1: {embeddings[0][:5]}")