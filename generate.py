import ollama
from retrieve import retrieve


def generate_answer(query: str, top_k=3) -> dict:

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Build context from retrieved chunks
    context = "\n\n".join(
        [f"[{c['chunk_id']}] {c['text']}" for c in chunks]
    )

    # Step 3: Create grounded prompt
    prompt = f"""Answer the question using ONLY the context below.

If the context does not contain enough information to answer the question,
say: "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

    # Step 4: Generate answer using local Ollama model
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return {
        "query": query,
        "retrieved_chunks": chunks,
        "answer": answer
    }


if __name__ == "__main__":

    result = generate_answer(
        "What is the refund policy for enterprise customers?"
    )

    print("Query:", result["query"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieved chunks:")

    for c in result["retrieved_chunks"]:
        print(
            f"  [{c['score']:.3f}] "
            f"{c['chunk_id']}: "
            f"{c['text'][:100]}"
        )