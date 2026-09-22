from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(raw_dir: str) -> list[dict]:
    docs = []

    for filepath in Path(raw_dir).glob("*.txt"):
        text = filepath.read_text(encoding="utf-8")

        docs.append({
            "source": filepath.name,
            "text": text
        })

    return docs


def chunk_documents(
    docs: list[dict],
    chunk_size=300,
    chunk_overlap=50
) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    chunk_id = 0

    for doc in docs:
        splits = splitter.split_text(doc["text"])

        for split in splits:
            chunks.append({
                "chunk_id": f"c{chunk_id}",
                "text": split,
                "source": doc["source"]
            })

            chunk_id += 1

    return chunks


if __name__ == "__main__":

    docs = load_documents("data/raw")

    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)

    print(f"Created {len(chunks)} chunks")

    for c in chunks:
        print(f"\n[{c['chunk_id']}] from {c['source']} | length={len(c['text'])}")
        print(c["text"])