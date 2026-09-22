# RAG Failure Taxonomy — Draft

## 1. Retrieval Miss
The retriever fails to return the information needed to answer the user's query.

## 2. Chunking Boundary Failure
Important information is split across chunk boundaries, causing incomplete or unusable retrieved context.

## 3. Ranking Failure
Relevant information is retrieved but ranked below less relevant information.

## 4. Reranking Failure
The reranking stage incorrectly changes the ordering of retrieved candidates and reduces the relevance of the final context.

## 5. Context-Ignored Hallucination
The retrieved context contains the information needed to answer the query, but the generator produces an unsupported or contradictory answer.

## 6. Ambiguous Query
The user's query is unclear or underspecified, causing retrieval to target the wrong information.