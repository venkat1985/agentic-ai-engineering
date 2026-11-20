from rag_basics.rag_pipeline import vector_store
from rag_basics.semantic_search import documents


def inspect_retrieval(question: str,k: int = 4):
    documents=vector_store.similarity_search_with_score(question,k=k)
    print("\n======QUESTION=========")
    print(question)

    for i, (document,score) in enumerate(documents,start=1):
        print(f"\n=========CHUNK {i}===============")
        print("CONTENT:")
        print(document.page_content)
        print("\nMETADATA:")
        print(document.metadata)

        print("SIMILARITY SCORE:",score)

if __name__ == "__main__":
    question="what happens when the circuit breaker open?"
    inspect_retrieval(question,k=4)

