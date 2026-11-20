from rag_basics.rag_pipeline import vector_store

question="what happens when the circuit breaker opens?"

relevant_texts={
    "Circuit breakers protect applications from repeated",
    "Most of the times, it sends the traffic"
}

def is_relevant(document):
    return any(text in document.page_content  for text in relevant_texts)

def evaluate_retrieval(k: int):
    documents=vector_store.similarity_search(question,k)
    relevant_retrieved=sum(1 for doc in documents if is_relevant(doc))
    total_relevant=len(relevant_texts)

    precision=relevant_retrieved/len(documents) if documents else 0

    recall=relevant_retrieved/total_relevant if total_relevant else 0

    hit_rate=1 if relevant_retrieved > 0 else 0

    print(f"\n==========K={k}============")
    print("Relevant retrieved:",relevant_retrieved)
    print(f"Precision@{k}: {precision:.2f}")
    print(f"Recall@{k}: {recall:.2f}")
    print(f"Hit Rate@{k}: {hit_rate:.2f}")

if __name__ == "__main__":
    for k in [1,2,4]:
        evaluate_retrieval(k)