import re
from rank_bm25 import BM25Okapi

from rag_basics.bm25_retrieval import tokenized_documents
from rag_basics.rag_pipeline import vector_store

def tokenize(text):
    return re.findall(r"[a-z0-9_]+",text.lower())

documents=[
    {
        "id":"payment-doc-chunk-0",
        "text":"""Payment authorization requests are sent to the issuer.The issuer can approve or decline the transaction.Declines may happen because of insufficient funds."""
    },
    {
        "id": "payment-doc-chunk-1",
        "text":"""
        Payment processing can experience timeouts. A timeout occurs when a downstream service does not respond within the configured response timeout
        """
    },
    {
        "id":"payment-doc-chunk-2",
        "text":"""
        Circuit breakers protect applications from repeated downstream failures.when the failure threshold reached, the circuit breaker opens and rejects new requests until recovery is attempted
        """
    },
    {
        "id": "payment-doc-chunk-3",
        "text": """
        Most of the times, it sends the traffic to the other acquirers to process Payments.
        """
    }
]

tokenized_documents=[
    tokenize(doc["text"]) for doc in documents
]
bm25=BM25Okapi(tokenized_documents)

def search_bm25(question,k=3):
    scores=bm25.get_scores(tokenize(question))
    ranked=sorted(
        zip(documents,scores),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        (doc["id"],float(score)) for doc,score in ranked if score >0
    ][:k]
def search_semantic(question,k=3):
    results=vector_store.similarity_search_with_score(question,k=k)
    return [
        (doc.metadata["chunk_id"],float(score))
        for doc,score in results
    ]
if __name__ == "__main__":
    questions=[
        "What happens when the circuit breaker opens?",
        "Why is the downstream service not responding?",
        "Why does an issuer decline a payment?",
        "What happens when traffic is routed to another acquirer?"
    ]
    for question in questions:
        print("\n"+"="*60)
        print("QUESTION:",question)

        print("\nBM25 RESULTS")
        for chunk_id, score in search_bm25(question):
            print(f"{chunk_id}: {score:.4f}")
        print("\nSEMANTIC RESULTS")
        for chunk_id,score in search_semantic(question):
            print(f"{chunk_id}: {score:.4f}")
