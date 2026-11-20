import re
from rank_bm25 import BM25Okapi

documents=[
    {
        "id":"doc-1",
        "text": "Payment authorization requests are sent to the issuer"
    },
    {
        "id":"doc-2",
        "text":"Payment processing can experience downstream timeouts"
    },
    {
        "id":"doc-3",
        "text":"circuit breakers open after repeated downstream failures."
    },
    {
        "id":"doc-4",
        "text":"Traffic can be routed to alternative acquirers."
    },
    {
        "id":"doc-5",
        "text":"SocketTimeoutException occurs when a downstream HTTP read times out."
    }
]

def tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())
tokenized_documents=[
    tokenize(doc["text"]) for doc in documents
]
bm25=BM25Okapi(tokenized_documents)

def search_bm25(question,k=3):
    query_tokens=tokenize(question)
    scores=bm25.get_scores(query_tokens)
    ranked=sorted(zip(documents,scores),key=lambda item: item[1],reverse=True)
    return ranked[:k]

if __name__ == "__main__":
    question="What causes SocketTimeoutException?"
    results = search_bm25(question, k=3)
    print("\nQuestion:", question)
    for doc, score in results:
        print(
            f"ID: {doc['id']} | "
            f"Score: {score:.4f} | "
            f"Text: {doc['text']}"
        )


