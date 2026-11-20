from rag_basics.rag_pipeline import vector_store

evaluation_dataset=[
    {
        "question":"What happens when the circuit breaker opens?",
        "relevant_ids":{
            "payment-doc-chunk-2",
            "payment-doc-chunk-3"
        }
    },
    {
        "question": "What causes payment processing timeouts?",
        "relevant_ids":{
            "payment-doc-chunk-1"
        }
    },
    {
        "question":"Why does an issuer decline a payment?",
        "relevant_ids":{
            "payment-doc-chunk-0"
        }
    },
    {
        "question": "What happens when traffic is routed to another acquirer?",
        "relevant_ids": {
            "payment-doc-chunk-3"
        }
    }
]


def reciprocal_rank(retrieved_ids,relevant_ids):
    for rank, chunk_id in enumerate(retrieved_ids,start=1):
        if chunk_id in relevant_ids:
            return 1.0/rank
    return 0.0

def calculate_metrics(retrieved_ids,relevant_ids):
    relevant_retrieved=len(set(retrieved_ids) & relevant_ids)
    precision=(relevant_retrieved/len(retrieved_ids) if retrieved_ids else 0.0)
    recall=(relevant_retrieved/len(relevant_ids) if relevant_ids else 0.0)
    hit_rate=1.0 if relevant_retrieved >0 else 0.0
    rr=reciprocal_rank(retrieved_ids,relevant_ids)
    return{
        "precision":precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "reciprocal_rank": rr
    }

def evaluate_benchmark(k=3):
    all_metrics=[]
    for item in evaluation_dataset:
        question=item["question"]
        relevant_ids=item["relevant_ids"]

        documents=vector_store.similarity_search(question,k=k)
        retrieved_ids=[doc.metadata["chunk_id"] for doc in documents]
        metrics=calculate_metrics(retrieved_ids,relevant_ids)

        all_metrics.append(metrics)
        print("\nQuestion:",question)
        print("Retrieved:",retrieved_ids)
        print("Precision:",round(metrics["precision"],4))
        print("Recall:",round(metrics["recall"],4))
        print("Hit Rate:", round(metrics["hit_rate"], 4))
        print("Reciprocal Rank:",round(metrics["reciprocal_rank"], 4))
    print("\n==========BENCHMARK SUMMARY========")
    print("Total Questions:",len(all_metrics))

    for metric_name in [
        "precision",
        "recall",
        "hit_rate",
        "reciprocal_rank"
    ]:
        average=sum(metrics[metric_name] for metrics in all_metrics)/len(all_metrics)
        print(f"Mean {metric_name}: {average:.4f}")

if __name__=="__main__":
    for k in [1, 2, 3, 4]:
        print(f"\n========== K={k} ==========")
        evaluate_benchmark(k=k)


