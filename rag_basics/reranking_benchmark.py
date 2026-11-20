from rag_basics.retrieval_benchmark import (evaluation_dataset,calculate_metrics)
from rag_basics.hybrid_retrieval import hybrid_search
from rag_basics.reranking import rerank

def evaluate_retriever(name,search_function,k=2):
    all_metrics=[]
    print(f"\n========={name}==========")

    for item in evaluation_dataset:
        question=item["question"]
        relevant_ids=item["relevant_ids"]
        results=search_function(question,k=k)
        retrieved_ids=[chunk_id for chunk_id,score in results]
        metrics=calculate_metrics(retrieved_ids,relevant_ids)
        all_metrics.append(metrics)
        print("\nQuestion:",question)
        print("Retrieved:",retrieved_ids)
        print("Precision:",round(metrics["precision"],4))
        print("Recall:",round(metrics["recall"],4))
        print("MRR contribution:",round(metrics["reciprocal_rank"],4))
    print(f"\n============{name} SUMMARY ===========")
    for metric_name in [
        "precision",
        "recall",
        "hit_rate",
        "reciprocal_rank"
    ]:
        average=sum(metrics[metric_name] for metrics in all_metrics)/len(all_metrics)
        print(f"Mean {metric_name}: {average:.4f}")
if __name__ == "__main__":
    evaluate_retriever("HYBRID RRF",lambda question,k: hybrid_search(question,k=k,candidate_k=4),k=2)
    evaluate_retriever("CROSS-ENCODER RANKING",lambda question,k: rerank(question,candidate_k=4,final_k=k),k=2)

