from rag_basics.retrieval_comparison import (search_bm25,search_semantic)

def reciprocal_rank_fusion(result_lists,constant=60):
    fused_scores={}

    for results in result_lists:
        for rank,(chunk_id,score) in enumerate(results,start=1):
            if chunk_id not in fused_scores:
                fused_scores[chunk_id]=0.0
            fused_scores[chunk_id]+=(1.0/(constant+rank))

    return sorted(fused_scores.items(),key=lambda item: item[1],reverse=True)

def hybrid_search(question,k=3,candidate_k=4):
    bm25_results=search_bm25(question,k=candidate_k)
    semantic_results=search_semantic(question,k=candidate_k)

    fused_results=reciprocal_rank_fusion([
        bm25_results,
        semantic_results
    ])
    return fused_results[:k]

if __name__ == "__main__":
    question="Why does an issuer decline a payment?"
    results=hybrid_search(question,3)
    print("\nQuestion:",question)
    print("\n======= HYBRID RESULTS =========")

    for chunk_id,score in results:
        print(f"{chunk_id}: {score:.6f}")