from sentence_transformers import CrossEncoder

import torch
from rag_basics.hybrid_retrieval import hybrid_search
from rag_basics.retrieval_comparison import documents

reranker=CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2",
                      device="cpu")

print("\n===== CROSS-ENCODER DIAGNOSTIC =====")

test_pairs = [
    (
        "Why was payment declined?",
        "The issuer declined because of insufficient funds."
    )
]

inputs = reranker.tokenizer(
    test_pairs,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.inference_mode():
    encoder_outputs = reranker.model.base_model(**inputs)
    logits = reranker.model(**inputs).logits

print(
    "Encoder output finite:",
    torch.isfinite(
        encoder_outputs.last_hidden_state
    ).all().item()
)

print("Raw logits:", logits)
print("Logits finite:", torch.isfinite(logits).all().item())

for name, param in reranker.model.named_parameters():
    if not torch.isfinite(param).all():
        print("INVALID MODEL WEIGHTS:", name)
        break
else:
    print("All model weights are finite")

document_lookup={
    doc["id"]: doc["text"] for doc in documents
}

def rerank(question, candidate_k=4, final_k=2):
    candidates=hybrid_search(question,k=candidate_k,candidate_k=candidate_k)

    if not candidates:
        return []
    pairs=[
        (question,document_lookup[chunk_id]) for chunk_id,rrf_score in candidates
    ]
    scores=reranker.predict(pairs)

    print("DEBUG candidates:", candidates)
    print("DEBUG pairs:", pairs)
    print("DEBUG scores:", scores)

    scored_documents=[(chunk_id,float(score)) for (chunk_id,_),score in zip(candidates,scores)]

    scored_documents.sort(key=lambda item: item[1],reverse=True)
    return scored_documents[:final_k]

if __name__ == "__main__":

    print("\n===== RAW MODEL TEST =====")

    test_pairs = [
        (
            "Why was payment declined?",
            "The issuer declined because of insufficient funds."
        )
    ]

    inputs = reranker.tokenizer(
        test_pairs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to("cpu")
        for key, value in inputs.items()
    }

    print("Token IDs:", inputs["input_ids"])
    print("Attention mask:", inputs["attention_mask"])

    with torch.inference_mode():
        outputs = reranker.model(**inputs)

    print("Raw logits:", outputs.logits)
    print("Logits finite:", torch.isfinite(outputs.logits).all().item())

    question="Why does an issuer decline payment?"
    print("\n========RRF CANDIATATES===========")
    candidates=hybrid_search(question,k=4,candidate_k=4)

    for chunk_id,score in candidates:
        print(f"{chunk_id}: {score:.6f}")
    print("\n=======RERANKED RESULTS==========")
    result=rerank(question,candidate_k=4,final_k=2)
    print("DEBUG final results:", result)

    for chunk_id, score in result:
        print(f"{chunk_id}: {score:.4f}")
