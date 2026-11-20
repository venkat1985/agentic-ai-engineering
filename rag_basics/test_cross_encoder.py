from sentence_transformers import CrossEncoder
import numpy as np

model = CrossEncoder(
    "cross-encoder/ms-marco-TinyBERT-L2-v2",
    device="cpu"
)

pairs = [
    (
        "Why was payment declined?",
        "The issuer declined because of insufficient funds."
    ),
    (
        "Why was payment declined?",
        "The circuit breaker opened after repeated timeouts."
    )
]

scores = model.predict(pairs)

print("Scores:", scores)
print("All scores finite:", np.isfinite(scores).all())
