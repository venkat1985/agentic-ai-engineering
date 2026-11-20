from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv()

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

documents=[
"Issuer declined the payment authorization request",
"Payment processing timed out while contacting the bank",
"Circuit breaker opened after repeated downstream failures",
"Customer successfully completed a card transaction",
"The weather is sunny in Bentonville"
]

document_vectors=embeddings.embed_documents(documents)

def cosine_similarity(a,b):
    a=np.array(a)
    b=np.array(b)

    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
question="Why did the downstream payment service stop responding?"

question_vector=embeddings.embed_query(question)

results=[]

for i, document_vector in enumerate(document_vectors):
    score=cosine_similarity(question_vector,document_vector)
    results.append((documents[i],score))
results.sort(key=lambda x: x[1],reverse=True)

print("Question:",question)
k=2
print("\nTop 3 matching documents")
for document,score in results[:k]:
    print(f"Score :{score:.4f} | {document}")

