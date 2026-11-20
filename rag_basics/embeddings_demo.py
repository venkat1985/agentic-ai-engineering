from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import embeddings
import numpy as np

load_dotenv()

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

sentences=[
    "Payment authorization failed",
    "Card transaction was declined",
    "The weather is sunny today"
]

vectors=embeddings.embed_documents(sentences)

def cosine_similarity(a,b):
    a=np.array(a)
    b=np.array(b)

    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


print("Payment vs Card:",cosine_similarity(vectors[0], vectors[1]))
print("Payment vs Weather:",
      cosine_similarity(vectors[0], vectors[2]))