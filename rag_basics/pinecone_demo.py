import os
from dotenv import load_dotenv

from pinecone import Pinecone,ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter



load_dotenv()

INDEX_NAME="payment-rag-learning"

document="""
Payment authorization requests are sent to the issuer.
The issuer can approve or decline the transaction.
Declines may happen because of insufficient funds.

Payment processing can experience timeouts.
A timeout occurs when a downstream service does not
respond within the configured  response timeout.

Circuit breakers protect applications from repeated 
downstream failures.When the failure threshold is reached,
the circuit breaker opens and rejects new requests until recovery is attempted.

Most of the times, it sends the traffic to the other acquirers to process Payments.

"""
splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=40)
chunks=splitter.split_text(document)
print("Total chunks:",len(chunks))

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

pc=Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if not pc.has_index(INDEX_NAME):
    pc.create_index(name=INDEX_NAME,dimension=1536,metric="cosine",spec=ServerlessSpec(cloud="aws",region="us-east-1"))

vector_store=PineconeVectorStore(index_name=INDEX_NAME,embedding=embeddings,namespace="rag-lesson-2-4-v2")
ids = [f"payment-doc-chunk-{i}" for i in range(len(chunks))]

vector_store.add_texts(texts=chunks,ids=ids, metadatas=[
        {"chunk_id": chunk_id}
        for chunk_id in ids
    ])
print("Documents stored in Pinecone")

question="what happens when the circuit breaker opens?"

results=vector_store.similarity_search_with_score(question,k=2)

for i, (doc,score) in enumerate(results):
    print(f"\n---Result {i+1}---")
    print("Score:",round(float(score),4))
    print("Content:",doc.page_content)




