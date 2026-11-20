import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage,HumanMessage
from pydantic.v1.validators import anystr_strip_whitespace

load_dotenv()

INDEX_NAME="payment-rag-learning"
NAMESPACE="rag-lesson-2-4-v2"

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")

vector_store=PineconeVectorStore(index_name=INDEX_NAME,embedding=embeddings,namespace=NAMESPACE)
#initialize the LLM
model=ChatOpenAI(model="gpt-5",temperature=0)

def generate_answer(question:str,context:str)->str:
    response = model.invoke([
        SystemMessage(content=(
            "Answer only using the provided context."
            "Do not invent missing details or casual relationships."
            "If the answer is unavailable,say insufficient context."
        )),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{question}"
        )
    ])
    return response.content


def ask_rag_with_context(question: str) -> dict:
    documents=vector_store.similarity_search(question,k=2)
    context="\n\n".join(doc.page_content for doc in documents)
    answer=generate_answer(question,context)
    return{
        "question":question,
        "context": context,
        "answer":answer,
        "documents":documents
    }

def ask_rag(question:str)->str:
    result=ask_rag_with_context(question)
    return result["answer"]

# --------------------------------------------------
# 6. Main program
# --------------------------------------------------
if __name__ == "__main__":
    # Option A: Store your question in a variable
    question = "What happens when the circuit breaker opens?"
    answer=ask_rag(question)
    print("\nFINAL ANSWER:")
    print(answer)

    #Option B: Read questions from terminal
    while True:
        question=input("\nEnter your question (or 'exit'): ".strip())

        if question.lower() == "exit":
            break
        answer=ask_rag(question)
        print("\nFINAL ANSWER:")
        print(answer)





