from pydantic import BaseModel,Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage

from rag_basics.rag_pipeline import ask_rag_with_context


class FaithfulnessResult(BaseModel):
    faithful:bool= Field(description="True if every factual claim is supported by context")
    reasoning:str=Field(description="Explain any unsupported claims")

#Step 2: Initialize the Judge
judge=(ChatOpenAI(model="gpt-5",temperature=0)
       .with_structured_output(FaithfulnessResult))

def evaluate_faithfulness(question: str,context: str,answer: str):
    response=judge.invoke([
        SystemMessage(content=(
            "You are a strict RAG faithfulness evaluator. "
            "Evaluate whether every factual claim in the answer "
            "is supported by the provided context. "
            "Do not use outside knowledge. "
            "If any factual claim is unsupported, set faithful=False. "
            "An answer that explicitly says 'insufficient context' "
            "without making other factual claims is faithful. "
            "Explain your decision."

        )),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}"
        ))

    ])

    return response

def run_evaluation():

    question="what happens when circuit breaker opens?"


    rag_result=ask_rag_with_context(question)

    context=rag_result["context"]
    answer=rag_result["answer"]

    judge_result=evaluate_faithfulness(question,context,answer)

    print("\n==========QUESTION===========")
    print(question)

    print("\n========RETRIEVED CONTEXT==========")
    print(context)

    print("\n=======RAG ANSWER============")
    print(answer)

    print("\n======JUDGE RESULT============")
    print("Faithful:",judge_result.faithful)
    print("Reasoning:",judge_result.reasoning)

    incorrect_answer = (
        "When the circuit breaker opens, it rejects requests "
        "and automatically refunds every failed payment within 5 seconds."
    )

    negative_result = evaluate_faithfulness(
        question,
        context,
        incorrect_answer
    )

    print("\n========== NEGATIVE TEST ==========")
    print("Faithful:", negative_result.faithful)
    print("Reasoning:", negative_result.reasoning)

if __name__ == "__main__":
    run_evaluation()






