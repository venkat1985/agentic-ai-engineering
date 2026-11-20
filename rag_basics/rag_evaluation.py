from rag_basics.rag_pipeline import ask_rag

test_cases=[
    {
        "question":"what happens when the circuit breaker opens?",
        "expected_keywords":["rejects","requests"]
    },
    {
        "question":"what causes a payment timeout?",
        "expected_keywords": ["downstream","respond"]
    },
    {
        "question": "what is the circuit breaker failure threshold percentage?",
        "expected_keywords":["insufficient context"]
    }
]

def evaluate_rag():
    passed=0

    for i,test in enumerate(test_cases,start=1):
        question=test["question"]
        expected_keywords=test["expected_keywords"]
        print(f"\n========TEST {i} ==========")
        print("QUESTION:",question)
        answer=ask_rag(question)
        print("ACTUAL ANSWER:",answer)

        #simple keyword-based evaluation
        answer_lower=answer.lower()
        success=all(keyword.lower() in answer_lower for keyword in expected_keywords)
        if success:
            print("RESULT: PASS")
            passed += 1
        else:
            print("RESULT: FAIL")
        print("EXPECTED KEYWORDS:",expected_keywords)
        print("\n========== SUMMARY ==========")
        print(f"Passed: {passed}/{len(test_cases)}")
        print(f"Failed: {len(test_cases)-passed}")
if __name__ == "__main__":
    evaluate_rag()
