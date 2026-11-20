from multi_agent.graph import graph

result=graph.invoke({
    "question": "Analyze earthquake activity in the past 24 hours",
    "magnitude_analysis": "",
    "final_report": ""
})

print("\n==========MAGNITUDE ANALYSIS===========")
print(result["magnitude_analysis"])

print("\n=========FINAL REPORT ===========")
print(result["final_report"])