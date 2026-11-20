import sys

from earthquake_coordinator.graph import graph

def main():

    question=(
        "" .join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Analyze earthquake activity in the last 24 hours."
    )

    initial_state={
        "question": question,
        "earthquakes": [],
        "selected_agents": [],
        "coordinated_reason":"",
        "analyses":[],
        "final_report":""
    }

    result=graph.invoke(initial_state)
    print("\n=========FINAL REPORT===========\n")
    print(result["final_report"])

if __name__ == "__main__":
    main()