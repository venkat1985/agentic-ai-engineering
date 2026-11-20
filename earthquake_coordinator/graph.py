from langgraph.graph import StateGraph,START,END
from langgraph.types import Send
from earthquake_tools import get_earthquakes
from earthquake_coordinator.state import EarthquakeState

from earthquake_coordinator.agents import (
    coordinator_agent, worker,final_agent)


def fetch_agent(state):

    earthquakes=get_earthquakes()
    print(f"\n Fetched {len(earthquakes)} earthquakes")

    return{
        "earthquakes": earthquakes
    }

def dispatch_agents(state):
    earthquakes=state["earthquakes"]
    selected_agents=state["selected_agents"]

    tasks=[]

    for agent_name in selected_agents:

        if agent_name == "magnitude":
            agent_input=earthquakes
        elif agent_name == "geographic":
            valid_events=[
                e for e in earthquakes
                if e.get("magnitude") is not None
            ]

            strongest_events=sorted(valid_events,key=lambda e: e["magnitude"],reverse=True)[:20]

            agent_input=[
                {
                    "place": e.get("place"),
                    "magnitude": e.get("magnitude"),
                    "latitude": e.get("latitude"),
                    "longitude": e.get("longitude")
                }
                for e in strongest_events
            ]
            print(
                f"Geographic agent input reduced: "
                f"{len(earthquakes)} -> {len(agent_input)} events"
            )
        elif agent_name == "significant":
            agent_input=sorted(
                [
                    e for e in earthquakes
                    if e.get("magnitude") is not None
                       and e["magnitude"] >= 4.5
                ],
                key=lambda e: e["magnitude"],
                reverse=True
            )[:20]
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        tasks.append(Send(
            "worker",
            {
                "agent_type":agent_name,
                "question":state["question"],
                "agent_input":agent_input
            }
        ))
    return tasks

builder= StateGraph(EarthquakeState)

builder.add_node("fetch",fetch_agent)
builder.add_node("coordinator",coordinator_agent)
builder.add_node("worker",worker)
builder.add_node("final",final_agent)

builder.add_edge(START,"fetch")
builder.add_edge("fetch","coordinator")
builder.add_conditional_edges("coordinator",dispatch_agents)
builder.add_edge("worker","final")
builder.add_edge("final",END)

graph=builder.compile()