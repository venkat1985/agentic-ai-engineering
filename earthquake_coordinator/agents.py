from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from typing import Literal
from earthquake_coordinator.observabilty import invoke_with_metrics

load_dotenv()

model=init_chat_model("gpt-5",temperature=0)

class CoordinatorDecision(BaseModel):
    selected_agents: list[
        Literal[
        "magnitude",
        "geographic",
        "significant"
        ]
    ]
    reason: str

coordinator_model=model.with_structured_output(CoordinatorDecision,include_raw=True)

import time
def coordinator_agent(state):
    earthquakes=state["earthquakes"]
    prompt=f"""
    You are earthquake analysis coordinator.
    
    User question:
    {state["question"]}
    
    Total earthquakes available:
    {len(earthquakes)}
    
    Available agents:
    
    magnitude:
    Analyze strongest earthquakes and magnitude distribution.
    
    geographic:
    Analyze geographic patterns and regional clusters.
    
    significant:
    Analyze significant earthquakes with magnitude >= 4.5.
    
    select only the agents required to answer the user's question.
    
    You can select one, two or all three agents.
    """

    start=time.perf_counter()
    result=coordinator_model.invoke(prompt)
    duration=time.perf_counter()-start
    decision=result["parsed"]
    raw=result["raw"]

    if decision is None:
        raise ValueError(f"Coordinator strucutred output field: {result['parsing_error']}")

    usage=raw.usage_metadata or {}
    print("\n=======Coordinator Metrics==========")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Input tokens: {usage.get('input_tokens','N/A')}")
    print(f"Output tokens: {usage.get('output_tokens', 'N/A')}")
    print(f"Input tokens: {usage.get('input_tokens', 'N/A')}")






    selected_agents=list(dict.fromkeys(decision.selected_agents))

    if not selected_agents:
        selected_agents= ["magnitude"]

    print("\n========COORDINATOR DECISION===========")
    print("Selected:",selected_agents)
    print("Reason:",decision.reason)

    return{
        "selected_agents": selected_agents,
        "coordinated_reason": decision.reason
    }

import json

def magnitude_agent(state):
    question=state["question"]
    earthquakes=state["agent_input"]

    valid_events=[
        e for e in earthquakes
        if e.get("magnitude") is not None
    ]
    strongest=sorted(valid_events,key=lambda e: e["magnitude"],reverse=True)[:10]
    prompt=f"""
    You are an earthquake magnitude analysis specialist.
    
    User question:
    {question}
    
    Strongest earthquakes:
    {json.dumps(strongest,indent=2)}
    
    Analyze;
    1. Strongest earthquakes.
    2. Magnitude patterns.
    3. Notable high-magnitude events.
    
    Use only the provided data.
    """

    response=invoke_with_metrics(model,prompt,"magnitude")

    return{

        "analyses":[
            {
                "agent":"magnitude",
                "result": response.content
            }
        ]

    }
def geographic_agent(state):

    question=state["question"]
    locations=state["agent_input"]

    prompt= f"""
    You are an earthquake geographic analysis specialist.
    
    User question:
    {question}
    
    Earthquake location data:
    {json.dumps(locations,indent=2)}
    
    Analyze:
    1. Regions experiencing earthquake activity.
    2. Geographic concentrations.
    3. Locations of stronger earthquakes.
    4. Notable regional patterns.
    
    Use only the provided data.
    Do not invent geographic statistics.
    """
    response=invoke_with_metrics(model,prompt,"geographic")

    return{
        "analyses":[
            {
                "agent": "geographic",
                "result": response.content
            }
        ]
    }

def significant_agent(state):
    question=state["question"]
    significant_events=state["agent_input"]

    prompt=f"""
    You are a significant earthquake analysis specialist.
    
    User question:
    {question}
    
    Significant earthquake events:
    {json.dumps(significant_events,indent=2)}
    
    Analyze:
    1. Strongest significant earthquakes.
    2. Locations and depths of significant events.
    3. Events that deserve further investigation.
    4. Notable patterns among significant events.
    
    Only use the provided earthquake data.
    Do not invent damage, casualties, or tsunami warnings.
    """
    response=invoke_with_metrics(model,prompt,"significant")

    return{
        "analyses":[
            {
                "agent": "significant",
                "result": response.content
            }
        ]
    }
import time
def worker(state):
    agent_type=state["agent_type"]
    start = time.perf_counter()
    print(f"\n========EXECUTING {agent_type.upper()} AGENT============")
    try:
        if agent_type == "magnitude":
            return magnitude_agent(state)
        elif agent_type == "geographic":
            return geographic_agent(state)
        elif agent_type == "significant":
            return significant_agent(state)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    finally:
        duration = time.perf_counter() - start
        print(f"{agent_type} completed in {duration:.2f} seconds")

def final_agent(state):
    question=state["question"]
    analyses=state["analyses"]
    prompt=f"""
    You are the final earthquake reporting agent.
    
    Original user question:
    {question}
    
    specialist analysis results:
    {json.dumps(analyses,indent=2)}
    
    Combine the specialist findings into a clear report.
    
    Include:
    1. Executive Summary
    2. key findings
    3. Magnitude analysis if available.
    4. Geopraphic analysis if available.
    5. Significant earthquake analysis if available.
    6. Overall conclusion.
    
    Use only specialist findings.
    Do not invent information.
    """

    response=invoke_with_metrics(model,prompt,"final")

    return {
        "final_report": response.content
    }


