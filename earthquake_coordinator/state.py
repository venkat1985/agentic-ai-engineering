from typing import TypedDict,Annotated
import operator

class EarthquakeState(TypedDict):
    question: str
    earthquakes: list
    selected_agents: list
    coordinated_reason: str
    analyses:Annotated[list,operator.add]
    final_report: str

class WorkerState(TypedDict):
    #Individual task dispatched using send
    agent_type: str
    question: str
    agent_input: list


