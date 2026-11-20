from typing import TypedDict, Annotated

import operator


class MultiAgentState(TypedDict):
    question: str
    earthquakes: list
    analyses:Annotated[list,operator.add]
    final_report: str

