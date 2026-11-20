from dotenv import load_dotenv



load_dotenv()

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph,START,END,MessagesState
from langgraph.prebuilt import ToolNode,tools_condition

from earthquake_tools import (
    get_strongest_earthquakes,
    get_earthquakes_by_magnitude,
    get_earthquakes
)

@tool
def strongest_earthquakes(limit: int =5):
    """Find strongest earthquakes in the past 24 hours."""
    return get_strongest_earthquakes(limit)
@tool
def earthquakes_by_magnitude(min_magnitude: float =4.5):
    """Find earthquakes above specified magnitude"""
    return get_earthquakes_by_magnitude(min_magnitude)
@tool
def earthquake_count():
    """Get the total number of earthquakes in the past 24 hours"""
    return len(get_earthquakes())

tools = [
    strongest_earthquakes,
    earthquakes_by_magnitude,  # Use decorated tool
    earthquake_count
]

model=init_chat_model("gpt-5",temperature=0)
model_with_tools=model.bind_tools(tools)

#Agent Node
def call_model(state: MessagesState):
    response=model_with_tools.invoke(state["messages"])
    return {"messages":[response]}

#Build Graph
builder=StateGraph(MessagesState)


builder.add_node("agent",call_model)
builder.add_node("tools",ToolNode(tools))

# Entry point
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent",tools_condition)


builder.add_edge("tools","agent")

graph=builder.compile()

#
if __name__ == "__main__":
    question=(
        "What are  the strongest earthquakes in the past 24 hours?"
        "Also identify earthquakes above magnitude 5."
    )
    result=graph.invoke(({
        "messages":[HumanMessage(content=question)]
    }))

    print("\nFINAL ANSWER:\n")
    print(result["messages"][-1].content)