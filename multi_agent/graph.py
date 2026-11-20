from langgraph.graph import StateGraph, START, END

from multi_agent.state import MultiAgentState
from multi_agent.agents import magnitude_agent,fetch_agent, final_agent, geographic_agent

builder=StateGraph(MultiAgentState)

builder.add_node("magnitude_agent",magnitude_agent)
builder.add_node("geographic_agent",geographic_agent)
builder.add_node("fetch_agent",fetch_agent)
builder.add_node("final_agent",final_agent)


builder.add_edge(START,"fetch_agent")

builder.add_edge("fetch_agent","magnitude_agent")
builder.add_edge("fetch_agent","geographic_agent")


builder.add_edge(["magnitude_agent","geographic_agent"],"final_agent")
builder.add_edge("final_agent",END)

graph=builder.compile()