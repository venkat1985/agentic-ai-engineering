import operator


import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Annotated



load_dotenv()

from langchain.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage, AnyMessage


def get_coordinates(city: str)->dict:
    """Resolve a city name to its coordinates and timezone."""
    geo=requests.get("https://geocoding-api.open-meteo.com/v1/search",params={"name":city,"count":1},timeout=10)
    geo.raise_for_status()
    places=geo.json().get("results",[])
    if not places:
        return {"error":f"City not found: {city}"}
    place=places[0]
    return{
        "name":place["name"],
        "latitude":place["latitude"],
        "longitude":place["longitude"],
        "country":place["country"],
        "timezone":place.get("timezone","auto")
    }
@tool
def get_forcast(city: str,days: int = 2) -> dict:
    """
    Get today's and tomorrow weather forcast for a city, including minimum and maximum temperatures and rain probability
    """
    if not 1<= days<=16:
        return {"error": "days must be between 1 and 16"}
    location=get_coordinates(city)
    if "error" in location:
        return location
    response=requests.get( "https://api.open-meteo.com/v1/forecast",params={
        "latitude":location["latitude"],
        "longitude":location["longitude"],
        "daily":(
            "temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max"
        ),
        "forcast_days":days,
        "timezone":location["timezone"]},
        timeout=10
    )
    response.raise_for_status()
    data = response.json()

    # Important: return the forecast data
    return {
        "city": location["name"],
        "days": days,
        "forecast": data.get("daily", {}),
    }



@tool
def get_weather(city: str) -> dict:
    """Get current weather conditions for a city, including
    temperature, feels-like temperature, humidity, and wind."""

    location = get_coordinates(city)
    if "error" in location:
        return location


    #Step 2: Get current weather

    response=requests.get("https://api.open-meteo.com/v1/forecast",params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "apparent_temperature,precipitation,"
                "wind_speed_10m,weather_code"
            ),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": location["timezone"]
        },
    timeout=10)

    return {
        "city":location["name"],
        "country":location["country"],
        "weather":response.json()["current"]
    }
class MessageState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    llm_calls:int

model=init_chat_model("gpt-5",temperature=0)
tools=[get_weather,get_forcast]
tools_by_name={tool.name:tool for tool in tools}
model_with_tools=model.bind_tools(tools)

def weather_llm_call(state: dict)->dict:
    return{
        "messages":[
            model_with_tools.invoke(
                [SystemMessage(content=(
                    "You are helpful weather assistant. "
                    "Use get_weather for current conditions. "
                    "Use get_forcast for forcasts. "
                    "Never invent weather data."
                ))

            ]+state["messages"])
        ],
        "llm_calls": state.get("llm_calls",0)+1
    }
def weather_tool_node(state: dict):
    """Perform tool call"""
    result=[]
    for tool_call in state["messages"][-1].tool_calls:
        tool=tools_by_name[tool_call["name"]]
        observation=tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation,tool_call_id=tool_call["id"]))
    return {"messages":result}
def weather_should_continue(state:dict):
    """Decide if we should continue the loop or stop based on whether the LLM made a tool call"""
    messages=state["messages"]
    if messages[-1].tool_calls:
        return "tool_node"
    return END

weather_agent_builder=StateGraph(MessageState)
weather_agent_builder.add_node("llm_call", weather_llm_call)
weather_agent_builder.add_node("tool_node", weather_tool_node)
weather_agent_builder.add_edge(START, "llm_call")
weather_agent_builder.add_edge("tool_node", "llm_call")
weather_agent_builder.add_conditional_edges("llm_call", weather_should_continue, ["tool_node", END])

weather_agent=weather_agent_builder.compile()
from IPython.display import Image, display

display(Image(weather_agent.get_graph(xray=True).draw_mermaid_png()))

#Invoke
from langchain.messages import HumanMessage
messages=[HumanMessage(content="What is the weather in Dallas,Texas now and in next 2-3 days")]
messages=weather_agent.invoke({"messages":messages})
for m in messages["messages"]:
    m.pretty_print()








