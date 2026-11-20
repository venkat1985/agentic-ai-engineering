from dotenv import load_dotenv


load_dotenv()

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage

model=init_chat_model("gpt-5",temperature=0)

@tool
def add(a: int, b:int) -> int:
    """
        Add `a` and `b`

        Args:
            a: First Input
            b: Second Input
    """
    return a+b
@tool
def substract(a: int,b: int)->int:
    """
        Substract `a` and `b`

        Args:
            a: First Input
            b: Second Input
    """
    return a-b
@tool
def multiply(a: int, b: int) -> int:
    """
       Multiply `a` and `b`
       Args:
           a: First Input
           b: Second Input
    """
    return a*b
@tool
def devide(a: int, b: int) -> float:
    """
        Devide `a` and `b`
        Args:
            a: First Argument
            b: Second Argument
    """
    return a/b

#Argument with LLM Tools
tools=[add,substract,multiply,devide]
tools_by_name={tool.name:tool for tool in tools}
model_with_tools=model.bind_tools(tools)

#Define State
from langchain.messages import AnyMessage
from typing_extensions import TypedDict,Annotated
import operator

class MessageState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    llm_calls:int

#Define Model node
from langchain.messages import SystemMessage

def llm_call(state: dict):
    """ LLM decides weather to make a call to tool or not """
    return{

        "messages":[
            model_with_tools.invoke([
                SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")
            ]+state["messages"])
        ],
        "llm_calls": state.get("llm_calls",0)+1
    }

def tool_node(state:dict):
    """ Performs the Tool call """
    result=[]
    for tool_call in state["messages"][-1].tool_calls:
        tool=tools_by_name[tool_call["name"]]
        observation=tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation,tool_call_id=tool_call["id"]))
    return {"messages":result}

from typing import Literal
from langgraph.graph import StateGraph,START,END

def should_continue(state: MessageState)->Literal["tool_node",END]:
    messages=state["messages"]
    last_message=messages[-1]

    if(last_message.tool_calls):
        return "tool_node"
    #Otherwise, we stop
    return END

agent_builder=StateGraph(MessageState)
agent_builder.add_node("llm_call",llm_call)
agent_builder.add_node("tool_node",tool_node)

agent_builder.add_edge(START,"llm_call")
agent_builder.add_conditional_edges("llm_call",should_continue,["tool_node",END])
agent_builder.add_edge("tool_node","llm_call")

agent=agent_builder.compile()

from IPython.display import Image,display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

#Invoke
from langchain.messages import HumanMessage
messages=[HumanMessage(content="Multiply 10 and 8")]
messages=agent.invoke({"messages":messages})

for m in messages["messages"]:
    m.pretty_print()




