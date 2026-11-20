from typing import TypedDict,Annotated
from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

class ChatState(TypedDict):
    messages:Annotated[list,add_messages]

llm=ChatOpenAI(model="gpt-5",temperature=0)

def chatbot_node(state: ChatState):
    messages=state["messages"]
    response=llm.invoke(messages)
    return{
        "messages":[response]
    }

graph_builder=StateGraph(ChatState)
graph_builder.add_node("chatbot",chatbot_node)
graph_builder.set_entry_point("chatbot")
graph_builder.add_edge("chatbot",END)

graph=graph_builder.compile()

conversation_history=[]
print("Chatbot started. Type 'exit' to stop\n")

while True:
    user_input=input("You: ")
    if user_input.lower() == "exit":
        break
    conversation_history.append(HumanMessage(content=user_input))

    result=graph.invoke({"messages":conversation_history})

    ai_message=result["messages"][-1]

    print(f"Bot: {ai_message.content}\n")

    conversation_history.append(AIMessage(content=ai_message.content))
