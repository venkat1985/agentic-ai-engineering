from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    question: str
    search_needed: bool
    search_results: str
    final_answer: str

llm=ChatOpenAI(model="gpt-5",temperature=0)

#Router node

def router_node(state: AgentState):
    question=state["question"]

    prompt=f"""
    Decide if this question needs web search.
    
    question:{question}
    Reply only YES or NO.
    """

    response=llm.invoke(prompt).content

    needs_search="YES" in response.upper()

    return {
        "search_needed": needs_search
    }
def search_node(state: AgentState):
    question=state["question"]

    #Fake search result
    result=f"""
    Search results for question: {question}
    Langgraph is a framework for building stateful multi agent workflows.
    """
    return {
        "search_results": result
    }
def direct_answer_node(state: AgentState):
    question=state["question"]
    response=llm.invoke(question)
    return{
        "final_answer": response.content
    }
def summarize_node(state: AgentState):
    question=state["question"]
    search_results=state["search_results"]
    prompt=f"""
    Answer the question using search results.
    Question:
    {question}
    
    Search results:
    {search_results}
    """
    response=llm.invoke(prompt)
    return{
        "final_answer": response.content
    }
def route_decision(state: AgentState):
    if(state["search_needed"]):
        return "search"
    return "direct_answer"

graph_builder=StateGraph(AgentState)
graph_builder.add_node("router",router_node)
graph_builder.add_node("search",search_node)
graph_builder.add_node("direct_answer",direct_answer_node)
graph_builder.add_node("summarize",summarize_node)

graph_builder.add_edge(START,"router")
graph_builder.add_conditional_edges("router",route_decision,{"search":"search","direct_answer":"direct_answer"})
graph_builder.add_edge("search","summarize")
graph_builder.add_edge("summarize",END)
graph_builder.add_edge("direct_answer",END)

graph=graph_builder.compile()
while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    result = graph.invoke({
        "question": question
    })

    print("\nAnswer:")
    print(result["final_answer"])





