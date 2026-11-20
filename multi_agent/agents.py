from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from earthquake_tools import get_earthquakes

load_dotenv()

model=init_chat_model("gpt-5",temperature=0)

def magnitude_agent(state):
    question=state["question"]

    earthquakes= state["earthquakes"]

    strongest=sorted(earthquakes,key=lambda x: x["magnitude"] or 0,reverse=True)[:5]

    significant=[
        event for event in earthquakes
        if (event["magnitude"] or 0) >=5
    ]

    earthquake_data={
        "total_earthquakes": len(earthquakes),
        "strongest_earthquakes": strongest,
        "earthquakes_magnitude_5_plus": significant
    }

    prompt=f"""
    You are an earthquake magnitude analysis specialist.
    
    User question:
    {question}
    
    Actual USGS earthquake data:
    {earthquake_data}
    
    Analyze the data. Identify the strongest earthquakes,
    significant events,locations  and magnitude patterns.
    
    Only report facts supported by the supplied data.
    """

    response=model.invoke(prompt)

    return{
        "analyses":[
            f"Magnitude Analysis: {response.content}"
        ]

    }



def geographic_agent(state):
    earthquakes= state["earthquakes"]
    #Group earth quake events by location
    regions= {}

    for event in earthquakes:
        location=event["place"] or "Unknown"

        #Use the text before the first comma as simple
        #Geographic grouping for this learning exercise.

        region=location.split(",")[-1].strip()

        regions[region]=regions.get(region,0)+1
    prompt = f"""
    You are geographic earthquake analysis specialist.
    
    User question:
    {state["question"]}
    
    Total earthquakes: {len(earthquakes)}
    
    Earthquake counts grouped by location:
    {regions}
    
    Identify regions with the highest activity.
    Explain geographic concentration and notable patterns.
    
    Only report findings supported by the supplied data.
    """

    response=model.invoke(prompt)
    return{
        "analyses":[
            f"geographic analysis: {response.content}"
        ]
    }

def final_agent(state):
    prompt=f"""
    You are earthquake analysis coordinator.
    
    User question:
    {state["question"]}
    
    Magnitude analysis:
    {state["analyses"]}
    
    Geographic analysis:
    {state["analyses"]}
    
    
    Prepare a clear final report based on findings.
    """

    response=model.invoke(prompt)
    return {
        "final_report": response.content
    }
def fetch_agent(state):
    earthquakes=get_earthquakes()
    print(f"Fetched {len(earthquakes)} earthquakes")
    return{
        "earthquakes": earthquakes
    }

