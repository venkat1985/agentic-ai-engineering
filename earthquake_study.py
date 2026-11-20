import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()

@tool
def get_earthquake_events() ->dict:
    """ Get Earth quake Information"""
    url= (
        "https://earthquake.usgs.gov/"
        "earthquakes/feed/v1.0/summary/all_day.geojson"
    )

    responses=requests.get(url,timeout=30)
    responses.raise_for_status()

    data=responses.json()
    events=data["features"]
    print("Earthquakes Today:",len(events))

    strongest=sorted(events,key=lambda event:event["properties"].get("mag") or 0,reverse=True,)[:5]

    for event in strongest:
        props=event["properties"]
        print(props["mag"],props["place"])

    return events


