from http.client import responses

import requests
from datetime import timezone,datetime
from langchain.tools import tool

USGS_URL=("https://earthquake.usgs.gov/earthquakes/feed/v1.0/"
    "summary/all_day.geojson")

def get_earthquakes():
    response=requests.get(USGS_URL,timeout=20)
    response.raise_for_status()

    data=response.json()

    earthquakes=[]

    for feature in data["features"]:
        properties=feature["properties"]
        coordinates=feature["geometry"]["coordinates"]
        earthquakes.append({
            "id": feature["id"],
            "magnitude": properties["mag"],
            "place": properties["place"],
            "time": datetime.fromtimestamp(properties["time"]/1000,tz=timezone.utc).isoformat(),
            "depth_km": coordinates[2],
            "url":properties["url"]
        })
    return earthquakes

def get_strongest_earthquakes(limit=5):
    earth_quakes=get_earthquakes()
    return sorted(earth_quakes,key=lambda x: x["magnitude"] or 0,reverse=True)[:limit]
def get_earthquakes_by_magnitude(min_maginitude=4.5):
    earth_quakes=get_earthquakes()
    return[
        earthquake
        for earthquake in earth_quakes
        if earthquake["magnitude"] is not None and earthquake["magnitude"] >= min_maginitude
    ]

if __name__ == "__main__":
    print("Strongest earthquakes:")
    for earthquake in get_strongest_earthquakes():
        print(earthquake)
    print("\nEarth magnitude 4.5+:")
    for earthquake in get_earthquakes_by_magnitude():
        print(earthquake)
@tool
def strongest_earthquakes(limit: int = 5):
    """Find the strongest earthquakes in the past 24 hours"""
    return get_strongest_earthquakes(limit)



