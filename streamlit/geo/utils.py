from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st


def get_lat_lon(street, city, state, country, postalcode):
    response = requests.get("https://geocode.maps.co/search/", params={
        "street": street,
        "city": city,
        "state": state,
        "country": country,
        "postalcode": postalcode,
        "api_key": st.secrets["MAPS_API_KEY"]
    })

    data = response.json()

    return (data[0]["lat"], data[0]["lon"]) if len(data) > 0 else (None,None)

def calculate_haversine_distance(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    r = 6371000

    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg)
    lon2 = np.radians(lon2_deg)

    return 2*r*np.arcsin(
        np.sqrt(
            np.square(np.sin((lat2-lat1)/2)) +
            np.cos(lat1) * np.cos(lat2) * np.square(np.sin((lon2-lon1)/2))
        )
    )

def calculate_distance_to_cities(lat_deg, lon_deg):
    distance_to_sanfrancisco = calculate_haversine_distance(lat_deg, lon_deg, 37.773972, -122.431297)
    distance_to_la = calculate_haversine_distance(lat_deg, lon_deg, 34.052235, -118.243683)
    distance_to_sandiego = calculate_haversine_distance(lat_deg, lon_deg, 32.715736, -117.161087)
    distance_to_sanjose = calculate_haversine_distance(lat_deg, lon_deg, 37.335480, -121.893028)
    return [distance_to_la, distance_to_sandiego, distance_to_sanjose, distance_to_sanfrancisco]

def calculate_distante_to_coast(lat_deg, lon_deg):
    print(Path.cwd())
    beaches = pd.read_csv(Path("beach/California_Beach.csv"))
    min_distance = np.inf

    for index, row in beaches.iterrows():
        curr_distance = calculate_haversine_distance(lat_deg, lon_deg, row["latitude"], row["longitude"])
        if curr_distance < min_distance:
            min_distance = curr_distance

    return min_distance