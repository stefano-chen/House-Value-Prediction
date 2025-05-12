import numpy as np
import pandas as pd
import requests


def get_lat_lon(street, city, state, country, postalcode, api_key):
    response = requests.get("https://geocode.maps.co/search/", params={
        "street": street,
        "city": city,
        "state": state,
        "country": country,
        "postalcode": postalcode,
        "api_key": api_key
    })

    data = response.json()

    return (float(data[0]["lat"]), float(data[0]["lon"])) if len(data) > 0 else (None, None)


def calculate_haversine_distance(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    r = 6371000

    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg)
    lon2 = np.radians(lon2_deg)

    return 2 * r * np.arcsin(
        np.sqrt(
            np.square(np.sin((lat2 - lat1) / 2)) +
            np.cos(lat1) * np.cos(lat2) * np.square(np.sin((lon2 - lon1) / 2))
        )
    )


def calculate_distance_to_cities(lat_deg, lon_deg):
    cities = {
        "sanfrancisco": [37.773972, -122.431297],
        "la": [34.052235, -118.243683],
        "sandiego": [32.715736, -117.161087],
        "sanjose": [37.335480, -121.893028]
    }

    distances = []

    for city, coord in cities.items():
        distances.append(calculate_haversine_distance(lat_deg, lon_deg, coord[0], coord[1]))

    return distances


def calculate_distance_to_coast(lat_deg, lon_deg, coast_coord: pd.DataFrame):
    min_distance = np.inf
    for index, row in coast_coord.iterrows():
        curr_distance = calculate_haversine_distance(lat_deg, lon_deg, row["latitude"], row["longitude"])
        if curr_distance < min_distance:
            min_distance = curr_distance

    return min_distance
