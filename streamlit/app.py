from turtledemo.sorting_animate import start_ssort

import streamlit as st
import requests

comet_key = st.secrets["COMET_API_KEY"]
mongodb_uri = st.secrets["MONGODB_URI"]
maps_key = st.secrets["MAPS_API_KEY"]


def get_lat_lon(street, city, state, country, postalcode):
    response = requests.get("https://geocode.maps.co/search/", params={
        "street": street,
        "city": city,
        "state": state,
        "country": country,
        "postalcode": postalcode,
        "api_key": maps_key
    })

    data = response.json()

    return data[0]["lat"], data[0]["lon"] if len(data) > 0 else None,None


# Page Config
st.set_page_config(page_title="House Value Predictor", layout="centered")

# Header
st.markdown(
    """
    <div style='background: linear-gradient(to right, #4e54c8, #8f94fb); padding: 2rem; border-radius: 10px; text-align: center; color: white;'>
        <h1>🏡 House Value Predictor</h1>
        <p>Estimate your house's market value instantly using our machine learning model.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("### Enter Property Details")


# Form for inputs
with st.form(key='prediction_form', enter_to_submit=False):
    form_state = {}
    col1, col2 = st.columns(2)

    with col1:
        form_state["street"] = st.text_input("Street", placeholder="7890 East San Pablo St.")
        form_state["state"] = st.text_input("State", placeholder="CA")
        form_state["postal_code"] = st.text_input("Postal Code", placeholder="90660")
        median_income = st.number_input("Median Income in the Neighborhood (USD)", min_value=0.0, step=1000.0)
        population = st.number_input("Population in the Neighborhood", min_value=0, step=1)
        households = st.number_input("Households in the Neighborhood", min_value=0, step=1)

    with col2:
        city = st.text_input("City", placeholder="Pico Rivera")
        country = st.text_input("Country", placeholder="USA")
        median_age = st.number_input("House Age", min_value=0, step=1)
        n_rooms = st.number_input("Number of Rooms", min_value=0, step=1)
        n_bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
        people_per_house = st.number_input("People per House", min_value=0, step=1)

    submit = st.form_submit_button("Predict Value")

if submit:

    predicted_value = 0

    st.success(f"🏠 Estimated House Value: **${predicted_value:,.2f}**")

# About the model
st.markdown("---")
st.markdown("### 📈 About the Model")
st.markdown(
    """
    This model uses machine learning to estimate the market value of a house.
    It considers various features like:

    - **Area** of the property in square feet
    - Number of **bedrooms** and **bathrooms**
    - **Location** (e.g., city or zip code)

    """
)
