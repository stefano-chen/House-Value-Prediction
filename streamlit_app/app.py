import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from classes.Model import Model
from classes.MongoDB import MongoDBLogger
from geo import utils


# Functions declarations

def extract_features(form, latitude, longitude):
    coast_df = pd.read_csv(Path("beach/California_Beach.csv"))
    coast_dist = utils.calculate_distance_to_coast(latitude, longitude, coast_df)

    sanfrancisco, la, sandiego, sanjose = utils.calculate_distance_to_cities(latitude, longitude)

    form["Median_Income"] /= 10000
    form["Bedrooms_Ratio"] /= form["Rooms_Per_House"]

    return {
        "Median_Income": form["Median_Income"],
        "Median_Age": form["Median_Age"],
        "Population": form["Population"],
        "Households": form["Households"],
        "Latitude": latitude,
        "Longitude": longitude,
        "Distance_to_coast": coast_dist,
        "Distance_to_LA": la,
        "Distance_to_SanDiego": sandiego,
        "Distance_to_SanJose": sanjose,
        "Distance_to_SanFrancisco": sanfrancisco,
        "Rooms_Per_House": form["Rooms_Per_House"],
        "Bedrooms_Ratio": form["Bedrooms_Ratio"],
        "People_Per_House": form["People_Per_House"]
    }


def check_form_fields(form: dict, address: dict) -> bool:
    for key in form.keys():
        if form[key] == 0:
            return False

    for key in address.keys():
        if address[key] == "":
            return False

    return True


def create_prediction_info(model, in_features, prediction_value, confidence_value):
    log_info = dict(in_features)

    log_info["Prediction"] = prediction_value
    log_info["Confidence"] = confidence_value
    log_info["Model_Version"] = model.get_version()
    log_info["Date"] = datetime.now(timezone.utc).isoformat()

    return log_info


def log_prediction(info):
    mongo_logger = MongoDBLogger(mongodb_uri)
    mongo_logger.connect()
    try:
        mongo_logger.log("HVP", "predictions", info)
    except RuntimeError as error:
        print(f"Logging Error:{error}", file=sys.stderr)
    finally:
        mongo_logger.close()


def load_model():
    if "model" not in st.session_state:
        model = Model(comet_key)

        with st.spinner("Downloading Model, Please wait", show_time=True):
            model.download_model("HVP")
        st.session_state["model"] = model


# Page Config
st.set_page_config(page_title="House Value Predictor", layout="centered")

comet_key = st.secrets["COMET_API_KEY"]
mongodb_uri = st.secrets["MONGODB_URI"]
maps_key = st.secrets["MAPS_API_KEY"]

load_model()

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
    form_address = {}
    col1, col2 = st.columns(2)

    with col1:
        form_address["street"] = st.text_input("Street", placeholder="1510 San Pablo St")
        form_address["state"] = st.text_input("State", placeholder="CA")
        form_address["postalcode"] = st.text_input("Postal Code", placeholder="90033")
        form_state["Median_Income"] = st.number_input("Median Income in the Neighborhood (USD)", min_value=0.0,
                                                      step=1000.0)
        form_state["Population"] = st.number_input("Population in the Neighborhood", min_value=0, step=1)
        form_state["Households"] = st.number_input("Households in the Neighborhood", min_value=0, step=1)

    with col2:
        form_address["city"] = st.text_input("City", placeholder="Los Angeles")
        form_address["country"] = st.text_input("Country", placeholder="USA")
        form_state["Median_Age"] = st.number_input("House Age", min_value=0, step=1)
        form_state["Rooms_Per_House"] = st.number_input("Number of Rooms", min_value=0, step=1)
        form_state["Bedrooms_Ratio"] = st.number_input("Number of Bedrooms", min_value=0, step=1)
        form_state["People_Per_House"] = st.number_input("People per House", min_value=0, step=1)
    submit = st.form_submit_button("Predict Value")

if submit:
    if not check_form_fields(form_state, form_address):
        st.error("Form not filled")
    else:
        lat, lon = utils.get_lat_lon(**form_address, api_key=maps_key)
        if lat is None or lon is None:
            st.error("Address not Found")
        else:
            with st.spinner("Calculating, Please wait", show_time=True):
                features = extract_features(form_state, lat, lon)
                predicted_value, confidence = st.session_state.model.predict(pd.DataFrame([features]))
                prediction_info = create_prediction_info(st.session_state["model"], features, predicted_value,
                                                         confidence)
                log_prediction(prediction_info)

            st.success(
                "🏠 Estimated House Value: {value:,.2f} USD with confidence: {conf:,.2f}%".format(value=predicted_value,
                                                                                                 conf=confidence))

# About the model
st.markdown("---")
st.markdown("### 📈 About the Model")
st.markdown(
    """
    This model uses machine learning to estimate the market value of a house located in California.
    It considers various features such as:

    - **Neighborhood** well-being
    - Number of **rooms** and **bedrooms**
    - **Location** (e.g., city or zip code)

    """
)
