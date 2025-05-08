import streamlit as st
from geo import utils

comet_key = st.secrets["COMET_API_KEY"]
mongodb_uri = st.secrets["MONGODB_URI"]


def check_form_fields(form: dict, address: dict) -> bool:
    for key in form.keys():
        if form[key] == 0:
            return False

    for key in address.keys():
        if address[key] == "":
            return False

    return True

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
    address={}
    col1, col2 = st.columns(2)

    with col1:
        address["street"] = st.text_input("Street", placeholder="1510 San Pablo St")
        address["state"] = st.text_input("State", placeholder="CA")
        address["postalcode"] = st.text_input("Postal Code", placeholder="90033")
        form_state["Median_Income"] = st.number_input("Median Income in the Neighborhood (USD)", min_value=0.0, step=1000.0)
        form_state["Population"] = st.number_input("Population in the Neighborhood", min_value=0, step=1)
        form_state["Households"] = st.number_input("Households in the Neighborhood", min_value=0, step=1)

    with col2:
        address["city"] = st.text_input("City", placeholder="Los Angeles")
        address["country"] = st.text_input("Country", placeholder="USA")
        form_state["Median_Age"] = st.number_input("House Age", min_value=0, step=1)
        form_state["Rooms_Per_House"] = st.number_input("Number of Rooms", min_value=0, step=1)
        form_state["Bedrooms_Ratio"] = st.number_input("Number of Bedrooms", min_value=0, step=1)
        form_state["People_Per_House"] = st.number_input("People per House", min_value=0, step=1)

    submit = st.form_submit_button("Predict Value")

if submit:
    if not check_form_fields(form_state, address):
        st.error("Form not filled")
    else:
        lat, lon = utils.get_lat_lon(**address)

        if lat is None or lon is None:
            st.error("Address not Found")
        else:

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
