from flask import request

def test_version_request(client):
    response = client.get("/version")
    assert "version" in response.get_json()

def test_health_request(client):
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_request(client):
    request_body = {
        "Median_Income": 8.3252,
        "Median_Age": 41,
        "Population": 322,
        "Households": 126,
        "Latitude": 37.88,
        "Longitude": -122.23,
        "Distance_to_coast": 9263.04077285038,
        "Distance_to_LA": 556529.1583418,
        "Distance_to_SanDiego": 735501.80698384,
        "Distance_to_SanJose": 67432.5170008434,
        "Distance_to_SanFrancisco": 21250.2137667799,
        "Rooms_Per_House": 6.984126984126984,
        "Bedrooms_Ratio": 0.14659090909090908,
        "People_Per_House": 2.5555555555555554
    }

    response = client.post('/predict', data=request_body)

    assert "prediction" in response.get_json()
    assert "confidence" in response.get_json()

