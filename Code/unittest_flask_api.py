import pytest
import json
from flask import Flask
from MarchMadnessDataAnalysis_v2 import app  # Adjust this import if needed

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_simulate_tourney(client):
    # Mock input data for testing
    test_data = {
        "input": "1"  # Replace with valid input for `simulate_tourney`
    }

    # Send a POST request to the API endpoint
    response = client.post(
        "/simulateTourney",
        data=json.dumps(test_data),
        content_type="application/json"
    )

    # Log response details for debugging
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Data: {response.data.decode()}")

    # Validate the response
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    response_json = response.get_json()
    assert response_json is not None, "Expected JSON response"
    print(f"Response JSON: {response_json}")  # To inspect the actual JSON response


