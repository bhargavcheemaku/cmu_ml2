import pytest
from mmapp import app  # Import the Flask app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_ping(client):
    response = client.get('/ping')
    assert response.status_code == 200  # Validate status code
    assert response.get_json() == {"message": "pong"}  # Validate JSON response
