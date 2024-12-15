from locust import HttpUser, task, between

class SimulateTourneyUser(HttpUser):
    # Wait time between tasks (to mimic real user behavior)
    wait_time = between(1, 5)
    
    @task
    def simulate_tourney(self):
        # Payload to be sent in the POST request
        payload = {
            "input": "1"  
        }
        
        # Sending POST request to the /spredict endpoint
        response = self.client.post("/predict", json=payload)
        
        # Print the response or log for debugging
        if response.status_code == 200:
            print("Request successful:", response.json())
        else:
            print("Request failed with status code:", response.status_code)

