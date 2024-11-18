import requests

# Define the feedback data
data = {
    "sentence": "Test sentence",
    "prediction": "Appropriate",
    "feedback": "Agree"
}

# Send a POST request to the feedback route
response = requests.post("http://127.0.0.1:5000/feedback", json=data)

# Print the response
print("Response:", response.json())
