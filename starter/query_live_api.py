"""
Script that POSTs to the deployed API using the requests module
and returns the result of model inference and the status code.
"""
import requests

# Update this URL to your deployed app URL (e.g., Render)
API_URL = "https://scalable-ml-pipeline.onrender.com"

payload = {
    "age": 52,
    "workclass": "Self-emp-inc",
    "fnlgt": 287927,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 55,
    "native-country": "United-States",
}

response = requests.post(f"{API_URL}/predict", json=payload)

print(f"Status code: {response.status_code}")
print(f"Result: {response.json()}")
