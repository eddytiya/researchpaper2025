import requests
import json

url = "http://127.0.0.1:5000/predict"

# Example JSON data matching your model columns
# Replace 'col1', 'col2', ... with actual feature names your model expects
data = [
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
]

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    # First, print the raw response text
    print("Raw Response:", response.text)
    # Then safely parse JSON if possible
    try:
        print("JSON Response:", response.json())
    except json.decoder.JSONDecodeError:
        print("Response is not valid JSON.")
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
