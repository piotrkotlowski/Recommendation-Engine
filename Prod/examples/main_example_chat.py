import requests

url = "http://127.0.0.1:8002/get_reco"   
payload = {
    "user_id": 6,
    "text": "I want to buy some good dog food for my cute puppy."
}

try:
    r = requests.post(url, json=payload)
    r.raise_for_status()
    print("Success:", r.json())
except requests.exceptions.HTTPError as err:
    print("HTTP error:", err)
except Exception as err:
    print("Other error:", err)
