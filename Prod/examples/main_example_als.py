import requests 


url = "http://127.0.0.1:8001/recommendation"
payload= {
    "user_id":1,
    "n_recs":5
}

try: 
    r=requests.post(url,json=payload)
    r.raise_for_status()
    print('Sucess:',r.json())
except requests.exceptions.HTTPError as err: 
    print("HTTP error:", err)
except Exception as err: 
    print("Other error: ",err)
