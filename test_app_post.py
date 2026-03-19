import requests

url = "http://127.0.0.1:5000/"
data = {"news": "This is a test news article about something fake."}

try:
    response = requests.post(url, data=data)
    print("Status Code:", response.status_code)
    # print("Response Content:", response.text)
    if "Result" in response.text or "Prediction" in response.text:
        print("Success: Found result in response")
    else:
        print("Failure: Could not find result in response")
except Exception as e:
    print("Error:", e)
