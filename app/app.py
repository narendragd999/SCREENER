import requests
from urllib.parse import unquote

session = requests.Session()

print("Getting Chartink homepage...")

r = session.get(
    "https://chartink.com",
    headers={
        "User-Agent": "Mozilla/5.0"
    }
)

print("Homepage Status:", r.status_code)

xsrf = session.cookies.get("XSRF-TOKEN")

print("XSRF Found:", xsrf is not None)

if xsrf:
    xsrf = unquote(xsrf)

headers = {
    "content-type": "application/json",
    "x-requested-with": "XMLHttpRequest",
    "x-xsrf-token": xsrf,
    "referer": "https://chartink.com/screener",
    "origin": "https://chartink.com",
    "user-agent": "Mozilla/5.0"
}

payload = {
    "scan_clause": "( {33489} ( [0] 5 minute close > [0] 5 minute ema( close,9 ) and [ -1 ] 5 minute close <= [ -1 ] 5 minute ema( close,9 ) ) )",
    "debug_clause": "",
    "column_clause": ""
}

print("Calling Chartink API...")

r = session.post(
    "https://chartink.com/screener/process",
    headers=headers,
    json=payload
)

print("Status:", r.status_code)
print("Response:")
print(r.text[:5000])