import httpx
try:
    r = httpx.get('http://127.0.0.1:11434/api/tags', timeout=5.0)
    print(f"Status: {r.status_code}")
    print(f"JSON: {r.json()}")
except Exception as e:
    print(f"Error: {e}")
