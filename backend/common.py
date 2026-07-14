import json
from datetime import datetime
import requests


def emit_to_transcriber(event):
    print(json.dumps(event), flush=True)
    
    try:
        server_url = "http://127.0.0.1:8000/transcribe"
        requests.post(server_url, json=event, timeout=0.5)
    except Exception as e:
        # Silently ignore errors sending to server to avoid breaking main flow
        pass


def emit_to_websocket(event):
    print(json.dumps(event), flush=True)
    
    try:
        server_url = "http://127.0.0.1:3011/event"
        requests.post(server_url, json=event, timeout=0.5)
    except Exception as e:
        # Silently ignore errors sending to server to avoid breaking main flow
        pass