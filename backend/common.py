import json
from datetime import datetime
import requests


def emit_to_transcriber(event):
    print(json.dumps(event), flush=True)
    
    try:
        server_url = "http://transcriber:8000/transcribe"
        requests.post(server_url, json=event, timeout=0.5)
    except Exception as e:
        # Silently ignore errors sending to server to avoid breaking main flow
        pass


def emit_to_server(event):
    print(json.dumps(event), flush=True)
    
    try:
        server_url = "http://server:3011/event"
        requests.post(server_url, json=event, timeout=0.5)
    except Exception as e:
        # Silently ignore errors sending to server to avoid breaking main flow
        pass