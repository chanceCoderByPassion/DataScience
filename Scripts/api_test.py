import requests
import os

url = "https://api.aitklabs.in/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    "Content-Type": "application/json"
}

payload = {
    "model": "gemini-3.8-flash",
    "messages": [
        {
            "role": "user",
            "content": "Multiply 15 by 20 using the available tool."
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ],
    "tool_choice": "required"
}

r = requests.post(
    url,
    headers=headers,
    json=payload
)

print(r.status_code)
print(r.text)