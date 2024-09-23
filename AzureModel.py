import requests

from dotenv import load_dotenv
import os 

load_dotenv()

def query_azure_openai(query):
    
    api_key =  os.getenv('API_KEY')
    endpoint = os.getenv('ENDPOINT')


    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 150
    }

    response = requests.post(endpoint, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = query_azure_openai(user_input)
    print(f"Bot: {response}")
