import ollama

# A simple test tool the model can call
def get_weather(city: str) -> str:
    """Fake weather lookup for testing tool-calling."""
    return f"It's sunny and 25°C in {city}."

response = ollama.chat(
    model='qwen2.5:7b',
    messages=[{'role': 'user', 'content': 'What is the weather like in Mumbai?'}],
    tools=[{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a city',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string', 'description': 'The city name'}
                },
                'required': ['city']
            }
        }
    }]
)

print("Full response:")
print(response)

# Check if the model actually decided to call our tool
if response['message'].get('tool_calls'):
    print("\n✅ Tool calling works!")
    for call in response['message']['tool_calls']:
        print(f"Model wants to call: {call['function']['name']} with args: {call['function']['arguments']}")
else:
    print("\n❌ Model did not call the tool — it answered directly instead:")
    print(response['message']['content'])