import ollama
from pricing_agent import what_if_price_change, whatif_tool_definition

# Simulate a user asking about a specific ride's price
user_question = "I have a 2.5 mile ride at 6pm on a Friday during rush hour, no rain, no surge, in a Lyft XL. Is $25 a fair price?"

messages = [{'role': 'user', 'content': user_question}]

response = ollama.chat(
    model='qwen2.5:7b',
    messages=messages,
    tools=[whatif_tool_definition]
)

print("=== Step 1: Model's initial response ===")
print(response['message'])

# If the model decided to call our tool, actually run it and feed the result back
if response['message'].get('tool_calls'):
    for call in response['message']['tool_calls']:
        if call['function']['name'] == 'what_if_price_change':
            args = call['function']['arguments']
            print(f"\n=== Step 2: Calling real API with args: {args} ===")
            
            result = what_if_price_change(**args)
            print(f"API returned: {result}")
            
            # Add the tool call and its result back into the conversation
            messages.append(response['message'])
            messages.append({
                'role': 'tool',
                'content': str(result)
            })
            
            # Ask the model to now explain the result in plain English
            final_response = ollama.chat(model='qwen2.5:7b', messages=messages)
            print("\n=== Step 3: Model's final explanation ===")
            print(final_response['message']['content'])
else:
    print("\nModel answered directly without calling the tool:")
    print(response['message']['content'])