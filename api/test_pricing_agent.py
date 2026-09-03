import ollama
from api.pricing_agent import what_if_price_change, whatif_tool_definition

def ask_agent(user_question):
    print(f"\n{'='*60}\nUSER: {user_question}\n{'='*60}")
    messages = [
        {'role': 'system', 'content': 'You are a helpful pricing assistant. Always respond in English only.'},
        {'role': 'user', 'content': user_question}
    ]
    response = ollama.chat(model='qwen2.5:7b', messages=messages, tools=[whatif_tool_definition])

    if response['message'].get('tool_calls'):
        for call in response['message']['tool_calls']:
            if call['function']['name'] == 'what_if_price_change':
                args = call['function']['arguments']
                print(f"Tool called with: {args}")
                result = what_if_price_change(**args)
                print(f"API result: {result}")
                messages.append(response['message'])
                messages.append({'role': 'tool', 'content': str(result)})
                final_response = ollama.chat(model='qwen2.5:7b', messages=messages)
                print(f"\nAGENT: {final_response['message']['content']}")
    else:
        print(f"AGENT (no tool call): {response['message']['content']}")

ask_agent("Is $80 fair for a 3 mile Uber XL ride at 2am on a Tuesday, no rain, no surge?")
ask_agent("A 5 mile Lyft Shared ride on Sunday afternoon, 2x surge, is $8 reasonable?")
ask_agent("hey is 15 bucks ok for a short 1.5 mile lyft ride during rush hour on a rainy Wednesday?")