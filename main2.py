from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature",
                },
            },
            "required": ["location"],
        },
    }
]

def get_weather(location: str, unit: str = "fahrenheit"):
    print('fn-call-01-anthropic')
    return f"{location} BLAH BLAH BLAH!!!! ({unit})"


FUNCTIONS = {"get_weather": get_weather}

messages = [
    {"role": "user", "content": "What's the weather like in San Francisco?"}
]

while True:
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        tools=tools,
        messages=messages,
    )

    if message.stop_reason != "tool_use":
        break

    tool_uses = [b for b in message.content if b.type == "tool_use"]

    messages.append({"role": "assistant", "content": message.content})

    tool_results = []
    for tool in tool_uses:
        fn = FUNCTIONS.get(tool.name)
        result = fn(**tool.input) if fn else f"Unknown tool: {tool.name}"
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool.id,
            "content": str(result),
        })

    messages.append({"role": "user", "content": tool_results})

for block in message.content:
    if block.type == "text":
        print(block.text)