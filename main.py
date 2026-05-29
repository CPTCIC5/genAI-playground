import json
from openai import OpenAI
from dotenv import load_dotenv
from rag import query as rag_query
from rag2 import query as bing_data

load_dotenv()

client = OpenAI()

tools = [
    {"type": "web_search"},
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia",
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": (
            "Search the Stanford SQuAD knowledge base for passages relevant to a "
            "question. Use this whenever the user asks a factual question that may "
            "be answered by Wikipedia-style articles. Returns the top 3 matching text chunks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_bing_query",
        "description": (
            "Search the  Bing Search History knowledge base for passages relevant to a "
            "question. Use this whenever the user asks a Bing related question"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]



def get_weather(location: str):
    print('fn-call-01', location)
    return f"{location} BLAH BLAH BLAH!!!!"

def search_knowledge_base(query: str):
    print('fn-call-rag', query)
    hits = rag_query(query)
    return "\n\n---\n\n".join(hits)

def search_bing_query(query: str):
    print('fn-call-rag02', query)
    data= bing_data(query)
    return f"\n\n--\n\n{data}"


FUNCTIONS = {"get_weather": get_weather, "search_knowledge_base": search_knowledge_base, "search_bing_query": search_bing_query}

inp= input("Input for LLM")
input_messages = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": inp}
        ],
    }
]

while True:
    resp = client.responses.create(
        model="gpt-4o",
        tools=tools,
        input=input_messages,
        instructions="You are an helpful assistant which helps the end user with providing whatever is asked by your common sense"
    )

    function_calls = [item for item in resp.output if item.type == "function_call"]

    if not function_calls:
        break

    input_messages += resp.output

    for call in function_calls:
        fn = FUNCTIONS.get(call.name)
        args = json.loads(call.arguments)
        result = fn(**args) if fn else f"Unknown function: {call.name}"

        input_messages.append({
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": str(result),
        })

print(resp.output_text)