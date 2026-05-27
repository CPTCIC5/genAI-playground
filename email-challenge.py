import json
from openai import OpenAI
from dotenv import load_dotenv

from gmail import (
    get_service,
    list_messages,
    get_message,
    get_message_content,
    send_email,
)
from pydantic import BaseModel

class Output(BaseModel):
    output_text: str
load_dotenv()

client = OpenAI()
service = get_service()

tools = [
    {
        "type": "function",
        "name": "list_emails",
        "description": "List recent emails from the user's Gmail inbox. Returns id, subject, from, and snippet for each.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of emails to return.",
                },
                "query": {
                    "type": "string",
                    "description": "Gmail search query, e.g. 'from:foo@bar.com' or 'subject:invoice'. Empty for none.",
                },
            },
            "required": ["max_results", "query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "read_email",
        "description": "Read the full content (subject, from, date, body) of a specific email by its id.",
        "parameters": {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The id of the email to read.",
                },
            },
            "required": ["message_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "send_email",
        "description": "Send an email from the user's Gmail account.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Subject line."},
                "body": {"type": "string", "description": "Plain-text body."},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def tool_list_emails(max_results: int, query: str):
    msgs = list_messages(service, max_results=max_results, query=query)
    summaries = []
    for m in msgs:
        full = get_message(service, m["id"])
        if full:
            c = get_message_content(full)
            summaries.append({
                "id": c["id"],
                "subject": c["subject"],
                "from": c["from"],
                "snippet": c["snippet"],
            })
    return summaries


def tool_read_email(message_id: str):
    msg = get_message(service, message_id)
    if not msg:
        return {"error": f"Message {message_id} not found"}
    return get_message_content(msg)


def tool_send_email(to: str, subject: str, body: str):
    result = send_email(service, to, subject, body)
    if result is None:
        return {"error": "Failed to send email"}
    return {"id": result.get("id"), "status": "sent"}


FUNCTIONS = {
    "list_emails": tool_list_emails,
    "read_email": tool_read_email,
    "send_email": tool_send_email,
}



inp = input("Ask the email agent: ")
input_messages = [
    {
        "role": "user",
        "content": [{"type": "input_text", "text": inp}],
    }
]

while True:
    resp = client.responses.parse(
        instructions="You are an assistant that helps the user interact with their Gmail inbox -- listing, reading, and sending emails. Use the provided tools when needed.",
        model="gpt-5",
        tools=tools,
        input=input_messages,
        text_format=Output
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
            "output": json.dumps(result, default=str),
        })

print(resp.output_text)
