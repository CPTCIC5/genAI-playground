import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

from gmail import (
    get_service,
    list_messages,
    get_message,
    get_message_content,
)

load_dotenv()

client = OpenAI()
service = get_service()

# ---------------------------------------------------------------------------
# Agent setup (mirrors email-challenge.py, but send_email is mocked so the
# eval doesn't actually send mail).
# ---------------------------------------------------------------------------

tools = [
    {
        "type": "function",
        "name": "list_emails",
        "description": "List recent emails from the user's Gmail inbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer"},
                "query": {"type": "string"},
            },
            "required": ["max_results", "query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "read_email",
        "description": "Read the full content of a specific email by id.",
        "parameters": {
            "type": "object",
            "properties": {"message_id": {"type": "string"}},
            "required": ["message_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "send_email",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def tool_list_emails(max_results: int, query: str):
    msgs = list_messages(service, max_results=max_results, query=query)
    out = []
    for m in msgs:
        full = get_message(service, m["id"])
        if full:
            c = get_message_content(full)
            out.append({
                "id": c["id"],
                "subject": c["subject"],
                "from": c["from"],
                "snippet": c["snippet"],
            })
    return out


def tool_read_email(message_id: str):
    msg = get_message(service, message_id)
    if not msg:
        return {"error": f"Message {message_id} not found"}
    return get_message_content(msg)


def tool_send_email_mock(to: str, subject: str, body: str):
    # No real send during eval -- just record the intent.
    return {"id": "mock-message-id", "status": "sent", "to": to, "subject": subject, "body": body}


FUNCTIONS = {
    "list_emails": tool_list_emails,
    "read_email": tool_read_email,
    "send_email": tool_send_email_mock,
}


def run_agent(prompt: str):
    """Run the email agent on a single prompt. Returns (final_text, tool_calls)."""
    input_messages = [
        {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
    ]
    tool_calls_made = []

    while True:
        resp = client.responses.create(
            instructions="You are an assistant that helps the user interact with their Gmail inbox -- listing, reading, and sending emails. Use the provided tools when needed.",
            model="gpt-4o",
            tools=tools,
            input=input_messages,
        )

        function_calls = [item for item in resp.output if item.type == "function_call"]
        if not function_calls:
            return resp.output_text, tool_calls_made

        input_messages += resp.output

        for call in function_calls:
            args = json.loads(call.arguments)
            tool_calls_made.append({"name": call.name, "args": args})

            fn = FUNCTIONS.get(call.name)
            result = fn(**args) if fn else f"Unknown function: {call.name}"

            input_messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, default=str),
            })


# ---------------------------------------------------------------------------
# LLM-as-a-judge
# ---------------------------------------------------------------------------

class Judgement(BaseModel):
    passed: bool
    score: int  # 1-5
    reasoning: str


def judge(prompt: str, criteria: str, agent_output: str, tool_calls: list):
    """Use an LLM to grade the agent's response against criteria."""
    judge_input = f"""You are evaluating an email-assistant agent.

USER PROMPT:
{prompt}

EVAL CRITERIA:
{criteria}

TOOL CALLS THE AGENT MADE:
{json.dumps(tool_calls, indent=2)}

AGENT FINAL RESPONSE:
{agent_output}

Grade strictly. `passed` = true only if the criteria are clearly met. Score 1 (terrible) to 5 (perfect)."""

    resp = client.responses.parse(
        model="gpt-4o",
        input=[{"role": "user", "content": [{"type": "input_text", "text": judge_input}]}],
        text_format=Judgement,
    )
    return resp.output_parsed


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "list_recent_emails",
        "prompt": "Show me my 3 most recent emails.",
        "criteria": "Agent must call `list_emails` with max_results=3 and present a readable summary (subject/sender) of the returned emails.",
    },
    {
        "name": "search_by_sender",
        "prompt": "Find emails from LinkedIn.",
        "criteria": "Agent must call `list_emails` with a `query` filter containing 'linkedin' (case-insensitive) and report what it found.",
    },
    {
        "name": "summarize_latest",
        "prompt": "Read my latest email and give me a one-sentence summary.",
        "criteria": "Agent must call `list_emails` then `read_email` on a returned id, and produce a concise one-sentence summary of that email's body.",
    },
    {
        "name": "send_email",
        "prompt": "Send an email to friend@example.com with subject 'Hello' and body 'Just checking in.'",
        "criteria": "Agent must call `send_email` exactly once with to='friend@example.com', subject='Hello', and a body matching 'Just checking in.', then confirm to the user.",
    },
]


def main():
    results = []
    for case in TEST_CASES:
        print(f"\n=== {case['name']} ===")
        print(f"Prompt: {case['prompt']}")

        output, calls = run_agent(case["prompt"])
        print(f"Tool calls: {[c['name'] for c in calls]}")
        print(f"Agent output: {output[:200]}{'...' if len(output) > 200 else ''}")

        verdict = judge(case["prompt"], case["criteria"], output, calls)
        status = "PASS" if verdict.passed else "FAIL"
        print(f"Judge: {status} (score {verdict.score}/5) -- {verdict.reasoning}")

        results.append({"name": case["name"], "passed": verdict.passed, "score": verdict.score})

    print("\n" + "=" * 50)
    passed = sum(1 for r in results if r["passed"])
    print(f"SUMMARY: {passed}/{len(results)} passed")
    for r in results:
        mark = "✓" if r["passed"] else "✗"
        print(f"  {mark} {r['name']}: {r['score']}/5")


if __name__ == "__main__":
    main()
