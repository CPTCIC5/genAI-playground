import json
import sqlite3
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# read-only connection — model cannot break the DB even if it tries
connection = sqlite3.connect("file:chinook.db?mode=ro", uri=True)
cursor = connection.cursor()


# --- the three buttons the agent can press ---

def list_tables():
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall()]


def get_schema(table_name):
    cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;",
        (table_name,),
    )
    row = cursor.fetchone()
    return row[0] if row else f"No table named {table_name}"


def run_sql(query):
    try:
        rows = cursor.execute(query).fetchall()
        return str(rows[:50])  # cap so the model can't dump huge results into context
    except Exception as e:
        return f"SQL error: {e}"  # feed the error back — the model will self-correct


# --- tool schemas the model sees ---

tools = [
    {
        "type": "function",
        "name": "list_tables",
        "description": "List all tables in the database.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "get_schema",
        "description": "Get the CREATE TABLE statement for one table.",
        "parameters": {
            "type": "object",
            "properties": {"table_name": {"type": "string"}},
            "required": ["table_name"],
        },
    },
    {
        "type": "function",
        "name": "run_sql",
        "description": "Run a read-only SELECT query and return the rows.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

# map tool names to the python functions that implement them
TOOL_FUNCS = {
    "list_tables": lambda args: list_tables(),
    "get_schema": lambda args: get_schema(args["table_name"]),
    "run_sql": lambda args: run_sql(args["query"]),
}

instructions = """
You are a SQL agent for a SQLite database.
First explore with list_tables and get_schema, then write SELECT queries with run_sql.
When you have the answer, reply to the user in plain English — no SQL in the final message.
"""


# --- the agent loop ---

def sql_agent(question, max_steps=6):
    input_items = [{"role": "user", "content": question}]

    for step in range(max_steps):
        resp = client.responses.create(
            model="gpt-4o",
            instructions=instructions,
            tools=tools,
            input=input_items,
        )

        calls = [item for item in resp.output if item.type == "function_call"]

        # no tool calls means the model is done — return its final answer
        if not calls:
            return resp.output_text

        # add the model's turn (its tool calls) to history
        for item in resp.output:
            input_items.append(item.model_dump())

        # execute each tool call and feed the result back
        for call in calls:
            args = json.loads(call.arguments)
            result = TOOL_FUNCS[call.name](args)
            print(f"[step {step}] {call.name}({args}) -> {str(result)[:120]}")
            input_items.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": str(result),
            })

    return "Max steps reached without an answer."


if __name__ == "__main__":
    print(sql_agent("Can you list me all my employees?"))
