import json
import re
import sqlite3
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()


connection = sqlite3.connect("chinook.db")
cursor = connection.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

instructions = """
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {dialect} query to run,
            then look at the results of the query and return the answer. Unless the user
            specifies a specific number of examples they wish to obtain, always limit your
            query to at most {top_k} results.
            The Tables in DB are {tables}.
            You can order the results by a relevant column to return the most interesting
            examples in the database. Never query for all the columns from a specific table,
            only ask for the relevant columns given the question.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
            Output the final SQL query only.
            """.format(dialect="sqlite", top_k=5, tables=tables)


def sql_agent(query_data):
    resp = client.responses.create(
        model="gpt-4o",
        instructions=instructions,
        input=query_data)
    output = resp.output_text
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", output, re.DOTALL | re.IGNORECASE)
    sql = match.group(1).strip() if match else output.strip()
    print(sql)
    data = cursor.execute(sql).fetchall()
    print(data)
    return str(data)


tools = [
    {
        "type": "function",
        "name": "sql_agent",
        "description": (
            "Query the Chinook SQLite database for music store data "
            "(artists, albums, tracks, customers, invoices, employees). "
            "Use this whenever the user asks a question that can be answered "
            "from the Chinook DB."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_data": {
                    "type": "string",
                    "description": "Natural-language question to translate into a SQL query.",
                }
            },
            "required": ["query_data"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

FUNCTIONS = {"sql_agent": sql_agent}


def query(text: str) -> str:
    input_messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text}
            ],
        }
    ]

    while True:
        resp = client.responses.create(
            model="gpt-4o",
            tools=tools,
            input=input_messages,
            instructions="You are a helpful assistant which answers questions by the user, using the sql_agent tool when the question is about the Chinook database."
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
    return resp.output_text


if __name__ == "__main__":
    inp = input("Input for LLM: ")
    query(inp)
