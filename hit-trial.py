from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client= OpenAI()


"""
# Function to create a file with the Files API
def create_file(file_path):
  with open(file_path, "rb") as file_content:
    result = client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id

# Getting the file ID
file_id = create_file("lol.jpg")
print(file_id)
"""


resp=client.responses.create(
    instructions="You're a professional chef and you work at home of the user(s) and make food for there family",
    model="gpt-4o",
    tools=[{"type": "web_search"}],
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Here is the photo of my fridge procurement right now, What shall we make at lunch and dinner today?"},
                {"type": "input_image", "file_id": "file-JEzLxSHh69Nr5h1h6yVwrv"}
            ]
        }
    ]

)

print(resp.output_text)
