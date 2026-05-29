from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import pymupdf4llm

load_dotenv()
client = OpenAI()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="bing_questions")


def split_data(file_path, chunk_size=1000):
    text = pymupdf4llm.to_markdown(file_path)
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def embed(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in response.data]


def index_data(file_path, chunk_size=1000):
    chunks = split_data(file_path, chunk_size)
    embeddings = embed(chunks)
    ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)
    print(f"Indexed {len(chunks)} chunks.")


def query(text):
    input_embedding = client.embeddings.create(
        input=text, model="text-embedding-3-small"
    ).data[0].embedding
    res = collection.query(query_embeddings=input_embedding, n_results=3)
    return res['documents'][0]


if __name__ == '__main__':
    index_data('research-paper.pdf')
