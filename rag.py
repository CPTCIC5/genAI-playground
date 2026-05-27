import json
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
load_dotenv()
client= OpenAI()


def split_json(file_path, chunk_size=1000, overlap=100):
    with open(file_path, 'r') as f:
        data = json.load(f)

    texts = []
    for article in data['data']:
        for para in article['paragraphs']:
            texts.append(para['context'])

    full_text = "\n\n".join(texts)

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(full_text), step):
        chunks.append(full_text[i:i + chunk_size])

    embeddings = []
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        resp = client.embeddings.create(
            model='text-embedding-3-large',
            input=chunks[i:i + batch_size],
        )
        embeddings.extend(d.embedding for d in resp.data)

    return chunks, embeddings


chroma = chromadb.PersistentClient(path='./chroma_db')
collection = chroma.get_or_create_collection('stanford')

if collection.count() == 0:
    print("Empty collection — embedding and storing...")
    chunks, embeddings = split_json('stanford-data.json', chunk_size=500)
    batch = 5000
    for i in range(0, len(chunks), batch):
        collection.add(
            ids=[str(j) for j in range(i, i + len(chunks[i:i + batch]))],
            embeddings=embeddings[i:i + batch],
            documents=chunks[i:i + batch],
        )
    print(f"Stored {collection.count()} vectors in ./chroma_db")
else:
    print(f"Collection already has {collection.count()} vectors — skipping embedding.")


def query(text):
    q_emb = client.embeddings.create(
        model='text-embedding-3-large',
        input=text,
    ).data[0].embedding
    res = collection.query(query_embeddings=[q_emb], n_results=3)
    return res['documents'][0]
