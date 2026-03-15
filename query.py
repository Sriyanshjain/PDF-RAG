import os
from openai import AzureOpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="pdf_chunks")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="my-embedding-model"
    )
    return response.data[0].embedding

query = "What is the significance of the relationship between the warrior and the charioteer?"

query_embedding = get_embedding(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
print("Top 3 relevant chunks:" )
context = "\n\n".join(results["documents"][0])
print(context)

chat_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer the question based on the provided context. If the context doesn't contain the answer, say you don't know. Be concise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)

print("LLM answer:")
print(chat_response.choices[0].message.content)
# print(f"Total chunks: {len(chunks)}")
# print("---")
# print("First chunk:")
# print(chunks[0])
# print("---")
# print("Second chunk:")
# print(chunks[1])

# print("Getting embedding for first chunk...")
# embedding = get_embedding(chunks[0])
# print(f"Embedding length: {len(embedding)}")
# print(f"First 5 values: {embedding[:5]}")

