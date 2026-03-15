import os
import pdfplumber
from openai import AzureOpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.create_collection(name="pdf_chunks")

with pdfplumber.open("mypdf.pdf") as pdf:
    all_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"
    print(all_text)

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="my-embedding-model"
    )
    return response.data[0].embedding

chunks = chunk_text(all_text)
print("Embedding all chunks...")

for i, chunk in enumerate(chunks):
    embedding = get_embedding(chunk)
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]
    )
    print(f"Embedded chunk {i + 1}/{len(chunks)}")

print("Done.")