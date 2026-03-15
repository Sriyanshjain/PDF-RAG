import os
from openai import AzureOpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

# your client setup here
chat_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="pdf_chunks")

# Get a few chunks
results = collection.get(ids=["0", "5", "10", "15", "20"], include=["documents"])

for i, doc in enumerate(results["documents"]):
    response = chat_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate one specific question that this text answers, and provide the correct answer. Format: Q: ...\nA: ..."},
            {"role": "user", "content": doc}
        ]
    )
    print(f"Chunk {results['ids'][i]}:")
    
    print(response.choices[0].message.content)
    print("---")