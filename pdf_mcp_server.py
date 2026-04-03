from mcp.server.fastmcp import FastMCP
from openai import AzureOpenAI
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

embedding_client = AzureOpenAI(
    api_key=os.getenv("AZURE_EMBEDDING_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT")
)

# Load existing embeddings
chroma_client = chromadb.PersistentClient(path="C:/Users/sriyjain/Desktop/learn/pdf_RAG/chroma_db")
collection = chroma_client.get_collection(name="pdf_chunks")

def get_embedding(text):
    response = embedding_client.embeddings.create(
        input=text,
        model="my-embedding-model"
    )
    return response.data[0].embedding

mcp = FastMCP("PDF Knowledge Server")

@mcp.tool()
def search_pdf(query: str) -> str:
    """Search the PDF document for information about the Bhagavadgita, Krishna, Arjuna, or related topics."""
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return "\n\n".join(results["documents"][0])

if __name__ == "__main__":
    mcp.run()