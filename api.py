import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AzureOpenAI
import chromadb
from fastapi import UploadFile, File
import pdfplumber

load_dotenv()

# Clients
embedding_client = AzureOpenAI(
    api_key=os.getenv("AZURE_EMBEDDING_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT")
)

chat_client = AzureOpenAI(
    api_key=os.getenv("AZURE_CHAT_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_CHAT_ENDPOINT")
)

# Vector DB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="pdf_chunks")

def get_embedding(text):
    response = embedding_client.embeddings.create(
        input=text,
        model="my-embedding-model"
    )
    return response.data[0].embedding

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks



app = FastAPI()

# Request model
class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(question: Question):
    query_embedding = get_embedding(question.query)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    context = "\n\n".join(results["documents"][0])
    
    response = chat_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question.query}"}
        ]
    )
    
    return {"answer": response.choices[0].message.content}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Check file type
    if not file.filename.endswith('.pdf'):
        return {"error": "Only PDF files allowed"}
    
    content = await file.read()
    temp_path = f"temp_{file.filename}"
    
    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        
        all_text = ""
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
        
        if not all_text.strip():
            return {"error": "Could not extract text. PDF might be scanned images."}
        
        chunks = chunk_text(all_text)
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            collection.add(
                ids=[f"{file.filename}_{i}"],
                embeddings=[embedding],
                documents=[chunk]
            )
        
        return {"message": f"Uploaded {file.filename}", "chunks": len(chunks)}
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/documents")
def list_documents():
    # Get all items from collection
    results = collection.get()
    
    # Extract unique document names from IDs
    doc_names = set()
    for id in results["ids"]:
        # IDs are like "filename.pdf_0", "filename.pdf_1"
        # Split by last underscore to get filename
        parts = id.rsplit("_", 1)
        if len(parts) > 1:
            doc_names.add(parts[0])
        else:
            doc_names.add(id)
    
    return {"documents": list(doc_names), "total_chunks": len(results["ids"])}