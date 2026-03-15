import os
import streamlit as st
from openai import AzureOpenAI
import chromadb
import json
from dotenv import load_dotenv
from ddgs import DDGS
from datetime import datetime

today = datetime.now().strftime("%B %d, %Y")
load_dotenv()

# Your clients
embedding_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

chat_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Load existing embeddings
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="pdf_chunks")

def get_embedding(text):
    response = embedding_client.embeddings.create(
        input=text,
        model="my-embedding-model"
    )
    return response.data[0].embedding

def ask(question):
    query_embedding = get_embedding(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    context = "\n\n".join(results["documents"][0])
    
    response = chat_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based on the provided context. If not found, say there is no data in this context in the pdf."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

# Tool 1: Search PDF
def search_pdf(query):
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return "\n\n".join(results["documents"][0])

# Tool 2: Calculator
def calculate(expression):
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Error: could not calculate"


def web_search(query):
    results = DDGS().text(query, max_results=3)
    if results:
        return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    return "No results found"
# Define tools for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_pdf",
            "description": "Search the PDF document for information about the Bhagavadgita, Krishna, Arjuna, or related topics",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '2 + 2' or '100 * 0.15'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or topics not in the PDF",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Map function names to actual functions
tool_functions = {
    "search_pdf": search_pdf,
    "calculate": calculate,
    "web_search": web_search
}
# Streamlit UI
st.title("Chat with your PDF")

question = st.text_input("Ask a question:")



def run_agent(user_question):
    st.write(f"User: {user_question}")
    print("\n=====\n")
    # print("---")
    
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Today's date is {today}. Use tools when needed. For questions about the Bhagavadgita, Krishna, Arjuna, or the PDF content, use search_pdf. For math, use calculate. For current events, news, or general knowledge not in the PDF, use web_search. When using web_search, keep queries simple and do not add specific dates."},
        {"role": "user", "content": user_question}
    ]
    
    # First call: LLM decides what to do
    response = chat_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )
    
    message = response.choices[0].message
    
    # Check if LLM wants to use a tool
    if message.tool_calls:
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            st.write(f"Agent decides: Use {tool_name}")
            st.write(f"With arguments: {tool_args}")
            
            # Call the actual function
            result = tool_functions[tool_name](**tool_args)
            st.write(f"Tool result: {result[:200]}...")  # Print first 200 chars
            st.write("---")
            
            # Add tool call and result to messages
            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
        
        # Second call: LLM generates final answer using tool result
        final_response = chat_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        st.write(f"Agent: {final_response.choices[0].message.content}")
    
    else:
        # No tool needed, LLM answered directly
        st.write(f"Agent (direct): {message.content}")


if question:
    with st.spinner("Thinking..."):
         run_agent(question)
    #st.write(answer)
# Test it
# run_agent("What is the relationship between Krishna and Arjuna?")
# print("\n=====\n")
# run_agent("What is 245 * 38?")
# print("\n=====\n")
# run_agent("What is the capital of France?")
