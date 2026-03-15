import os
from openai import AzureOpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

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

# Get embeddings for similar and different sentences
sentence1 = "Krishna and Arjuna are friends"
sentence2 = "Arjuna and Krishna have a friendship"
sentence3 = "The Bhagavadgita is a sacred text"

emb1 = get_embedding(sentence1)
emb2 = get_embedding(sentence2)
emb3 = get_embedding(sentence3)

print(f"Embedding dimensions: {len(emb1)}")
print(f"First 5 values of sentence 1: {emb1[:5]}")
print()

# Calculate cosine similarity
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)

sim_1_2 = cosine_similarity(emb1, emb2)
sim_1_3 = cosine_similarity(emb1, emb3)

print(f"Similarity between similar sentences: {sim_1_2:.4f}")
print(f"Similarity between different sentences: {sim_1_3:.4f}")