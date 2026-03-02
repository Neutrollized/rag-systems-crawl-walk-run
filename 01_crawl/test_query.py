import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from google.cloud import discoveryengine_v1 as discoveryengine
from dotenv import load_dotenv


load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM   = os.getenv("EMBEDDING_DIM", 768)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    output_dimensionality=EMBEDDING_DIM,
    project=PROJECT_ID,
    location=LOCATION,
    vertexai=True,
    task_type="retrieval_query"  # IMPORTANT: Use 'retrieval_query' for searching
)


vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

query = "How many vacation days am I entitled to?"
initial_results = vectorstore.similarity_search(query, k=5) # Get top 5 relevant chunks

records = []
print(f"\n--- Results for: \"{query}\" ---\n")
for i, doc in enumerate(initial_results):
    source = doc.metadata.get("source", "Unknown")
    print(f"Chunk {i+1} | Source: {source}:")
    print(f"{doc.page_content[:200]}...\n") # Print first 200 chars
