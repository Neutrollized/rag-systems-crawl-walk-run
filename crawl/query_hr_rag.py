import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from google.cloud import discoveryengine_v1 as discoveryengine

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBEDDING_MODEL = "text-embedding-005"

embeddings = GoogleGenerativeAIEmbeddings(
    model=f"models/{EMBEDDING_MODEL}",
    project=PROJECT_ID,
    location=LOCATION,
    vertexai=True,
    task_type="retrieval_query"  # IMPORTANT: Use 'retrieval_query' for searching
)


# 2. Load the existing database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. Perform a Search
#query = "How do I set up a cluster in Azure Databricks?"
query = "How many vacation days am I entitled to?"
initial_results = vectorstore.similarity_search(query, k=5) # Get top 5 relevant chunks

# 4. Display Results
records = []
print(f"\n--- Results for: \"{query}\" ---\n")
for i, doc in enumerate(initial_results):
    source = doc.metadata.get("source", "Unknown")
    print(f"Result {i+1} (Source: {source}):")
    print(f"{doc.page_content[:200]}...\n") # Print first 300 chars
