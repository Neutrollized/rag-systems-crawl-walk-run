import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from google.cloud import discoveryengine_v1 as discoveryengine


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


current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(current_dir, "..", "chroma_db")
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

def query_hr(query: str, relevant_chunks: int = 5):
    """
    Search the internal HR knowledge base.
    The 'query' argument must be a valid, double-quoted string representing the user's search.

    Args:
        query: Double-quoted search term based on user query
        relevant_chunks: Max number of chunks to return that meet semantic search
    """
    try:
        #print(f"DEBUG: Agent is calling query_hr with query: {query} in {DB_PATH}")
        results = vectorstore.similarity_search(query, k=relevant_chunks)
        #print(f"DEBUG: results: {results}")
        
        if not results:
            return {"status": "no_results_found", "data": []}

        formatted_data = []
        for doc in results:
            formatted_data.append({
                "content": doc.page_content,
                "source": os.path.basename(doc.metadata.get("source", "Unknown")),  # only want the filename
                "page": doc.metadata.get("page", "N/A") + 1                         # counter starts at 0
            })

        return {
            "status": "success",
            "count": len(formatted_data),
            "results": formatted_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


#----------------------------------------------------------------
# NOTE: This is unused in the 'crawl' phase,
#       but you can import 'query_hr_v2' instead in agent.py
#       to see how your results may have improved with 
#       reranking incorporated into the result retrieval.
#       I will expand on this concept in the 'walk' phase
#----------------------------------------------------------------
def query_hr_v2(query: str, relevant_chunks: int = 5, threshold: float = 0.5):
    """
    Search the internal HR knowledge base.
    The 'query' argument must be a valid, double-quoted string representing the user's search.

    Args:
        query: Double-quoted search term based on user query
        relevant_chunks: Max number of chunks to return that meet semantic search
        threshold: Threshold for relevance score
    """
    try:
        results_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=relevant_chunks)
        
        filtered_results = [
            doc for doc, score in results_with_scores if score >= threshold
        ]

        if not filtered_results:
            return "No highly relevant documents found for this query"

        formatted_data = []
        for doc in filtered_results:
            formatted_data.append({
                "content": doc.page_content,
                "source": os.path.basename(doc.metadata.get("source", "Unknown")),  # only want the filename
                "page": doc.metadata.get("page", "N/A") + 1                         # counter starts at 0
            })

        return {
            "status": "success",
            "count": len(formatted_data),
            "results": formatted_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

