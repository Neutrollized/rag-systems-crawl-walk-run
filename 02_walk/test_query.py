import os
import sys
import time
import httpx
import tarfile
from pathlib import Path
from dotenv import load_dotenv
from typing import Tuple

import json
import cohere
import lancedb


#---------------------------
# setup/config
#---------------------------
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-english-light-v3.0")
EMBEDDING_DIM   = os.getenv("EMBEDDING_DIM", 384)
RERANKING_MODEL = os.getenv("RERANKING_MODEL", "rerank-v4.0-fast")


#------------------------
# helper functions
#------------------------
def query_rag(user_query: str, threshold: float = 0.9) -> tuple[list, list]:
    """Perform semantic search of user query against vector database

    Args:
        user_query (str): User query, should be double-quoted

    Returns:
        A tuple of two lists
        - A list of top candidate response outputs
        - A list of corresponding top candidate response output sources
    """
    # convert query to vector
    co = cohere.ClientV2()

    query_vector = co.embed(
        texts=[user_query],
        model=EMBEDDING_MODEL,
        input_type="search_query", # Use search_query for the actual question
        output_dimension=int(EMBEDDING_DIM)
    ).embeddings.float[0]

    #query results
    RESULTS_N = 10
    # make sure you specify distance_type("cosine") as the default is Euclidean distance ("l2")
    # Cohere reranker integration: https://docs.lancedb.com/integrations/reranking/cohere
    results = tbl.search(query_vector).distance_type("cosine").limit(RESULTS_N).to_list()

    candidate_responses = []
    for res in results:
        candidate_responses.append({
            "content": res.get('text'),
            "source": res.get('source'),
            "heading": res.get('heading'),
            "page": res.get('page_no', 'N/A'),
            "search_distance": res.get('_distance')
        })

    # extracting just the text content for reranking
    documents_to_rerank = [doc['content'] for doc in candidate_responses]

    # rerank
    response = co.rerank(
        model=RERANKING_MODEL,
        query=user_query,
        documents=documents_to_rerank,
        top_n=len(candidate_responses),
    )

    reranked_results = []
    for res in response.results:
        # Use the 'index' from Cohere to grab the full original dictionary
        original_data = candidate_responses[res.index]
    
        reranked_results.append({
            "content": original_data["content"],
            "source": original_data["source"],
            "heading": original_data["heading"],
            "page": original_data["page"],
            "search_distance": original_data["search_distance"],
            "relevance_score": res.relevance_score
        })

    # filter results that meet threshold
    # Helper to filter based on current threshold
    def filter_by_score(current_threshold):
        return [item for item in reranked_results if item["relevance_score"] >= current_threshold]

    filtered_results = filter_by_score(threshold)

    # looping 
    while len(filtered_results) < 3:
        threshold -= 0.05
        filtered_results = filter_by_score(threshold)

    # Output formatting
    print(f"\nRERANKING RESULTS: \n-----------------------\n {json.dumps(filtered_results, indent=4)}")
    
    if filtered_results:
        best_match = filtered_results[0]
        print(f"\nBEST ANSWER: \n---------------------\n {best_match['content']}")
        print(f"\nSOURCE CITATION: \n---------------------\n {best_match['source']}, heading {best_match['heading']}, page {best_match['page']}")

    # Prepare the lists to return (satisfying the tuple[list, list] type hint)
    final_texts = [item["content"] for item in filtered_results]
    final_sources = [f"{item['source']}, heading {item['heading']}, page {item['page']}" for item in filtered_results]

    return final_texts, final_sources


#----------------
# main
#----------------
if __name__ == "__main__":
    db = lancedb.connect("./lancedb_data")
    tbl = db.open_table("bc_hr_policies")

    print(f"\n\n-----------------------\n--- BAD QUERY\n-----------------------\n")
    query_rag("Can you give me a chocolate chip cookie recipe?")

    time.sleep(2)

    print(f"\n\n-----------------------\n--- GOOD QUERY\n-----------------------\n")
    #query_rag("What's the harassment policy like?")
    query_rag("What's the policy for job offers?")
