import os
import sys
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
COHERE_API_KEY  = os.getenv("COHERE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-english-light-v3.0")
EMBEDDING_DIM   = os.getenv("EMBEDDING_DIM", 384)
RERANKING_MODEL = os.getenv("RERANKING_MODEL", "rerank-v4.0-fast")

current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(current_dir, "..", "lancedb_data")


#------------------------
# helper functions
#------------------------
def query_hr(user_query: str, num_results: int = 10, threshold: float = 0.6):
    """Perform semantic search of user query against vector database

    Args:
        user_query (str): User query, should be double-quoted
        num_results (int): Number of results to return in semantic search
        threshold (float): Reranking relevanc score threshold. Scores below threshold considered not relevant.

    """
    co = cohere.ClientV2()

    query_vector = co.embed(
        texts=[user_query],
        model=EMBEDDING_MODEL,
        input_type="search_query", # Use search_query for the actual question
        output_dimension=int(EMBEDDING_DIM)
    ).embeddings.float[0]

    # semantic search
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table("bc_hr_policies")
    # make sure you specify distance_type as the default is Euclidean distance ("l2")
    # Cohere reranker integration: https://docs.lancedb.com/integrations/reranking/cohere
    results = tbl.search(query_vector).distance_type("dot").limit(num_results).to_list()

    if not results:
        return {"status": "no_results_found", "data": []}

    candidate_responses = []
    for res in results:
        candidate_responses.append({
            "content": res.get('text'),
            "source": res.get('source'),
            "heading": res.get('heading'),
            "page": res.get('page_no', 'N/A')
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
            "relevance_score": res.relevance_score
        })

    def filter_by_score(current_threshold):
        return [item for item in reranked_results if item["relevance_score"] >= current_threshold]

    filtered_results = filter_by_score(threshold)

    if not filtered_results:
        return "No highly relevant documents found for this query" 

    return filtered_results
