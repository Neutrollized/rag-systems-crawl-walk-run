# RAG - "Crawl"

Here, we are going to be covering the basic components of RAG. 

[Medium blog post](https://medium.com/@glen.yu/implementing-a-rag-system-crawl-d553ed164611)
[DEV Community blog post](https://dev.to/gde/implementing-a-rag-system-crawl-5li)


## Prerequisites / Setup
You're going to first need a `.env` file with the following:
```
GCP_PROJECT_ID="[YOUR_PROJECT_ID_HERE]"
GCP_LOCATION="us-central1"

EMBEDDING_MODEL="gemini-embedding-001"
EMBEDDING_DIM=384
```

Location-wise, you can use the one closes to you, but not all locations will will have all the models available. I tend to find `us-central1` to be a pretty safe bet. A embedding dimensions, the default I've defined in my `embed_pdfs_to_chromadb.py` script is `768`. Here I'm starting off with `384` because I want something quick/lite for testing (you can set the dimensions up to a max of `3072` with the Gemini embedding model).

The higher the dimensions, the deeper the model tries to look, but it also extends the indexing and retrieval times, and not everything will benefit from using higher dimensions, because at some point you will hit diminishing returns. I see `768` or `1024` as fairly common, balanced setting for majority of use cases.


## Indexing Your PDFs 
The script first downloads the tarball and unpacks it into a local directory. The documents are then loaded using `PyPDFLoader` and split into chunks via the `RecursiveCharacterTextSplitter`. The text chunk are then embedded into vectors and finally indexed into your vector database. 

> [!NOTE]
> In this example, I chose to use ChromaDB as my vector database, but you can use others if you wish.
> In the "Walk" example, I will be using Milvus Lite instead.

> [!NOTE]
> You can also swap out Gemini for a different embedding model such as Cohere's that I use in the "Walk" example,
> but make sure you use the same model for retrieval as you do indexing.
