# RAG - "Walk"

Expanding on "Crawl", we're going to level-up our document processing by using [Docling](https://www.docling.ai/), and we're also going to switch it up a bit and use [Cohere's models](https://docs.cohere.com/docs/models) for embedding and reranking. We're also going to be using [LanceDB](https://lancedb.com/) in this example (I just want to show that many components are interchangeable)

> [!NOTE]
> You will need to sign up for a free Cohere account and API key to access the models


- [Medium blog post](https://medium.com/@glen.yu/implementing-a-rag-system-walk-c207aea0cd6b)
- [DEV Community blog post](https://dev.to/gde/implementing-a-rag-system-walk-4h76)


## Prerequisites / Setup
You're going to first need a `.env` file with the following:
```
COHERE_API_KEY='A1b2c3d4e5f6G7h8i9j10K11l12m13n14o15p16q'

EMBEDDING_MODEL="embed-english-light-v3.0"
EMBEDDING_DIM=384
RERANKING_MODEL="rerank-v4.0-fast"
```


## Hybrid Search
Added a Hybrid Search option which does semantic + BM25 search. While semantic search mainly covers "intent", it can miss/fail to interpret nuances such as a part number, specialized terminology, or acronyms (and generally and out-of-domain data). Hybrid search adds in BM25 (Best Matching 25), which look for exact character matches to potentially help cover for anything nuanced that might have been missed through semantic search alone.
