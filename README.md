# The "Crawl, Walk, Run" of Implementing a RAG System

In this GitHub repo, I'm going to be using the same dataset and build a Human Resources RAG Agent three different ways: from the very basics of of RAG running on your local laptop using mainly open-source tooling, to fully managed platform on Google Cloud's Vertex AI. Welcome to my "Crawl, Walk, Run with Retrieval-Augmented Generation"!

I will use the same dataset through the three phases. It's [British Columbia government's HR policy PDF documents](https://www2.gov.bc.ca/gov/content/careers-myhr/managers-supervisors/employee-labour-relations/conditions-agreements/policy/hr-policy-pdf). I've downloaded them all, bundled it in a tarball, and put them in a publicly readable GCS bucket access.


## The Goal
This is mean to be educational and if you don't know how Retrieval-Augmented Generation works, this will hopefully get you a acquainted. If you do already know how it works, then maybe I can offer some new perspectives and tools that you can try out to enhance the results of you existing RAG system.

### Crawl
The basics:
- Process PDFs
- Chunk + create embeddings
- Insert into local vector database
- Perform semantic search
- ADK agent to interact with the user 

### Walk
Builds on 'Crawl' phase:
- Improves document processing
- Improves chunking strategy
- Perform reranking after semantic search

### Run
Run a fully-managed RAG system that applies the concepts covered in "Crawl" and "Walk" phases:
- Vertex AI RAG Engine
- Model Armor to provide guardrails
