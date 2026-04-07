# RAG - "Run"

After local testing and optimizing in the "Crawl" and "Walk" phases, it's time to productionize this in Google Cloud. Here we are going to leverage [Vertex AI RAG Engine](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview) to host and serve the documents and [Model Armor](https://cloud.google.com/security/products/model-armor) to provide guardrails and protection against unsafe prompts.

[Medium blog post](https://medium.com/google-cloud/implementing-a-rag-system-run-25d4fd21ce26)
[DEV Community blog post](https://dev.to/gde/implementing-a-rag-system-run-148g)


## Prerequisites / Setup
Cloud Resource Manager API

Vertex AI User
Model Armor User

