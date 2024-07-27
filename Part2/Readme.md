# technical_test_DS_GenAI - Part2

This section has Generative AI skills technical test , the can load the Bruno_child_offers.pdf , to ask questions. 

### difficulties in the elaboration
Using open-source models can be more challenging and may not offer the same level of performance. Additionally, not having a vector database on the server, such as PostgreSQL with pgvector, complicates putting models into production. For this infrastructure, I propose two endpoints: one for chat and the other for vectorizing documents. 

Using models like openai or anthropic will be much more efficient than using open-source