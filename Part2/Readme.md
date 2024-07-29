# technical_test - Part2

This section has Generative AI skills technical test , the can load the Bruno_child_offers.pdf , to ask questions. 

### difficulties in the elaboration
Using open-source models can be more challenging and may not offer the same level of performance. Additionally, not having a vector database on the server, such as PostgreSQL with pgvector, complicates putting models into production. For this infrastructure, I propose two endpoints: one for chat and the other for vectorizing documents. 

Using models like openai or anthropic will be much more efficient than using open-source.

### Futere improve

* Add in each response and answers the number of tokens 
* Create some styles for frontend Streamlit app
* Use OpenLLM for using a bigger LLM model and using the information in dedicate vector database

## How to running in production with Docker
You have to generate your HuggingFace token and pass as parameter, the port is the default for streamlit.

1. `cd Part2` 

2. `docker build -t part2:test .`

3. `docker run -e HUGGINGFACEHUB_API_TOKEN='your_token' -p 8501 --rm part2:test`
