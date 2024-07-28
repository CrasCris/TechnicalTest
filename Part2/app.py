from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnableSequence
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

import streamlit as st


# This was for local machine develop, we will send docker run variable for the credentials to Hugging face hub
# from dotenv import load_dotenv
# load_dotenv()

hub_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="question-answering",
    max_new_tokens=100,
    top_p=0.6,
    do_sample=False,
    verbose=False,
    temperature=0.2
)

@st.cache_resource
def load_pdf():
    pdf_name = 'Bruno_child_offers.pdf'
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceBgeEmbeddings(model_name='all-MPNet-base-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20,length_function=len)
    ).from_loaders(loaders)
    return index

index = load_pdf()

chain = RetrievalQA.from_chain_type(
    llm=hub_llm,
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question'
)

st.title("Ask about Bruno child offers")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Enter your questions here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    system_message = "You are an assistant tasked with answering questions based on the provided PDF document. Respond concisely and avoid unnecessary repetition and try response only with the Final Answer."
    prompt_template = PromptTemplate(
        input_variables=["system_message", "question"],
        template="{system_message}\nQuestion: {question}\nAnswer:"
    )
    
    hub_chain = RunnableSequence(prompt_template | hub_llm)
    try:
        # Invoke the chain to get the result
        result = chain.invoke({'question': prompt})
        # Extract and clean the result
        cleaned_result = result['result'].strip().replace('Helpful Answer:', '').replace('�', '').replace('helpful answer:', '')
        st.chat_message('assistant').markdown(cleaned_result)
        st.session_state.messages.append({'role': 'assistant', 'content': cleaned_result})
    except Exception as e:
        st.error(f"Error: {e}")