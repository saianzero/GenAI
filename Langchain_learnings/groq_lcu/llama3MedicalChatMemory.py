import streamlit as st
import os
import time
import tempfile
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import UnstructuredFileLoader
from dotenv import load_dotenv

# Initialize the environment and API keys
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

# Setting up the Streamlit title
st.title("ChatGroq With Llama3")

# Initialize the ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""
Provide accurate and well-grounded responses based on the context provided. When addressing medical topics, ensure explanations are clear and straightforward, tailored to the user's level of understanding. Prioritize precision in your responses, adhering closely to the details of the user's query.

Questions:{input}

""")

# Handling file upload
uploaded_file = st.file_uploader("Upload your document", type=['pdf'])
temp_file = "./temp.pdf"

if uploaded_file:
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name 

# Function to generate vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = UnstructuredFileLoader(temp_file, strategy="fast")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

if st.button("Click to generate the Document Embeddings"):
    vector_embedding()
    st.write("Successfully generated Document Embeddings. The vector database is ready! ")

# User query input
prompt1 = st.text_input("What would you like to know based on this document?")

# Initialize chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def compile_history(history):
    context_parts = []
    for query, reply in history:
        context_parts.append(f"Q: {query}\nA: {reply}")
    return "\n".join(context_parts)

def generate_response(prompt_template, query, history):
    compiled_history = compile_history(history)
    full_prompt = prompt_template.from_template(f"""
    {prompt}
    Questions:{query}
    """)
    document_chain = create_stuff_documents_chain(llm, full_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': query})
    response_time = time.process_time() - start
    return {'answer': response['answer'], 'response_time': response_time}

# Process the user's query and update the chat history
if prompt1:
    response_data = generate_response(prompt, prompt1, st.session_state.chat_history)
    st.session_state.chat_history.append((prompt1, response_data['answer']))
    st.write(response_data['answer'])
    st.write("This response was generated in ", response_data['response_time'], "s")


    # Display the chat history
    with st.expander("Chat History"):
        for query, reply in reversed(st.session_state.chat_history):
            st.write(f"Q: {query}")
            st.write(f"A: {reply}")
            st.write("--------------------------------")

    # Display additional context if available
    if 'context' in response_data:
        with st.expander("Document Similarity Search"):
            for doc in response_data['context']:
                st.write(doc.page_content)
                st.write("--------------------------------")
