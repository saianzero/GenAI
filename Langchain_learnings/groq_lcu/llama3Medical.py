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
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from dotenv import load_dotenv

load_dotenv()

## load the GROQ And OpenAI API KEY 
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv('GROQ_API_KEY')

st.title("ChatGroq With Llama3")


llm=ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Provide accurate and well-grounded responses based on the context provided. When addressing medical topics, ensure explanations are clear and straightforward, tailored to the user's level of understanding. Prioritize precision in your responses, adhering closely to the details of the user's query.

<context>
{context}
<context>
Questions:{input}

"""
)



uploaded_file = st.file_uploader("Upload your document", type=['pdf'])
# print("test1 - uploaded file type: ", type(uploaded_file))



temp_file = "./temp.pdf"

if uploaded_file:
    
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name 

# print("test2 - temp file-path type: ", type(temp_file))


def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader = UnstructuredFileLoader(temp_file, strategy="fast")  
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


if st.button("Click to generate the Document Embeddings"):
    vector_embedding()
    st.write("Succesfully generated Document Embeddings. The vector database is ready! ")

prompt1=st.text_input("What would you like to know based on this document?")


if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    response_time = time.process_time()-start
    # print("Response time :",time.process_time()-start)
    print(response_time)
    st.write(response['answer'])
    st.write("This response was generated in ",response_time, "s")

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")





