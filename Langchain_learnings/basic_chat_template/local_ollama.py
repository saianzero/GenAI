# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user's queries"),
        ("user","Question:{question}")
    ]
)
## streamlit framework

st.title('Langchain Implementation With LLAMA2 API [Uncensored]')
input_text=st.text_input("What would you like to know about?")

# ollama llama2-uncensored LLM 
llm=Ollama(model="llama2-uncensored") # -> ollama run "model" in cmd
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))