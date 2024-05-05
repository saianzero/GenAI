from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

## web application instance
app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)

## Adding routes: url to functionality mappings
add_routes(
    app,
    ChatOpenAI(),
    path="/openai" ## examplewebapp.com/openai -> calls openai functionality
)

chat_openai_model = ChatOpenAI()
ollama_model = Ollama(model="llama2-uncensored")

model1 = chat_openai_model
model2 = ollama_model


prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 year old child with 100 words")

add_routes(
    app,
    prompt1|model1,
    path="/essay"

)

add_routes(
    app,
    prompt2|model2,
    path="/poem"


)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

