# Project Documentation (DOCter)


## Project Overview
DOCter is a Streamlit-based application that offers personalized healthcare insights by processing uploaded medical reports. The application utilizes advanced language models and document processing to provide accurate and relevant answers to user queries.

![alt text](https://i.imgur.com/gRcTTMP.png)

## Core Components
- **Language Models**: Utilizes `Groq LCU`, `Llama3` and `OpenAIEmbeddings` for natural language understanding and generating vector embeddings for documents.
- **Document Processing**: Supports uploading PDF documents, which are processed to extract text and convert them into searchable vector embeddings using `FAISS Vector database`.
- **User Interaction**: Features an interface for users to upload documents, submit questions, and receive answers. It maintains a session-based chat history for tracking interactions.

## Main Functionalities 
`RAG Pipeline implementation`

1. **Environment Initialization**:
   - Loads environment variables and configures API keys for Groq and OpenAI services.

2. **Streamlit Interface**:
   - Configures the homepage with a title and user instructions.
   - Includes a file uploader for PDFs and a text input for user queries.

3. **Document Upload and Processing**:
   - Temporarily stores and processes uploaded documents to split into manageable chunks.
   - Converts chunks into vector embeddings using OpenAI's model, with `FAISS` facilitating efficient similarity searches.

      ![alt text](https://i.imgur.com/1lk9K4a.png)

4. **Query Processing and Response Generation**:
   - Uses a template-driven approach for query generation to the language model `Llama3`.
   - Handles user queries with a retrieval chain that identifies relevant document sections based on vector embeddings.
   - Displays answers and the processing time taken.
      ![alt text](https://i.imgur.com/Nd67NHn.png)

5. **Chat History**:
   - Keeps a session-based log of queries and responses, accessible via an expandable section in the UI.
      ![alt text](https://i.imgur.com/PqBPDJg.png)

## Future Enhancements
- **Exception handling**
- **Input Sanitization and Guardrails**
- **Backend: FastAPI**
- **Frontend: React Next.js**
- **Integrate Chat Memory with 'save chat' option**
- **Integration of PubMed data source**
- **Cloud Integration**
