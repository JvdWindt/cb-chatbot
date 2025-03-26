import os

from dotenv import load_dotenv
from typing import Annotated
import json
import uuid

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, JSONLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langgraph.graph.message import add_messages


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize models
llm = ChatOpenAI(model="gpt-4o",
                api_key=api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              api_key=api_key)

# Initialize vector store
vector_store = Chroma("langchain_store", embeddings, persist_directory="chroma_db")

def pdf_processor(folder_path: str):
    """Loads, chunks and embeds all PDF files in the folder and loads them to the vector store"""
    # Get all PDF file names in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    # Load all PDFs
    docs = []
    for file_name in pdf_files:
        print(file_name)
        file_path = os.path.join(folder_path, file_name)  # Create full file path
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())  # Load and append all pages from the PDF

    # Split text into chunks
    print("Splitting")
    text_splitter = SemanticChunker(
        OpenAIEmbeddings(), 
        breakpoint_threshold_type="percentile")
    all_splits = text_splitter.split_documents(docs)


    # Index chunks, load them to the vector store
    print("Indexing")
    _ = vector_store.add_documents(documents=all_splits)


def json_processor(folder_path: str):
    """Loads, chunks and embeds all JSON files in the folder and loads them to the vector store"""
    # Get all json file names in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    # Load all json
    json_docs = []
    for file_name in json_files:
        print(file_name)
        file_path = os.path.join(folder_path, file_name)  # Create full file path
        with open(file_path, errors="ignore") as json_file:
            data = json.load(json_file)
            for d in data:
                jsonid = getattr(d, 'id', uuid.uuid4())  # Use 'id' if exists, else generate new
                json_doc = Document(
                    page_content=json.dumps(d),
                    metadata={"source": "json"},
                    id=jsonid
                )
                json_docs.append(json_doc)
            
    # Index JSON, load them to the vector store
    print("Indexing")
    _ = vector_store.add_documents(documents=json_docs)


# Execute the funtions above
folder_path = r"C:\Users\Johannes\Documents\Coolblue\Assessment Prompt Engineer"
pdf_processor(folder_path)
json_processor(folder_path)