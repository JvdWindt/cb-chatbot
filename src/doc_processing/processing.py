from dotenv import load_dotenv
import os

import bs4

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph

from typing_extensions import List, TypedDict


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o",
                api_key=api_key)

embeddings = OpenAIEmbeddings(model="gpt-4o",
                              api_key=api_key)

vector_store = InMemoryVectorStore(embeddings)


# Load and chunk contents of the blog
# Load PDF (process in seperate processing .py)
folder_path = r"C:\Users\Johannes\Documents\Coolblue\Assessment Prompt Engineer"

# Get all PDF file names in the folder
pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

# Load all PDFs
docs = []
for file_name in pdf_files:
    print(file_name)
    file_path = os.path.join(folder_path, file_name)  # Create full file path
    loader = PyPDFLoader(file_path)
    docs.extend(loader.load())  # Load and append all pages from the PDF
    #print(docs)

#print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# print(all_splits)

# Index chunks
# _ = vector_store.add_documents(documents=all_splits)

