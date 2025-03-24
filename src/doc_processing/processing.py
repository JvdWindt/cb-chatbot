from dotenv import load_dotenv

import bs4

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph

from typing_extensions import List, TypedDict


load_dotenv()

llm = ChatOpenAI("gpt-4o")

embeddings = OpenAIEmbeddings(model="gpt-4o")

vector_store = InMemoryVectorStore(embeddings)


# Load and chunk contents of the blog
# Load PDF (process in seperate processing .py)
# Load JSON
# Load CSV

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

