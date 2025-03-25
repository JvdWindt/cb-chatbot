import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import List, TypedDict

import bs4

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from state import State


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o",
                api_key=api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              api_key=api_key)

vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

def retrieve(question: str):
    """Retrieves relevant documents from the vector store"""
    retrieved_docs = vector_store.similarity_search(question)
    return retrieved_docs


def retrieve_context(state: State):
    """Retrieves relevant documents and updates the state."""
    user_input = state["messages"][-1].content  # Get the latest user message
    docs = retrieve(user_input)  # Fetch relevant documents
    context = "\n\n".join(doc.page_content for doc in docs)  # Format context

    state["context"] = [context]  # Update state with context
    return state  # Return updated state