from dotenv import load_dotenv
from typing import Annotated

import bs4

from langchain import hub
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from typing_extensions import List, TypedDict

load_dotenv()

llm = ChatOpenAI("gpt-4o")
embeddings = OpenAIEmbeddings(model="gpt-4o")
vector_store = InMemoryVectorStore(embeddings)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# # Compile application and test
# # Maybe put this in chatbot.py?
# graph_builder = StateGraph(State)
# graph_builder.add_node("retrieve", retrieve)
# graph_builder.add_node("generate", generate)
# graph = graph_builder.compile()