from typing import Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from IPython.display import Image, display

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from rag import retrieve, generate

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_node("retrieve", retrieve)
# graph_builder.add_node("generate", generate)
graph_builder.set_entry_point("chatbot")
# graph_builder.set_finish_point("generate")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()


try:
    graph_img = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(graph_img)
    print("Graph image saved as 'graph.png'.")
except Exception as e:
    print(f"Could not save graph image: {e}")


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break