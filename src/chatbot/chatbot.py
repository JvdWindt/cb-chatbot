from typing import Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from IPython.display import Image, display

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

#from rag import retrieve, generate

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: list

graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    for event in graph.stream({"messages": [{"role": "system",
                                             "content": "Je bent een assistent op de klantenservice van Coolblue. Je bent gespecialiseerd in vragen over support."},
                                            {"role": "user", 
                                             "content": user_input}]},
                                             config=config,
                                             stream_mode="values"):
        event["messages"][-1].pretty_print()


# Build te graph
graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_node("retrieve", retrieve)
# graph_builder.add_node("generate", generate)
graph_builder.set_entry_point("chatbot")
# graph_builder.set_finish_point("generate")
graph_builder.set_finish_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


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
        user_input = "No input avaiable"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break