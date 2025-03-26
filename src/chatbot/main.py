from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from state import State
from rag import retrieve_context
from chatbot import chatbot

# Build the graph nodes
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve_context)
graph_builder.add_node("chatbot", chatbot)

# Define graph flow
graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "chatbot")
graph_builder.set_finish_point("chatbot")

# Compile the graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Save graph structure.
try:
    graph_img = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(graph_img)
except Exception as e:
    print(f"Could not save graph image: {e}")

# Function to run the chatbot
def run_chatbot(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({
        "messages": [{"role": "user", "content": user_input}]
    },
    config=config)
    result["messages"][-1].pretty_print()

# Run chatbot loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    run_chatbot(user_input)
