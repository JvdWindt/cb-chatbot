from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

# Define shared state
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Chat history
    context: list  # Retrieved documents