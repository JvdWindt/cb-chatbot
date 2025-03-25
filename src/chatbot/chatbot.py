from langchain_openai import ChatOpenAI

from state import State 

llm = ChatOpenAI(model="gpt-4o")

def chatbot(state: State):
    """Generates a chatbot response using context."""
    state["messages"].append({
        "role": "system",
        "content": """Je bent een assistent op de klantenservice van Coolblue. Je doet alles voor een glimlach. 
        
        Je bent gespecialiseerd in vragen over support, je geeft geen product advies. Als de klant een probleem heeft probeer je te helpen tot een oplossing te komen."
        
        Gebruik een vriendelijke, informele, maar professionele toon.
        """
    })
    state["messages"].append({
        "role": "user",
        "content": f"### user input: {state['messages'][-1]['content']}\n\n### context:\n{state['context'][0]}"
    })
    
    # Generate response
    return {"messages": [llm.invoke(state["messages"])]}
