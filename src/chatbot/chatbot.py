from langchain_openai import ChatOpenAI

from state import State 

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o")

def chatbot(state: State):
    """Generates a chatbot response using context."""
    state["messages"].append({
        "role": "system",
        "content": """Je bent een assistent op de klantenservice van Coolblue. 
        
        Je bent gespecialiseerd in customer support. 
        Je beantwoordt supportvragen en als de klant een probleem heeft probeer je te helpen tot een oplossing te komen door vragen te stellen. 
        Je geeft geen productadvies"
        
        Gebruik een vriendelijke, informele, maar professionele toon. Je doet alles voor een glimlach bij de klant.
        """
    })
    state["messages"].append({
        "role": "user",
        "content": f"### user input: {state['messages'][-1]['content']}\n\n### context:\n{state['context'][0]}"
    })
    
    # Generate response
    return {"messages": [llm.invoke(state["messages"])]}
