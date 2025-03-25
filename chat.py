import os

from dotenv import load_dotenv
import openai

load_dotenv()

def chat_with_openai(prompt, api_key):
    """
    Takes a prompt and an api key, sends that prompt to the 
    openai model and returns the response.
    """
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Get API key from env file
    api_key = os.getenv("OPENAI_API_KEY")

    # Start conversation
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        response = chat_with_openai(user_input, api_key)
        print(f"Chatbot: {response}")