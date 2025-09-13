from dotenv import load_dotenv
import os




from langchain_groq import ChatGroq
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")


def simple_chat_without_memory(user_query):
    response = llm.invoke(user_query)
    return response.content


response1 = simple_chat_without_memory("Hi")
print(response1)