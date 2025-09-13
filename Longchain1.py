from langchain_groq import ChatGroq
import getpass
import os

import os
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment")

print(33)
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)