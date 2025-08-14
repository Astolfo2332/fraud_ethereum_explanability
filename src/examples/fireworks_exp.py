import os

from langchain_fireworks import ChatFireworks
from dotenv import load_dotenv

load_dotenv()

llm = ChatFireworks(model="accounts/fireworks/models/gpt-oss-20b",
                temperature=0.7)

if __name__ == "__main__":
    r = llm.invoke("Hello, how are you?")
    print(r)

