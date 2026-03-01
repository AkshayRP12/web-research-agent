from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from ddgs import DDGS
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            output = ""
            for r in results:
                output += f"Title: {r['title']}\nSummary: {r['body']}\nURL: {r['href']}\n\n"
            return output if output else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"

search_tool = Tool(
    name="WebSearch",
    func=search_web,
    description="Use this to search the web for any topic or question."
)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=[search_tool], prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

query = input("What do you want to research? ")
response = agent_executor.invoke({"input": query})

print("\n===== FINAL ANSWER =====")
print(response["output"])