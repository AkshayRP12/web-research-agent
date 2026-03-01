import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from ddgs import DDGS
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Research", page_icon=None, layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #F5F2ED;
    font-family: 'DM Mono', monospace;
}

[data-testid="stAppViewContainer"] {
    background-color: #F5F2ED;
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 680px !important;
    padding: 4rem 2rem 2rem 2rem !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Title area */
.title-block {
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #D4CFC8;
}

.title-main {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #1A1814;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 0.4rem;
}

.title-main em {
    font-style: italic;
    color: #8B7355;
}

.title-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #A09880;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 300;
}

/* Messages */
.msg-user {
    margin: 1.5rem 0;
    display: flex;
    justify-content: flex-end;
}

.msg-user-bubble {
    background: #1A1814;
    color: #F5F2ED;
    padding: 0.8rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    font-weight: 300;
    max-width: 85%;
    line-height: 1.6;
    letter-spacing: 0.01em;
}

.msg-assistant {
    margin: 1.5rem 0;
}

.msg-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #A09880;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    font-weight: 300;
}

.msg-assistant-bubble {
    background: transparent;
    color: #1A1814;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    font-weight: 300;
    line-height: 1.8;
    letter-spacing: 0.01em;
    border-left: 2px solid #8B7355;
    padding-left: 1rem;
}

/* Divider */
.msg-divider {
    border: none;
    border-top: 1px solid #E8E3DC;
    margin: 0.5rem 0;
}

/* Thinking state */
.thinking {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #A09880;
    letter-spacing: 0.08em;
    animation: pulse 1.5s infinite;
    padding-left: 1rem;
    border-left: 2px solid #D4CFC8;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Input area */
.stChatInput {
    margin-top: 2rem;
}

.stChatInput > div {
    border: 1px solid #D4CFC8 !important;
    border-radius: 0 !important;
    background: #FDFAF6 !important;
    box-shadow: none !important;
}

.stChatInput textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #1A1814 !important;
    background: transparent !important;
    font-weight: 300 !important;
    letter-spacing: 0.01em !important;
}

.stChatInput textarea::placeholder {
    color: #B8B0A4 !important;
    font-style: italic;
}

/* Spinner override */
.stSpinner > div {
    border-top-color: #8B7355 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D4CFC8; }

</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="title-block">
    <div class="title-main">Research <em>Agent</em></div>
    <div class="title-sub">Llama 3.3 &nbsp;&middot;&nbsp; DuckDuckGo &nbsp;&middot;&nbsp; LangChain</div>
</div>
""", unsafe_allow_html=True)

# Setup agent
@st.cache_resource
def setup_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
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
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True
    )
    return agent_executor

agent_executor = setup_agent()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-user-bubble">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-assistant">
            <div class="msg-label">Response</div>
            <div class="msg-assistant-bubble">{message["content"]}</div>
        </div>
        <hr class="msg-divider">
        """, unsafe_allow_html=True)

# Input
if query := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f"""
    <div class="msg-user">
        <div class="msg-user-bubble">{query}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("searching..."):
        response = agent_executor.invoke({"input": query})
        answer = response["output"]

    st.markdown(f"""
    <div class="msg-assistant">
        <div class="msg-label">Response</div>
        <div class="msg-assistant-bubble">{answer}</div>
    </div>
    <hr class="msg-divider">
    """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})