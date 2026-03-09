import streamlit as st
import os
import time
import base64
import urllib.parse
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from ddgs import DDGS
import pypdf

load_dotenv()

# ======================== PAGE CONFIG ========================
st.set_page_config(page_title="Multi-Agent Research", page_icon=None, layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] { background-color: #F5F2ED; font-family: 'DM Mono', monospace; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 4rem 2rem 2rem 2rem !important; }
#MainMenu, footer, header { visibility: hidden; }

.title-block { margin-bottom: 3rem; padding-bottom: 2rem; border-bottom: 1px solid #D4CFC8; }
.title-main { font-family: 'DM Serif Display', serif; font-size: 2.8rem; color: #1A1814; letter-spacing: -0.02em; line-height: 1; margin-bottom: 0.4rem; }
.title-main em { font-style: italic; color: #8B7355; }
.title-sub { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #A09880; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 300; }

.msg-user { margin: 1.5rem 0; display: flex; justify-content: flex-end; }
.msg-user-bubble { background: #1A1814; color: #F5F2ED; padding: 0.8rem 1.2rem; font-family: 'DM Mono', monospace; font-size: 0.82rem; font-weight: 300; max-width: 85%; line-height: 1.6; letter-spacing: 0.01em; }
.msg-assistant { margin: 1.5rem 0; }
.msg-label { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #A09880; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 300; }
.msg-assistant-bubble { background: transparent; color: #1A1814; font-family: 'DM Mono', monospace; font-size: 0.82rem; font-weight: 300; line-height: 1.8; letter-spacing: 0.01em; border-left: 2px solid #8B7355; padding-left: 1rem; }
.msg-divider { border: none; border-top: 1px solid #E8E3DC; margin: 0.5rem 0; }

.model-tag { display: inline-block; padding: 0.2rem 0.6rem; margin: 0.2rem; font-size: 0.7rem; background: #E8E3DC; color: #1A1814; border-radius: 4px; border: 1px solid #D4CFC8; }
.model-tag.success { background: #D4E1CD; border-color: #AFCAA1; }
.model-tag.error { background: #E1CDCD; border-color: #CAA1A1; }

.stChatInput > div { border: 1px solid #D4CFC8 !important; border-radius: 0 !important; background: #FDFAF6 !important; box-shadow: none !important; }
.stChatInput textarea { font-family: 'DM Mono', monospace !important; font-size: 0.82rem !important; color: #1A1814 !important; background: transparent !important; font-weight: 300 !important; }
.stChatInput textarea::placeholder { color: #B8B0A4 !important; font-style: italic; }
.stSpinner > div { border-top-color: #8B7355 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D4CFC8; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-block">
    <div class="title-main">Multi AI <em>Agent</em></div>
    <div class="title-sub">Gemini &nbsp;&middot;&nbsp; Parallel Execution &nbsp;&middot;&nbsp; File Uploads &nbsp;&middot;&nbsp; LangChain</div>
</div>
""", unsafe_allow_html=True)


# ======================== TOOLS ========================
@tool
def search_web(query: str) -> str:
    """Use this to search the web for any topic or question."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            output = ""
            for r in results:
                output += f"Title: {r.title}\nSummary: {r.body}\nURL: {r.href}\n\n"
            return output if output else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def generate_image(prompt: str) -> str:
    """Use this tool exclusively to generate images based on a user description using Nano Banana Pro. Return the Markdown image link."""
    encoded = urllib.parse.quote(prompt)
    return f"![Generated via Nano Banana Pro](https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&nologo=true)"


TOOLS = [search_web, generate_image]

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. You have access to tools that you should use if needed."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# ======================== AGENT SETUP ========================
@st.cache_resource
def setup_models():
    """Create Gemini agent variants (kept to 2 to stay within free-tier rate limits)."""
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        return []

    model_list = [
        ("Gemini 2.0 Flash", ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", temperature=0.2, google_api_key=google_key
        )),
        ("Gemini 2.0 Flash Lite", ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite", temperature=0.5, google_api_key=google_key
        )),
    ]

    agent_executors = []
    for name, llm in model_list:
        try:
            agent = create_tool_calling_agent(llm, TOOLS, AGENT_PROMPT)
            executor = AgentExecutor(
                agent=agent, tools=TOOLS, verbose=False,
                max_iterations=5, handle_parsing_errors=True
            )
            agent_executors.append((name, executor))
        except Exception:
            pass

    return agent_executors


agents = setup_models()


def run_single_agent(name, executor, inputs, max_retries=3):
    """Run a single agent with retry on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            res = executor.invoke(inputs)
            return name, res["output"], True
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = (attempt + 1) * 15  # 15s, 30s, 45s
                time.sleep(wait)
                continue
            return name, err, False
    return name, "Rate limit exceeded after retries", False


# ======================== SIDEBAR ========================
with st.sidebar:
    st.markdown("<h3 style='font-family: \"DM Serif Display\", serif;'>📁 File Uploads</h3>", unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])
    uploaded_img = st.file_uploader("Upload Image (Vision context)", type=["png", "jpg", "jpeg"])

    pdf_text = ""
    if uploaded_pdf:
        with st.spinner("Extracting PDF..."):
            try:
                pdf_reader = pypdf.PdfReader(uploaded_pdf)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        pdf_text += extracted + "\n"
                st.success(f"✅ Extracted {len(pdf_text)} chars from PDF")
            except Exception as e:
                st.error(f"PDF extraction failed: {e}")

    img_b64 = None
    if uploaded_img:
        img_b64 = base64.b64encode(uploaded_img.getvalue()).decode("utf-8")
        st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)
        st.success("✅ Image ready for Gemini vision")

    st.markdown("---")
    st.markdown("<h3 style='font-family: \"DM Serif Display\", serif;'>🤖 Active Models</h3>", unsafe_allow_html=True)
    if not agents:
        st.warning("No GOOGLE_API_KEY found in .env!")
    for name, _ in agents:
        st.markdown(f"<div class='model-tag success'>{name}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.65rem; color: #A09880; line-height: 1.6;'>
        <strong>Required .env key:</strong><br>
        GOOGLE_API_KEY
    </div>
    """, unsafe_allow_html=True)


# ======================== CHAT STATE ========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f'<div class="msg-user"><div class="msg-user-bubble">{message["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        model_label = message.get("model", "Judge")
        st.markdown(
            f'<div class="msg-assistant">'
            f'<div class="msg-label">Response • {model_label}</div>'
            f'<div class="msg-assistant-bubble">{message["content"]}</div>'
            f'</div><hr class="msg-divider">',
            unsafe_allow_html=True,
        )


# ======================== INPUT & PROCESSING ========================
if query := st.chat_input("Ask anything or upload files..."):
    display_query = query
    if uploaded_pdf:
        display_query += f"  📄 {uploaded_pdf.name}"
    if uploaded_img:
        display_query += f"  🖼️ {uploaded_img.name}"

    st.markdown(
        f'<div class="msg-user"><div class="msg-user-bubble">{display_query}</div></div>',
        unsafe_allow_html=True,
    )
    st.session_state.messages.append({"role": "user", "content": display_query})

    final_query = query
    if pdf_text:
        final_query += f"\n\n[Extracted PDF Content]:\n{pdf_text[:5000]}"

    if img_b64:
        input_content = [
            {"type": "text", "text": final_query},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]
    else:
        input_content = final_query

    inputs = {
        "input": input_content,
        "chat_history": st.session_state.langchain_history,
    }

    st.session_state.langchain_history.append(HumanMessage(content=final_query))

    with st.spinner("⚡ Running Gemini models..."):
        results = []
        statuses = []

        for name, agent_exec in agents:
            agent_name, output, success = run_single_agent(name, agent_exec, inputs)
            results.append((agent_name, output))
            if success:
                statuses.append(f"<div class='model-tag success'>✓ {agent_name}</div>")
            else:
                statuses.append(f"<div class='model-tag error'>✗ {agent_name}</div>")
            # Delay between models to respect free-tier rate limits
            if len(agents) > 1:
                time.sleep(10)

        st.markdown(
            "<div style='margin: 1rem 0;'>" + "".join(statuses) + "</div>",
            unsafe_allow_html=True,
        )

        # --- Judge / Evaluator (uses Gemini 2.0 Flash Lite to save quota) ---
        successful_results = [(n, o) for n, o in results if o and "Rate limit" not in o]

        if len(successful_results) == 0:
            best_answer = "All models failed. You may have hit Gemini's free-tier rate limit. Please wait a minute and try again."
        elif len(successful_results) == 1:
            best_answer = successful_results[0][1]
        else:
            time.sleep(10)  # wait before judge call
            eval_prompt = f"User asked: {query}\n\nHere are answers from different Gemini model variants:\n"
            for name, ans in successful_results:
                eval_prompt += f"--- {name} ---\n{ans}\n\n"
            eval_prompt += (
                "Evaluate these answers and provide the single best, most accurate, "
                "and most complete response. Do not mention that you are a judge or "
                "the evaluation process. Just provide the final synthesized response. "
                "If an image was generated (Markdown link), include it EXACTLY as provided."
            )

            try:
                judge = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-lite", temperature=0,
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                )
                best_answer = judge.invoke(eval_prompt).content
            except Exception:
                # If judge fails due to rate limit, use the first successful result
                best_answer = successful_results[0][1]

        st.session_state.langchain_history.append(AIMessage(content=best_answer))
        st.session_state.messages.append({
            "role": "assistant", "content": best_answer, "model": "Judge (Gemini)"
        })

        st.markdown(
            f'<div class="msg-assistant">'
            f'<div class="msg-label">Response • Judge (Gemini)</div>'
            f'<div class="msg-assistant-bubble">{best_answer}</div>'
            f'</div><hr class="msg-divider">',
            unsafe_allow_html=True,
        )