# Web Research Agent

An agentic AI that searches the web in real time and answers questions using a ReAct reasoning loop. Built with LangChain, Llama 3.3, and Streamlit.

---

## How it works

The agent follows the ReAct loop on every query:

1. **Thought** — the LLM reasons about what to do
2. **Action** — it calls the web search tool
3. **Observation** — it reads the search results
4. **Final Answer** — it synthesizes a clean response

---

## Stack

| Layer | Tool |
|---|---|
| LLM | Llama 3.3 via Groq API |
| Agent Framework | LangChain |
| Search Tool | DuckDuckGo |
| UI | Streamlit |
| Language | Python 3.12 |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/AkshayRP12/web-research-agent.git
cd web-research-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Project Structure

```
web-research-agent/
├── app.py              # Streamlit UI
├── agent.py            # Terminal version of the agent
├── requirements.txt    # Dependencies
├── .env                # API keys (never committed)
└── .gitignore
```

---

## What I learned

- How LLMs work as the reasoning brain of an agent
- The ReAct (Reasoning + Acting) loop that powers most AI agents
- How to connect LLMs to real world tools like web search
- LangChain for agent orchestration
- Building and styling a chat UI with Streamlit
- Git and GitHub workflow

---

## Roadmap

- [ ] Add conversation memory
- [ ] Deploy on Streamlit Cloud
- [ ] Add more tools (calculator, Wikipedia, weather)
- [ ] Multi-agent system

---

## License

MIT
