# Multi-Agent Research Agent

A premium, agentic AI platform that conducts deep research by orchestrating multiple Gemini models in parallel. It features autonomous web searching, document analysis, and image generation through a minimalist, bespoke interface.

---

## Key Features

- **Parallel Model Execution** — Runs `Gemini 2.0 Flash` and `Flash Lite` simultaneously to compare results.
- **AI "Judge" Evaluation** — A specialized judge model synthesizes the best, most accurate answer from parallel responses.
- **Multimodal Context** — Upload images for vision-based reasoning and PDFs for grounded document research.
- **ReAct Reasoning Loop** — Uses the Reasoning + Acting pattern for autonomous tool usage and fact-finding.
- **Image Generation** — Built-in `Nano Banana Pro` tool for creating visuals directly from descriptions.
- **Premium UI** — Minimalist typography, custom HSL color palettes, and glassmorphism-inspired design.

---

## Technical Stack

| Layer | Technology |
|---|---|
| **Core LLM** | Google Gemini 2.0 (Flash & Lite) |
| **Agent Framework** | LangChain |
| **Search Engine** | DuckDuckGo Search |
| **Image Generation** | Pollinations AI (Nano Banana Pro) |
| **Document Parsing** | PyPDF |
| **Interface** | Streamlit (Custom Styled) |

---

## Getting Started

### 1. Configuration

Ensure you have a `.env` file in the project root with your Google API Key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/AkshayRP12/web-research-agent.git
cd web-research-agent

# Set up virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch

```bash
streamlit run app.py
```

---

## Project Structure

```text
web-research-agent/
├── app.py              # Main Multi-Agent Application (Streamlit)
├── agent.py            # Legacy Terminal Reasoning Agent
├── requirements.txt    # Project Dependencies
├── .env                # Secret Keys (Local Only)
└── .streamlit/         # UI Configuration
```

---

## Future Roadmap

- [x] Integrate Gemini 2.0 Models
- [x] Implement Multi-Agent Parallel Reasoning
- [x] Add PDF and Image Upload Support
- [x] Premium Minimalist UI Overhaul
- [x] Image Generation Integration
- [ ] Support for Local LLMs (Ollama)
- [ ] Export Research Reports as PDF/Markdown
- [ ] Multi-Agent specialized "Expert" personas

---

## License

MIT
