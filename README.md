### Responsibility

**PandasAI Agent** helps you answer data-related questions about Excel or CSV files using a local LLM (e.g., Llama 3.1) via PandasAI. Simply provide the file path and your natural language question—the agent will query the data and return the answer.

### Details

* Framework: LangChain
* Tools used: PandasAI Tools, Coral MCP Tools
* AI model: OpenAI GPT-4.1 / Llama3.1 via Ollama
* Date added: 04/06/25
* Licence: MIT

### Install Dependencies

Install all required packages:

```bash
pip install langchain langchain_mcp_adapters langchain_openai pandasai python-dotenv anyio
pip install pandas openpyxl
```

### Configure Environment Variables

```bash
export OPENAI_API_KEY=sk-xxx
```

**How to obtain API keys:**

* **OPENAI\_API\_KEY:**
  Sign up at [platform.openai.com](https://platform.openai.com/), go to “API Keys” under your account, and click “Create new secret key.”

### Run agent command

Make sure Pllama is running in your local machine, then run:

```bash
python 1-langchain-pandasai-agent.py
```

### Example output

```bash
Question: What are the total number of columns in the coral_public_repo_docs.xlsx
Answer: The total number of columns in the coral_public_repo_docs.xlsx is 4.
```

### Creator details

* Name: Xinxing
* Affiliation: Coral Protocol
* Contact: xinxing@coralprotocol.org



