### Responsibility

**PandasAI Agent** helps you answer data-related questions about Excel or CSV files using a local LLM (e.g., Qwen3) via PandasAI. Simply provide the file path and your natural language question—the agent will query the data and return the answer.

### Details

* Framework: LangChain
* Tools used: PandasAI Tools, Coral MCP Tools
* AI model: Qwen3
* Date added: 04/06/25
* Licence: MIT

### Install Dependencies

Install all required packages:

```bash
pip install langchain langchain_mcp_adapters langchain_ollama pandasai python-dotenv anyio
pip install pandas openpyxl
```

### Install and Run Ollama (for Local LLM)

PandasAI Agent uses Ollama to run local LLM Qwen3. Please make sure you have Ollama installed and the desired model downloaded before running the agent.

**Step 1: Install Ollama**

* **Linux/macOS:**
  Follow the official instructions: [https://ollama.com/download](https://ollama.com/download)
  Or run:

  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

* **Windows:**
  Download the installer from [Ollama’s website](https://ollama.com/download).

**Step 2: Download Qwen3 model**

To pull the latest Qwen3 model:

```bash
ollama pull qwen3:latest
```

**Step 3: Start Ollama Service**

Ollama usually starts automatically. If not, start it manually:

```bash
ollama serve
```

**Step 4: Verify the model is running**

You can check with:

```bash
ollama list
```

and

```bash
ollama run qwen3:latest
```

Make sure no errors occur and Ollama is running at `http://localhost:11434`.

---

### Run agent command

Make sure Ollama is running in your local machine, then run:

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



