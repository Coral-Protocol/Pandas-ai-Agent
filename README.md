## Responsibility

**PandasAI Agent** helps you answer data-related questions about Excel or CSV files using a local LLM (e.g., Llama 3.1/Qwen3) via PandasAI. Simply provide the file path and your natural language question—the agent will query the data and return the answer.

## Details

* Framework: LangChain
* Tools used: PandasAI Tools, Coral MCP Tools
* AI model: Llama3.1/Qwen3 via Ollama
* Date added: 04/06/25
* Licence: MIT

## Use the Agent

### 1.Install and Run Ollama (for Local LLM)

<details>

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

**Step 2: Download Local model**

To pull the latest llama3.1/Qwen3 model:

```bash
ollama pull llama3.1:latest
```

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

Make sure no errors occur and Ollama is running at `http://localhost:11434`.

</details>

### 2.Clone & Install Dependencies

Run [Interface Agent](https://github.com/Coral-Protocol/Coral-Interface-Agent)
<details>


If you are trying to run Open Deep Research agent and require an input, you can either create your agent which communicates on the coral server or run and register the Interface Agent on the Coral Server. In a new terminal clone the repository:


```bash
git clone https://github.com/Coral-Protocol/Coral-Interface-Agent.git
```
Navigate to the project directory:
```bash
cd Coral-Interface-Agent
```

Install `uv`:
```bash
pip install uv
```
Install dependencies from `pyproject.toml` using `uv`:
```bash
uv sync
```

Configure API Key
```bash
export OPENAI_API_KEY=
```

Run the agent using `uv`:
```bash
uv run python 0-langchain-interface.py
```

</details>

Agent Installation

<details>

Clone the repository:
```bash
git clone https://github.com/Coral-Protocol/Pandas-ai-Agent.git
```

Navigate to the project directory:
```bash
cd Pandas-ai-Agent
```

Install `uv`:
```bash
pip install uv
```

Install dependencies from `pyproject.toml` using `uv`:
```bash
uv sync
```

This command will read the `pyproject.toml` file and install all specified dependencies in a virtual environment managed by `uv`.

Copy the client sse.py from utils to mcp package
```bash
cp -r utils/sse.py .venv/lib/python3.10/site-packages/mcp/client/sse.py
```

OR Copy this for windows
```bash
cp -r utils\sse.py .venv\Lib\site-packages\mcp\client\sse.py
```

</details>

### 3.Run Agent

<details>
  
Run the agent using `uv`:
  
```bash
uv run 1-langchain-PandasAiAgent.py
```
</details>

### 4.Example 

<details>
Input:

```bash
Question: What are the total number of columns in the coral_public_repo_docs.xlsx
```
Output:

```bash
Answer: The total number of columns in the coral_public_repo_docs.xlsx is 4.
```

**🎬 [Watch Video Demo](https://youtu.be/aq6du6XRzGE)**

</details>

## Creator details

* Name: Xinxing
* Affiliation: Coral Protocol
* Contact: [Discord](https://discord.com/invite/Xjm892dtt3)
