## [PandasAI Agent](https://github.com/Coral-Protocol/Pandas-ai-Agent)

The PandasAI Agent helps you answer data-related questions about Excel or CSV files using a local LLM (e.g., Llama4) via PandasAI. Simply provide the file path and your natural language question—the agent will query the data and return the answer.

## Responsibility
The PandasAI Agent enables natural language querying of tabular data (Excel/CSV) using a local LLM through PandasAI, making data analysis accessible and conversational

## Details
- **Framework**: LangChain
- **Tools used**: PandasAI Tools, Coral MCP Tools
- **AI model**: Llama4 via Ollama
- **Date added**: 04/06/25
- **Reference**: [PandasAI Agent](https://pandas-ai.com/)
- **License**: MIT

## Use the Agent

### 1. Install and Run Ollama (for Local LLM)
<details>

PandasAI Agent uses Ollama to run local LLMs. Please make sure you have Ollama installed and the desired model downloaded before running the agent.

**Step 1: Install Ollama**

- **Linux/macOS:**
  Follow the official instructions: [https://ollama.com/download](https://ollama.com/download)
  Or run:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- **Windows:**
  Download the installer from [Ollama's website](https://ollama.com/download).

**Step 2: Download Local model**

```bash
ollama pull llama4:latest
```

**Step 3: Start Ollama Service**

Ollama usually starts automatically. If not, start it manually:
```bash
ollama serve
```

**Step 4: Verify the model is running**

```bash
ollama list
```
Make sure no errors occur and Ollama is running at `http://localhost:11434`.

</details>

### 2. Clone & Install Dependencies

<details>  

Ensure that the [Coral Server](https://github.com/Coral-Protocol/coral-server) is running on your system and the [Interface Agent](https://github.com/Coral-Protocol/Coral-Interface-Agent) is running on the Coral Server.  

```bash
# Clone the PandasAI Agent repository
git clone https://github.com/Coral-Protocol/Pandas-ai-Agent.git

# Navigate to the project directory
cd Pandas-ai-Agent

# Install `uv`:
pip install uv

# Install dependencies from `pyproject.toml` using `uv`:
uv sync
```
This command will read the `pyproject.toml` file and install all specified dependencies in a virtual environment managed by `uv`.

</details>

### 3. Configure Environment Variables
<details>

Get the API Key:
[OpenAI](https://platform.openai.com/api-keys)

Create a .env file in the project root:
```bash
cp -r env_sample .env
```

Add your API keys and any other required environment variables to the .env file.

</details>

### 4. Run Agent
<details>

Run the agent using `uv`:
```bash
uv run 1-langchain-PandasAiAgent.py
```
</details>

### 5. Example
<details>

```bash
# Input:
Question: What are the total number of columns in the coral_public_repo_docs.xlsx

# Output:
Answer: The total number of columns in the coral_public_repo_docs.xlsx is 4.
```

**🎬 [Watch Video Demo](https://youtu.be/aq6du6XRzGE)**

</details>

## Creator Details
- **Name**: Xinxing
- **Affiliation**: Coral Protocol
- **Contact**: [Discord](https://discord.com/invite/Xjm892dtt3)
