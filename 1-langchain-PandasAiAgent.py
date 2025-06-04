import asyncio
import os
import json
import logging
from typing import List
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain.memory import ConversationSummaryMemory
from langchain_core.memory import BaseMemory
from dotenv import load_dotenv
from anyio import ClosedResourceError
import urllib.parse
import subprocess
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

base_url = "http://localhost:5555/devmode/exampleApplication/privkey/session1/sse"
params = {
    "waitForAgents": 1,
    "agentId": "pandasai_agent",
    "agentDescription": """I am a `pandasai_agent`, responsible for answering data-related questions about Excel or CSV files using only the available tools..
                           You should let me know the file name and question."""
}
query_string = urllib.parse.urlencode(params)
MCP_SERVER_URL = f"{base_url}?{query_string}"
AGENT_NAME = "pandasai_agent"

# Validate API keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

def get_tools_description(tools):
    return "\n".join(f"Tool: {t.name}, Schema: {json.dumps(t.args).replace('{', '{{').replace('}', '}}')}" for t in tools)
    
@tool
def query_xlsx_with_llama(
    file_path: str,
    question: str,
    api_base: str = "http://localhost:11434/v1",
    model: str = "llama3.1:latest"
) -> str:
    """
    Query an Excel file using a local LLM via PandasAI.

    Args:
        file_path (str): Path to the Excel file.
        question (str): Natural language query to ask about the data.
        api_base (str): URL of the local LLM API. Defaults to Ollama's localhost endpoint.
        model (str): Model name/tag for Ollama LLM. Defaults to "llama3.1:latest".

    Returns:
        str: The answer to the question.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For any LLM or DataFrame-related errors.
    """
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    try:
        ollama_llm = LocalLLM(api_base=api_base, model=model)
        df = SmartDataframe(file_path, config={"llm": ollama_llm})
        answer = df.chat(output_type='string', query=question)
        return answer
    except Exception as e:
        raise Exception(f"Failed to process query '{question}' on '{file_path}': {str(e)}")



class HeadSummaryMemory(BaseMemory):
    def __init__(self, llm, head_n=3):
        super().__init__()
        self.head_n = head_n
        self._messages = []
        self.summary_memory = ConversationSummaryMemory(llm=llm)

    def save_context(self, inputs, outputs):
        user_msg = inputs.get("input") or next(iter(inputs.values()), "")
        ai_msg = outputs.get("output") or next(iter(outputs.values()), "")
        self._messages.append({"input": user_msg, "output": ai_msg})
        if len(self._messages) > self.head_n:
            self.summary_memory.save_context(inputs, outputs)

    def load_memory_variables(self, inputs):
        messages = []
        
        for i in range(min(self._head_n, len(self._messages))):
            msg = self._messages[i]
            messages.append(HumanMessage(content=msg['input']))
            messages.append(AIMessage(content=msg['output']))
        # summary
        if len(self._messages) > self._head_n:
            summary_var = self.summary_memory.load_memory_variables(inputs).get("history", [])
            if summary_var:
                
                if isinstance(summary_var, str):
                    messages.append(HumanMessage(content="[Earlier Summary]\n" + summary_var))
                elif isinstance(summary_var, list):
                    messages.extend(summary_var)
        return {"history": messages}

    def clear(self):
        self._messages.clear()
        self.summary_memory.clear()

    @property
    def memory_variables(self):
        return {"history"}
    
    @property
    def head_n(self):
        return self._head_n

    @head_n.setter
    def head_n(self, value):
        self._head_n = value

    @property
    def summary_memory(self):
        return self._summary_memory

    @summary_memory.setter
    def summary_memory(self, value):
        self._summary_memory = value

async def create_pandasai_agent(client, tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are `pandasai_agent`, responsible for answering data-related questions about Excel or CSV files using only the available tools. Follow this workflow:

        1. Use `wait_for_mentions(timeoutMs=60000)` to wait for instructions from other agents.
        2. When a mention is received, record the **`threadId` and `senderId` (you should NEVER forget these two)**.
        3. Check if the message contains a valid `file_path` and a `query` (the user's natural language question about the data).
        4. Call `query_xlsx_with_llama(file_path = ..., question = ...)` to get the answer to the question.
            - If you fail to call `query_xlsx_with_llama`, double-check the input parameters and call it again.
        5. Analyze the returned answer:
            - If the answer is meaningful, proceed to the next step.
            - If the answer is empty, inconclusive, or you encounter an error, set content to `"error"` or an appropriate message.
        6. Use `send_message(senderId=..., mentions=[senderId], threadId=..., content="your answer")` to reply to the sender with your answer.
        7. Always respond to the sender, even if the answer is empty or inconclusive.
        8. Wait 2 seconds and repeat from step 1.
        
        Tools: {get_tools_description(tools)}"""),
        ("placeholder", "{history}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = ChatOpenAI(
        model="gpt-4.1-2025-04-14",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
        max_tokens=32768
    )

    '''model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )'''

    memory = HeadSummaryMemory(llm=model, head_n=4)


    agent = create_tool_calling_agent(model, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, memory=memory, max_iterations=100 ,verbose=True)

async def main():
    max_retries = 5
    retry_delay = 5  # seconds

    github_token = os.getenv("GITHUB_ACCESS_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN environment variable is required")

    for attempt in range(max_retries):
        try:
            async with MultiServerMCPClient(
                connections = {
                    "coral": {
                        "transport": "sse", 
                        "url": MCP_SERVER_URL, 
                        "timeout": 600, 
                        "sse_read_timeout": 600
                    }
                }
            ) as client:
                logger.info(f"Connected to MCP server at {MCP_SERVER_URL}")
                coral_tool_names = [
                    "list_agents",
                    "create_thread",
                    "add_participant",
                    "remove_participant",
                    "close_thread",
                    "send_message",
                    "wait_for_mentions",
                ]

                tools = client.get_tools()

                tools = [
                    tool for tool in tools
                    if tool.name in coral_tool_names
                ]

                tools += [query_xlsx_with_llama]

                logger.info(f"Tools Description:\n{get_tools_description(tools)}")
                agent_executor = await create_pandasai_agent(client, tools)
                await agent_executor.ainvoke({})

        except ClosedResourceError as e:
            logger.error(f"ClosedResourceError on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                logger.error("Max retries reached. Exiting.")
                raise
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                logger.error("Max retries reached. Exiting.")
                raise

if __name__ == "__main__":
    asyncio.run(main())
