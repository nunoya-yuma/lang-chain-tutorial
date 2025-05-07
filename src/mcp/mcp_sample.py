"""
This is a sample code to demonstrate how to use the MCP client with
LangChain and LangGraph.
Reference page
https://langchain-ai.github.io/langgraph/agents/mcp/#use-mcp-tools

It uses the MCP client to connect to a math server and a weather server,
and sends messages to them.
Before running this code, make sure to start the weather server on port 8000
using the following command:
uv run src/mcp/mcp_weather_server.py
The math server is started automatically by the MCP client.
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import asyncio
import dotenv
import os
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import uuid
from contextlib import asynccontextmanager


class WeatherAndMathAgent:
    def __init__(self):
        self._client = None
        self._agent = None
        self._model = init_chat_model("gpt-4o-mini", model_provider="openai")
        self._memory = MemorySaver()

        thread_id = str(uuid.uuid4())
        self._config = {"configurable": {"thread_id": thread_id}}
        print(f"Thread ID: {thread_id}")

    @asynccontextmanager
    async def session(self):
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        math_server_path = str(current_dir / "mcp_math_server.py")

        self._client = MultiServerMCPClient(
            {
                "math": {
                    "command": "uv",
                    "args": ["run", math_server_path],
                    "transport": "stdio",
                },
                "weather": {
                    # Ensure your start your weather server on port 8000
                    "url": "http://localhost:8000/sse",
                    "transport": "sse",
                }
            }
        )

        try:
            await self._client.__aenter__()

            self._agent = create_react_agent(
                self._model,
                self._client.get_tools(),
                checkpointer=self._memory,
            )
            yield self
        finally:
            await self._client.__aexit__(None, None, None)

    async def send_message(self, message):
        result = ""

        structured_message = {
            "messages": [HumanMessage(content=message)],
        }

        async for weather_res in self._agent.astream(
            structured_message,
            self._config,
            stream_mode="values",
        ):
            weather_res["messages"][-1].pretty_print()
            result += weather_res["messages"][-1].content

        return result


async def main():
    dotenv.load_dotenv()

    async with WeatherAndMathAgent().session() as agent:
        _ = await agent.send_message("what's (3 + 5) x 12")
        _ = await agent.send_message(
            "I live in Osaka. Do you know where that is?")
        _ = await agent.send_message(
            "What is the weather at my current location?")


if __name__ == "__main__":
    asyncio.run(main())
