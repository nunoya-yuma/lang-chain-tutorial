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
import argparse
from typing import List


class AgentWithMCP:
    """
    Agent that integrates with MCP (Model Context Protocol) servers.

    This class provides functionality to interact with both stdio-based and
    SSE-based MCP servers, allowing dynamic registration of different types and
    handling communication through a unified interface.

    Attributes:
        _client: MCP client instance
        _agent: Agent instance for handling requests
        _mcp_config: Configuration dictionary for MCP servers
        _model: Language model instance
        _memory: Memory manager for maintaining conversation state
    """

    def __init__(self, model_provider="openai"):
        """
        Initialize the agent with a specified model provider.

        Args:
            model_provider (str): Provider name to use:
                - "openai": Uses GPT-4 mini model
                - "google_genai": Uses Gemini flash model

        Raises:
            ValueError: If an unsupported model provider is specified
        """
        self._client = None
        self._agent = None
        self._mcp_config = {}

        # Set model based on provider
        if model_provider == "openai":
            self._model = init_chat_model(
                "gpt-4o-mini",
                model_provider="openai"
            )
        elif model_provider == "google_genai":
            self._model = init_chat_model(
                "gemini-2.0-flash",
                model_provider="google_genai"
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

        self._memory = MemorySaver()

        thread_id = str(uuid.uuid4())
        self._config = {"configurable": {"thread_id": thread_id}}
        print(f"Thread ID: {thread_id}")
        print(f"Using model provider: {model_provider}")

    def register_mcp_stdio(self, name: str, command: List[str]) -> None:
        """
        Register a stdio-based MCP server.

        This method configures a server that communicates through standard
        input/output. The server will be started using the provided command
        when the session begins.

        Args:
            name (str): Unique identifier for this MCP server
            command (List[str]): Command and args to start the server.
                [0] is the command, [1:] are the arguments.
        """
        self._mcp_config[name] = {
            "command": command[0],
            "args": command[1:],
            "transport": "stdio"
        }

    def register_mcp_sse(self, name: str, url: str) -> None:
        """
        Register an SSE-based MCP server.

        This method configures a server that communicates through Server-Sent
        Events (SSE). The server should already be running and accessible at
        the provided URL.

        Args:
            name (str): Unique identifier for this MCP server
            url (str): Full URL of the SSE endpoint
        """
        self._mcp_config[name] = {
            "url": url,
            "transport": "sse"
        }

    @asynccontextmanager
    async def session(self):
        """
        Create and manage a session with all registered MCP servers.

        This context manager handles the lifecycle of MCP server connections:
        1. Validates the configuration and state
        2. Establishes connections to all registered servers
        3. Creates an agent with access to server tools
        4. Cleans up connections when the session ends

        Yields:
            AgentWithMCP: The agent instance with active connections to
                         registered MCP servers

        Raises:
            ValueError: If any of these conditions are met:
                      - No MCP servers have been registered
                      - MCP client has already been created
                      - Agent has already been created
                      - No language model has been specified
                      - No memory system has been specified
        """
        if self._mcp_config == {}:
            raise ValueError("No MCP servers registered")
        if self._client is not None:
            raise ValueError("MCP client already created")
        if self._agent is not None:
            raise ValueError("MCP agent already created")
        if self._model is None:
            raise ValueError("No model specified")
        if self._memory is None:
            raise ValueError("No memory specified")

        self._client = MultiServerMCPClient(self._mcp_config)
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

    async def send_message(self, message: str) -> str:
        """
        Send a message to the agent and receive a response.

        This method sends a message to the agent, which may interact with
        registered MCP servers to generate a response. The interaction is
        streamed, with each response part being displayed as it's generated.

        Args:
            message (str): The message to send to the agent

        Returns:
            str: The complete response from the agent

        Note:
            This method should only be called within a session context.
        """
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

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Weather and Math Agent using MCP"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "google_genai"],
        default="google_genai",
        help="Model provider to use (openai or google_genai)"
    )
    args = parser.parse_args()

    agent = AgentWithMCP(
        model_provider=args.provider
    )

    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    math_server_path = str(current_dir / "mcp_math_server.py")
    agent.register_mcp_stdio("math", ["uv", "run", math_server_path])
    agent.register_mcp_sse("weather", "http://localhost:8000/sse")

    async with agent.session() as agent_session:
        _ = await agent_session.send_message("what's (3 + 5) x 12")
        _ = await agent_session.send_message(
            "I live in Osaka. Do you know where that is?")
        _ = await agent_session.send_message(
            "What is the weather at my current location?")


if __name__ == "__main__":
    asyncio.run(main())
