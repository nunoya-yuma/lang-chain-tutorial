import dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()

memory = MemorySaver()
model = init_chat_model("gpt-4o-mini", model_provider="openai")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

for step in agent_executor.stream(
    {"messages": [HumanMessage(content="what is the weather?")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
