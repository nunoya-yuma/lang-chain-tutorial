from langchain_core.messages import SystemMessage, trim_messages
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import Sequence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage
import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.messages import AIMessage

dotenv.load_dotenv()
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# # Define a new graph
# workflow = StateGraph(state_schema=MessagesState)


# # Define the function that calls the model
# def call_model(state: MessagesState):
#     response = model.invoke(state["messages"])
#     return {"messages": response}


# # Define the (single) node in the graph
# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)

# # Add memory
# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)
# config = {"configurable": {"thread_id": "abc123"}}

# query = "Hi! I'm Bob."

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()  # output contains all messages in
# state

# query = "What's my name?"

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# # Now let's try a different thread
# config = {"configurable": {"thread_id": "abc234"}}
# # LLM does not have memory of previous conversation
# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# # Now return to the first thread
# config = {"configurable": {"thread_id": "abc123"}}
# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# ==============================================================================================

# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system",
#          "You talk like a pirate. Answer all questions to the best of your ability.",
#          ),
#         MessagesPlaceholder(
#             variable_name="messages"),
#     ])
# workflow = StateGraph(state_schema=MessagesState)


# def call_model(state: MessagesState):
#     prompt = prompt_template.invoke(state)
#     response = model.invoke(prompt)
#     return {"messages": response}


# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)

# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)
# config = {"configurable": {"thread_id": "abc345"}}
# query = "Hi! I'm Jim."

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# query = "What is my name?"

# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# ==============================================================================================

# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system",
#          "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
#          ),
#         MessagesPlaceholder(
#             variable_name="messages"),
#     ])


# class State(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     language: str


# workflow = StateGraph(state_schema=State)


# def call_model(state: State):
#     prompt = prompt_template.invoke(state)
#     response = model.invoke(prompt)
#     return {"messages": [response]}


# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)

# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)

# config = {"configurable": {"thread_id": "abc456"}}
# query = "Hi! I'm Bob."
# language = "Spanish"

# input_messages = [HumanMessage(query)]
# output = app.invoke(
#     {"messages": input_messages, "language": language},
#     config,
# )
# output["messages"][-1].pretty_print()

# query = "What is my name?"

# input_messages = [HumanMessage(query)]
# output = app.invoke(
#     {"messages": input_messages},
#     config,
# )
# output["messages"][-1].pretty_print()

# ==============================================================================================
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
         ),
        MessagesPlaceholder(
            variable_name="messages"),
    ])


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


trimmer = trim_messages(
    max_tokens=75,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

workflow = StateGraph(state_schema=State)


def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc567"}}
query = "What is my name?"
language = "English"

input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc678"}}
query = "What math problem did I ask?"
language = "English"

input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()
