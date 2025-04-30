import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# messages = [
#     SystemMessage("Translate the following from English into Italian"),
#     HumanMessage("hi!"),
# ]

# for token in model.stream(messages):
#     print(token.content, end="|")

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

print(prompt)
print(prompt.to_messages())

response = model.invoke(prompt)
print(response.content)
