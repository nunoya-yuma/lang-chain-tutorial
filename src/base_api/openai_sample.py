import dotenv
from langchain.chat_models import init_chat_model

dotenv.load_dotenv()
model = init_chat_model("gpt-4o-mini", model_provider="openai")
res = model.invoke("Hello, world!")
print(res)
