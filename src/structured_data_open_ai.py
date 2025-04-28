from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Optional
from typing import Union
import dotenv


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(
        description="A conversational response to the user's query")


class FinalResponse(BaseModel):
    final_output: Union[Joke, ConversationalResponse]


def structured_output(llm):
    structured_llm = llm.with_structured_output(FinalResponse)
    # res = structured_llm.invoke("Tell me how to make a sandwich")
    res = structured_llm.invoke("Tell me a joke about birds")
    print(res)


def stream_output(llm):
    json_schema = {
        "title": "joke",
        "description": "Joke to tell user.",
        "type": "object",
        "properties": {
            "setup": {
                "type": "string",
                "description": "The setup of the joke",
            },
            "punchline": {
                "type": "string",
                "description": "The punchline to the joke",
            },
            "rating": {
                "type": "integer",
                "description": "How funny the joke is, from 1 to 10",
                "default": None,
            },
        },
        "required": ["setup", "punchline"],
    }
    structured_llm = llm.with_structured_output(json_schema)
    for chank in structured_llm.stream("Tell me a joke about birds"):
        print(chank)


if __name__ == "__main__":
    dotenv.load_dotenv()
    llm = init_chat_model(
        "gpt-4o-mini",
        model_provider="openai",
        temperature=0.9)

    # stream_output(llm)
    structured_output(llm)
