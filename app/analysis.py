import json
from typing import Any, Dict, List

from fastapi.responses import StreamingResponse
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from langchain_core.pydantic_v1 import Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import time


llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)


class ClassifyUserIntent(BaseModel):
    """An enum value to classify user intent."""
    intent: str = Field(
        description="The classified intent of the user query, must be one of: 'Summarization', 'Translation', 'Paraphrasing', 'Role-play', 'Miscellaneous'."
    )

system = """
You are an intent classification system. The correctness of the classification is crucial.

We provide you with the intents and their descriptions:
- Summarization: When the user asks for a summary of a document or text.
- Translation: When the user asks to translate text into another language.
- Paraphrasing: When the user asks to rephrase or reword a sentence or text.
- Role-play: When the user asks to simulate a conversation or scenario.
- Miscellaneous: For queries that do not fall under any of the above categories.

You are given a user query and you have to classify it into one of these intent categories. Only respond with the intent class. If the query does not match any of the descriptions, output 'Miscellaneous'.
You are not allowed to add a class on your own , JUST USE THE GIVEN INTENTS !

Now take a deep breath and classify the following user query.
"""

classify_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),
    ]
)

structured_classifier_llm = llm.with_structured_output(ClassifyUserIntent)

classifier_chain = classify_prompt |structured_classifier_llm

async def classify_single_instance(text: str) -> str:
    try:
        result = await classifier_chain.ainvoke(text)
        if isinstance(result, str):
            return result
        else:
            return result.intent if result is not None else 'Miscellaneous'
    except Exception as e:
        print(f"Error during classification: {e}")
        return 'Miscellaneous'




