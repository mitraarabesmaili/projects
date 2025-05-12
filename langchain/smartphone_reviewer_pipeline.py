"""
This script uses a LangChain pipeline with a language model to generate a professional review of a smartphone.
It identifies the product's key features, analyzes its advantages and disadvantages, and formats the result
into a structured review. The pipeline uses OpenAI's GPT model and LangChain's Expression Language (LCEL).
"""

import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI model
model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    openai_api_base="https://api.gilas.io/v1/"
)

# Prompt for extracting product features
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a tech reviewer who writes clear, concise summaries for consumers."),
        ("human", "What are the most important specifications and capabilities of the smartphone called {product_name}?"),
    ]
)

def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a tech reviewer who helps consumers make smart decisions."),
            ("human", "Based on the following features: {features}, what are the strongest selling points of this phone?"),
        ]
    )
    return pros_template.format_prompt(features=features)

def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a tech reviewer who helps consumers make smart decisions."),
            ("human", "Considering these features: {features}, what potential downsides or limitations should buyers be aware of?"),
        ]
    )
    return cons_template.format_prompt(features=features)

def combine_pros_cons(pros, cons):
    return f"Top Strengths:\n{pros}\n\nThings to Consider:\n{cons}"

pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name": "Samsung Galaxy S24"})

print(result)
