import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a thank you note for this positive feedback: {feedback}."),
])

negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a response addressing this negative feedback: {feedback}."),
])

neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a request for more details for this neutral feedback: {feedback}."),
])

escalate_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a message to escalate this feedback to a human agent: {feedback}."),
])

classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
])

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()
chain = classification_chain | branches

def handle_feedback(feedback: str) -> str:
    """
    Classifies the sentiment of the feedback and generates an appropriate response.

    Args:
        feedback (str): The user-submitted feedback text.

    Returns:
        str: A generated response based on the classified sentiment.
    """
    return chain.invoke({"feedback": feedback})


if __name__ == "__main__":
    #  review = "The product is terrible. It broke after just one use and the quality is very poor."
    review = "The product is okay. It works as expected but nothing exceptional." # neutral
    result = handle_feedback(review)
    print(result)
