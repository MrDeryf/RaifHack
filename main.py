import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import faiss
import re
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS

# Подключаем все переменные из окружения
load_dotenv()
# Подключаем ключ для LLM-модели
LLM_API_KEY = os.getenv("LLM_API_KEY")
# Подключаем ключ для EMBEDDER-модели
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://ai-for-finance-hack.up.railway.app/",
    api_key=EMBEDDER_API_KEY,
)

vectorstore = FAISS.load_local(
    "faiss_store", embedder, allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_bank_info",
    "Ищи информацию для разных финансовых вопросов",
)


llm_model = ChatOpenAI(
    api_key=LLM_API_KEY,
    base_url="https://ai-for-finance-hack.up.railway.app/",
    model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
)
response_model = llm_model


def generate_query_or_respond(state: MessagesState):
    response = response_model.bind_tools(
        [retriever_tool],
        tool_choice="retrieve_bank_info",
    ).invoke(state["messages"])
    return {"messages": [response]}


GENERATE_PROMPT = """
Ты - AI-ассистент банка. Ответь на вопрос клиента используя предоставленную информацию.

{context}

Вопрос: {question}

Инструкции:
1. Ответь строго на основе предоставленной информации
2. Если информации недостаточно, честно скажи об этом
3. Будь точным и полезным
4. Форматируй ответ для лучшей читаемости
5. Не упоминай что используешь базу знаний или документы

Ответ:"""


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


if __name__ == "__main__":
    workflow = StateGraph(MessagesState)

    # Define the nodes we will cycle between
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    graph = workflow.compile()

    # from IPython.display import Image, display

    # display(Image(graph.get_graph().draw_mermaid_png()))

    for chunk in graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Как просрочка по «беспроцентному» займу скажется на переплате/ПСК?",
                }
            ]
        }
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")
