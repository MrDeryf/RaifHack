import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI
from tqdm import tqdm

from langchain_core.vectorstores import InMemoryVectorStore

# Подключаем все переменные из окружения
load_dotenv()
# Подключаем ключ для LLM-модели
LLM_API_KEY = os.getenv("LLM_API_KEY")
# Подключаем ключ для EMBEDDER-модели
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
)

data_path = "./train_data.csv"


print("Загрузка тренировочных данных...")

try:
    # Чтение CSV с обработкой кавычек
    df = pd.read_csv(data_path, quotechar='"', delimiter=",")
    df = df.iloc[:10, :]
except Exception as e:
    print(f"Ошибка чтения CSV: {e}")

# Создание документов LangChain
processed_docs = []
for _, row in df.iterrows():
    # Объединяем аннотацию и текст для лучшего контекста
    content = f"Тема: {row.get('annotation', '')}\n\n{row.get('text', '')}"
    metadata = {
        "id": row.get("id", ""),
        "tags": row.get("tags", ""),
        "source": "bank_knowledge_base",
    }
    doc = Document(page_content=content, metadata=metadata)
    processed_docs.append(doc)

# Разбиение на чанки
doc_splits = text_splitter.split_documents(processed_docs)
documents = [doc.page_content for doc in doc_splits]
documents_with_metadata = doc_splits

print(f"Обработано {len(documents)} текстовых фрагментов")


embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://ai-for-finance-hack.up.railway.app/",
    api_key=EMBEDDER_API_KEY,
)

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=embedder
)

retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Ищи информацию для разных финансовых вопросов",
)


llm_model = ChatOpenAI(
    api_key=LLM_API_KEY,
    base_url="https://ai-for-finance-hack.up.railway.app/",
    model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
)
response_model = llm_model


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
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


for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Чем отличаются госгарантии по деньгам на эскроу от гарантий по накоплениям в НПФ?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")
