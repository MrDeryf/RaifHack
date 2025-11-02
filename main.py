import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import faiss
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool

# Подключаем все переменные из окружения
load_dotenv()
# Подключаем ключ для LLM-модели
LLM_API_KEY = os.getenv("LLM_API_KEY")
# Подключаем ключ для EMBEDDER-модели
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")


class BankAssistantRAG:
    def __init__(
        self,
        llm_model="openrouter/mistralai/mistral-small-3.2-24b-instruct:free",
        embedder="text-embedding-3-small",
    ):
        self.llm_model = llm_model
        self.embedder = embedder
        self.index = None
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
        )
        self.initialize_embedder()

    def load_and_process_data(
        self,
        data_path="./train_data.csv",
    ):
        """Загрузка и обработка банковских данных"""
        print("Загрузка тренировочных данных...")

        try:
            # Чтение CSV с обработкой кавычек
            df = pd.read_csv(data_path, quotechar='"', delimiter=",")
        except Exception as e:
            print(f"Ошибка чтения CSV: {e}. Использую альтернативный метод...")
            # df = self._alternative_csv_reading(data_path)

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
        doc_splits = self.text_splitter.split_documents(processed_docs)
        self.documents = [doc.page_content for doc in doc_splits]
        self.documents_with_metadata = doc_splits  # Сохраняем документы с метаданными

        # print(f"Обработано {len(self.documents)} текстовых фрагментов")
        return self.documents

    def initialize_embedder(self):
        """Инициализация эмбеддера через OpenAI API"""
        print("Инициализация эмбеддинг модели...")
        self.embedder_client = OpenAI(
            base_url="https://ai-for-finance-hack.up.railway.app/",
            api_key=EMBEDDER_API_KEY,
        )

    def get_embeddings(self, texts):
        """Получение эмбеддингов через API"""
        embeddings = []
        for text in tqdm(texts, desc="Генерация эмбеддингов"):
            try:
                response = self.embedder_client.embeddings.create(
                    model=self.embedder, input=text
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            except Exception as e:
                print(f"Ошибка получения эмбеддинга: {e}")
                embeddings.append([0] * 1536)

        return np.array(embeddings)

    def build_knowledge_base(self, documents):
        """Построение векторной базы знаний"""
        print("Построение векторной базы знаний...")

        embeddings = self.get_embeddings(documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype("float32"))

        print(f"Векторная база знаний построена: {self.index.ntotal} векторов")

    def search_similar_documents(self, query, k=5):
        """Поиск релевантных документов"""
        if self.index is None:
            raise ValueError("Векторная база знаний не построена")

        query_embedding = self.get_embeddings([query])[0]
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append(
                    {
                        "content": self.documents[idx],
                        "similarity": distance,
                        "rank": i + 1,
                    }
                )

        return results

    def create_retriever_tool(self):
        """Создание инструмента для поиска в стиле LangChain"""

        # Создаем кастомный retriever
        class CustomRetriever:
            def __init__(self, rag_system):
                self.rag_system = rag_system

            def invoke(self, query):
                results = self.rag_system.search_similar_documents(query, k=3)
                # Конвертируем в формат LangChain Document
                docs = []
                for result in results:
                    docs.append(
                        Document(
                            page_content=result["content"],
                            metadata={
                                "similarity": result["similarity"],
                                "rank": result["rank"],
                            },
                        )
                    )
                return docs

        custom_retriever = CustomRetriever(self)

        # Создаем инструмент
        retriever_tool = create_retriever_tool(
            custom_retriever,
            "bank_knowledge_search",
            "Поиск информации о банковских продуктах, кредитах, вкладах, ипотеке и финансовых услугах.",
        )

        return retriever_tool


# # Функция для генерации ответа по заданному вопросу, вы можете изменять ее в процессе работы, однако
# # просим оставить структуру обращения, т.к. при запуске на сервере, потребуется корректно указанный путь
# # для формирования ответов. Также не вставляйте ключ вручную, поскольку при запуске ключ подтянется автоматически
# def answer_generation(question):
#     # Подключаемся к модели
#     client = OpenAI(
#         # Базовый url - сохранять без изменения
#         base_url="https://ai-for-finance-hack.up.railway.app/",
#         # Указываем наш ключ, полученный ранее
#         api_key=LLM_API_KEY,
#     )
#     # Формируем запрос к клиенту
#     response = client.chat.completions.create(
#         # Выбираем любую допступную модель из предоставленного списка
#         model="openrouter/mistralai/mistral-small-3.2-24b-instruct:free",
#         # Формируем сообщение
#         messages=[
#             {
#                 "role": "user",
#                 "content": [{"type": "text", "text": f"Ответь на вопрос: {question}"}],
#             }
#         ],
#     )
#     # Формируем ответ на запрос и возвращаем его в результате работы функции
#     return response.choices[0].message.content


# # Блок кода для запуска. Пожалуйста оставляйте его в самом низу вашего скрипта,
# # при необходимости добавить код - опишите функции выше и вставьте их вызов в блок после if
# # в том порядке, в котором они нужны для запуска решения, пути к файлам оставьте неизменными.
# if __name__ == "__main__":
#     # Считываем список вопросов
#     questions = pd.read_csv("./questions.csv")
#     # Выделяем список вопросов
#     questions_list = questions["Вопрос"].tolist()
#     # Создаем список для хранения ответов
#     answer_list = []
#     # Проходимся по списку вопросов
#     for current_question in tqdm(questions_list, desc="Генерация ответов"):
#         # Отправляем запрос на генерацию ответа
#         answer = answer_generation(question=current_question)
#         # Добавляем ответ в список
#         answer_list.append(answer)
#     # Добавляем в данные список ответов
#     questions["Ответы на вопрос"] = answer_list
#     # Сохраняем submission
#     questions.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    assistant = BankAssistantRAG()

    # Загрузка и обработка данных
    processed_docs = assistant.load_and_process_data("./train_data.csv")

    # Построение векторного хранилища
    assistant.build_knowledge_base(processed_docs)

    # # Создание инструмента поиска (для возможного использования в агентах)
    # retrieval_tool = assistant.create_retrieval_tool()
    query = "Как просрочка по «беспроцентному» займу скажется на переплате/ПСК?"
    results = assistant.search_similar_documents(query=query, k=5)
    for i, result in enumerate(results):
        print(f"Результат {i + 1}:")
        print(f"  Сходство: {result['similarity']:.3f}")
        print(f"  Содержимое: {result['content'][:200]}...")
