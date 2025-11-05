#!/usr/bin/env python3

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.core.embedder import Embedder
from app.clients.qdrant_client import QdrantClientWrapper
from app.core.knowledge_manager import KnowledgeBaseManager

async def main():
    print("Запуск инициализации базы знаний AI-брокера...")

    try:
        print("Инициализация эмбеддера...")
        embedder = Embedder(model_name=settings.embedder_model)

        print("Инициализация Qdrant клиента...")
        qdrant_client = QdrantClientWrapper(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )

        print("Инициализация менеджера базы знаний...")
        kb_manager = KnowledgeBaseManager(
            qdrant_client=qdrant_client,
            embedder=embedder,
            source_path=settings.knowledge_base_path
        )

        print("Загрузка базы знаний AI-брокера...")
        await kb_manager.initialize_knowledge_base()

        print("База знаний AI-брокера успешно инициализирована!")

        info = kb_manager.get_knowledge_base_info()
        print(f"Загружено финансовых вопросов: {info['total_entries']}")
        print(f"Категории: {info['categories']}")
        print("Темы: акции, облигации, ETF, ИИС, налоги, дивиденды")

    except FileNotFoundError:
        print(f"Ошибка: Файл базы знаний не найден: {settings.knowledge_base_path}")
        print("Создайте файл data/knowledge_base.csv с финансовыми вопросами и ответами")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
