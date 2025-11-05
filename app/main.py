import logging
import asyncio
from loguru import logger
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.core.embedder import Embedder
from app.clients.qdrant_client import QdrantClientWrapper
from app.clients.chatwoot_client import ChatwootClient
from app.core.knowledge_manager import KnowledgeBaseManager
from app.core.assistant import SupportAssistant
from app.api.api import SupportAssistantAPI

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )

    logger.add(
        "/var/log/support-assistant/app.log",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        level="INFO"
    )

async def create_app():
    logger.info("Запуск инициализации Support Assistant...")

    try:
        logger.info("Инициализация эмбеддера...")
        embedder = Embedder(model_name=settings.embedder_model)

        model_info = embedder.get_model_info()
        logger.info(f"Модель эмбеддингов: {model_info['model_name']}")
        logger.info(f"Размерность векторов: {model_info['embedding_dimension']}")

        logger.info("Инициализация Qdrant клиента...")
        qdrant_client = QdrantClientWrapper(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name="support_kb"
        )

        logger.info("Инициализация Chatwoot клиента...")
        chatwoot_client = ChatwootClient(
            base_url=settings.chatwoot_base_url,
            api_token=settings.chatwoot_api_token,
            account_id=settings.chatwoot_account_id
        )

        chatwoot_healthy = await chatwoot_client.health_check()
        if chatwoot_healthy:
            logger.info("Подключение к Chatwoot успешно")
        else:
            logger.warning("Chatwoot недоступен, проверьте настройки")

        logger.info("Инициализация менеджера базы знаний...")
        kb_manager = KnowledgeBaseManager(
            qdrant_client=qdrant_client,
            embedder=embedder,
            source_path=settings.knowledge_base_path
        )

        logger.info("Загрузка базы знаний...")
        await kb_manager.initialize_knowledge_base()
        logger.info("База знаний успешно загружена")


        logger.info("Инициализация AI-ассистента...")
        assistant = SupportAssistant(
            qdrant_client=qdrant_client,
            chatwoot_client=chatwoot_client,
            embedder=embedder,
            top_k=3,
            private=True
        )

        logger.info("Создание FastAPI приложения...")
        api = SupportAssistantAPI(assistant=assistant, kb_manager=kb_manager)
        app = api.get_app()

        logger.info("Support Assistant успешно инициализирован!")
        logger.info(f"API будет доступно по адресу: http://{settings.api_host}:{settings.api_port}")
        logger.info(f"Документация API: http://{settings.api_host}:{settings.api_port}/docs")

        return app

    except Exception as e:
        logger.error(f"Критическая ошибка инициализации: {e}")
        raise

if __name__ == "__main__":
    import uvicorn

    setup_logging()

    logger.info("Запуск Support Assistant...")

    try:
        uvicorn.run(
            "app.main:create_app",
            host=settings.api_host,
            port=settings.api_port,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Ошибка запуска приложения: {e}")
        sys.exit(1)
