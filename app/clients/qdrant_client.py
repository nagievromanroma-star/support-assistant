import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
import uuid

# Настраиваем логирование
logger = logging.getLogger(__name__)

class QdrantClientWrapper:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "support_kb"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self._connect()
    def _connect(self):
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Успешное подключение к Qdrant: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {e}")
            raise
    def create_collection(self, vector_size: int = 384):
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Коллекция '{self.collection_name}' создана с размерностью {vector_size}")
        except Exception as e:
            logger.error(f"Ошибка создания коллекции: {e}")
            raise
    def add_points(self, embeddings: List[List[float]], payloads: List[Dict[str, Any]]):
        try:
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
                for embedding, payload in zip(embeddings, payloads)
            ]
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            logger.info(f" Добавлено {len(points)} точек в коллекцию '{self.collection_name}'")
            return operation_info
        except Exception as e:
            logger.error(f"Ошибка добавления точек: {e}")
            raise
    def search(self, query_embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            results = []
            for result in search_results:
                results.append({
                    "score": result.score,
                    "payload": result.payload,
                    "id": result.id
                })
            logger.info(f"Найдено {len(results)} результатов поиска")
            return results
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            raise
    def collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections()
            return any(collection.name == self.collection_name for collection in collections.collections)
        except Exception as e:
            logger.error(f"Ошибка проверки коллекции: {e}")
            return False
