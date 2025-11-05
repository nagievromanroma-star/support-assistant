import logging
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-ru"):
        self.model_name = model_name
        logger.info(f" Загрузка модели эмбеддингов: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    def embed_text(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(text)
            embedding_list = embedding.tolist()
            logger.debug(f"Создан эмбеддинг для текста: '{text[:50]}...'")
            return embedding_list
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддинга: {e}")
            raise
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts)
            embeddings_list = embeddings.tolist()
            logger.info(f"Создано {len(embeddings_list)} эмбеддингов")
            return embeddings_list
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддингов: {e}")
            raise

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.model.max_seq_length
        }
