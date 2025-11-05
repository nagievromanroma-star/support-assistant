import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import asyncio

from .embedder import Embedder
from ..clients.qdrant_client import QdrantClientWrapper

logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    def __init__(self, qdrant_client: QdrantClientWrapper, embedder: Embedder, source_path: str = "./data/knowledge_base.csv"):
        self.qdrant_client = qdrant_client
        self.embedder = embedder
        self.source_path = Path(source_path)
        
        logger.info(f"Менеджер базы знаний инициализирован. Источник: {source_path}")
    
    def load_knowledge_base(self) -> pd.DataFrame:
        try:
            if not self.source_path.exists():
                logger.error(f"Файл базы знаний не найден: {self.source_path}")
                raise FileNotFoundError(f"Файл базы знаний не найден: {self.source_path}")
            
            df = pd.read_csv(self.source_path)
            
            required_columns = ['question', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"В файле отсутствуют обязательные колонки: {missing_columns}")
                raise ValueError(f"Отсутствуют колонки: {missing_columns}")
            
            if 'category' not in df.columns:
                df['category'] = 'general'
            else:
                df['category'] = df['category'].fillna('general')
            
            logger.info(f"Загружено {len(df)} записей из базы знаний")
            logger.debug(f"Колонки в данных: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка загрузки базы знаний: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
        texts = []
        payloads = []
        
        logger.info("Подготовка данных для векторизации...")
        
        for index, row in df.iterrows():
            try:
                question = str(row.get('question', '')).strip()
                answer = str(row.get('answer', '')).strip()
                category = str(row.get('category', 'general')).strip()
                
                text = f"Вопрос: {question} Ответ: {answer}"
                texts.append(text)
                
                payload = {
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "original_text": text,
                    "index": index
                }
                payloads.append(payload)
                
            except Exception as e:
                logger.warning(f"Ошибка обработки строки {index}: {e}")
                continue
        
        logger.info(f"Подготовлено {len(texts)} текстов для векторизации")
        return texts, payloads
    
    async def initialize_knowledge_base(self):
        try:
            logger.info("Начало инициализации базы знаний...")
            
            df = self.load_knowledge_base()
            texts, payloads = self.prepare_data(df)
            
            if not texts:
                logger.error("Нет данных для загрузки в базу знаний")
                return
            
            logger.info("Создание эмбеддингов...")
            embeddings = self.embedder.embed_texts(texts)
            
            vector_size = len(embeddings[0]) if embeddings else 384
            logger.info(f"Создание коллекции с размерностью {vector_size}...")
            self.qdrant_client.create_collection(vector_size)
            
            logger.info("Загрузка данных в Qdrant...")
            self.qdrant_client.add_points(embeddings, payloads)
            
            logger.info("База знаний успешно инициализирована в Qdrant!")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации базы знаний: {e}")
            raise
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        try:
            df = self.load_knowledge_base()
            
            info = {
                "total_entries": len(df),
                "categories": df['category'].value_counts().to_dict() if 'category' in df.columns else {},
                "source_file": str(self.source_path),
                "file_exists": self.source_path.exists()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о БЗ: {e}")
            return {"error": str(e)}
