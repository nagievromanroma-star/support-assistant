import logging
from typing import List, Dict, Any
from .embedder import Embedder
from ..clients.qdrant_client import QdrantClientWrapper
from ..clients.chatwoot_client import ChatwootClient

logger = logging.getLogger(__name__)

class SupportAssistant:
    def __init__(
        self,
        qdrant_client: QdrantClientWrapper,
        chatwoot_client: ChatwootClient,
        embedder: Embedder,
        top_k: int = 3,
        private: bool = True
    ):
        self.qdrant_client = qdrant_client
        self.chatwoot_client = chatwoot_client
        self.embedder = embedder
        self.top_k = top_k
        self.private = private
        
        logger.info(f"AI-ассистент инициализирован (top_k: {top_k}, private: {private})")

    async def process_message(self, conversation_id: int, message_text: str) -> bool:
        try:
            logger.info(f"Обработка сообщения в беседе {conversation_id}: '{message_text}'")
            
            query_embedding = self.embedder.embed_text(message_text)
            logger.debug("Эмбеддинг запроса создан")
            
            search_results = self.qdrant_client.search(query_embedding, self.top_k)
            logger.debug(f"Найдено {len(search_results)} релевантных ответов")
            
            if not search_results:
                response = self._format_no_results_response(message_text)
            else:
                response = self._format_response(search_results, message_text)

            success = await self.chatwoot_client.send_message(
                conversation_id=conversation_id,
                message=response,
                private=self.private
            )

            if success:
                logger.info(f"Ответ успешно отправлен в беседу {conversation_id}")
            else:
                logger.error(f"Ошибка отправки ответа в беседу {conversation_id}")

            return success

        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {e}")
            return False

    def _format_response(self, search_results: List[Dict[str, Any]], original_question: str) -> str:
        try:
            response_parts = [
                "Автоматический помощник нашел следующие ответы:",
                f"Ваш вопрос: \"{original_question}\"",
                "",
                "---",
                ""
            ]

            for i, result in enumerate(search_results, 1):
                payload = result["payload"]
                score = result["score"]

                response_parts.extend([
                    f"{i}. {payload.get('question', 'Вопрос')}",
                    f"   Ответ: {payload.get('answer', '')}",
                    f"   Категория: {payload.get('category', 'general')}",
                    f"   Релевантность: {score:.2f}",
                    ""
                ])

            response_parts.extend([
                "---",
                "",
                "Если эти ответы не решают вашу проблему, ожидайте ответа оператора."
            ])

            formatted_response = "\n".join(response_parts)
            logger.debug("Ответ отформатирован")
            return formatted_response

        except Exception as e:
            logger.error(f"Ошибка форматирования ответа: {e}")
            return "Произошла ошибка при поиске ответа. Ожидайте ответа оператора."

    def _format_no_results_response(self, original_question: str) -> str:
        response = (
            "Автоматический помощник\n\n"
            f"Ваш вопрос: \"{original_question}\"\n\n"
            "К сожалению, я не нашел подходящего ответа в базе знаний.\n\n"
            "Ожидайте ответа оператора, который поможет решить вашу проблему."
        )

        logger.info("Ответы не найдены в базе знаний")
        return response

    async def health_check(self) -> Dict[str, Any]:
        health_status = {
            "assistant": "operational",
            "components": {}
        }

        try:
            qdrant_health = self.qdrant_client.collection_exists()
            health_status["components"]["qdrant"] = {
                "status": "operational" if qdrant_health else "down",
                "collection_exists": qdrant_health
            }

            chatwoot_health = await self.chatwoot_client.health_check()
            health_status["components"]["chatwoot"] = {
                "status": "operational" if chatwoot_health else "down",
                "api_accessible": chatwoot_health
            }

            try:
                model_info = self.embedder.get_model_info()
                health_status["components"]["embedder"] = {
                    "status": "operational",
                    "model_info": model_info
                }
            except Exception as e:
                health_status["components"]["embedder"] = {
                    "status": "down",
                    "error": str(e)
                }

            all_healthy = all(
                comp["status"] == "operational"
                for comp in health_status["components"].values()
            )
            health_status["overall"] = "healthy" if all_healthy else "degraded"

        except Exception as e:
            health_status["assistant"] = "error"
            health_status["error"] = str(e)
            logger.error(f"Ошибка проверки здоровья: {e}")

        return health_status

    def update_settings(self, top_k: int = None, private: bool = None):
        if top_k is not None:
            self.top_k = top_k
            logger.info(f"Обновлен top_k: {top_k}")

        if private is not None:
            self.private = private
            logger.info(f"Обновлен режим private: {private}")
