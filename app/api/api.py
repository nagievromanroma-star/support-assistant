import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from ..core.assistant import SupportAssistant
from ..core.knowledge_manager import KnowledgeBaseManager

logger = logging.getLogger(__name__)

class ChatwootWebhook(BaseModel):
    event: str
    account_id: int
    conversation: Dict[str, Any]
    message: Optional[Dict[str, Any]] = None

class KnowledgeBaseReload(BaseModel):
    force: bool = False

class SupportAssistantAPI:
    def __init__(self, assistant: SupportAssistant, kb_manager: KnowledgeBaseManager):
        self.assistant = assistant
        self.kb_manager = kb_manager

        self.app = FastAPI(
            title="Support Assistant API",
            description="AI-powered support assistant integrated with Chatwoot",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        self._setup_routes()

        logger.info("FastAPI приложение инициализировано")

    def _setup_routes(self):
        @self.app.get("/")
        async def root():
            return {
                "message": "Support Assistant API is running",
                "version": "1.0.0",
                "endpoints": {
                    "docs": "/docs",
                    "health": "/health",
                    "webhook": "/webhook/chatwoot",
                    "kb_reload": "/kb/reload"
                }
            }

        @self.app.get("/health")
        async def health():
            try:
                health_status = await self.assistant.health_check()
                return {
                    "status": "healthy",
                    "service": "support-assistant",
                    "components": health_status
                }
            except Exception as e:
                logger.error(f"Ошибка проверки здоровья: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/webhook/chatwoot")
        async def chatwoot_webhook(webhook: ChatwootWebhook, background_tasks: BackgroundTasks, request: Request):
            try:
                logger.info(f"Получен вебхук: {webhook.event}")

                webhook_data = await request.json()
                logger.debug(f"Данные вебхука: {str(webhook_data)[:500]}...")

                if webhook.event == "message_created" and webhook.message:
                    messagetype = webhook.message.get("message_type", "")

                    if messagetype == "outgoing":
                        logger.info("Игнорируем исходящее сообщение от оператора")
                        return {"status": "ignored", "reason": "outgoing_message"}

                    if webhook.message.get("sender", {}).get("type") == "agent_bot":
                        logger.info("Игнорируем сообщение от бота")
                        return {"status": "ignored", "reason": "bot_message"}

                    conversation_id = webhook.conversation.get("id")
                    message_content = webhook.message.get("content", "")

                    if conversation_id and message_content:
                        logger.info(f"Обработка сообщения в беседе {conversation_id}: '{message_content[:50]}...'")


                        background_tasks.add_task(
                            self.assistant.process_message,
                            conversation_id,
                            message_content
                        )

                        logger.info(f"Задача обработки сообщения добавлена для беседы {conversation_id}")
                        return {"status": "processing", "conversation_id": conversation_id}
                    else:
                        logger.warning("Недостаточно данных в вебхуке")
                        return {"status": "skipped", "reason": "insufficient_data"}
                else:
                    logger.info(f"Игнорируем событие: {webhook.event}")
                    return {"status": "ignored", "reason": f"event_{webhook.event}"}
            except Exception as e:
                logger.error(f"Ошибка обработки вебхука: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.post("/kb/reload")
        async def reload_knowledge_base(reload_data: KnowledgeBaseReload = None):
            try:
                logger.info("Запуск перезагрузки базы знаний...")


                await self.kb_manager.initialize_knowledge_base()


                logger.info("База знаний успешно перезагружена")
                return {
                    "status": "success",
                    "message": "Knowledge base reloaded successfully"
                }
            except Exception as e:
                logger.error(f"Ошибка перезагрузки БЗ: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/kb/info")
        async def get_knowledge_base_info():
            try:
                info = self.kb_manager.get_knowledge_base_info()
                return {
                    "status": "success",
                    "data": info
                }
            except Exception as e:
                logger.error(f"Ошибка получения информации о БЗ: {e}")
                raise HTTPException(status_code=500, detail=str(e))


        @self.app.get("/config")
        async def get_config():
            return {
                "top_k": self.assistant.top_k,
                "private_messages": self.assistant.private,
                "status": "active"
            }

    def get_app(self):
        return self.app
