import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ChatwootClient:
    def __init__(self, base_url: str, api_token: str, account_id: int):
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.account_id = account_id
        
        self.headers = {
            "Content-Type": "application/json",
            "api_access_token": self.api_token
        }
        
        logger.info(f"Chatwoot клиент инициализирован для {self.base_url}")
    
    async def send_message(self, conversation_id: int, message: str, private: bool = True) -> bool:
        url = f"{self.base_url}/api/v1/accounts/{self.account_id}/conversations/{conversation_id}/messages"
        
        data = {
            "content": message,
            "message_type": "outgoing",
            "private": private
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=self.headers)
                
            if response.status_code == 200:
                message_type = "приватное" if private else "публичное"
                logger.info(f"{message_type} сообщение отправлено в беседу {conversation_id}")
                return True
            else:
                logger.error(f"Ошибка отправки сообщения: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения: {e}")
            return False
    
    async def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/api/v1/accounts/{self.account_id}/conversations/{conversation_id}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers)
                
            if response.status_code == 200:
                conversation_data = response.json()
                logger.debug(f"Получена информация о беседе {conversation_id}")
                return conversation_data
            else:
                logger.error(f"Ошибка получения беседы: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при получении беседы: {e}")
            return None
    
    async def create_private_note(self, conversation_id: int, note: str) -> bool:
        url = f"{self.base_url}/api/v1/accounts/{self.account_id}/conversations/{conversation_id}/messages"
        
        data = {
            "content": note,
            "message_type": "outgoing",
            "private": True,
            "content_type": "text",
            "content_attributes": {}
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=self.headers)
                
            if response.status_code == 200:
                logger.info(f"Приватная заметка добавлена в беседу {conversation_id}")
                return True
            else:
                logger.error(f"Ошибка создания приватной заметки: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при создании приватной заметки: {e}")
            return False
    
    async def health_check(self) -> bool:
        url = f"{self.base_url}/api/v1/accounts/{self.account_id}/dashboard"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers)
                
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Chatwoot API недоступно: {e}")
            return False
