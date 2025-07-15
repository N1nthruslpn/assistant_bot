import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pytz # Импортируем pytz для работы с часовыми поясами

from config import BotConfig # Импортируем BotConfig

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Управляет историей диалогов пользователей.
    В текущей реализации хранит историю в памяти.
    Для масштабирования может быть заменен на класс,
    работающий с базой данных (например, Redis, PostgreSQL).
    """
    def __init__(self, config: BotConfig):
        self.config = config
        self.user_conversations: Dict[int, Dict[str, Any]] = {}
        self.cleanup_task: asyncio.Task = None

        # Системный промпт будет генерироваться динамически
        self.base_system_prompt_text: str = (
            'Ты — дружелюбный и полезный ИИ-ассистент, разработанный Google. '
            'Твоя цель — отвечать на вопросы пользователей и вести с ними диалог, '
            'сохраняя контекст беседы. Ты также можешь взаимодействовать с Google Календарем пользователя '
            'для создания, просмотра и УДАЛЕНИЯ событий. Отвечай на русском языке. '
            'Используй доступные инструменты, когда это уместно. '
            'Если пользователь просит создать событие, убедись, что у тебя есть название, время начала и время окончания. '
            'Если время не указано, попроси его уточнить. '
            'Если пользователь просит список событий, ты можешь запросить количество или временной диапазон. '
            'При запросе на удаление события, если пользователь указывает название события, сначала используй инструмент `list_calendar_events` с параметром `summary_keyword` для поиска событий по этому названию. '
            'Если найдены одно или несколько событий, выведи их список с ID и попроси пользователя подтвердить, какое именно событие он хочет удалить, указав его ID. '
            'Если найдено только одно событие, спроси подтверждение на его удаление, указав ID. '
            'Если событий по названию не найдено, сообщи об этом пользователю. '
            'Всегда уточняй временные рамки для событий, если они не указаны явно. '
            'Используй формат ISO 8601 для времени.'
            'Всегда используй текущую дату и время, предоставленные в этом промпте, для определения относительного времени (например, "сегодня", "завтра").'
            'Твой часовой пояс по умолчанию для всех операций с календарем - Europe/Moscow.'
            'Помимо работы с календарем, ты также можешь просто общаться с пользователем, давать советы, предлагать идеи для досуга или занятий. '
            'Если ты предлагаешь какое-либо занятие, и пользователь соглашается, предложи сразу же добавить его в календарь, уточнив время.'
        )
        # Ответ модели на системный промпт, чтобы диалог начался корректно
        self.system_prompt_response: Dict[str, Any] = {
            'role': 'model',
            'parts': [{'text': 'Привет! Я готов к общению.'}]
        }

    def _generate_system_prompt(self) -> Dict[str, Any]:
        """
        Генерирует системный промпт с текущей датой и временем.
        """
        current_time_utc = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        
        # Получаем текущее время в часовом поясе Москвы
        moscow_tz = pytz.timezone('Europe/Moscow')
        current_time_moscow = datetime.now(moscow_tz).isoformat(timespec='seconds') # Используем isoformat для МСК

        prompt_with_time = (
            f"Текущая дата и время (UTC): {current_time_utc}. "
            f"Текущая дата и время (Moscow Time): {current_time_moscow}. "
            f"{self.base_system_prompt_text}"
        )
        return {
            'role': 'user',
            'parts': [{'text': prompt_with_time}]
        }

    def _get_initial_history(self) -> List[Dict[str, Any]]:
        """
        Возвращает начальную историю диалога, включающую системный промпт
        с текущим временем и его ожидаемый ответ от модели.
        """
        return [self._generate_system_prompt(), self.system_prompt_response]

    def get_history(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Возвращает историю диалога для пользователя.
        Инициализирует новую историю, если её нет или она устарела.
        Обновляет системный промпт текущим временем.
        """
        current_time = datetime.now()
        if user_id not in self.user_conversations:
            logger.info(f"Инициализирована новая история для пользователя {user_id}, так как её не было.")
            self.user_conversations[user_id] = {
                'history': self._get_initial_history(),
                'last_active': current_time
            }
        else:
            last_active_time = self.user_conversations[user_id]['last_active']
            if current_time - last_active_time > timedelta(minutes=self.config.conversation_lifetime_minutes):
                logger.info(f"Диалог с пользователем {user_id} сброшен из-за неактивности (более {self.config.conversation_lifetime_minutes} минут).")
                self.user_conversations[user_id]['history'] = self._get_initial_history()
                # Сообщение о сбросе будет отправлено ботом, а не менеджером
            else:
                # Обновляем системный промпт в существующей истории, чтобы время было актуальным
                self.user_conversations[user_id]['history'][0] = self._generate_system_prompt()
        
        self.user_conversations[user_id]['last_active'] = current_time
        return self.user_conversations[user_id]['history']

    def update_history(self, user_id: int, message: Dict[str, Any]):
        """
        Добавляет сообщение в историю диалога пользователя.
        """
        if user_id in self.user_conversations:
            self.user_conversations[user_id]['history'].append(message)
            self.user_conversations[user_id]['last_active'] = datetime.now()
        else:
            logger.warning(f"Попытка обновить историю для несуществующего пользователя {user_id}. Инициализация новой истории.")
            self.user_conversations[user_id] = {
                'history': self._get_initial_history() + [message],
                'last_active': datetime.now()
            }

    def reset_history(self, user_id: int):
        """
        Сбрасывает историю диалога для указанного пользователя.
        """
        self.user_conversations[user_id] = {
            'history': self._get_initial_history(),
            'last_active': datetime.now()
        }
        logger.info(f"Диалог с пользователем {user_id} сброшен.")

    async def start_cleanup_task(self):
        """Запускает фоновую задачу по очистке старых диалогов."""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_old_conversations_loop())
            logger.info("Фоновая задача очистки старых диалогов запущена.")

    async def stop_cleanup_task(self):
        """Останавливает фоновую задачу по очистке старых диалогов."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                logger.info("Фоновая задача очистки успешно отменена.")

    async def _cleanup_old_conversations_loop(self):
        """
        Цикл для периодической очистки устаревших диалогов.
        """
        while True:
            await asyncio.sleep(self.config.cleanup_interval_seconds)
            current_time = datetime.now()
            users_to_remove = []

            for user_id, data in list(self.user_conversations.items()): # Итерируем по копии, чтобы безопасно удалять
                last_active_time = data['last_active']
                if current_time - last_active_time > timedelta(minutes=self.config.conversation_lifetime_minutes):
                    users_to_remove.append(user_id)

            for user_id in users_to_remove:
                del self.user_conversations[user_id]
                logger.info(f"Диалог с пользователем {user_id} удален из памяти из-за неактивности.")

