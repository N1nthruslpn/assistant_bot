import logging
from typing import Dict, Any, List, Callable, Coroutine

from google_calendar_api import GoogleCalendarAPI # Импортируем наш API для Google Календаря

logger = logging.getLogger(__name__)

class ToolManager:
    """
    Управляет доступными инструментами (функциями), которые может вызывать ИИ.
    """
    def __init__(self, google_calendar_api: GoogleCalendarAPI):
        self.google_calendar_api = google_calendar_api
        # Словарь для сопоставления имен функций из Gemini с методами класса
        self._available_tools: Dict[str, Callable[..., Coroutine[Any, Any, str]]] = {
            "create_calendar_event": self.google_calendar_api.create_event,
            "list_calendar_events": self.google_calendar_api.list_events,
            "delete_calendar_event": self.google_calendar_api.delete_event, # Добавляем новую функцию
        }

        # Схема инструментов, которую мы передаем в Gemini API
        # Это позволяет Gemini знать, какие функции доступны и как их вызывать.
        self._tools_schema: List[Dict[str, Any]] = [
            {
                "function_declarations": [
                    {
                        "name": "create_calendar_event",
                        "description": "Создает новое событие в Google Календаре пользователя. Требует название события, время начала и время окончания. Время должно быть в формате ISO 8601 (например, '2025-07-15T10:00:00+03:00').",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "summary": { "type": "STRING", "description": "Название или краткое описание события." },
                                "start_time": { "type": "STRING", "description": "Время начала события в формате ISO 8601." },
                                "end_time": { "type": "STRING", "description": "Время окончания события в формате ISO 8601." },
                                "description": { "type": "STRING", "description": "Подробное описание события (необязательно)." },
                                "location": { "type": "STRING", "description": "Место проведения события (необязательно)." }
                            },
                            "required": ["summary", "start_time", "end_time"]
                        }
                    },
                    {
                        "name": "list_calendar_events",
                        "description": "Получает список предстоящих событий из Google Календаря пользователя. Можно указать количество событий, временной диапазон или ключевое слово для фильтрации по названию события. Время должно быть в формате ISO 8601 (например, '2025-07-15T00:00:00Z'). При получении списка событий также возвращается их ID, который может быть использован для удаления.",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "max_results": { "type": "INTEGER", "description": "Максимальное количество событий для получения (по умолчанию 10)." },
                                "time_min": { "type": "STRING", "description": "Минимальное время начала события в формате ISO 8601." },
                                "time_max": { "type": "STRING", "description": "Максимальное время начала события в формате ISO 8601." },
                                "summary_keyword": { "type": "STRING", "description": "Ключевое слово для фильтрации событий по названию." }
                            }
                        }
                    },
                    {
                        "name": "delete_calendar_event",
                        "description": "Удаляет событие из Google Календаря по его уникальному идентификатору (ID). ID события можно получить, запросив список событий. Если пользователь просит удалить событие по названию, сначала используй list_calendar_events с параметром summary_keyword, чтобы найти ID, а затем уточни у пользователя, какое именно событие удалить, если найдено несколько совпадений, или запроси подтверждение, если найдено одно.",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "event_id": { "type": "STRING", "description": "Уникальный идентификатор события, которое нужно удалить." }
                            },
                            "required": ["event_id"]
                        }
                    }
                ]
            }
        ]

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Возвращает схему доступных инструментов для Gemini API."""
        return self._tools_schema

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        Выполняет указанный инструмент с переданными аргументами.
        """
        if tool_name in self._available_tools:
            try:
                result = await self._available_tools[tool_name](**kwargs)
                logger.info(f"Инструмент '{tool_name}' успешно выполнен. Результат: {result[:100]}...")
                return result
            except Exception as e:
                logger.error(f"Ошибка при выполнении инструмента '{tool_name}': {e}", exc_info=True)
                return f"Ошибка при выполнении инструмента '{tool_name}': {e}"
        else:
            logger.warning(f"Попытка выполнить неизвестный инструмент: {tool_name}")
            return f"Ошибка: Инструмент '{tool_name}' не найден."

