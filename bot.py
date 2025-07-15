import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable, Coroutine
from dataclasses import dataclass # Используем dataclass для конфигурации

import aiohttp # Используем aiohttp для асинхронных HTTP-запросов
from aiogram import Bot, Dispatcher, types, F
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BotConfig:
    """
    Класс для хранения и валидации конфигурации бота, загружаемой из переменных окружения.
    """
    telegram_bot_token: str
    gemini_api_key: str
    conversation_lifetime_minutes: int = 30
    cleanup_interval_seconds: int = 60 * 5 # Каждые 5 минут
    gemini_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    gemini_max_retries: int = 3
    gemini_base_delay: float = 1.0 # Начальная задержка для повторных попыток в секундах
    # Максимальное количество итераций для цикла вызова инструментов
    # Предотвращает бесконечные циклы в случае некорректного поведения ИИ
    max_tool_call_iterations: int = 5

    @classmethod
    def load_from_env(cls):
        """
        Загружает конфигурацию из переменных окружения.
        """
        load_dotenv()
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        gemini_key = os.getenv("GEMINI_API_KEY")

        if not telegram_token:
            raise ValueError("Переменная окружения 'TELEGRAM_BOT_TOKEN' не установлена.")
        if not gemini_key:
            raise ValueError("Переменная окружения 'GEMINI_API_KEY' не установлена.")

        return cls(telegram_bot_token=telegram_token, gemini_api_key=gemini_key)

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

        # Системный промпт для модели Gemini
        self.system_prompt: Dict[str, Any] = {
            'role': 'user',
            'parts': [{
                'text': (
                    'Ты — дружелюбный и полезный ИИ-ассистент, разработанный Google. '
                    'Твоя цель — отвечать на вопросы пользователей и вести с ними диалог, '
                    'сохраняя контекст беседы. Ты также можешь взаимодействовать с Google Календарем пользователя '
                    'для создания и просмотра событий. Отвечай на русском языке. '
                    'Используй доступные инструменты, когда это уместно. '
                    'Если пользователь просит создать событие, убедись, что у тебя есть название, время начала и время окончания. '
                    'Если время не указано, попроси его уточнить. '
                    'Если пользователь просит список событий, ты можешь запросить количество или временной диапазон.'
                )
            }]
        }
        # Ответ модели на системный промпт, чтобы диалог начался корректно
        self.system_prompt_response: Dict[str, Any] = {
            'role': 'model',
            'parts': [{'text': 'Привет! Я готов к общению.'}]
        }

    def _get_initial_history(self) -> List[Dict[str, Any]]:
        """
        Возвращает начальную историю диалога, включающую системный промпт
        и его ожидаемый ответ от модели.
        """
        return [self.system_prompt, self.system_prompt_response]

    def get_history(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Возвращает историю диалога для пользователя.
        Инициализирует новую историю, если её нет или она устарела.
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

class ToolManager:
    """
    Управляет доступными инструментами (функциями), которые может вызывать ИИ.
    """
    def __init__(self):
        # Словарь для сопоставления имен функций из Gemini с методами класса
        # В реальном приложении здесь была бы интеграция с Google Calendar API
        # с использованием OAuth2.0 для авторизации пользователя.
        # Для пет-проекта мы просто имитируем действия.
        self._available_tools: Dict[str, Callable[..., Coroutine[Any, Any, str]]] = {
            "create_calendar_event": self._mock_create_calendar_event,
            "list_calendar_events": self._mock_list_calendar_events,
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
                        "description": "Получает список предстоящих событий из Google Календаря пользователя. Можно указать количество событий и временной диапазон. Время должно быть в формате ISO 8601 (например, '2025-07-15T00:00:00Z').",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "max_results": { "type": "INTEGER", "description": "Максимальное количество событий для получения (по умолчанию 10)." },
                                "time_min": { "type": "STRING", "description": "Минимальное время начала события в формате ISO 8601." },
                                "time_max": { "type": "STRING", "description": "Максимальное время начала события в формате ISO 8601." }
                            }
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

    # --- Имитационные функции для Google Календаря ---
    async def _mock_create_calendar_event(self, summary: str, start_time: str, end_time: str, description: str = None, location: str = None) -> str:
        """Имитирует создание события в календаре.
        Возвращает строку с результатом имитации.
        """
        logger.info(f"Имитация создания события: '{summary}' с {start_time} по {end_time}. Описание: '{description}', Место: '{location}'")
        # Здесь могла бы быть логика вызова Google Calendar API
        return f"Событие '{summary}' успешно создано в календаре (имитация). Время: с {start_time} по {end_time}."

    async def _mock_list_calendar_events(self, max_results: int = 10, time_min: str = None, time_max: str = None) -> str:
        """Имитирует получение списка событий из календаря.
        Возвращает строку с имитацией списка событий.
        """
        logger.info(f"Имитация получения {max_results} событий. Диапазон: с {time_min} по {time_max}")
        # Пример возвращаемых данных. В реальном приложении здесь были бы реальные данные из календаря.
        mock_events = [
            {"summary": "Встреча с командой", "start": "2025-07-16T10:00:00+03:00", "end": "2025-07-16T11:00:00+03:00"},
            {"summary": "Обед с коллегами", "start": "2025-07-16T13:00:00+03:00", "end": "2025-07-16T14:00:00+03:00"},
            {"summary": "Сдача проекта", "start": "2025-07-17T09:00:00+03:00", "end": "2025-07-17T17:00:00+03:00"},
            {"summary": "Созвон по новому функционалу", "start": "2025-07-18T14:30:00+03:00", "end": "2025-07-18T15:00:00+03:00"},
            {"summary": "Планирование отпуска", "start": "2025-07-20T11:00:00+03:00", "end": "2025-07-20T12:00:00+03:00"}
        ]
        
        # Фильтрация по времени (простая имитация)
        filtered_events = []
        try:
            time_min_dt = datetime.fromisoformat(time_min) if time_min else None
            time_max_dt = datetime.fromisoformat(time_max) if time_max else None
        except ValueError:
            return "Ошибка: Некорректный формат времени в запросе. Используйте ISO 8601."

        for event in mock_events:
            event_start = datetime.fromisoformat(event['start'])
            if time_min_dt and event_start < time_min_dt:
                continue
            if time_max_dt and event_start > time_max_dt:
                continue
            filtered_events.append(event)

        if filtered_events:
            # Сортируем события по времени начала
            filtered_events.sort(key=lambda x: datetime.fromisoformat(x['start']))
            
            events_str = "\n".join([f"- {e['summary']} (с {e['start']} по {e['end']})" for e in filtered_events[:max_results]])
            return f"Вот несколько предстоящих событий (имитация):\n{events_str}"
        return "Событий не найдено (имитация)."

class GeminiService:
    """
    Отвечает за взаимодействие с Gemini API, включая отправку запросов,
    обработку ответов, логику повторных попыток и обработку вызовов инструментов.
    """
    def __init__(self, config: BotConfig, http_session: aiohttp.ClientSession, tool_manager: ToolManager):
        self.config = config
        self.http_session = http_session
        self.tool_manager = tool_manager
        self.gemini_api_url: str = f"{self.config.gemini_api_base_url}?key={self.config.gemini_api_key}"

    async def get_gemini_response_with_tools(self, chat_history: List[Dict[str, Any]]) -> str:
        """
        Отправляет историю чата в Gemini API, обрабатывает вызовы инструментов
        и возвращает окончательный текстовый ответ.
        """
        gemini_response_content = None
        # Копируем историю для модификации в цикле вызова инструментов
        current_chat_history = list(chat_history)

        for iteration in range(self.config.max_tool_call_iterations):
            try:
                # Отправляем запрос к Gemini с текущей историей и доступными инструментами
                response_from_gemini = await self._call_gemini_api(current_chat_history, self.tool_manager.get_tools_schema())

                if not response_from_gemini:
                    gemini_response_content = "Извините, не удалось получить ответ от ИИ."
                    logger.error(f"Gemini API вернул пустой или некорректный ответ на итерации {iteration + 1}.")
                    break

                # Проверяем, является ли ответ вызовом функции
                if response_from_gemini.get('candidates') and \
                   response_from_gemini['candidates'][0].get('content') and \
                   response_from_gemini['candidates'][0]['content'].get('parts') and \
                   response_from_gemini['candidates'][0]['content']['parts'][0].get('function_call'):

                    function_call = response_from_gemini['candidates'][0]['content']['parts'][0]['function_call']
                    function_name = function_call['name']
                    function_args = function_call.get('args', {})

                    logger.info(f"ИИ запросил вызов функции: '{function_name}' с аргументами: {function_args} на итерации {iteration + 1}.")

                    # Добавляем вызов функции в историю для Gemini
                    current_chat_history.append({
                        'role': 'model',
                        'parts': [{'function_call': function_call}]
                    })

                    # Выполняем функцию через ToolManager
                    tool_output = await self.tool_manager.execute_tool(function_name, **function_args)
                    
                    # Добавляем результат выполнения функции в историю как tool_response
                    current_chat_history.append({
                        'role': 'function',
                        'parts': [{'function_response': {'name': function_name, 'response': {'content': tool_output}}}]
                    })
                    # Продолжаем цикл, чтобы Gemini мог обработать результат функции
                    # и сгенерировать окончательный ответ пользователю.

                elif response_from_gemini.get('candidates') and \
                     response_from_gemini['candidates'][0].get('content') and \
                     response_from_gemini['candidates'][0]['content'].get('parts') and \
                     response_from_gemini['candidates'][0]['content']['parts'][0].get('text'):
                    # Если это текстовый ответ, завершаем цикл
                    gemini_response_content = response_from_gemini['candidates'][0]['content']['parts'][0]['text']
                    logger.info(f"Получен текстовый ответ от Gemini на итерации {iteration + 1}.")
                    break
                else:
                    logger.warning(f"Неожиданный тип ответа от Gemini на итерации {iteration + 1}: {response_from_gemini}")
                    gemini_response_content = "Извините, ИИ вернул непонятный ответ."
                    break

            except Exception as e:
                logger.error(f"Ошибка в цикле обработки сообщений/инструментов на итерации {iteration + 1}: {e}", exc_info=True)
                gemini_response_content = "Произошла внутренняя ошибка при обработке запроса к ИИ."
                break
        else: # Если цикл завершился без break (достигнут max_tool_call_iterations)
            logger.warning(f"Достигнуто максимальное количество итераций вызова инструментов ({self.config.max_tool_call_iterations}).")
            gemini_response_content = "Извините, я не смог завершить обработку вашего запроса после нескольких попыток использования инструментов."

        return gemini_response_content or "Извините, не удалось получить окончательный ответ от ИИ."

    async def _call_gemini_api(self, chat_history: List[Dict[str, Any]], tools_schema: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Внутренний метод для выполнения запроса к Gemini API с логикой повторных попыток.
        """
        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            "contents": chat_history,
            "tools": tools_schema # Добавляем схему инструментов
        }

        for attempt in range(self.config.gemini_max_retries):
            try:
                async with self.http_session.post(self.gemini_api_url, headers=headers, data=json.dumps(payload)) as response:
                    response.raise_for_status() # Вызывает исключение для ошибок HTTP (4xx или 5xx)
                    return await response.json()
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP ошибка от Gemini API на попытке {attempt + 1}: {e.status} - {e.message}")
                if e.status >= 500 or e.status == 429: # Повторяем для серверных ошибок или Too Many Requests
                    if attempt < self.config.gemini_max_retries - 1:
                        delay = self.config.gemini_base_delay * (2 ** attempt)
                        logger.info(f"Повторная попытка через {delay:.2f} секунд...")
                        await asyncio.sleep(delay)
                    else:
                        raise # Перебрасываем ошибку после всех попыток
                else: # Для других клиентских ошибок (например, 400 Bad Request) не повторяем
                    raise # Перебрасываем ошибку для неповторяемых HTTP ошибок
            except aiohttp.ClientError as e:
                logger.error(f"Сетевая ошибка при запросе к Gemini API на попытке {attempt + 1}: {e}")
                if attempt < self.config.gemini_max_retries - 1:
                    delay = self.config.gemini_base_delay * (2 ** attempt)
                    logger.info(f"Повторная попытка через {delay:.2f} секунд...")
                    await asyncio.sleep(delay)
                else:
                    raise # Перебрасываем ошибку после всех попыток
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка декодирования JSON ответа от Gemini на попытке {attempt + 1}: {e}")
                raise # Ошибка парсинга JSON обычно не решается повторной попыткой
            except Exception as e:
                logger.critical(f"Неизвестная критическая ошибка при вызове Gemini API на попытке {attempt + 1}: {e}", exc_info=True)
                raise # Критическая ошибка, перебрасываем
        return {} # Достигнуто, если все попытки исчерпаны и ошибки были перехвачены

class TelegramBotApp:
    """
    Основной класс приложения, который оркестрирует работу всех компонентов бота.
    """
    def __init__(self, config: BotConfig):
        self.config = config
        self.bot = Bot(token=self.config.telegram_bot_token)
        self.dp = Dispatcher()
        self.http_session: aiohttp.ClientSession = None

        self.conversation_manager = ConversationManager(self.config)
        self.tool_manager = ToolManager()
        self.gemini_service = GeminiService(self.config, self.http_session, self.tool_manager) # http_session будет инициализирован позже

        self.register_handlers()
        logger.info("Приложение TelegramBotApp инициализировано.")

    def register_handlers(self):
        """
        Регистрирует все обработчики сообщений в диспетчере aiogram.
        """
        self.dp.message.register(self.send_welcome, F.text == '/start')
        self.dp.message.register(self.reset_conversation, F.text == '/reset')
        self.dp.message.register(self.handle_message, F.text) # Обработчик для всех текстовых сообщений

    async def send_welcome(self, message: types.Message):
        """
        Обработчик команды /start. Приветствует пользователя и инициализирует
        новую историю диалога.
        """
        user_id = message.from_user.id
        # get_history уже инициализирует или сбрасывает историю
        self.conversation_manager.get_history(user_id) 
        logger.info(f"Пользователь {user_id} начал диалог.")
        await message.reply(
            "Привет! Я бот с ИИ. Я буду отвечать на твои сообщения. "
            f"История диалога сохраняется {self.config.conversation_lifetime_minutes} минут. "
            "Чтобы начать новый диалог, используй команду /reset."
        )

    async def reset_conversation(self, message: types.Message):
        """
        Обработчик команды /reset. Сбрасывает историю диалога для пользователя.
        """
        user_id = message.from_user.id
        self.conversation_manager.reset_history(user_id)
        logger.info(f"Диалог с пользователем {user_id} сброшен по команде.")
        await message.reply("Диалог сброшен. Можете начать новую беседу.")

    async def handle_message(self, message: types.Message):
        """
        Основной обработчик текстовых сообщений.
        Получает историю, отправляет запрос к Gemini и отправляет ответ.
        """
        user_id = message.from_user.id
        user_text = message.text

        if not user_text:
            logger.warning(f"Получено пустое сообщение от пользователя {user_id}. Игнорируем.")
            return

        logger.info(f"Получено сообщение от пользователя {user_id}: '{user_text}'")

        # Получаем историю диалога (она будет инициализирована/сброшена при необходимости)
        chat_history = self.conversation_manager.get_history(user_id)
        
        # Если история была сброшена из-за неактивности, сообщаем пользователю
        # (это можно улучшить, чтобы отправлять сообщение только один раз)
        if len(chat_history) == 2 and chat_history[0] == self.conversation_manager.system_prompt:
             # Проверяем, было ли сообщение о сбросе отправлено ранее
            if not getattr(message, '_reset_notified', False): # Используем флаг на объекте сообщения
                await message.reply(f"Диалог был сброшен из-за неактивности (более {self.config.conversation_lifetime_minutes} минут).")
                setattr(message, '_reset_notified', True) # Устанавливаем флаг
        

        # Добавляем сообщение пользователя в историю
        self.conversation_manager.update_history(user_id, {
            'role': 'user',
            'parts': [{'text': user_text}]
        })

        # Отправляем индикатор "печатает..."
        await self.bot.send_chat_action(message.chat.id, 'typing')

        # Получаем ответ от Gemini, который включает логику вызова инструментов
        gemini_response_content = await self.gemini_service.get_gemini_response_with_tools(
            self.conversation_manager.get_history(user_id) # Передаем актуальную историю
        )

        # Добавляем окончательный ответ Gemini в историю диалога пользователя
        self.conversation_manager.update_history(user_id, {
            'role': 'model',
            'parts': [{'text': gemini_response_content}]
        })

        # Отправляем ответ пользователю в Telegram
        await message.reply(gemini_response_content)
        logger.info(f"Отправлен окончательный ответ пользователю {user_id}.")

    async def on_startup(self):
        """
        Функция, выполняемая при запуске бота.
        Инициализирует aiohttp.ClientSession и запускает фоновую задачу очистки.
        """
        logger.info("Запуск aiohttp.ClientSession...")
        self.http_session = aiohttp.ClientSession()
        # Передаем инициализированную сессию в GeminiService
        self.gemini_service.http_session = self.http_session
        
        await self.conversation_manager.start_cleanup_task()
        logger.info("Бот готов к приему сообщений.")

    async def on_shutdown(self):
        """
        Функция, выполняемая при остановке бота.
        Закрывает aiohttp.ClientSession и отменяет фоновую задачу очистки.
        """
        logger.info("Закрытие aiohttp.ClientSession...")
        if self.http_session:
            await self.http_session.close()
        await self.conversation_manager.stop_cleanup_task()
        logger.info("Бот остановлен.")

    async def run(self):
        """
        Основная функция для запуска бота.
        Регистрирует функции on_startup и on_shutdown в диспетчере
        и запускает опрос обновлений Telegram.
        """
        self.dp.startup.register(self.on_startup)
        self.dp.shutdown.register(self.on_shutdown)
        await self.dp.start_polling(self.bot, skip_updates=True)

if __name__ == '__main__':
    # Создаем экземпляр бота и запускаем его
    try:
        config = BotConfig.load_from_env()
        app = TelegramBotApp(config)
        asyncio.run(app.run())
    except ValueError as ve:
        logger.critical(f"Ошибка конфигурации: {ve}. Убедитесь, что переменные окружения установлены корректно.", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Бот остановлен вручную (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Произошла критическая ошибка при работе бота: {e}", exc_info=True)
