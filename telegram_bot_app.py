import asyncio
import logging
import aiohttp
from aiogram import Bot, Dispatcher, types, F

from config import BotConfig
from conversation_manager import ConversationManager
from google_calendar_api import GoogleCalendarAPI
from tool_manager import ToolManager
from gemini_service import GeminiService

logger = logging.getLogger(__name__)

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
        self.google_calendar_api = GoogleCalendarAPI(self.config) # Инициализируем Google Calendar API
        self.tool_manager = ToolManager(self.google_calendar_api) # Передаем его в ToolManager
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
        # Исправлено: Проверяем по длине истории, а не по атрибуту system_prompt
        # Так как system_prompt теперь генерируется динамически, его прямое сравнение
        # может быть ненадежным. Проверка len(chat_history) == 2 (системный промпт + ответ модели)
        # является хорошим индикатором свежего диалога.
        if len(chat_history) == 2:
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

