import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass, field # Используем dataclass для конфигурации

import aiohttp
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

class TelegramAIBot:
    """
    Класс для управления Telegram-ботом с интеграцией Google Gemini AI.
    Инкапсулирует всю логику бота, включая взаимодействие с Telegram API
    и Gemini API, а также управление состоянием диалогов.
    """
    def __init__(self, config: BotConfig):
        """
        Инициализирует бота, диспетчер и загружает конфигурацию.
        """
        self.config = config
        self.bot = Bot(token=self.config.telegram_bot_token)
        self.dp = Dispatcher()

        self.gemini_api_url: str = f"{self.config.gemini_api_base_url}?key={self.config.gemini_api_key}"

        # Словарь для хранения истории диалогов пользователей
        # Ключ: user_id (int)
        # Значение: {'history': list, 'last_active': datetime}
        # 'history' - список сообщений в формате Gemini API: [{'role': 'user', 'parts': [{'text': '...'}]}, ...]
        # 'last_active' - время последнего сообщения от пользователя
        self.user_conversations: Dict[int, Dict[str, Any]] = {}

        # Системный промпт для модели Gemini
        # Это начальное сообщение, которое задает контекст и "личность" ИИ
        self.system_prompt: Dict[str, Any] = {
            'role': 'user',
            'parts': [{'text': 'Ты — дружелюбный и полезный ИИ-ассистент, разработанный Google. Твоя цель — отвечать на вопросы пользователей и вести с ними диалог, сохраняя контекст беседы. Отвечай на русском языке.'}]
        }
        # Ответ модели на системный промпт, чтобы диалог начался корректно
        # Это помогает ИИ понять, что системный промпт был "услышан" и принят
        self.system_prompt_response: Dict[str, Any] = {
            'role': 'model',
            'parts': [{'text': 'Привет! Я готов к общению.'}]
        }

        self.http_session: aiohttp.ClientSession = None # Сессия aiohttp для HTTP-запросов
        self.cleanup_task: asyncio.Task = None # Задача для фоновой очистки

        self.register_handlers()
        logger.info("Бот инициализирован и готов к запуску.")

    def _get_initial_history(self) -> List[Dict[str, Any]]:
        """
        Возвращает начальную историю диалога, включающую системный промпт
        и его ожидаемый ответ от модели.
        """
        return [self.system_prompt, self.system_prompt_response]

    async def _cleanup_old_conversations(self):
        """
        Периодически очищает устаревшие диалоги из памяти.
        Диалог считается устаревшим, если с момента последнего сообщения
        прошло больше self.config.conversation_lifetime_minutes.
        """
        while True:
            await asyncio.sleep(self.config.cleanup_interval_seconds)
            current_time = datetime.now()
            users_to_remove = []

            for user_id, data in self.user_conversations.items():
                last_active_time = data['last_active']
                if current_time - last_active_time > timedelta(minutes=self.config.conversation_lifetime_minutes):
                    users_to_remove.append(user_id)

            for user_id in users_to_remove:
                del self.user_conversations[user_id]
                logger.info(f"Диалог с пользователем {user_id} удален из памяти из-за неактивности.")

    async def get_gemini_response(self, chat_history: List[Dict[str, Any]]) -> str:
        """
        Отправляет историю чата в Gemini API и возвращает ответ.
        Использует асинхронную HTTP-сессию с повторными попытками.
        """
        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            "contents": chat_history
        }

        for attempt in range(self.config.gemini_max_retries):
            try:
                async with self.http_session.post(self.gemini_api_url, headers=headers, data=json.dumps(payload)) as response:
                    response.raise_for_status()  # Вызывает исключение для ошибок HTTP (4xx или 5xx)
                    result = await response.json()

                    if result and result.get('candidates') and len(result['candidates']) > 0 and \
                       result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
                       len(result['candidates'][0]['content']['parts']) > 0:
                        logger.info(f"Получен успешный ответ от Gemini API на попытке {attempt + 1}.")
                        return result['candidates'][0]['content']['parts'][0]['text']
                    else:
                        logger.warning(f"Неожиданная структура ответа от Gemini на попытке {attempt + 1}: {result}")
                        # Если структура ответа некорректна, не повторяем попытку, это не поможет
                        return "Извините, не удалось получить ответ от ИИ. Пожалуйста, попробуйте еще раз."
            except aiohttp.ClientResponseError as e:
                # Обработка HTTP ошибок (например, 429 Too Many Requests, 5xx Server Error)
                logger.error(f"HTTP ошибка от Gemini API на попытке {attempt + 1}: {e.status} - {e.message}")
                if e.status >= 500 or e.status == 429: # Повторяем для серверных ошибок или Too Many Requests
                    if attempt < self.config.gemini_max_retries - 1:
                        delay = self.config.gemini_base_delay * (2 ** attempt)
                        logger.info(f"Повторная попытка через {delay:.2f} секунд...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Все {self.config.gemini_max_retries} попыток исчерпаны для HTTP ошибки.")
                        return "Извините, произошла ошибка на стороне ИИ-сервиса. Пожалуйста, попробуйте позже."
                else: # Для других клиентских ошибок (например, 400 Bad Request) не повторяем
                    logger.error(f"Неповторяемая HTTP ошибка от Gemini API: {e.status} - {e.message}")
                    return "Извините, произошла ошибка при запросе к ИИ. Проверьте правильность запроса."
            except aiohttp.ClientError as e:
                # Обработка сетевых ошибок (например, проблемы с подключением)
                logger.error(f"Сетевая ошибка при запросе к Gemini API на попытке {attempt + 1}: {e}")
                if attempt < self.config.gemini_max_retries - 1:
                    delay = self.config.gemini_base_delay * (2 ** attempt)
                    logger.info(f"Повторная попытка через {delay:.2f} секунд...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Все {self.config.gemini_max_retries} попыток исчерпаны для сетевой ошибки.")
                    return "Извините, произошла ошибка при подключении к ИИ. Пожалуйста, попробуйте позже."
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка декодирования JSON ответа от Gemini на попытке {attempt + 1}: {e}")
                # Ошибка парсинга JSON обычно не решается повторной попыткой
                return "Извините, произошла ошибка при обработке ответа от ИИ."
            except Exception as e:
                logger.critical(f"Неизвестная критическая ошибка при получении ответа от Gemini на попытке {attempt + 1}: {e}", exc_info=True)
                # Критическая ошибка, не повторяем
                return "Извините, произошла непредвиденная ошибка. Пожалуйста, попробуйте позже."

        # Этот код будет достигнут, только если все попытки исчерпаны без успеха
        return "Извините, не удалось получить ответ от ИИ после нескольких попыток. Пожалуйста, попробуйте позже."


    async def send_welcome(self, message: types.Message):
        """
        Обработчик команды /start. Приветствует пользователя и инициализирует
        новую историю диалога с системным промптом.
        """
        user_id = message.from_user.id
        self.user_conversations[user_id] = {
            'history': self._get_initial_history(),
            'last_active': datetime.now()
        }
        logger.info(f"Пользователь {user_id} начал диалог. История инициализирована.")
        await message.reply(
            "Привет! Я бот с ИИ. Я буду отвечать на твои сообщения. "
            f"История диалога сохраняется {self.config.conversation_lifetime_minutes} минут. "
            "Чтобы начать новый диалог, используй команду /reset."
        )

    async def reset_conversation(self, message: types.Message):
        """
        Обработчик команды /reset. Сбрасывает историю диалога для пользователя,
        начиная новую беседу с системным промптом.
        """
        user_id = message.from_user.id
        self.user_conversations[user_id] = {
            'history': self._get_initial_history(),
            'last_active': datetime.now()
        }
        logger.info(f"Диалог с пользователем {user_id} сброшен по команде.")
        await message.reply("Диалог сброшен. Можете начать новую беседу.")

    async def handle_message(self, message: types.Message):
        """
        Основной обработчик текстовых сообщений.
        Управляет жизненным циклом диалога, добавляет сообщения в историю,
        отправляет запросы к Gemini и отправляет ответы пользователю.
        """
        user_id = message.from_user.id
        user_text = message.text

        if not user_text: # Проверяем, что сообщение не пустое
            logger.warning(f"Получено пустое сообщение от пользователя {user_id}. Игнорируем.")
            return

        current_time = datetime.now()

        logger.info(f"Получено сообщение от пользователя {user_id}: '{user_text}'")

        # Проверяем, есть ли история для пользователя, и если нет, инициализируем
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = {
                'history': self._get_initial_history(),
                'last_active': current_time
            }
            logger.info(f"Инициализирована новая история для пользователя {user_id}, так как её не было.")
        else:
            # Проверяем время последнего сообщения для сброса диалога по неактивности
            last_active_time = self.user_conversations[user_id]['last_active']
            if current_time - last_active_time > timedelta(minutes=self.config.conversation_lifetime_minutes):
                self.user_conversations[user_id]['history'] = self._get_initial_history()
                logger.info(f"Диалог с пользователем {user_id} сброшен из-за неактивности (более {self.config.conversation_lifetime_minutes} минут).")
                await message.reply(f"Диалог был сброшен из-за неактивности (более {self.config.conversation_lifetime_minutes} минут).")

        # Обновляем время последней активности пользователя
        self.user_conversations[user_id]['last_active'] = current_time

        # Добавляем сообщение пользователя в историю диалога
        self.user_conversations[user_id]['history'].append({
            'role': 'user',
            'parts': [{'text': user_text}]
        })

        # Отправляем индикатор "печатает..." в чат
        await self.bot.send_chat_action(message.chat.id, 'typing')

        # Получаем ответ от Gemini AI
        gemini_response = await self.get_gemini_response(self.user_conversations[user_id]['history'])

        # Добавляем ответ Gemini в историю диалога
        self.user_conversations[user_id]['history'].append({
            'role': 'model',
            'parts': [{'text': gemini_response}]
        })

        # Отправляем ответ пользователю в Telegram
        await message.reply(gemini_response)
        logger.info(f"Отправлен ответ пользователю {user_id}.")

    def register_handlers(self):
        """
        Регистрирует все обработчики сообщений в диспетчере aiogram.
        """
        self.dp.message.register(self.send_welcome, F.text == '/start')
        self.dp.message.register(self.reset_conversation, F.text == '/reset')
        # Регистрируем handle_message как общий обработчик для всех текстовых сообщений,
        # которые не являются командами, уже обработанными выше.
        self.dp.message.register(self.handle_message, F.text)

    async def on_startup(self):
        """
        Функция, выполняемая при запуске бота.
        Инициализирует aiohttp.ClientSession и запускает фоновую задачу очистки.
        """
        logger.info("Запуск aiohttp.ClientSession...")
        self.http_session = aiohttp.ClientSession()
        logger.info("Запуск фоновой задачи очистки старых диалогов...")
        self.cleanup_task = asyncio.create_task(self._cleanup_old_conversations())
        logger.info("Бот готов к приему сообщений.")

    async def on_shutdown(self):
        """
        Функция, выполняемая при остановке бота.
        Закрывает aiohttp.ClientSession и отменяет фоновую задачу очистки.
        """
        logger.info("Закрытие aiohttp.ClientSession...")
        if self.http_session:
            await self.http_session.close()
        logger.info("Отмена фоновой задачи очистки...")
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task # Ожидаем завершения задачи, чтобы избежать RuntimeWarning
            except asyncio.CancelledError:
                logger.info("Фоновая задача очистки успешно отменена.")
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
        bot_instance = TelegramAIBot(config)
        asyncio.run(bot_instance.run())
    except ValueError as ve:
        logger.critical(f"Ошибка конфигурации: {ve}. Убедитесь, что переменные окружения установлены корректно.", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Бот остановлен вручную (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Произошла критическая ошибка при работе бота: {e}", exc_info=True)
