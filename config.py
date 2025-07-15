import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

@dataclass
class BotConfig:
    """
    Класс для хранения и валидации конфигурации бота, загружаемой из переменных окружения.
    """
    telegram_bot_token: str
    gemini_api_key: str
    conversation_lifetime_minutes: int = 30
    cleanup_interval_seconds: int = 60 * 5  # Каждые 5 минут
    gemini_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    gemini_max_retries: int = 3
    gemini_base_delay: float = 1.0  # Начальная задержка для повторных попыток в секундах
    # Максимальное количество итераций для цикла вызова инструментов
    # Предотвращает бесконечные циклы в случае некорректного поведения ИИ
    max_tool_call_iterations: int = 5
    google_calendar_credentials_file: str = "credentials.json" # Имя файла с учетными данными Google
    google_calendar_token_file: str = "token.json" # Имя файла для сохранения токена авторизации

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

        return cls(
            telegram_bot_token=telegram_token,
            gemini_api_key=gemini_key
        )

