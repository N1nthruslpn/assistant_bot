import asyncio
import logging

from config import BotConfig
from telegram_bot_app import TelegramBotApp

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

