import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List

from config import BotConfig
from tool_manager import ToolManager

logger = logging.getLogger(__name__)

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
                # Исправлено: используем 'functionCall' вместо 'function_call'
                if response_from_gemini.get('candidates') and \
                   response_from_gemini['candidates'][0].get('content') and \
                   response_from_gemini['candidates'][0]['content'].get('parts') and \
                   response_from_gemini['candidates'][0]['content']['parts'][0].get('functionCall'):

                    function_call = response_from_gemini['candidates'][0]['content']['parts'][0]['functionCall']
                    function_name = function_call['name']
                    function_args = function_call.get('args', {})

                    logger.info(f"ИИ запросил вызов функции: '{function_name}' с аргументами: {function_args} на итерации {iteration + 1}.")

                    # Добавляем вызов функции в историю для Gemini
                    current_chat_history.append({
                        'role': 'model',
                        'parts': [{'function_call': function_call}] # Здесь оставляем snake_case для внутренней обработки
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

