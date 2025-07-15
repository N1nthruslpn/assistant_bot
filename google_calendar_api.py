import datetime
import os
import logging

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import BotConfig

logger = logging.getLogger(__name__)

class GoogleCalendarAPI:
    """
    Класс для взаимодействия с Google Calendar API.
    Обрабатывает авторизацию и выполняет операции с календарем.
    """
    SCOPES = ['https://www.googleapis.com/auth/calendar.events'] # Область доступа для создания/изменения/удаления событий

    def __init__(self, config: BotConfig):
        self.config = config
        self.creds = None
        self._authenticate()

    def _authenticate(self):
        """
        Авторизует пользователя для доступа к Google Calendar API.
        Если токен существует, использует его. В противном случае,
        запускает процесс OAuth 2.0 для получения нового токена.
        """
        if os.path.exists(self.config.google_calendar_token_file):
            self.creds = Credentials.from_authorized_user_file(self.config.google_calendar_token_file, self.SCOPES)
        
        # Если нет действительных учетных данных, или они просрочены, обновить или получить новые
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                logger.info("Токен Google Календаря просрочен, обновляем...")
                self.creds.refresh(Request())
            else:
                logger.info("Запуск OAuth потока для Google Календаря...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.config.google_calendar_credentials_file, self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            # Сохраняем учетные данные для следующего запуска
            with open(self.config.google_calendar_token_file, 'w') as token:
                token.write(self.creds.to_json())
            logger.info(f"Учетные данные Google Календаря сохранены в {self.config.google_calendar_token_file}")

    def _get_service(self):
        """Возвращает объект сервиса Google Calendar API."""
        if not self.creds or not self.creds.valid:
            self._authenticate() # Повторная попытка аутентификации, если креды недействительны
        return build('calendar', 'v3', credentials=self.creds)

    async def create_event(self, summary: str, start_time: str, end_time: str, description: str = None, location: str = None) -> str:
        """
        Создает новое событие в Google Календаре.
        Время должно быть в формате ISO 8601 (например, '2025-07-15T10:00:00+03:00').
        """
        try:
            service = self._get_service()
            event = {
                'summary': summary,
                'location': location,
                'description': description,
                'start': {
                    'dateTime': start_time,
                    'timeZone': 'Europe/Moscow', # Укажите ваш часовой пояс
                },
                'end': {
                    'dateTime': end_time,
                    'timeZone': 'Europe/Moscow', # Укажите ваш часовой пояс
                },
            }

            event = service.events().insert(calendarId='primary', body=event).execute()
            logger.info(f"Событие создано: {event.get('htmlLink')}")
            return f"Событие '{summary}' успешно создано в календаре. Ссылка: {event.get('htmlLink')}"
        except HttpError as error:
            logger.error(f"Произошла ошибка при создании события: {error}", exc_info=True)
            return f"Не удалось создать событие: {error}"
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при создании события: {e}", exc_info=True)
            return f"Непредвиденная ошибка при создании события: {e}"

    async def list_events(self, max_results: int = 10, time_min: str = None, time_max: str = None, summary_keyword: str = None) -> str:
        """
        Получает список предстоящих событий из Google Календаря.
        Время должно быть в формате ISO 8601.
        Можно указать ключевое слово для фильтрации по названию события.
        """
        try:
            service = self._get_service()
            now = datetime.datetime.utcnow().isoformat() + 'Z'  # Текущее время в UTC
            
            # Если time_min не указан, используем текущее время
            if not time_min:
                time_min = now
            
            events_result = service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])

            filtered_events = []
            if summary_keyword:
                for event in events:
                    if summary_keyword.lower() in event.get('summary', '').lower():
                        filtered_events.append(event)
            else:
                filtered_events = events

            if not filtered_events:
                return "Предстоящих событий не найдено." if not summary_keyword else f"Событий с названием '{summary_keyword}' не найдено."

            events_str = "Вот несколько предстоящих событий:\n"
            for event in filtered_events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                # Добавляем ID события, чтобы его можно было использовать для удаления
                events_str += f"- {event['summary']} (с {start} по {end}) ID: {event['id']}\n"
            return events_str
        except HttpError as error:
            logger.error(f"Произошла ошибка при получении событий: {error}", exc_info=True)
            return f"Не удалось получить события: {error}"
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при получении событий: {e}", exc_info=True)
            return f"Непредвиденная ошибка при получении событий: {e}"

    async def delete_event(self, event_id: str) -> str:
        """
        Удаляет событие из Google Календаря по его ID.
        """
        try:
            service = self._get_service()
            service.events().delete(calendarId='primary', eventId=event_id).execute()
            logger.info(f"Событие с ID '{event_id}' успешно удалено.")
            return f"Событие с ID '{event_id}' успешно удалено из календаря."
        except HttpError as error:
            if error.resp.status == 404:
                return f"Событие с ID '{event_id}' не найдено или уже удалено."
            logger.error(f"Произошла ошибка при удалении события: {error}", exc_info=True)
            return f"Не удалось удалить событие: {error}"
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при удалении события: {e}", exc_info=True)
            return f"Непредвиденная ошибка при удалении события: {e}"

