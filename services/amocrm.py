"""
Сервис для работы с AmoCRM API.
Получение данных о звонках и сохранение примечаний.
"""
import asyncio
import time
import httpx
import ssl
import logging
from typing import Optional, Dict, Any
from config import AMOCRM_DOMAIN, AMOCRM_ACCESS_TOKEN, AMOCRM_VERIFY_SSL, MANAGERS

logger = logging.getLogger(__name__)


class AmoCRMService:
    """Класс для работы с AmoCRM API"""
    
    def __init__(self):
        self.base_url = f"https://{AMOCRM_DOMAIN}/api/v4"
        self.verify_ssl = AMOCRM_VERIFY_SSL
        self.headers = {
            "Authorization": f"Bearer {AMOCRM_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        # Кэш enum-значений select-полей: field_id -> ({value_lower: enum_id}, valid_until)
        self._enum_cache: Dict[int, tuple] = {}
    
    async def resolve_enum_id(self, field_id: int, value_name: str, ttl: int = 3600) -> Optional[int]:
        """
        enum_id значения select-поля сделки по ИМЕНИ (case-insensitive), кэш на ttl.

        Хардкод enum_id уже приводил к инциденту: поле Источник пересоздали,
        все зашитые id умерли, и PATCH получал 400. Резолв по имени переживает
        пересоздание поля; при недоступности API используется последний кэш.
        """
        if not value_name:
            return None
        now = time.time()
        cached = self._enum_cache.get(field_id)
        mapping = None
        if not cached or cached[1] < now:
            try:
                async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                    response = await client.get(
                        f"{self.base_url}/leads/custom_fields/{field_id}",
                        headers=self.headers,
                    )
                    response.raise_for_status()
                    enums = response.json().get("enums") or []
                mapping = {str(e["value"]).strip().lower(): int(e["id"]) for e in enums}
                self._enum_cache[field_id] = (mapping, now + ttl)
            except Exception as e:
                logger.warning(f"⚠️ resolve_enum_id: не смог обновить enums поля {field_id}: {e}")
                if cached:
                    mapping = cached[0]  # протухший кэш лучше, чем ничего
        else:
            mapping = cached[0]
        if mapping is None:
            return None
        enum_id = mapping.get(value_name.strip().lower())
        if enum_id is None:
            logger.warning(f"⚠️ Значение '{value_name}' не найдено среди enums поля {field_id}")
        return enum_id

    async def list_call_events(self, from_ts: int, limit: int = 100) -> list:
        """События звонков (incoming_call/outgoing_call) начиная с from_ts."""
        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.get(
                    f"{self.base_url}/events",
                    headers=self.headers,
                    params={
                        "filter[type][0]": "incoming_call",
                        "filter[type][1]": "outgoing_call",
                        "filter[created_at][from]": from_ts,
                        "limit": limit,
                    },
                )
                if response.status_code == 204:
                    return []
                response.raise_for_status()
                events = response.json().get("_embedded", {}).get("events", [])
        except Exception as e:
            logger.warning(f"⚠️ list_call_events: {e}")
            return []
        out = []
        for e in events:
            try:
                out.append({
                    "entity_type": "leads" if e["entity_type"] == "lead" else "contacts",
                    "entity_id": int(e["entity_id"]),
                    "note_id": int(e["value_after"][0]["note"]["id"]),
                    "created_at": int(e["created_at"]),
                })
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        return out

    async def list_analysis_note_times(self, lead_id: int, since_ts: int) -> list:
        """created_at примечаний «АНАЛИЗ ЗВОНКА» сделки начиная с since_ts, по возрастанию."""
        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.get(
                    f"{self.base_url}/leads/{lead_id}/notes",
                    headers=self.headers,
                    params={
                        "filter[note_type][0]": "common",
                        "order[id]": "desc",  # order[created_at] на notes игнорируется amoCRM
                        "limit": 100,
                    },
                )
                if response.status_code == 204:
                    return []
                response.raise_for_status()
                notes = response.json().get("_embedded", {}).get("notes", [])
        except Exception as e:
            logger.warning(f"⚠️ list_analysis_note_times({lead_id}): {e}")
            return []
        times = [
            n.get("created_at", 0) for n in notes
            if "АНАЛИЗ ЗВОНКА" in ((n.get("params") or {}).get("text") or "")
            and n.get("created_at", 0) >= since_ts
        ]
        return sorted(times)

    async def close_lead_as_spam(self, lead_id: int, loss_reason_id: int = 22393318) -> bool:
        """Закрывает сделку в 143 («закрыто и не реализовано») с причиной СПАМ (22393318)."""
        ok, err = await self._patch_lead(lead_id, {"status_id": 143, "loss_reason_id": loss_reason_id})
        if not ok:
            logger.error(f"❌ Не удалось закрыть сделку #{lead_id} как спам: {err}")
        return ok

    async def complete_task(self, task_id: int, result_text: str = "") -> bool:
        """Завершает задачу с текстом результата."""
        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.patch(
                    f"{self.base_url}/tasks/{task_id}",
                    headers=self.headers,
                    json={"is_completed": True, "result": {"text": result_text or "Выполнено"}},
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка завершения задачи #{task_id}: {e}")
            return False

    async def get_open_tasks(self, lead_id: int) -> list:
        """Открытые (невыполненные) задачи сделки."""
        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.get(
                    f"{self.base_url}/tasks",
                    headers=self.headers,
                    params={
                        "filter[entity_type]": "leads",
                        "filter[entity_id]": lead_id,
                        "filter[is_completed]": 0,
                        "limit": 10,
                    },
                )
                if response.status_code == 204:
                    return []
                response.raise_for_status()
                return response.json().get("_embedded", {}).get("tasks", [])
        except Exception as e:
            logger.warning(f"⚠️ get_open_tasks({lead_id}): {e}")
            return []

    async def create_task(
        self,
        lead_id: int,
        text: str,
        complete_till: int,
        responsible_user_id: Optional[int] = None,
        task_type_id: int = 1,
    ) -> bool:
        """Создаёт задачу, привязанную к сделке (task_type_id=1 — «Связаться»)."""
        payload = {
            "task_type_id": task_type_id,
            "text": text,
            "complete_till": int(complete_till),
            "entity_id": lead_id,
            "entity_type": "leads",
        }
        if responsible_user_id:
            payload["responsible_user_id"] = int(responsible_user_id)
        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.post(
                    f"{self.base_url}/tasks",
                    headers=self.headers,
                    json=[payload],
                )
                response.raise_for_status()
                logger.info(f"✅ Задача создана для сделки #{lead_id} (срок {complete_till})")
                return True
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Ошибка создания задачи для #{lead_id}: {e} | {e.response.text[:300]}")
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка создания задачи для #{lead_id}: {e}")
            return False

    async def get_recent_calls(self, minutes: int = 10) -> list:
        """
        Получает список недавних звонков из AmoCRM.
        Точно как в Make.com: GET /api/v4/events с фильтрами
        
        Args:
            minutes: За сколько минут искать звонки
            
        Returns:
            Список событий звонков
        """
        import time
        try:
            # Время "от" в Unix timestamp
            from_timestamp = int(time.time()) - (minutes * 60)
            logger.info(f"🕐 Ищем звонки с timestamp: {from_timestamp} (последние {minutes} мин)")
            
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                # Точный URL из Make.com:
                # /api/v4/events?filter[type][0]=outgoing_call&filter[type][1]=incoming_call&filter[created_at][from]=...
                response = await client.get(
                    f"{self.base_url}/events",
                    headers=self.headers,
                    params={
                        "filter[type][0]": "outgoing_call",
                        "filter[type][1]": "incoming_call",
                        "filter[created_at][from]": from_timestamp
                    }
                )
                
                if response.status_code == 204:
                    logger.info("Нет звонков (204 No Content)")
                    return []
                    
                response.raise_for_status()
                data = response.json()
                
                events = data.get("_embedded", {}).get("events", [])
                logger.info(f"Найдено {len(events)} звонков за последние {minutes} минут")
                return events
                
        except Exception as e:
            logger.error(f"Ошибка получения звонков: {e}")
            return []
    
    async def get_recent_notes(self, entity_type: str, entity_id: int, limit: int = 5) -> list:
        """
        Получает последние примечания сущности.
        Используется когда webhook не содержит note_id.
        
        Args:
            entity_type: Тип сущности (lead, contact, company)
            entity_id: ID сущности
            limit: Количество примечаний
            
        Returns:
            Список примечаний
        """
        try:
            # Преобразуем entity_type
            type_map = {
                "lead": "leads",
                "contact": "contacts",
                "company": "companies",
                "leads": "leads",
                "contacts": "contacts",
                "companies": "companies"
            }
            api_type = type_map.get(entity_type, entity_type)
            
            url = f"{self.base_url}/{api_type}/{entity_id}/notes"
            logger.info(f"Запрос примечаний: {url}")
            
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.get(
                    url,
                    headers=self.headers,
                    # order[id]=desc обязателен: order[created_at] на notes-эндпоинте
                    # amoCRM МОЛЧА игнорирует и отдаёт старейшие ноты
                    params={"limit": limit, "order[id]": "desc"}
                )
                
                if response.status_code == 204:
                    logger.info(f"Нет примечаний для {api_type}/{entity_id}")
                    return []
                    
                response.raise_for_status()
                data = response.json()
                
                notes = data.get("_embedded", {}).get("notes", [])
                logger.info(f"Найдено {len(notes)} примечаний для {api_type}/{entity_id}")
                return notes
                
        except Exception as e:
            logger.error(f"Ошибка получения примечаний: {e}")
            return []
    
    async def get_note_with_recording(self, entity_type: str, entity_id: int, note_id: int) -> Optional[Dict[str, Any]]:
        """
        Получает примечание с записью звонка.
        Как в Make.com: GET /api/v4/{entity_type}/{entity_id}/notes/{note_id}
        
        Args:
            entity_type: Тип сущности (leads, contacts, companies)
            entity_id: ID сущности
            note_id: ID примечания
            
        Returns:
            Данные примечания с params.link
        """
        try:
            # Преобразуем entity_type как в Make: switch(entity_type; "contact"; "contacts"; ...)
            type_map = {
                "lead": "leads",
                "contact": "contacts",
                "company": "companies"
            }
            api_type = type_map.get(entity_type, entity_type)
            
            url = f"{self.base_url}/{api_type}/{entity_id}/notes/{note_id}"
            logger.info(f"Запрос примечания: {url}")
            
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.get(url, headers=self.headers)
                
                if response.status_code == 204:
                    logger.warning(f"Примечание не найдено (204)")
                    return None
                    
                response.raise_for_status()
                data = response.json()
                logger.info(f"Получено примечание: {data}")
                return data
                
        except Exception as e:
            logger.error(f"Ошибка получения примечания: {e}")
            return None
    
    async def process_call_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Обрабатывает событие звонка и получает ссылку на запись.
        Логика из Make.com:
        1. Из события берём entity_type, entity_id, value_after[].note.id
        2. Запрашиваем примечание
        3. Из примечания берём params.link
        
        Args:
            event: Событие звонка из API
            
        Returns:
            Словарь с данными для обработки или None
        """
        try:
            event_id = event.get("id")
            event_type = event.get("type")  # incoming_call или outgoing_call
            entity_type = event.get("entity_type")  # lead, contact, company
            entity_id = event.get("entity_id")
            created_by = event.get("created_by")
            created_at = event.get("created_at")  # Unix timestamp (сек), время события в AmoCRM
            
            logger.info(f"Обработка события #{event_id}: {event_type} для {entity_type}/{entity_id}")
            
            # Ищем note.id в value_after
            value_after = event.get("value_after", [])
            note_id = None
            for item in value_after:
                if isinstance(item, dict) and "note" in item:
                    note_id = item["note"].get("id")
                    break
            
            if not note_id:
                logger.warning(f"Нет note_id в событии #{event_id}")
                return None
            
            logger.info(f"Найден note_id: {note_id}")
            
            # Получаем примечание с записью
            note_data = await self.get_note_with_recording(entity_type, entity_id, note_id)
            
            if not note_data:
                logger.warning(f"Не удалось получить примечание {note_id}")
                return None
            
            # Извлекаем ссылку на запись из params.link
            params = note_data.get("params", {})
            record_link = params.get("link")
            
            if not record_link:
                logger.warning(f"Нет ссылки на запись в примечании {note_id}")
                return None
            
            logger.info(f"✅ Найдена ссылка на запись: {record_link[:50]}...")
            
            # Извлекаем телефон из params
            phone = params.get("phone", "")
            
            return {
                "event_id": event_id,
                "event_type": event_type,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "note_id": note_id,
                "record_url": record_link,
                "created_by": created_by,
                "created_at": created_at,
                "phone": phone,
                "params": params
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки события: {e}")
            return None
    
    async def get_call_events_for_entity(self, entity_id: int, entity_type: str) -> list:
        """
        Получает события звонков для конкретной сущности (контакта или сделки).
        
        Args:
            entity_id: ID сущности
            entity_type: Тип сущности (contacts, leads)
            
        Returns:
            Список событий звонков
        """
        try:
            # Преобразуем entity_type для API
            api_entity_type = "contact" if entity_type == "contacts" else "lead"
            
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.get(
                    f"{self.base_url}/events",
                    headers=self.headers,
                    params={
                        "filter[entity]": api_entity_type,
                        "filter[entity_id]": entity_id,
                        "filter[type][0]": "outgoing_call",
                        "filter[type][1]": "incoming_call"
                    }
                )
                
                if response.status_code == 204:
                    logger.info(f"Нет звонков для {entity_type}/{entity_id}")
                    return []
                    
                response.raise_for_status()
                data = response.json()
                
                events = data.get("_embedded", {}).get("events", [])
                logger.info(f"Найдено {len(events)} звонков для {entity_type}/{entity_id}")
                return events
                
        except Exception as e:
            logger.error(f"Ошибка получения звонков для {entity_type}/{entity_id}: {e}")
            return []
    
    async def download_call_recording(self, url: str, max_retries: int = 3) -> bytes:
        """
        Скачивает аудиофайл записи звонка.
        Обходит проверку SSL для серверов с невалидными сертификатами.
        При 404 делает повторные попытки с задержкой (запись может быть ещё не готова).
        
        Args:
            url: URL записи звонка
            max_retries: Максимальное количество попыток при 404
            
        Returns:
            Бинарные данные аудиофайла
        """
        logger.info(f"📥 Скачиваем запись: {url[:80]}...")
        
        # Создаём SSL контекст, который полностью игнорирует проверку сертификатов
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        
        last_error = None
        # Задержки между попытками: 30с, 60с, 90с
        retry_delays = [30, 60, 90]
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True, 
                    timeout=120.0, 
                    verify=ssl_ctx
                ) as client:
                    response = await client.get(url)
                    
                    # Если требует авторизации, пробуем с ней
                    if response.status_code in [401, 403]:
                        response = await client.get(url, headers=self.headers)
                    
                    # Если 404 и есть ещё попытки — ждём и повторяем
                    if response.status_code == 404 and attempt < max_retries - 1:
                        delay = retry_delays[attempt]
                        logger.warning(f"⏳ Запись не готова (404), попытка {attempt + 1}/{max_retries}. Ждём {delay}с...")
                        await asyncio.sleep(delay)
                        continue
                    
                    response.raise_for_status()
                    
                    content_length = len(response.content)
                    if attempt > 0:
                        logger.info(f"✅ Скачано с попытки {attempt + 1}: {content_length} байт")
                    else:
                        logger.info(f"✅ Скачано: {content_length} байт")
                    
                    return response.content
                    
            except Exception as e:
                last_error = e
                # Если это НЕ 404, не ретраим
                if "404" not in str(e) or attempt >= max_retries - 1:
                    logger.error(f"❌ Ошибка скачивания (попытка {attempt + 1}/{max_retries}): {e}")
                    raise
                delay = retry_delays[attempt]
                logger.warning(f"⏳ Ошибка скачивания (попытка {attempt + 1}/{max_retries}): {e}. Ждём {delay}с...")
                await asyncio.sleep(delay)
        
        raise last_error or Exception("Не удалось скачать запись после всех попыток")
    
    async def get_lead(self, lead_id: int) -> Optional[Dict[str, Any]]:
        """
        Получает данные сделки.
        
        Args:
            lead_id: ID сделки
            
        Returns:
            Данные сделки
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/leads/{lead_id}",
                    headers=self.headers,
                    params={"with": "contacts"}
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Ошибка получения сделки {lead_id}: {e}")
            raise
    
    async def get_contact(self, contact_id: int) -> Optional[Dict[str, Any]]:
        """
        Получает данные контакта.
        
        Args:
            contact_id: ID контакта
            
        Returns:
            Данные контакта
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/contacts/{contact_id}",
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Ошибка получения контакта {contact_id}: {e}")
            raise
    
    async def add_note_to_entity(self, entity_id: int, text: str, entity_type: str = "leads") -> bool:
        """
        Добавляет примечание к сущности (сделке, контакту, компании).
        
        Args:
            entity_id: ID сущности
            text: Текст примечания
            entity_type: Тип сущности (leads, contacts, companies)
            
        Returns:
            True если успешно
        """
        try:
            # Приводим entity_type к множественному числу если нужно
            if entity_type == "lead":
                entity_type = "leads"
            elif entity_type == "contact":
                entity_type = "contacts"
            elif entity_type == "company":
                entity_type = "companies"
            
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.post(
                    f"{self.base_url}/{entity_type}/{entity_id}/notes",
                    headers=self.headers,
                    json=[{
                        "note_type": "common",
                        "params": {
                            "text": text
                        }
                    }]
                )
                
                if response.status_code == 400:
                    error_text = response.text
                    try:
                        error_json = response.json()
                        logger.error(f"AmoCRM вернул 400 для {entity_type}/{entity_id}: {error_json}")
                    except (ValueError, Exception):
                        logger.error(f"AmoCRM вернул 400 для {entity_type}/{entity_id}: {error_text}")
                    # Пробуем получить больше информации об ошибке
                    logger.error(f"Запрос был: POST {self.base_url}/{entity_type}/{entity_id}/notes")
                    logger.error(f"Текст примечания (первые 200 символов): {text[:200]}")
                
                response.raise_for_status()
                logger.info(f"Примечание добавлено к {entity_type}/{entity_id}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка добавления примечания к {entity_type}/{entity_id}: {e}")
            raise
    
    async def add_note_to_lead(self, lead_id: int, text: str) -> bool:
        """Обратная совместимость"""
        return await self.add_note_to_entity(lead_id, text, "leads")
    
    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Получает данные пользователя (менеджера).
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Данные пользователя
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/users/{user_id}",
                    headers=self.headers
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Ошибка получения пользователя {user_id}: {e}")
            return None
    
    def get_manager_name(self, user_id: int) -> str:
        """
        Получает имя менеджера по ID.
        Сначала ищет в локальном словаре, потом в API.
        
        Args:
            user_id: ID пользователя в AmoCRM
            
        Returns:
            Имя менеджера
        """
        # Сначала ищем в локальном словаре
        if str(user_id) in MANAGERS:
            return MANAGERS[str(user_id)]
        
        return f"Менеджер #{user_id}"
    
    async def get_active_lead_for_contact(self, contact_id: int) -> Optional[int]:
        """
        Получает ID АКТИВНОЙ сделки, привязанной к контакту.
        
        Закрытые сделки (status_id = 142 - успех, 143 - провал) игнорируются!
        
        Args:
            contact_id: ID контакта
            
        Returns:
            ID активной сделки или None
        """
        # Статусы "закрытых" сделок (не добавляем примечания туда)
        CLOSED_STATUSES = {
            142,  # Успешно реализовано
            143,  # Закрыто и не реализовано
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                # 1. Получаем связи контакта
                response = await client.get(
                    f"{self.base_url}/contacts/{contact_id}/links",
                    headers=self.headers
                )
                
                if response.status_code == 204:
                    logger.info(f"У контакта {contact_id} нет связей")
                    return None
                    
                response.raise_for_status()
                data = response.json()
                
                # 2. Собираем ID всех связанных сделок
                links = data.get("_embedded", {}).get("links", [])
                lead_ids = [
                    link.get("to_entity_id") 
                    for link in links 
                    if link.get("to_entity_type") == "leads"
                ]
                
                if not lead_ids:
                    logger.info(f"У контакта {contact_id} нет сделок")
                    return None
                
                logger.info(f"🔍 Контакт {contact_id} имеет {len(lead_ids)} сделок: {lead_ids}")
                
                # 3. Проверяем статус каждой сделки
                for lead_id in lead_ids:
                    try:
                        lead_response = await client.get(
                            f"{self.base_url}/leads/{lead_id}",
                            headers=self.headers
                        )
                        
                        if lead_response.status_code == 200:
                            lead_data = lead_response.json()
                            status_id = lead_data.get("status_id")
                            lead_name = lead_data.get("name", "")
                            
                            logger.info(f"  Сделка #{lead_id} '{lead_name}': статус {status_id}")
                            
                            # Если сделка НЕ закрыта - используем её
                            if status_id not in CLOSED_STATUSES:
                                logger.info(f"✅ Найдена активная сделка #{lead_id}")
                                return lead_id
                            else:
                                logger.info(f"  ⏭️ Сделка #{lead_id} закрыта, пропускаем")
                                
                    except Exception as e:
                        logger.warning(f"Не удалось проверить сделку {lead_id}: {e}")
                
                logger.info(f"❌ Все сделки контакта {contact_id} закрыты")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка получения активной сделки для контакта {contact_id}: {e}")
            return None
    
    async def create_lead_for_contact(
        self, 
        contact_id: int, 
        contact_name: str = "",
        phone: str = "",
        responsible_user_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Создаёт новую сделку и привязывает к контакту.
        
        Args:
            contact_id: ID контакта
            contact_name: Имя контакта (для названия сделки)
            phone: Телефон (для названия сделки)
            responsible_user_id: Ответственный менеджер
            
        Returns:
            ID созданной сделки или None
        """
        try:
            # Формируем название сделки
            lead_name = f"Входящий звонок: {contact_name or phone or contact_id}"
            
            # Данные для создания сделки
            lead_data = [{
                "name": lead_name,
                "_embedded": {
                    "contacts": [{"id": contact_id}]
                }
            }]
            
            # Добавляем ответственного если есть
            if responsible_user_id:
                lead_data[0]["responsible_user_id"] = responsible_user_id
            
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.post(
                    f"{self.base_url}/leads",
                    headers=self.headers,
                    json=lead_data
                )
                
                if response.status_code == 400:
                    logger.error(f"Ошибка создания сделки: {response.text}")
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                # Получаем ID созданной сделки
                leads = data.get("_embedded", {}).get("leads", [])
                if leads:
                    lead_id = leads[0].get("id")
                    logger.info(f"✅ Создана сделка #{lead_id} для контакта #{contact_id}")
                    return lead_id
                
                return None
                
        except Exception as e:
            logger.error(f"Ошибка создания сделки для контакта {contact_id}: {e}")
            return None
    
    async def get_or_create_lead_for_contact(
        self, 
        contact_id: int,
        phone: str = "",
        responsible_user_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Получает АКТИВНУЮ сделку контакта или создаёт новую.
        
        Логика:
        - Если есть активная (не закрытая) сделка → возвращаем её
        - Если все сделки закрыты или нет сделок → создаём новую
        
        Args:
            contact_id: ID контакта
            phone: Телефон для названия сделки
            responsible_user_id: Ответственный менеджер
            
        Returns:
            ID сделки (существующей активной или новой)
        """
        # Ищем АКТИВНУЮ сделку (не закрытую)
        lead_id = await self.get_active_lead_for_contact(contact_id)
        
        if lead_id:
            logger.info(f"✅ Используем активную сделку #{lead_id}")
            return lead_id
        
        # Нет активной сделки - создаём новую
        logger.info(f"📝 Создаём новую сделку для контакта #{contact_id}...")
        
        contact = await self.get_contact(contact_id)
        contact_name = contact.get("name", "") if contact else ""
        
        new_lead_id = await self.create_lead_for_contact(
            contact_id=contact_id,
            contact_name=contact_name,
            phone=phone,
            responsible_user_id=responsible_user_id
        )
        
        if not new_lead_id:
            logger.error(f"❌ Не удалось создать сделку для контакта #{contact_id}")
            return None
        
        if new_lead_id == contact_id:
            logger.error(f"⚠️ ВНИМАНИЕ: create_lead_for_contact вернул ID контакта {contact_id} вместо ID сделки!")
            return None
        
        logger.info(f"✅ Создана новая сделка #{new_lead_id} для контакта #{contact_id}")
        return new_lead_id

    async def _patch_lead(self, lead_id: int, body: dict) -> tuple:
        """PATCH /api/v4/leads/{id}. Возвращает (успех, текст ошибки amoCRM)."""
        try:
            async with httpx.AsyncClient(timeout=30.0, verify=self.verify_ssl) as client:
                response = await client.patch(
                    f"{self.base_url}/leads/{lead_id}",
                    headers=self.headers,
                    json=body,
                )
                response.raise_for_status()
                return True, ""
        except httpx.HTTPStatusError as e:
            return False, f"{e} | ответ amoCRM: {e.response.text[:500]}"
        except Exception as e:
            return False, str(e)

    async def update_lead_fields(
        self,
        lead_id: int,
        custom_fields_values: list = None,
        price: int = None,
        name: str = None,
    ) -> bool:
        """
        Обновляет поля сделки через PATCH /api/v4/leads/{id}.

        Сначала всё одним запросом; при ошибке (например, протухший enum_id
        одного из полей — amoCRM отбрасывает ВЕСЬ батч) повторяем по одному
        полю за запрос: записываем всё валидное и видим в логе, какое поле отбито.
        """
        body = {}
        if custom_fields_values:
            body["custom_fields_values"] = custom_fields_values
        if price is not None:
            body["price"] = price
        if name is not None:
            body["name"] = name

        if not body:
            logger.info(f"⏭️ Нечего обновлять в сделке #{lead_id}")
            return True

        ok, err = await self._patch_lead(lead_id, body)
        if ok:
            logger.info(f"✅ Поля сделки #{lead_id} обновлены: {list(body.keys())}")
            return True

        logger.error(f"❌ Ошибка обновления полей сделки #{lead_id}: {err}")

        # Фолбэк: по одному полю за запрос
        parts = []
        for cf in (custom_fields_values or []):
            parts.append(({"custom_fields_values": [cf]}, f"field_id={cf.get('field_id')}"))
        if price is not None:
            parts.append(({"price": price}, "price"))
        if name is not None:
            parts.append(({"name": name}, "name"))
        if len(parts) < 2:
            return False

        saved = 0
        for part_body, label in parts:
            part_ok, part_err = await self._patch_lead(lead_id, part_body)
            if part_ok:
                saved += 1
            else:
                logger.error(f"  ❌ Сделка #{lead_id}, {label} отбито: {part_err}")
        logger.info(f"🔁 Сделка #{lead_id}: фолбэк по одному полю — записано {saved}/{len(parts)}")
        return saved > 0

    async def update_contact_name(self, contact_id: int, name: str) -> bool:
        """
        Обновляет имя контакта через PATCH /api/v4/contacts/{id}.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0, verify=self.verify_ssl) as client:
                response = await client.patch(
                    f"{self.base_url}/contacts/{contact_id}",
                    headers=self.headers,
                    json={"name": name},
                )
                response.raise_for_status()
                logger.info(f"✅ Имя контакта #{contact_id} обновлено: {name}")
                return True
        except Exception as e:
            logger.error(f"❌ Ошибка обновления имени контакта #{contact_id}: {e}")
            return False


# Синглтон для использования во всём приложении
amocrm_service = AmoCRMService()
