"""
Главный файл приложения.
FastAPI сервер с webhook endpoint для AmoCRM.

Запуск:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import logging
import asyncio
import json
import httpx
import os
import re
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import JSONResponse

from config import PORT, DEBUG, AMOCRM_DOMAIN, STT_PROVIDER, MIN_CALL_SECONDS, validate_config
from services.amocrm import amocrm_service
from services.transcription import transcription_service
from services.analysis import analysis_service
from services.telegram import telegram_service
from services import alerts

# ============== Маппинг work_type → ИМЯ значения поля "Интерес" (field_id=212083) ==============
# enum_id резолвится по имени через amocrm_service.resolve_enum_id — хардкод id
# уже приводил к инциденту (пересоздание поля Источник). Имена сверены с CRM 25.08.2026.
WORK_TYPE_TO_INTEREST_NAME = {
    "межевание": "Межевание",
    "вынос": "Выносв натуру",
    "вынос в натуру": "Выносв натуру",
    "вынос границ": "Выносв натуру",
    "топосъёмка": "Топосъёмка",
    "топосъемка": "Топосъёмка",
    "топографическая съёмка": "Топосъёмка",
    "топографическая съемка": "Топосъёмка",
    "техплан": "Техплан",
    "технический план": "Техплан",
    "техпаспорт": "Техпаспорт",
    "технический паспорт": "Техпаспорт",
    "акт обследования": "Акт обследования",
    "схема": "Схема на КПТ",
    "схема на кпт": "Схема на КПТ",
    "геодезическое сопровождение": "геодезическое сопровождение",
    "геодезия": "геодезическое сопровождение",
    "раздел": "Раздел ЗУ",
    "раздел зу": "Раздел ЗУ",
    "разбивка": "Разбивка",
    "исполнительная съемка": "Исполнительная съемка",
    "исполнительная съёмка": "Исполнительная съемка",
    "проектирование": "Проектирование",
    "геология": "Геология",
    "экология": "Экология",
    "подбор зу": "Подбор ЗУ",
    "межевание и тех план": "межевание и тех план",
    "межевание и техплан": "межевание и тех план",
    "прочие": "Прочие",
    "подача в рр через лк": "Подача в РР через ЛК",
    "все виды изысканий": "Все виды изысканий",
    "инженерные изыскания": "Все виды изысканий",
    "снижение кс": "Снижение КС",
    "ппт": "ППТ (ПМТ)",
    "гидрометеорология": "Гидрометеорология",
    "водоохранная зона": "Водоохранная зона",
}


# ============== Маппинг тег → ИМЯ значения поля "Источник" (field_id=212063) ==============
TAG_TO_SOURCE_NAME = {
    "whatsapp": "WhatsApp",
    "вотсап": "WhatsApp",
    "авито": "Авито",
    "avito": "Авито",
    "2гис": "2ГИС",
    "2gis": "2ГИС",
    "яндекс ук": "Яндекс УК",
    "seo": "SEO",
    "менеджер": "Менеджер",
    "lp new google": "LP new Google",
    "lp new яндекс": "LP new Яндекс",
    "profzem": "ProfZem",
    "гугл": "Гугл",
    "google": "Гугл",
    "1777": "1777",
    "исходящий": "Исходящий",
    "почта": "почта (уточняем источник трафика!)",
    "ltv": "LTV",
    "личные": "Личные",
    "рекомендация": "Рекомендация",
    "квиз": "Квиз",
}


def _match_tag_to_source(tags: list) -> str | None:
    """Ищет ИМЯ значения поля Источник по тегам сделки."""
    for tag in tags:
        tag_name = (tag.get("name") or "").lower().strip()
        if tag_name in TAG_TO_SOURCE_NAME:
            return TAG_TO_SOURCE_NAME[tag_name]
        # Частичное совпадение
        for keyword, source_name in TAG_TO_SOURCE_NAME.items():
            if keyword in tag_name or tag_name in keyword:
                return source_name
    return None


# Сокращения для названий сделок (стиль менеджера)
WORK_TYPE_SHORT_NAME = {
    "межевание": "МП",
    "межевой план": "МП",
    "техплан": "ТП",
    "технический план": "ТП",
    "топосъёмка": "ТС",
    "топосъемка": "ТС",
    "топографическая съёмка": "ТС",
    "топографическая съемка": "ТС",
    "вынос": "Вынос",
    "вынос в натуру": "Вынос",
    "вынос границ": "Вынос",
}


def _shorten_work_type(work_type_text: str) -> str:
    """Возвращает сокращение для названия сделки, или оригинал если нет сокращения."""
    if not work_type_text:
        return ""
    text = work_type_text.lower().strip()
    if text in WORK_TYPE_SHORT_NAME:
        return WORK_TYPE_SHORT_NAME[text]
    for keyword, short in WORK_TYPE_SHORT_NAME.items():
        if keyword in text:
            return short
    return work_type_text


def _match_call_result_name(call_result: str) -> str | None:
    """Сводит свободный call_result из анализа к значению поля «Итог звонка (AI)» (770463)."""
    if not call_result:
        return None
    t = call_result.lower()
    if "соглас" in t or "договор" in t:
        return "Согласие"
    if "отказ" in t or "не интерес" in t:
        return "Отказ"
    if "перезвон" in t:
        return "Перезвонить"
    if "дума" in t:
        return "Думает"
    return "Не определено"


def _match_work_type_name(work_type_text: str) -> str | None:
    """Ищет ИМЯ значения поля Интерес по тексту work_type из анализа."""
    if not work_type_text:
        return None
    text = work_type_text.lower().strip()
    # Точное совпадение
    if text in WORK_TYPE_TO_INTEREST_NAME:
        return WORK_TYPE_TO_INTEREST_NAME[text]
    # Частичное совпадение
    for keyword, name in WORK_TYPE_TO_INTEREST_NAME.items():
        if keyword in text:
            return name
    return None


def _parse_price(cost_text: str):
    """Извлекает числовую стоимость из текста (например '25 000 ₽' → 25000)."""
    if not cost_text:
        return None
    normalized = cost_text.lower().strip()
    if normalized in ("не обсуждали", "не указано", "не определено", ""):
        return None
    import re

    # Сначала ищем сумму рядом с валютой.
    with_currency = re.findall(r"(\d[\d\s]{0,14}\d|\d+)\s*(?:₽|руб(?:\.|ля|лей)?)", normalized)
    if with_currency:
        candidate = with_currency[0]
    else:
        # Фолбэк: любое число в тексте.
        any_numbers = re.findall(r"\d[\d\s]{0,14}\d|\d+", normalized)
        if not any_numbers:
            return None
        candidate = any_numbers[0]

    digits = re.sub(r"[^\d]", "", candidate)
    if digits:
        price = int(digits)
        # Санитарные границы: от 1 тыс. до 50 млн.
        if 1_000 <= price <= 50_000_000:
            return price
    return None


def _has_custom_field_value(custom_field: dict) -> bool:
    """Проверяет, что у custom field есть фактическое значение любого типа."""
    for val in (custom_field.get("values") or []):
        if not isinstance(val, dict):
            continue
        for key in ("value", "enum_id", "enum_code", "file_id", "text"):
            raw = val.get(key)
            if raw is None:
                continue
            if isinstance(raw, str):
                if raw.strip():
                    return True
                continue
            return True
    return False


async def auto_fill_lead_fields(lead_id: int, analysis, call_type_simple: str,
                                source_fallback_name: str | None = None):
    """
    Автоматически заполняет поля сделки на основе AI-анализа звонка.
    Заполняет ТОЛЬКО пустые поля — не перезаписывает то, что менеджер уже заполнил.
    """
    try:
        # Получаем текущие данные сделки
        lead_data = await amocrm_service.get_lead(lead_id)
        if not lead_data:
            logger.warning(f"⚠️ Не удалось получить сделку #{lead_id} для автозаполнения")
            return

        # Собираем уже заполненные поля
        existing_fields = set()
        for cf in (lead_data.get("custom_fields_values") or []):
            fid = cf.get("field_id")
            if fid and _has_custom_field_value(cf):
                existing_fields.add(fid)

        existing_price = lead_data.get("price", 0)

        custom_fields = []
        price_to_set = None

        # 0. Источник (field_id=212063, select; enum_id резолвится по имени):
        # по тегам сделки, для чатов без тегов — фолбэк по каналу (source_fallback_name)
        if 212063 not in existing_fields:
            tags = lead_data.get("_embedded", {}).get("tags", [])
            # Метка линии из названия «Входящий +7… (9383527800 - 2ГИС)» надёжнее тегов:
            # тег «Авито» интеграция вешает на все звонковые сделки без разбора
            source_name = None
            m = re.search(r"\(\s*\d+\s*-\s*([^)]+)\)", lead_data.get("name") or "")
            if m:
                source_name = _match_tag_to_source([{"name": m.group(1).strip()}])
            if not source_name and tags:
                source_name = _match_tag_to_source(tags)
            if not source_name and source_fallback_name:
                source_name = source_fallback_name
            if source_name:
                enum_id = await amocrm_service.resolve_enum_id(212063, source_name)
                if enum_id:
                    custom_fields.append({
                        "field_id": 212063,
                        "values": [{"enum_id": enum_id}]
                    })
                    tag_names = ", ".join(t.get("name", "") for t in tags)
                    logger.info(f"  📢 Источник (теги [{tag_names}] / канал): {source_name} → enum_id={enum_id}")

        # 1. Город (field_id=212029, text)
        if 212029 not in existing_fields:
            city = getattr(analysis, "client_city", "")
            if city and city.lower() not in ("не указано", "не определено", ""):
                custom_fields.append({
                    "field_id": 212029,
                    "values": [{"value": city}]
                })
                logger.info(f"  📍 Город: {city}")

        # 2. Интерес / work_type (field_id=212083, select; enum_id резолвится по имени)
        if 212083 not in existing_fields:
            work_type = getattr(analysis, "work_type", "")
            if work_type and work_type.lower() not in ("не обсуждали", "не указано", "не определено", ""):
                interest_name = _match_work_type_name(work_type)
                if interest_name:
                    enum_id = await amocrm_service.resolve_enum_id(212083, interest_name)
                    if enum_id:
                        custom_fields.append({
                            "field_id": 212083,
                            "values": [{"enum_id": enum_id}]
                        })
                        logger.info(f"  🔧 Интерес: {work_type} → {interest_name} (enum_id={enum_id})")

        # 3. Тип сделки (field_id=212099, select; enum_id резолвится по имени)
        if 212099 not in existing_fields:
            type_name = "входящий" if call_type_simple == "incoming" else "исходящий"
            enum_id = await amocrm_service.resolve_enum_id(212099, type_name)
            if enum_id:
                custom_fields.append({
                    "field_id": 212099,
                    "values": [{"enum_id": enum_id}]
                })
                logger.info(f"  📞 Тип сделки: {type_name} (enum_id={enum_id})")

        # 4. Схема оплаты (field_id=767917, text)
        if 767917 not in existing_fields:
            payment = getattr(analysis, "payment_terms", "")
            if payment and payment.lower() not in ("не обсуждали", "не указано", "не определено", ""):
                custom_fields.append({
                    "field_id": 767917,
                    "values": [{"value": payment}]
                })
                logger.info(f"  💳 Схема оплаты: {payment}")

        # 4.5. Итог звонка (field_id=770463, select) — итог ПОСЛЕДНЕГО звонка,
        # содержательные значения перезаписывают старые; «Не определено» — только в пустое поле
        outcome_name = _match_call_result_name(getattr(analysis, "call_result", "") or "")
        if outcome_name and (outcome_name != "Не определено" or 770463 not in existing_fields):
            enum_id = await amocrm_service.resolve_enum_id(770463, outcome_name)
            if enum_id:
                custom_fields.append({
                    "field_id": 770463,
                    "values": [{"enum_id": enum_id}]
                })
                logger.info(f"  🎯 Итог звонка: {outcome_name}")

        # 5. Бюджет сделки (встроенное поле price)
        if not existing_price or existing_price == 0:
            cost = getattr(analysis, "cost", "")
            parsed_price = _parse_price(cost)
            if parsed_price:
                price_to_set = parsed_price
                logger.info(f"  💰 Бюджет: {parsed_price}")

        # 6. Адрес объекта (field_id=768529, text)
        if 768529 not in existing_fields:
            location = getattr(analysis, "location", "")
            if location and location.lower() not in ("не указано", "не определено", ""):
                custom_fields.append({
                    "field_id": 768529,
                    "values": [{"value": location}]
                })
                logger.info(f"  📍 Адрес объекта: {location}")

        # 7. Название сделки: "{work_type} {location}" — только если текущее дефолтное
        name_to_set = None
        existing_name = lead_data.get("name", "") or ""
        # Считаем имя дефолтным, если оно:
        # - пустое
        # - начинается с "Входящий звонок" / "Исходящий звонок" (наша система)
        # - начинается с "Входящий +" / "Исходящий +" (AmoCRM автосоздание)
        # - начинается с "Входящий" / "Исходящий" и содержит номер телефона
        is_default_name = (
            not existing_name
            or existing_name.startswith("Входящий звонок")
            or existing_name.startswith("Исходящий звонок")
            or existing_name.startswith("Входящий +")
            or existing_name.startswith("Исходящий +")
            or bool(re.match(r"^(Входящий|Исходящий)\s+\+?\d", existing_name))
        )
        if is_default_name:
            work_type = getattr(analysis, "work_type", "")
            location = getattr(analysis, "location", "")
            if work_type and work_type.lower() not in ("не определено", "не обсуждали", ""):
                short_name = _shorten_work_type(work_type)
                # Привязка, чтобы «МП» не плодились неотличимыми: адрес → город → имя клиента
                empty = ("не указано", "не определено", "не обсуждали", "")
                city = getattr(analysis, "client_city", "") or ""
                client = _clean_client_name(getattr(analysis, "client_name", "")) or ""
                anchor = next(
                    (v.strip() for v in (location, city, client)
                     if v and v.strip() and v.lower() not in empty),
                    None,
                )
                name_to_set = f"{short_name} {anchor}"[:80] if anchor else short_name
                logger.info(f"  🏷️ Название сделки: {name_to_set}")

        # Отправляем PATCH если есть что обновлять
        if custom_fields or price_to_set is not None or name_to_set is not None:
            logger.info(f"📝 Автозаполнение сделки #{lead_id}: {len(custom_fields)} полей" +
                        (f" + бюджет {price_to_set}" if price_to_set else "") +
                        (f" + название '{name_to_set}'" if name_to_set else ""))
            updated = await amocrm_service.update_lead_fields(
                lead_id=lead_id,
                custom_fields_values=custom_fields if custom_fields else None,
                price=price_to_set,
                name=name_to_set,
            )
            if not updated:
                logger.warning(f"⚠️ PATCH сделки #{lead_id} вернул ошибку (см. лог выше)")
        else:
            logger.info(f"⏭️ Автозаполнение #{lead_id}: нечего обновлять (поля уже заполнены или данных нет)")

    except Exception as e:
        # Не валим основной пайплайн из-за ошибки автозаполнения
        logger.error(f"❌ Ошибка автозаполнения сделки #{lead_id}: {e}")


# ============== Имя клиента → карточка контакта ==============

CLIENT_NAME_STOPWORDS = {"клиент", "не представился", "не указано", "не определено", "неизвестно"}


def _clean_client_name(raw: Optional[str]) -> Optional[str]:
    """
    Готовит имя из анализа к записи в карточку: убирает скобки с организацией
    («Александр (Управление Росреестра)» → «Александр») и мусорные значения.
    """
    if not raw:
        return None
    name = re.sub(r"\([^)]*\)", " ", raw)
    name = re.sub(r"\s+", " ", name).strip(" ,.-— ")
    if not name or name.lower() in CLIENT_NAME_STOPWORDS:
        return None
    return name


def _is_default_contact_name(name: str) -> bool:
    """
    Дефолтные имена, которые можно перезаписывать: пустое, голый номер телефона
    или автоимя amoCRM вида «Входящий +78652951729 (9280058522 - Менеджер)».
    Тот же паттерн, что для названий сделок в auto_fill_lead_fields.
    """
    if not name:
        return True
    stripped = name.lstrip("+").replace(" ", "").replace("-", "")
    if stripped.isdigit():
        return True
    return bool(re.match(r"^(Входящий|Исходящий)\s+\+?\d", name))


async def _update_contact_name_from_analysis(analysis, entity_type: str, entity_id: int, lead_id: int):
    """
    Пишет имя клиента из анализа в карточку контакта.
    Контакт: entity_id вебхука (если вебхук был про контакт) или главный контакт
    сделки из _embedded.contacts — вебхуки телефонии приходят с entity_type="leads",
    поэтому путь через сделку основной.
    Перезаписываем ТОЛЬКО дефолтные имена — ручной ввод менеджера не трогаем.
    """
    try:
        client_name = _clean_client_name(getattr(analysis, "client_name", ""))
        if not client_name:
            return

        if entity_type.lower() in ("contact", "contacts"):
            contact_id = entity_id
        else:
            lead_data = await amocrm_service.get_lead(lead_id)
            contacts = ((lead_data or {}).get("_embedded") or {}).get("contacts") or []
            main_contacts = [c for c in contacts if c.get("is_main")]
            contact = (main_contacts or contacts or [None])[0]
            contact_id = contact.get("id") if contact else None

        if not contact_id:
            logger.info(f"⏭️ Имя контакта: у сделки #{lead_id} нет привязанного контакта")
            return

        contact_data = await amocrm_service.get_contact(contact_id)
        if not contact_data:
            return
        current_name = contact_data.get("name", "") or ""
        if not _is_default_contact_name(current_name):
            logger.info(f"⏭️ Имя контакта #{contact_id} заполнено вручную, не трогаем: {current_name!r}")
            return

        if await amocrm_service.update_contact_name(contact_id, client_name):
            logger.info(f"👤 Имя контакта #{contact_id}: {current_name!r} → {client_name!r}")

    except Exception as e:
        # Имя — вторично; не валим основной пайплайн
        logger.error(f"❌ Ошибка обновления имени контакта для сделки #{lead_id}: {e}")


_suspicious_alert_last = 0.0


async def _alert_suspicious_diarization(lead_id: int, reason: str):
    """Telegram-алерт о сомнительной диаризации, не чаще раза в 30 минут."""
    global _suspicious_alert_last
    if time.time() - _suspicious_alert_last < 1800:
        return
    _suspicious_alert_last = time.time()
    try:
        await telegram_service.send_message(
            "⚠️ <b>Диаризация под сомнением</b>\n\n"
            f"Сделка: #{lead_id}\n"
            f"Причина: {reason}\n\n"
            "Расшифровка сохранена, автозаполнение полей пропущено."
        )
    except Exception as e:
        logger.warning(f"⚠️ Не удалось отправить алерт о диаризации: {e}")


# Настраиваем логирование
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Безопасность: подавляем подробные логи HTTP-клиента (могут содержать токены в URL).
# Например, Telegram API использует URL вида /bot<TOKEN>/sendMessage — в INFO/DEBUG это утечка.
for _logger_name in ("httpx", "httpcore"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

# Кэш обработанных звонков, чтобы избежать дублей и петель
# В продакшене лучше использовать Redis, но для начала хватит и Set в памяти
PROCESSED_CALLS = set()
PROCESSED_LOCK = asyncio.Lock()


async def _get_audio_duration(audio_data: bytes) -> float:
    """Определяет длительность аудио через ffprobe."""
    import tempfile
    import os
    suffix = ".mp3"
    if audio_data[:4] == b'RIFF':
        suffix = ".wav"
    elif audio_data[:4] == b'OggS':
        suffix = ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_data)
        tmp_path = f.name
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return float(stdout.decode().strip())
    except Exception:
        return 0.0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _ensure_full_recording(
    audio_data: bytes,
    record_url: str,
    expected_duration: int,
    max_retries: int = 4,
) -> bytes:
    """
    Проверяет, что скачанная запись соответствует ожидаемой длительности.
    Если аудио слишком короткое (< 50% от ожидаемого) — ждёт и скачивает заново.
    Это решает проблему, когда vmclouds ещё не успел обработать полную запись.
    """
    actual_duration = await _get_audio_duration(audio_data)
    threshold = expected_duration * 0.5

    if actual_duration >= threshold:
        logger.info(
            f"✅ Длительность аудио OK: {actual_duration:.0f}с "
            f"(ожидалось {expected_duration}с)"
        )
        return audio_data

    logger.warning(
        f"⚠️ Аудио слишком короткое: {actual_duration:.0f}с из ожидаемых {expected_duration}с. "
        f"Запись ещё не готова на vmclouds — ждём и повторяем скачивание..."
    )

    # Задержки: 30с, 60с, 90с, 120с
    retry_delays = [30, 60, 90, 120]
    best_data = audio_data
    best_duration = actual_duration

    for attempt in range(max_retries):
        delay = retry_delays[min(attempt, len(retry_delays) - 1)]
        logger.info(
            f"⏳ Попытка {attempt + 1}/{max_retries}: ждём {delay}с..."
        )
        await asyncio.sleep(delay)

        try:
            new_data = await amocrm_service.download_call_recording(record_url)
            new_duration = await _get_audio_duration(new_data)
            logger.info(
                f"📊 Попытка {attempt + 1}: скачано {len(new_data)} байт, "
                f"длительность {new_duration:.0f}с (ожидаем {expected_duration}с)"
            )

            if new_duration > best_duration:
                best_data = new_data
                best_duration = new_duration

            if new_duration >= threshold:
                logger.info(
                    f"✅ Запись готова! {new_duration:.0f}с "
                    f"(ожидалось {expected_duration}с)"
                )
                return best_data
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при повторном скачивании: {e}")

    logger.warning(
        f"⚠️ Запись так и не стала полной после {max_retries} попыток. "
        f"Лучший результат: {best_duration:.0f}с из {expected_duration}с. "
        f"Используем что есть."
    )
    return best_data


async def is_processed_peek(record_url: str) -> bool:
    """Проверка БЕЗ пометки — для догоняющего цикла (пометит сам process_call)."""
    async with PROCESSED_LOCK:
        return record_url in PROCESSED_CALLS


async def is_already_processed(record_url: str) -> bool:
    """Проверяет, обрабатывался ли уже этот звонок по URL записи"""
    async with PROCESSED_LOCK:
        if record_url in PROCESSED_CALLS:
            return True
        # Ограничиваем размер кэша (храним последние 1000 записей)
        if len(PROCESSED_CALLS) > 1000:
            PROCESSED_CALLS.clear()
        PROCESSED_CALLS.add(record_url)
        return False


# ============== Авто-задача менеджеру по итогам звонка ==============
AUTO_TASK_ENABLED = os.getenv("AUTO_TASK_ENABLED", "true").strip().lower() in ("1", "true", "yes")

_WEEKDAYS = {
    "понедельник": 0, "вторник": 1, "среда": 2, "сред": 2, "четверг": 3,
    "пятниц": 4, "суббот": 5, "воскресен": 6,
}
_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_WORD_NUMS = {
    "первого": 1, "второго": 2, "третьего": 3, "четвертого": 4, "четвёртого": 4,
    "пятого": 5, "шестого": 6, "седьмого": 7, "восьмого": 8, "девятого": 9, "десятого": 10,
}
_MSK_TZ = timezone(timedelta(hours=3))


def _parse_next_contact_ts(text: str, now_utc: Optional[datetime] = None) -> Optional[int]:
    """
    Переводит фразу next_contact_date («пятница», «завтра», «1-го числа», «15 января»,
    «через неделю») в unix timestamp срока задачи. Времена — по МСК: будущий день → 10:00,
    «сегодня» → 18:00 (если уже поздно — завтра 10:00). Не распознали → None.
    """
    if not text:
        return None
    t = text.lower().strip()
    if t in ("не указано", "не определено", "не обсуждали", ""):
        return None

    if now_utc is None:
        now_msk = datetime.now(_MSK_TZ)
    else:
        now_msk = now_utc.replace(tzinfo=timezone.utc).astimezone(_MSK_TZ)
    today = now_msk.replace(hour=0, minute=0, second=0, microsecond=0)

    def ts_at(day: datetime, hour: int = 10) -> int:
        # day — aware (МСК), поэтому timestamp() не зависит от TZ хоста
        return int(day.replace(hour=hour).timestamp())

    def today_or_tomorrow() -> int:
        # «сегодня»: конец рабочего дня; если он уже близко/прошёл — завтра утром
        if now_msk.hour < 17:
            return ts_at(today, 18)
        return ts_at(today + timedelta(days=1), 10)

    if "сегодня" in t or "текущий день" in t:
        return today_or_tomorrow()
    if "послезавтра" in t:
        return ts_at(today + timedelta(days=2))
    if "завтра" in t or "следующий день" in t:
        return ts_at(today + timedelta(days=1))
    if "через недел" in t:
        return ts_at(today + timedelta(days=7))
    if "через месяц" in t:
        return ts_at(today + timedelta(days=30))
    m = re.search(r"через\s+(\d+)\s*(день|дня|дней)", t)
    if m:
        return ts_at(today + timedelta(days=int(m.group(1))))

    # День недели («в пятницу», «понедельник») → ближайший будущий; сегодня → сегодня 18:00
    for stem, wd in _WEEKDAYS.items():
        if stem in t:
            delta = (wd - today.weekday()) % 7
            if delta == 0:
                return today_or_tomorrow()
            return ts_at(today + timedelta(days=delta))

    # «15 января», «1 сентября»
    m = re.search(r"(\d{1,2})\s*([а-яё]+)", t)
    if m:
        day_num = int(m.group(1))
        for stem, month in _MONTHS.items():
            if m.group(2).startswith(stem):
                if 1 <= day_num <= 31:
                    year = today.year
                    try:
                        target = today.replace(year=year, month=month, day=day_num)
                    except ValueError:
                        return None
                    if target < today:
                        target = target.replace(year=year + 1)
                    return ts_at(target)

    # «1-го числа», «первого числа», «5 числа» → ближайшее такое число месяца
    m = re.search(r"(\d{1,2})(?:-?го)?\s*числ", t)
    day_num = int(m.group(1)) if m else None
    if day_num is None and "числ" in t:
        for word, n in _WORD_NUMS.items():
            if word in t:
                day_num = n
                break
    if day_num and 1 <= day_num <= 31:
        try:
            target = today.replace(day=day_num)
        except ValueError:
            target = None
        if target is None or target <= today:
            nxt = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
            try:
                target = nxt.replace(day=day_num)
            except ValueError:
                return None
        return ts_at(target)

    return None


async def _create_followup_task(lead_id: int, analysis, responsible_user_id: Optional[int]):
    """
    Ставит менеджеру задачу по итогам разговора — только когда LLM увидел договорённость
    (needs_task). Раньше задача создавалась на каждый анализ, и при пустых next_steps
    в CRM улетало «Связаться с клиентом по итогам звонка (AI)»: 04.09.2026 таких висело
    7 штук, а всего автозадач — 197 из 200 открытых.

    Срок: распознанная фраза next_contact_date («в пятницу», «15 марта») — как есть,
    это договорённость с клиентом. Не распознали — due_in_hours от LLM, и вот его уже
    прижимаем к рабочему окну.

    Дублей не плодим: ensure_task оставляет на сделке ровно одну открытую задачу.
    """
    try:
        if not getattr(analysis, "needs_task", False):
            logger.info(f"⏭️ Сделка #{lead_id}: договорённости нет — задача не ставится")
            return

        text = (getattr(analysis, "task_text", "") or "").strip()
        if not text:
            logger.warning(f"⏭️ Сделка #{lead_id}: needs_task=true, но текст пустой — задача не ставится")
            return

        next_contact = (getattr(analysis, "next_contact_date", "") or "").strip()
        if next_contact.lower() not in ("не указано", "не определено", "не обсуждали", ""):
            text += f"\nСлед. контакт: {next_contact}"

        complete_till = _parse_next_contact_ts(next_contact)
        if complete_till is None:
            hours = getattr(analysis, "due_in_hours", 24) or 24
            complete_till = _snap_to_work_hours(int(time.time()) + hours * 3600)

        result = await amocrm_service.ensure_task(
            lead_id=lead_id,
            text=text[:500],
            complete_till=complete_till,
            responsible_user_id=responsible_user_id,
        )
        if result:
            due = datetime.fromtimestamp(complete_till, _MSK_TZ)
            verb = "создана" if result["action"] == "created" else "обновлена"
            logger.info(
                f"📋 Задача #{result['task_id']} по сделке #{lead_id} {verb}: срок {due:%d.%m %H:%M} МСК"
            )
    except Exception as e:
        # Задача — вишенка, не роняем пайплайн
        logger.error(f"❌ Ошибка постановки авто-задачи для #{lead_id}: {e}")


# ============== Догоняющий цикл: подхват звонков, потерянных вебхуком ==============
# 25.08.2026 сервис завис и вебхуки пропали — звонки остались без расшифровки,
# пока их не догнали вручную. Этот цикл делает потерю вебхука нефатальной:
# раз в RECONCILE_INTERVAL сканируем звонки за RECONCILE_LOOKBACK и дообрабатываем
# те, у которых есть запись >= 60с, но нет примечания «АНАЛИЗ ЗВОНКА».
RECONCILE_ENABLED = os.getenv("RECONCILE_ENABLED", "true").strip().lower() in ("1", "true", "yes")
RECONCILE_INTERVAL = int(os.getenv("RECONCILE_INTERVAL", "180"))    # период скана, сек
RECONCILE_LOOKBACK = int(os.getenv("RECONCILE_LOOKBACK", "21600"))  # окно скана, сек (6 ч)
RECONCILE_MIN_AGE = int(os.getenv("RECONCILE_MIN_AGE", "600"))      # звонки моложе — оставляем вебхук-пути
_RECONCILE_CLAIM_SLACK = 60  # люфт: анализ не может появиться раньше звонка минус slack


def _match_covered(call_times: list, note_times: list) -> set:
    """Жадное паросочетание (как в scripts/backfill_week.py): какие звонки покрыты анализом."""
    covered, used = set(), [False] * len(note_times)
    for i, ct in enumerate(call_times):
        for j, nt in enumerate(note_times):
            if not used[j] and nt >= ct - _RECONCILE_CLAIM_SLACK:
                used[j] = True
                covered.add(i)
                break
    return covered


# --- Пропущенные звонки без перезвона + спам-автозакрытие (работают в том же цикле) ---
MISSED_CALL_TASK_ENABLED = os.getenv("MISSED_CALL_TASK_ENABLED", "true").strip().lower() in ("1", "true", "yes")
MISSED_CALL_GRACE = int(os.getenv("MISSED_CALL_GRACE", "7200"))  # сек: даём менеджерам перезвонить самим
SPAM_AUTOCLOSE_ENABLED = os.getenv("SPAM_AUTOCLOSE_ENABLED", "true").strip().lower() in ("1", "true", "yes")

_MISSED_HANDLED: set = set()      # note_id пропущенных, по которым задача уже есть (in-memory)
_SPAM_HANDLED: set = set()        # lead_id закрытых/проверенных спам-сделок
_SPAM_CHECKED_AT: dict = {}       # lead_id -> ts последней проверки нот (не чаще раза в 30 мин)


def _norm_phone(phone: str) -> str:
    """Последние 10 цифр — чтобы +7/8/7 варианты совпадали."""
    digits = re.sub(r"\D", "", str(phone or ""))
    return digits[-10:] if len(digits) >= 10 else digits


def _snap_to_work_hours(ts: int, start_h: int = 9, end_h: int = 18) -> int:
    """
    Прижимает срок задачи к рабочему окну 9–18 МСК: после конца дня или в выходной →
    следующий рабочий день 10:00, до начала дня → сегодня 10:00.

    Выходные пропускаем циклом, а не одним «+1 день»: пятница 19:00 без этого
    превращалась в субботу 10:00, и задача протухала ещё до того, как её кто-то увидел.
    """
    d = datetime.fromtimestamp(int(ts), _MSK_TZ)
    if d.weekday() >= 5 or d.hour >= end_h:
        d = (d + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
        while d.weekday() >= 5:
            d += timedelta(days=1)
    elif d.hour < start_h:
        d = d.replace(hour=10, minute=0, second=0, microsecond=0)
    return int(d.timestamp())


def _next_work_slot_ts() -> int:
    """Срок задачи «перезвонить»: через час, но в рабочее окно 9–18 МСК."""
    return _snap_to_work_hours(int(time.time()) + 3600)


async def _missed_calls_pass(all_calls: list, now: int):
    """Пропущенный входящий, по которому за MISSED_CALL_GRACE не было разговора, → задача."""
    answered = [c for c in all_calls if c["duration"] > 0]
    for c in all_calls:
        note = c["note"]
        if note.get("note_type") != "call_in" or c["duration"] > 0:
            continue
        if now - c["created_at"] < MISSED_CALL_GRACE:
            continue
        nid = note.get("id")
        if nid in _MISSED_HANDLED:
            continue
        phone = _norm_phone(c["phone"])
        if not phone:
            _MISSED_HANDLED.add(nid)
            continue
        # После пропущенного был состоявшийся разговор с этим номером (в любую сторону)?
        if any(_norm_phone(a["phone"]) == phone and a["created_at"] > c["created_at"] for a in answered):
            _MISSED_HANDLED.add(nid)
            continue
        ev = c["ev"]
        if ev["entity_type"] == "leads":
            lead_id = ev["entity_id"]
        else:
            lead_id = await amocrm_service.get_active_lead_for_contact(ev["entity_id"])
        if not lead_id or lead_id in _SPAM_HANDLED:
            _MISSED_HANDLED.add(nid)
            continue
        # Guard, переживающий рестарт. Пропускаем при ЛЮБОЙ открытой задаче, а не только
        # при своей: если менеджеру уже есть что делать по сделке, вторая задача про тот же
        # контакт — шум. Правило «одна открытая задача на сделку» здесь важнее полноты.
        if await amocrm_service.get_open_tasks(lead_id):
            _MISSED_HANDLED.add(nid)
            continue
        when = datetime.fromtimestamp(c["created_at"], _MSK_TZ).strftime("%H:%M")
        ok = await amocrm_service.create_task(
            lead_id=lead_id,
            text=f"Перезвонить: пропущенный звонок с +7{phone} в {when}, никто не перезвонил",
            complete_till=_next_work_slot_ts(),
            responsible_user_id=note.get("responsible_user_id"),
        )
        if ok:
            _MISSED_HANDLED.add(nid)
            logger.info(f"📵 Пропущенный без перезвона → задача по сделке #{lead_id} (звонок {when})")


async def _spam_autoclose_pass(all_calls: list, now: int):
    """Сделка со спам-вердиктом Block и без состоявшегося разговора → закрыть с причиной СПАМ."""
    seen = set()
    for c in all_calls:
        ev = c["ev"]
        if ev["entity_type"] != "leads":
            continue
        lead_id = ev["entity_id"]
        if lead_id in seen or lead_id in _SPAM_HANDLED:
            continue
        seen.add(lead_id)
        # Состоявшийся разговор по сделке → не автоспам, решает менеджер
        if any(x["duration"] >= MIN_CALL_SECONDS and x["ev"]["entity_type"] == "leads"
               and x["ev"]["entity_id"] == lead_id for x in all_calls):
            continue
        if now - _SPAM_CHECKED_AT.get(lead_id, 0) < 1800:
            continue
        _SPAM_CHECKED_AT[lead_id] = now
        notes = await amocrm_service.get_recent_notes("leads", lead_id, limit=10)
        def _txt(n):
            return ((n.get("params") or {}).get("text") or "")
        has_block = any("СПАМ-НОМЕР" in _txt(n) and "Block" in _txt(n) for n in notes)
        if not has_block:
            continue
        lead = await amocrm_service.get_lead(lead_id)
        if not lead or lead.get("status_id") in (142, 143):
            _SPAM_HANDLED.add(lead_id)
            continue
        if await amocrm_service.close_lead_as_spam(lead_id):
            _SPAM_HANDLED.add(lead_id)
            for t in await amocrm_service.get_open_tasks(lead_id):
                await amocrm_service.complete_task(t["id"], "Закрыто автоматически: спам-номер (вердикт Block)")
            logger.info(f"🚫 Сделка #{lead_id} закрыта как СПАМ (Block, разговора не было)")


async def _reconcile_once():
    """Один проход: найти звонки с записью без анализа и дообработать."""
    now = int(time.time())
    events = await amocrm_service.list_call_events(now - RECONCILE_LOOKBACK)
    if not events:
        return

    # Собираем ВСЕ звонки окна (включая пропущенные и короткие — они нужны
    # проходам missed/spam), затем фильтруем для дообработки анализом
    all_calls: list = []
    for ev in events:
        note = await amocrm_service.get_note_with_recording(
            ev["entity_type"].rstrip("s"), ev["entity_id"], ev["note_id"]
        )
        if not note:
            continue
        params = note.get("params") or {}
        try:
            duration = int(params.get("duration") or 0)
        except (ValueError, TypeError):
            duration = 0
        all_calls.append({
            "ev": ev, "note": note, "link": params.get("link"),
            "duration": duration, "phone": str(params.get("phone") or ""),
            "created_at": note.get("created_at", 0),
        })

    # Спам-проход раньше missed-прохода: на закрытую спам-сделку задача не ставится
    if SPAM_AUTOCLOSE_ENABLED:
        try:
            await _spam_autoclose_pass(all_calls, now)
        except Exception as e:
            logger.error(f"❌ Ошибка спам-прохода: {e}")
    if MISSED_CALL_TASK_ENABLED:
        try:
            await _missed_calls_pass(all_calls, now)
        except Exception as e:
            logger.error(f"❌ Ошибка прохода по пропущенным: {e}")

    # Группируем подходящие звонки по сделке (контактные резолвим в активную сделку)
    by_lead: dict = {}
    for c in all_calls:
        ev, note, link, duration = c["ev"], c["note"], c["link"], c["duration"]
        if now - ev["created_at"] < RECONCILE_MIN_AGE:
            continue
        if not link or duration < MIN_CALL_SECONDS:
            continue
        if ev["entity_type"] == "leads":
            lead_id = ev["entity_id"]
        else:
            # Сделку из догоняющего цикла НЕ создаём — только ищем активную
            lead_id = await amocrm_service.get_active_lead_for_contact(ev["entity_id"])
            if not lead_id:
                continue
        by_lead.setdefault(lead_id, []).append(
            {"ev": ev, "note": note, "link": link, "duration": duration}
        )

    for lead_id, calls in by_lead.items():
        calls.sort(key=lambda c: c["note"].get("created_at", 0))
        call_times = [c["note"].get("created_at", 0) for c in calls]
        note_times = await amocrm_service.list_analysis_note_times(
            lead_id, call_times[0] - _RECONCILE_CLAIM_SLACK
        )
        covered = _match_covered(call_times, note_times)
        for i, c in enumerate(calls):
            if i in covered:
                continue
            if await is_processed_peek(c["link"]):
                continue  # уже обработан/в работе у этого процесса
            # ВАЖНО: не вызывать здесь is_already_processed — она ПОМЕЧАЕТ url,
            # и process_call внутри себя видел «уже обработан» и молча выходил
            note = c["note"]
            logger.info(
                f"🩹 Reconcile: звонок без анализа — note {note.get('id')} "
                f"({c['duration']}с) → сделка #{lead_id}"
            )
            call_type = "incoming_call" if note.get("note_type") == "call_in" else "outgoing_call"
            try:
                await process_call(
                    entity_id=c["ev"]["entity_id"],
                    call_type=call_type,
                    record_url=c["link"],
                    call_created_at=note.get("created_at"),
                    responsible_user_id=note.get("responsible_user_id"),
                    phone=(note.get("params") or {}).get("phone", ""),
                    entity_type=c["ev"]["entity_type"],
                    call_direction=note.get("note_type"),
                    expected_duration=c["duration"],
                )
            except Exception as e:
                logger.error(f"❌ Reconcile: ошибка дообработки note {note.get('id')}: {e}")


# ============== Автозаполнение из чат-переписки (Авито / WhatsApp / прочие каналы) ==============
# Переписка сделок хранится на mcp-amocrm-server (вебхук add_message → БД на томе Railway).
# Когда диалог затихает, прогоняем его через AI-анализ: та же карточка, та же задача, что и для звонков.
CHAT_ANALYSIS_ENABLED = os.getenv("CHAT_ANALYSIS_ENABLED", "true").strip().lower() in ("1", "true", "yes")
CHAT_STORAGE_URL = os.getenv("CHAT_STORAGE_URL", "https://mcp-amocrm-server-production.up.railway.app").rstrip("/")
CHAT_QUIET_SECONDS = int(os.getenv("CHAT_QUIET_SECONDS", "1800"))   # диалог «затих» спустя, сек
CHAT_VOICE_TRANSCRIBE = os.getenv("CHAT_VOICE_TRANSCRIBE", "true").strip().lower() in ("1", "true", "yes")

_CHAT_ANALYZED: dict = {}       # lead_id -> ts, до которого переписка проанализирована
_CHAT_VOICE_CACHE: dict = {}    # message_id -> текст расшифровки голосового
CHAT_NOTE_MARKER = "АНАЛИЗ ПЕРЕПИСКИ"

# origin канала → имя значения поля «Источник» (только однозначные)
CHAT_ORIGIN_TO_SOURCE = {
    "avito": "Авито",
    "wappi": "WhatsApp",
    "whatsapp": "WhatsApp",
}


def _chat_origin_label(origin: str) -> str:
    o = (origin or "").lower()
    if "avito" in o:
        return "Авито"
    if "wappi" in o or "whatsapp" in o:
        return "WhatsApp"
    return origin or "чат"


def _chat_source_name(origin: str) -> str | None:
    o = (origin or "").lower()
    for key, name in CHAT_ORIGIN_TO_SOURCE.items():
        if key in o:
            return name
    return None


async def _fetch_chat_json(path: str) -> dict:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{CHAT_STORAGE_URL}{path}")
        response.raise_for_status()
        return response.json()


async def _transcribe_chat_voice(msg: dict) -> str:
    """Расшифровка голосового из чата (кэш по message_id; ошибки не фатальны)."""
    mid = msg.get("message_id") or str(msg.get("id"))
    if mid in _CHAT_VOICE_CACHE:
        return _CHAT_VOICE_CACHE[mid]
    text = "[голосовое сообщение]"
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(msg["media_url"])
            response.raise_for_status()
            audio = response.content
        if len(audio) > 1000:
            tr = await transcription_service.transcribe_audio(audio, speaker_labels=False)
            if (tr.full_text or "").strip():
                text = f"[голосовое: {tr.full_text.strip()}]"
    except Exception as e:
        logger.warning(f"⚠️ Не удалось расшифровать голосовое из чата ({mid}): {e}")
    if len(_CHAT_VOICE_CACHE) > 500:
        _CHAT_VOICE_CACHE.clear()
    _CHAT_VOICE_CACHE[mid] = text
    return text


def _chat_attachment_line(msg: dict) -> str:
    """Маркер вложения; имя файла берём из raw_payload — оно информативно."""
    mtype = msg.get("media_type")
    fname = ""
    try:
        fname = (json.loads(msg.get("raw_payload") or "{}").get("attachment") or {}).get("file_name") or ""
    except Exception:
        pass
    if mtype == "picture":
        return "[фото]"
    if mtype == "file":
        return f"[файл: {fname}]" if fname else "[файл]"
    return f"[вложение: {mtype}]"


async def _build_chat_dialog(messages: list) -> tuple[str, int]:
    """Хронологичный текст диалога. Возвращает (текст, число содержательных реплик)."""
    lines, meaningful = [], 0
    msgs = sorted(messages, key=lambda m: m.get("created_at") or 0)
    voice_budget = 8  # не расшифровываем больше N голосовых за один анализ
    for m in msgs:
        role = "Клиент" if m.get("is_incoming") else "Менеджер"
        when = datetime.fromtimestamp(m.get("created_at") or 0, _MSK_TZ).strftime("%d.%m %H:%M")
        text = (m.get("text") or "").strip()
        if not text and m.get("media_type") == "voice" and m.get("media_url"):
            if CHAT_VOICE_TRANSCRIBE and voice_budget > 0:
                voice_budget -= 1
                text = await _transcribe_chat_voice(m)
            else:
                text = "[голосовое сообщение]"
        elif not text and m.get("media_type"):
            text = _chat_attachment_line(m)
        if not text:
            continue
        if not text.startswith("[") or "голосовое:" in text or "файл:" in text:
            meaningful += 1
        lines.append(f"[{when}] {role}: {text}")
    return "\n".join(lines), meaningful


async def _chat_autofill_pass(now: int):
    """Затихшие диалоги с новыми входящими → анализ → поля + задача + примечание."""
    data = await _fetch_chat_json("/api/chat/recent?limit=100")
    by_lead: dict = {}
    for m in data.get("messages", []):
        lid = m.get("lead_id")
        if lid:
            by_lead.setdefault(int(lid), []).append(m)

    for lead_id, msgs in by_lead.items():
        last_ts = max(m.get("created_at") or 0 for m in msgs)
        if now - last_ts < CHAT_QUIET_SECONDS:
            continue  # диалог ещё активен — ждём затишья
        analyzed_to = _CHAT_ANALYZED.get(lead_id, 0)
        if analyzed_to == 0:
            # CRM-guard после рестарта: наша нота новее последнего сообщения?
            notes = await amocrm_service.get_recent_notes("leads", lead_id, limit=15)
            marks = [n.get("created_at", 0) for n in notes
                     if CHAT_NOTE_MARKER in ((n.get("params") or {}).get("text") or "")]
            if marks:
                analyzed_to = max(marks)
                _CHAT_ANALYZED[lead_id] = analyzed_to
        if not any((m.get("is_incoming") and (m.get("created_at") or 0) > analyzed_to) for m in msgs):
            continue  # новых входящих нет — рассылки/исходящие анализ не триггерят

        full = await _fetch_chat_json(f"/api/chat/lead/{lead_id}?limit=100")
        dialog, meaningful = await _build_chat_dialog(full.get("messages", []))
        if meaningful < 2:
            _CHAT_ANALYZED[lead_id] = last_ts
            continue

        origin = (msgs[0].get("origin") or "")
        channel = _chat_origin_label(origin)
        logger.info(f"💬 Анализ переписки [{channel}] по сделке #{lead_id} ({meaningful} реплик)")
        try:
            analysis = await analysis_service.analyze_chat(dialog, channel_name=channel)
        except Exception as e:
            logger.error(f"❌ Ошибка анализа переписки #{lead_id}: {e}")
            continue

        lead_data = await amocrm_service.get_lead(lead_id)
        responsible = (lead_data or {}).get("responsible_user_id")

        await auto_fill_lead_fields(
            lead_id, analysis, "incoming",
            source_fallback_name=_chat_source_name(origin),
        )
        await _update_contact_name_from_analysis(analysis, "leads", lead_id, lead_id)
        if AUTO_TASK_ENABLED:
            await _create_followup_task(lead_id, analysis, responsible)

        # ✉ — BMP-символ: 4-байтные эмодзи (💬) amoCRM молча вырезает из нот
        note_lines = [f"✉ {CHAT_NOTE_MARKER} (AI) [{channel}]", ""]
        if analysis.summary:
            note_lines += [analysis.summary, ""]
        note_lines.append(f" Работа: {analysis.work_type} |  Город: {analysis.client_city}")
        note_lines.append(f" Стоимость: {analysis.cost} |  Оплата: {analysis.payment_terms}")
        note_lines.append(f" Итог: {analysis.call_result} |  След. контакт: {analysis.next_contact_date}")
        if analysis.next_steps:
            note_lines += ["", "✅ Следующие шаги:"] + [f"- {s}" for s in analysis.next_steps]
        await amocrm_service.add_note_to_entity(lead_id, "\n".join(note_lines), "leads")

        _CHAT_ANALYZED[lead_id] = last_ts
        logger.info(f"✅ Переписка #{lead_id} проанализирована (до {datetime.fromtimestamp(last_ts, _MSK_TZ):%d.%m %H:%M})")


async def _reconcile_loop():
    logger.info(
        f"🩹 Догоняющий цикл включён: каждые {RECONCILE_INTERVAL}с, "
        f"окно {RECONCILE_LOOKBACK // 3600}ч, min age {RECONCILE_MIN_AGE}с"
    )
    while True:
        await asyncio.sleep(RECONCILE_INTERVAL)
        try:
            await _reconcile_once()
        except Exception as e:
            logger.error(f"❌ Ошибка догоняющего цикла: {e}")
        if CHAT_ANALYSIS_ENABLED:
            try:
                await _chat_autofill_pass(int(time.time()))
            except Exception as e:
                logger.error(f"❌ Ошибка чат-прохода: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Обработчик жизненного цикла приложения"""
    # Запуск
    logger.info("🚀 Запуск сервера транскрибации...")
    try:
        missing = validate_config()
        if missing:
            # Не валим процесс: Railway должен получить /health, а функциональность
            # будет зависеть от того, какие переменные заданы.
            logger.warning(f"⚠️ Не все переменные окружения заданы: {', '.join(missing)}")
        else:
            logger.info("✅ Конфигурация валидна")

        # Проверяем наличие ffmpeg (нужен для стерео-диаризации)
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if ffmpeg_path and ffprobe_path:
            logger.info(f"✅ ffmpeg найден: {ffmpeg_path}")
            logger.info(f"✅ ffprobe найден: {ffprobe_path}")
        else:
            logger.warning(f"⚠️ ffmpeg={'найден' if ffmpeg_path else 'НЕ НАЙДЕН'}, ffprobe={'найден' if ffprobe_path else 'НЕ НАЙДЕН'}")
            logger.warning("⚠️ Стерео-диаризация будет недоступна, fallback на mono Whisper")

        logger.info("🟢 Сервер запущен")
    except Exception as e:
        # Не валим старт: пусть поднимется хотя бы healthcheck.
        logger.error(f"❌ Ошибка конфигурации/старта: {e}")

    reconcile_task = None
    if RECONCILE_ENABLED:
        reconcile_task = asyncio.create_task(_reconcile_loop())
    else:
        logger.info("🩹 Догоняющий цикл выключен (RECONCILE_ENABLED=false)")

    yield

    if reconcile_task:
        reconcile_task.cancel()

    # Остановка
    logger.info("🛑 Остановка сервера...")
    # Не спамим в Telegram
    # await telegram_service.send_shutdown()
    logger.info("🔴 Сервер остановлен")


app = FastAPI(
    title="Voice Transcription Service",
    description="Сервис транскрибации звонков AmoCRM с диаризацией",
    version="1.0.0",
    lifespan=lifespan
)


async def _delayed_process_call(delay: int = 0, **kwargs):
    """Обёртка: ждёт delay секунд, затем вызывает process_call."""
    if delay > 0:
        await asyncio.sleep(delay)
    await process_call(**kwargs)


async def process_call(
    entity_id: int,
    call_type: str,
    record_url: str,
    call_created_at: Optional[int] = None,
    responsible_user_id: Optional[int] = None,
    phone: str = "",
    entity_type: str = "leads",
    call_direction: str = "call_in",
    expected_duration: Optional[int] = None,
):
    """
    Основная функция обработки звонка.
    Выполняется в фоновом режиме.
    """
    try:
        # 0. Проверяем дубликаты
        if await is_already_processed(record_url):
            logger.info(f"⏭️ Звонок {record_url[:50]}... уже обрабатывается или обработан, скипаем")
            return

        # ВАЖНО: если звонок привязан к контакту, находим АКТИВНУЮ сделку или создаём новую!
        # Логика согласно документации AmoCRM:
        # 1. Если звонок привязан к контакту → запрашиваем его сделки
        # 2. Если есть активная (не закрытая) сделка → используем её
        # 3. Если все сделки закрыты или нет сделок → создаём новую
        # 4. Добавляем примечание в найденную/созданную сделку
        
        lead_id = entity_id
        target_entity_type = entity_type
        
        # Нормализуем entity_type для проверки (AmoCRM может вернуть "contact" или "contacts")
        normalized_check = entity_type.lower()
        if normalized_check in ["contact", "contacts"]:
            logger.info(f"🔍 Звонок привязан к контакту #{entity_id}")
            logger.info(f"📋 Запрашиваем сделки контакта #{entity_id}...")
            
            # Получаем контакт для проверки
            contact = await amocrm_service.get_contact(entity_id)
            if contact:
                contact_name = contact.get("name", "")
                logger.info(f"📇 Контакт: {contact_name}")
            
            # Ищем активную сделку или создаём новую
            found_lead = await amocrm_service.get_or_create_lead_for_contact(
                contact_id=entity_id,
                phone=phone,
                responsible_user_id=responsible_user_id
            )
            
            if found_lead and found_lead != entity_id:
                # Убеждаемся, что получили ID сделки, а не контакта
                lead_id = found_lead
                target_entity_type = "leads"
                logger.info(f"✅ Используем сделку #{lead_id} для контакта #{entity_id}")
            else:
                # Крайний случай - не удалось создать сделку или вернулся тот же ID
                logger.error(f"❌ Не удалось найти/создать сделку для контакта #{entity_id}. Получено: {found_lead}")
                return
        
        logger.info(f"📞 Обработка звонка → {target_entity_type}/{lead_id}, тип: {call_type}")
        
        # 1. Получаем имя менеджера
        manager_name = "Менеджер"
        if responsible_user_id:
            manager_name = amocrm_service.get_manager_name(responsible_user_id)
            if manager_name.startswith("Менеджер #"):
                user = await amocrm_service.get_user(responsible_user_id)
                if user:
                    manager_name = user.get("name", manager_name)
        
        # 2. Скачиваем запись (если не загружена вручную)
        if record_url.startswith("uploaded://"):
            logger.error("❌ process_call вызван с uploaded:// URL - используйте process_uploaded_audio")
            return
        
        logger.info("📥 Скачиваем запись...")
        audio_data = await amocrm_service.download_call_recording(record_url)

        if len(audio_data) < 10000:
            logger.warning(f"⚠️ Файл слишком маленький ({len(audio_data)} байт)")
            # запись могла быть ещё не готова — снимаем метку, reconcile перепроверит
            async with PROCESSED_LOCK:
                PROCESSED_CALLS.discard(record_url)
            return

        # 2.1. Проверяем длительность скачанного аудио vs ожидаемой (защита от неготовой записи)
        if expected_duration and expected_duration > 10:
            audio_data = await _ensure_full_recording(
                audio_data, record_url, expected_duration
            )

        # 3. Транскрибируем
        logger.info("🎙️ Транскрибация...")
        transcription = await transcription_service.transcribe_audio(
            audio_data, speaker_labels=True, call_direction=call_direction,
        )
        # STT ответил — если до этого был зафиксирован сбой, сообщим о восстановлении
        await alerts.notify_recovered("OpenAI")

        if not (transcription.full_text or "").strip():
            logger.warning("⚠️ Пустая транскрибация (с диаризацией). Пробуем без диаризации...")
            transcription = await transcription_service.transcribe_audio(audio_data, speaker_labels=False)

        if len((transcription.full_text or "").strip()) < 50:
            logger.warning(
                f"⚠️ Транскрибация слишком короткая ({len((transcription.full_text or '').strip())} символов). "
                "Пробуем без диаризации для улучшения..."
            )
            fallback = await transcription_service.transcribe_audio(audio_data, speaker_labels=False)
            if len((fallback.full_text or "").strip()) > len((transcription.full_text or "").strip()):
                transcription = fallback
                logger.info("✅ Используем транскрипцию без диаризации (получилось длиннее)")

        if not (transcription.full_text or "").strip():
            logger.warning("⚠️ Транскрибация пустая даже после retry — пропускаем обработку")
            return

        # 3.1. Проверяем длительность звонка
        # Меньше 60 секунд = автоответчик, сброс, не состоялся — не обрабатываем
        if transcription.duration_seconds < MIN_CALL_SECONDS:
            logger.info(
                f"⏭️ Звонок слишком короткий ({transcription.duration_seconds:.0f} сек < {MIN_CALL_SECONDS} сек) — "
                "пропускаем обработку"
            )
            return

        # 4. Определяем роли
        if transcription.speakers:
            roles = transcription_service.identify_roles(transcription.speakers)
            formatted_transcript = transcription_service.format_with_roles(
                transcription.speakers, 
                roles
            )
        else:
            formatted_transcript = transcription.full_text or ""
        logger.info(f"📝 Транскрибация: {len(formatted_transcript)} символов")
        
        # 5. Анализируем через Claude
        logger.info("🤖 Анализ через Claude...")
        call_type_simple = "outgoing" if "outgoing" in call_type else "incoming"
        analysis = await analysis_service.analyze_call(
            formatted_transcript,
            call_type=call_type_simple,
            manager_name=manager_name,
            speakers=transcription.speakers,
            call_direction=call_direction,
        )

        # 5.5. Автозаполнение — только при достоверной диаризации (fail-closed):
        # если роли под сомнением, поля и имя контакта НЕ трогаем (чтобы не записать
        # данные менеджера как данные клиента), но расшифровку сохраняем и алертим.
        stats = getattr(analysis, "speaker_stats", None)
        suspicious_reasons = []
        if stats is not None and getattr(stats, "suspicious_diarization", False):
            suspicious_reasons.append(getattr(stats, "suspicious_reason", "") or "speaker_stats")
        if getattr(transcription, "roles_uncertain", False):
            suspicious_reasons.append("roles_uncertain")

        if suspicious_reasons:
            reason_text = ", ".join(suspicious_reasons)
            logger.warning(f"🚫 Диаризация под сомнением ({reason_text}) — автозаполнение пропущено")
            await _alert_suspicious_diarization(lead_id, reason_text)
        else:
            if target_entity_type == "leads":
                await auto_fill_lead_fields(lead_id, analysis, call_type_simple)

            # 5.6. Имя клиента → карточка контакта (через вебхук-контакт или контакт сделки)
            await _update_contact_name_from_analysis(analysis, entity_type, entity_id, lead_id)

            # 5.7. Задача менеджеру по итогам звонка (guard: не дублируем открытые)
            if AUTO_TASK_ENABLED and target_entity_type == "leads":
                await _create_followup_task(lead_id, analysis, responsible_user_id)

        # 6. Формируем примечание
        note_text = analysis_service.format_note(
            analysis,
            call_type=call_type_simple,
            duration_seconds=transcription.duration_seconds,
            manager_name=manager_name,
            # Фактический провайдер, а не настройка: при сбое Whisper
            # сработает автофолбэк, и в шапке должно стоять «AssemblyAI»
            stt_provider=getattr(transcription, "stt_provider", STT_PROVIDER),
        )
        
        # 7. Сохраняем в AmoCRM (в СДЕЛКУ!)
        logger.info(f"💾 Сохраняем примечание в {target_entity_type}/{lead_id}...")
        try:
            await amocrm_service.add_note_to_entity(lead_id, note_text, target_entity_type)
            logger.info(f"✅ Примечание успешно добавлено к {target_entity_type}/{lead_id}")

        except Exception as note_error:
            logger.error(f"❌ Ошибка добавления примечания к {target_entity_type}/{lead_id}: {note_error}")
            # Проверяем, может быть это ID контакта, а не сделки?
            if target_entity_type == "leads":
                logger.error(f"⚠️ ВНИМАНИЕ: Пытались добавить примечание к сделке #{lead_id}, но получили ошибку!")
                logger.error(f"⚠️ Возможно, {lead_id} - это ID контакта, а не сделки!")
            raise
        
        # 8. Отправляем красивый анализ в Telegram
        # Время: Railway работает в UTC, для Москвы всегда +3 часа.
        if call_created_at:
            ts = int(call_created_at)
            if ts > 10**12:
                ts = ts // 1000
            utc_dt = datetime.utcfromtimestamp(ts)
            moscow_dt = utc_dt + timedelta(hours=3)
            call_datetime = moscow_dt.strftime("%d.%m.%Y %H:%M")
            logger.info(f"🕐 Время звонка: UTC={utc_dt.strftime('%H:%M')} → МСК={call_datetime}")
        else:
            moscow_dt = datetime.utcnow() + timedelta(hours=3)
            call_datetime = moscow_dt.strftime("%d.%m.%Y %H:%M")
            logger.info(f"🕐 Время звонка (текущее): МСК={call_datetime}")
        amocrm_url = f"https://{AMOCRM_DOMAIN}/{target_entity_type}/detail/{lead_id}"
        
        tg_ok = await telegram_service.send_call_analysis(
            call_datetime=call_datetime,
            call_type=call_type_simple,
            phone=phone or "Не определён",
            manager_name=analysis.manager_name,
            client_name=analysis.client_name,
            summary=analysis.summary,
            amocrm_url=amocrm_url,
            record_url=record_url,
            client_city=analysis.client_city,
            work_type=analysis.work_type,
            cost=analysis.cost,
            payment_terms=analysis.payment_terms,
            call_result=analysis.call_result,
            next_contact_date=analysis.next_contact_date,
            next_steps=analysis.next_steps,
        )
        if not tg_ok:
            logger.warning(f"⚠️ Telegram: уведомление не отправлено (проверьте TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID)")
        
        logger.info(f"✅ Звонок для сделки #{lead_id} успешно обработан!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки звонка для сделки #{lead_id}: {e}")
        # Снимаем метку дедупликации: иначе упавший звонок навсегда числится
        # «обработанным» и догоняющий цикл его пропускает (случай 34857413 26.08)
        async with PROCESSED_LOCK:
            PROCESSED_CALLS.discard(record_url)
        # Обычные ошибки звонка НЕ шлём в Telegram (избегаем спама) — только логируем.
        # Но если это инфраструктурный сбой (кончился баланс / отозван ключ),
        # конвейер стоит целиком, и об этом нужно узнать сразу. Троттлинг внутри.
        await alerts.maybe_alert(e, lead_id=lead_id)


@app.get("/")
async def root():
    """Проверка работоспособности"""
    return {
        "status": "ok",
        "service": "Voice Transcription Service",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check для Railway"""
    return {"status": "healthy"}


@app.get("/test-telegram")
async def test_telegram():
    """
    Проверка отправки в Telegram.
    Вызови: GET /test-telegram — должно прийти тестовое сообщение.
    """
    ok = await telegram_service.send_message(
        "🧪 <b>Тест</b>: сервис транскрибации работает. Telegram подключён.",
        disable_notification=True
    )
    return {"telegram_ok": ok}


@app.post("/webhook/amocrm")
async def amocrm_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Webhook endpoint для AmoCRM.
    
    AmoCRM отправляет webhook когда создаётся примечание о звонке.
    Примечание уже содержит ссылку на запись (params.link).
    """
    try:
        # 1. Получаем данные от AmoCRM
        form_data = await request.form()
        body = dict(form_data)
        
        # Логируем ВСЕ ключи связанные с примечаниями для отладки
        note_keys = [k for k in body.keys() if '[note]' in k]
        if note_keys:
            logger.info(f"📨 Webhook примечание, ключей: {len(note_keys)}")
            # Логируем первые 10 ключей для отладки
            for k in note_keys[:10]:
                logger.info(f"  {k} = {body[k]}")
        else:
            # Это не примечание - другой тип webhook
            keys_preview = list(body.keys())[:5]
            logger.info(f"📨 Webhook (не примечание): {keys_preview}")
        
        # 2. Ищем примечание о звонке в webhook
        # AmoCRM отправляет: contacts[note][0][note][id], contacts[note][0][note][element_id], etc.
        note_id = None
        element_id = None  # ID контакта/сделки к которому привязано примечание
        entity_type = None
        note_type = None
        responsible_user_id = None
        
        for key, value in body.items():
            # Ищем note[id] - ID самого примечания
            if "[note][id]" in key and value:
                try:
                    note_id = int(value)
                except (ValueError, TypeError):
                    pass
            
            # Ищем element_id - ID сущности (контакта/сделки)
            if "[note][element_id]" in key and value:
                try:
                    element_id = int(value)
                except (ValueError, TypeError):
                    pass
            
            # Определяем тип сущности
            if "contacts[note]" in key:
                entity_type = "contacts"
            elif "leads[note]" in key:
                entity_type = "leads"
            
            # Тип примечания (call_in, call_out, common, etc.)
            if "[note][note_type]" in key and value:
                note_type = value
            
            # Ответственный
            if "[note][responsible_user_id]" in key and value:
                try:
                    responsible_user_id = int(value)
                except (ValueError, TypeError):
                    pass
        
        # 3. Если это не примечание - игнорируем (не спамим в лог)
        if not element_id or not entity_type:
            # Это webhook о создании контакта/сделки/задачи - не о звонке
            return JSONResponse(content={"status": "ignored", "reason": "not_a_note"}, status_code=200)
        
        # Логируем извлечённые данные для отладки
        logger.info(f"📋 Извлечено: note_id={note_id}, element_id={element_id}, entity={entity_type}, note_type={note_type}")
        
        # 4. Получаем данные примечания
        note_data = None
        
        if note_id:
            # Если note_id найден - запрашиваем конкретное примечание
            logger.info(f"📝 Запрос примечания #{note_id} для {entity_type}/{element_id}")
            note_data = await amocrm_service.get_note_with_recording(
                entity_type=entity_type.rstrip('s'),  # contacts -> contact
                entity_id=element_id,
                note_id=note_id
            )
        else:
            # Если note_id не найден - получаем последние примечания и ищем звонок
            logger.info(f"🔍 note_id не в webhook, ищем последние примечания {entity_type}/{element_id}")
            recent_notes = await amocrm_service.get_recent_notes(
                entity_type=entity_type,
                entity_id=element_id,
                limit=5
            )
            
            # Ищем примечание о звонке среди последних
            for note in recent_notes:
                if note.get("note_type") in ["call_in", "call_out"]:
                    note_data = note
                    logger.info(f"✅ Найдено примечание о звонке: #{note.get('id')}")
                    break
        
        if not note_data:
            logger.warning(f"⚠️ Не удалось найти примечание о звонке")
            return JSONResponse(content={"status": "note_not_found"}, status_code=200)
        
        # 6. Проверяем тип примечания
        actual_note_type = note_data.get("note_type")
        if actual_note_type not in ["call_in", "call_out"]:
            # Это обычное примечание, не звонок
            logger.info(f"⏭️ Примечание #{note_id} не звонок (тип: {actual_note_type})")
            return JSONResponse(content={"status": "not_a_call", "note_type": actual_note_type}, status_code=200)
        
        # 7. Извлекаем ссылку на запись и длительность
        params = note_data.get("params", {})
        record_url = params.get("link")
        phone = params.get("phone", "")
        call_duration = params.get("duration")  # длительность в секундах от АТС
        try:
            call_duration = int(call_duration) if call_duration else None
        except (ValueError, TypeError):
            call_duration = None

        if not record_url:
            logger.warning(f"⚠️ Примечание #{note_id} без записи")
            return JSONResponse(content={"status": "no_recording"}, status_code=200)
        
        logger.info(f"✅ Найден звонок! Тип: {actual_note_type}, запись: {record_url[:50]}...")
        
        # 8. Определяем тип звонка
        call_type = "incoming_call" if actual_note_type == "call_in" else "outgoing_call"
        
        # 9. Запускаем обработку в фоне.
        # onlinePBX отдаёт запись сразу — стартуем без паузы. Если файл окажется
        # неполным, _ensure_full_recording сам подождёт и перекачает (наследие vmclouds).
        initial_delay = 0

        raw_created_at = note_data.get("created_at")
        background_tasks.add_task(
            _delayed_process_call,
            delay=initial_delay,
            entity_id=element_id,
            call_type=call_type,
            record_url=record_url,
            call_created_at=raw_created_at,
            responsible_user_id=responsible_user_id or note_data.get("responsible_user_id"),
            phone=phone,
            entity_type=entity_type,
            call_direction=actual_note_type,
            expected_duration=call_duration,
        )
        
        return JSONResponse(content={"status": "processing", "note_id": note_id}, status_code=200)
        
    except Exception as e:
        logger.error(f"❌ Webhook ошибка: {e}")
        return JSONResponse(content={"status": "error"}, status_code=200)


@app.post("/upload-audio")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    lead_id: int = Form(...),
    call_type: str = Form("incoming_call"),
    phone: str = Form(""),
    manager_name: str = Form("Менеджер"),
    # Unix timestamp (секунды или миллисекунды) из AmoCRM, если ваш MCP/интеграция его знает
    call_created_at: Optional[int] = Form(None),
):
    """
    Загрузка аудиофайла вручную для транскрибации.
    
    Используй когда SSL сертификат не работает:
    1. Скачай запись вручную
    2. Загрузи через этот endpoint
    3. Результат появится в AmoCRM и Telegram
    
    Пример curl:
    curl -X POST https://voice-transcription-production.up.railway.app/upload-audio \
      -F "file=@recording.mp3" \
      -F "lead_id=12345" \
      -F "call_type=incoming_call" \
      -F "phone=+79001234567"
    """
    try:
        # Читаем файл
        audio_data = await file.read()
        logger.info(f"📤 Загружен файл: {file.filename}, размер: {len(audio_data)} байт")
        
        if len(audio_data) < 10000:
            raise HTTPException(status_code=400, detail="Файл слишком маленький")
        
        # Запускаем обработку напрямую (без скачивания)
        background_tasks.add_task(
            process_uploaded_audio,
            audio_data=audio_data,
            lead_id=lead_id,
            call_type=call_type,
            phone=phone,
            manager_name=manager_name,
            call_created_at=call_created_at,
        )
        
        return {
            "status": "processing",
            "lead_id": lead_id,
            "file_size": len(audio_data),
            "message": "Файл принят в обработку. Результат появится в Telegram и AmoCRM."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_uploaded_audio(
    audio_data: bytes,
    lead_id: int,
    call_type: str,
    phone: str,
    manager_name: str,
    call_created_at: Optional[int] = None,
    call_direction: str = "call_in",
):
    """Обработка загруженного аудио (без скачивания)"""
    try:
        logger.info(f"📞 Обработка загруженного аудио для сделки #{lead_id}")
        
        # Используем общую логику обработки (без скачивания)
        # 1. Транскрибируем
        logger.info("🎙️ Транскрибация...")
        transcription = await transcription_service.transcribe_audio(
            audio_data, speaker_labels=True, call_direction=call_direction,
        )
        await alerts.notify_recovered("OpenAI")

        if not (transcription.full_text or "").strip():
            logger.warning("⚠️ Пустая транскрибация (с диаризацией). Пробуем без диаризации...")
            transcription = await transcription_service.transcribe_audio(audio_data, speaker_labels=False)

        if len((transcription.full_text or "").strip()) < 50:
            logger.warning(
                f"⚠️ Транскрибация слишком короткая ({len((transcription.full_text or '').strip())} символов). "
                "Пробуем без диаризации для улучшения..."
            )
            fallback = await transcription_service.transcribe_audio(audio_data, speaker_labels=False)
            if len((fallback.full_text or "").strip()) > len((transcription.full_text or "").strip()):
                transcription = fallback
                logger.info("✅ Используем транскрипцию без диаризации (получилось длиннее)")

        if not (transcription.full_text or "").strip():
            logger.warning("⚠️ Транскрибация пустая даже после retry — пропускаем обработку")
            return

        # 1.1. Проверяем длительность звонка
        if transcription.duration_seconds < MIN_CALL_SECONDS:
            logger.info(
                f"⏭️ Звонок слишком короткий ({transcription.duration_seconds:.0f} сек < {MIN_CALL_SECONDS} сек) — "
                "пропускаем обработку"
            )
            return

        # 2. Определяем роли
        if transcription.speakers:
            roles = transcription_service.identify_roles(transcription.speakers)
            formatted_transcript = transcription_service.format_with_roles(
                transcription.speakers, 
                roles
            )
        else:
            formatted_transcript = transcription.full_text or ""
        logger.info(f"📝 Транскрибация: {len(formatted_transcript)} символов")
        
        # 3. Анализируем через Claude
        logger.info("🤖 Анализ через Claude...")
        call_type_simple = "outgoing" if "outgoing" in call_type else "incoming"
        analysis = await analysis_service.analyze_call(
            formatted_transcript,
            call_type=call_type_simple,
            manager_name=manager_name,
            speakers=transcription.speakers,
            call_direction=call_direction,
        )
        
        # 4. Формируем примечание
        note_text = analysis_service.format_note(
            analysis,
            call_type=call_type_simple,
            duration_seconds=transcription.duration_seconds,
            manager_name=manager_name,
            # Фактический провайдер, а не настройка: при сбое Whisper
            # сработает автофолбэк, и в шапке должно стоять «AssemblyAI»
            stt_provider=getattr(transcription, "stt_provider", STT_PROVIDER),
        )
        
        # 5. Сохраняем в AmoCRM (в СДЕЛКУ!)
        logger.info(f"💾 Сохраняем примечание в leads/{lead_id}...")
        await amocrm_service.add_note_to_entity(lead_id, note_text, "leads")
        logger.info(f"✅ Примечание успешно добавлено к leads/{lead_id}")

        # 6. Отправляем красивый анализ в Telegram
        # Время: Railway работает в UTC, для Москвы всегда +3 часа.
        if call_created_at:
            ts = int(call_created_at)
            if ts > 10**12:
                ts = ts // 1000
            utc_dt = datetime.utcfromtimestamp(ts)
            moscow_dt = utc_dt + timedelta(hours=3)
            call_datetime = moscow_dt.strftime("%d.%m.%Y %H:%M")
            logger.info(f"🕐 Время звонка (upload): UTC={utc_dt.strftime('%H:%M')} → МСК={call_datetime}")
        else:
            moscow_dt = datetime.utcnow() + timedelta(hours=3)
            call_datetime = moscow_dt.strftime("%d.%m.%Y %H:%M")
            logger.info(f"🕐 Время звонка (upload, текущее): МСК={call_datetime}")
        amocrm_url = f"https://{AMOCRM_DOMAIN}/leads/detail/{lead_id}"
        
        tg_ok = await telegram_service.send_call_analysis(
            call_datetime=call_datetime,
            call_type=call_type_simple,
            phone=phone or "Не определён",
            manager_name=analysis.manager_name,
            client_name=analysis.client_name,
            summary=analysis.summary,
            amocrm_url=amocrm_url,
            record_url="",
            client_city=analysis.client_city,
            work_type=analysis.work_type,
            cost=analysis.cost,
            payment_terms=analysis.payment_terms,
            call_result=analysis.call_result,
            next_contact_date=analysis.next_contact_date,
            next_steps=analysis.next_steps,
        )
        if not tg_ok:
            logger.warning(f"⚠️ Telegram: уведомление не отправлено (проверьте TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID)")
        
        logger.info(f"✅ Загруженный файл для сделки #{lead_id} обработан!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки загруженного файла: {e}")
        await alerts.maybe_alert(e, lead_id=lead_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=DEBUG
    )
