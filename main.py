"""
Главный файл приложения.
FastAPI сервер с webhook endpoint для AmoCRM.

Запуск:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import logging
import asyncio
import re
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import JSONResponse

from config import PORT, DEBUG, AMOCRM_DOMAIN, STT_PROVIDER, validate_config
from services.amocrm import amocrm_service
from services.transcription import transcription_service
from services.analysis import analysis_service
from services.telegram import telegram_service

# ============== Маппинг work_type → enum_id для поля "Интерес" (field_id=212083) ==============
WORK_TYPE_ENUM_MAP = {
    "межевание": 423475,
    "вынос": 423477,
    "вынос в натуру": 423477,
    "вынос границ": 423477,
    "топосъёмка": 423479,
    "топосъемка": 423479,
    "топографическая съёмка": 423479,
    "топографическая съемка": 423479,
    "техплан": 423495,
    "технический план": 423495,
    "техпаспорт": 1265209,
    "технический паспорт": 1265209,
    "акт обследования": 1276685,
    "схема": 423483,
    "схема на кпт": 423483,
    "геодезическое сопровождение": 1329477,
    "геодезия": 1329477,
    "раздел": 423507,
    "раздел зу": 423507,
    "разбивка": 423505,
    "исполнительная съемка": 1276379,
    "исполнительная съёмка": 1276379,
    "проектирование": 423491,
    "геология": 423493,
    "экология": 423497,
    "подбор зу": 1342589,
    "межевание и тех план": 1344645,
    "межевание и техплан": 1344645,
    "прочие": 1324619,
    "подача в рр через лк": 1344085,
    "все виды изысканий": 1314061,
    "снижение кс": 423489,
    "ппт": 423481,
    "гидрометеорология": 423499,
    "водоохранная зона": 423509,
}


# ============== Маппинг тег → enum_id для поля "Источник" (field_id=212063) ==============
TAG_TO_SOURCE_ENUM = {
    "2гис": 423423,       # 2GIS
    "2gis": 423423,
    "авито": 423433,      # Авито
    "avito": 423433,
    "seo": 423411,        # SEO
    "ltv": 423413,        # LTV
    "ltv-сарафан": 423415,
    "ltv-тиу": 423425,
    "ltv-пульс": 423427,
    "пульс цен": 423421,
    "рекомендация": 423429,
    "улица": 423431,
    "торги": 423437,
    "1777": 706911,
    "инстаграмм": 706913,
    "instagram": 706913,
    "юла": 707821,
    "личные": 709191,
    "оффлайн": 855121,
    "исходящий": 1237601,
    "агентство недвижимости": 1260789,
    "городской": 1276687,
    "яндекс ук": 1342883,
    "profzem": 1342911,
    "почта": 1343301,
    "кадуслуги": 1343847,
    "сайт кадуслуги": 1343847,
    "lp new яндекс": 423407,
    "lp new google": 423409,
    "менеджер": 423419,
}


def _match_tag_to_source(tags: list) -> int | None:
    """Ищет enum_id для поля Источник по тегам сделки."""
    for tag in tags:
        tag_name = (tag.get("name") or "").lower().strip()
        if tag_name in TAG_TO_SOURCE_ENUM:
            return TAG_TO_SOURCE_ENUM[tag_name]
        # Частичное совпадение
        for keyword, enum_id in TAG_TO_SOURCE_ENUM.items():
            if keyword in tag_name or tag_name in keyword:
                return enum_id
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


def _match_work_type_enum(work_type_text: str):
    """Ищет enum_id для поля Интерес по тексту work_type из анализа."""
    if not work_type_text:
        return None
    text = work_type_text.lower().strip()
    # Точное совпадение
    if text in WORK_TYPE_ENUM_MAP:
        return WORK_TYPE_ENUM_MAP[text]
    # Частичное совпадение
    for keyword, enum_id in WORK_TYPE_ENUM_MAP.items():
        if keyword in text:
            return enum_id
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


async def auto_fill_lead_fields(lead_id: int, analysis, call_type_simple: str):
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

        # 0. Источник по тегам (field_id=212063, select)
        if 212063 not in existing_fields:
            tags = lead_data.get("_embedded", {}).get("tags", [])
            if tags:
                source_enum_id = _match_tag_to_source(tags)
                if source_enum_id:
                    custom_fields.append({
                        "field_id": 212063,
                        "values": [{"enum_id": source_enum_id}]
                    })
                    tag_names = ", ".join(t.get("name", "") for t in tags)
                    logger.info(f"  📢 Источник (по тегам [{tag_names}]): enum_id={source_enum_id}")

        # 1. Город (field_id=212029, text)
        if 212029 not in existing_fields:
            city = getattr(analysis, "client_city", "")
            if city and city.lower() not in ("не указано", "не определено", ""):
                custom_fields.append({
                    "field_id": 212029,
                    "values": [{"value": city}]
                })
                logger.info(f"  📍 Город: {city}")

        # 2. Интерес / work_type (field_id=212083, select)
        if 212083 not in existing_fields:
            work_type = getattr(analysis, "work_type", "")
            if work_type and work_type.lower() not in ("не обсуждали", "не указано", "не определено", ""):
                enum_id = _match_work_type_enum(work_type)
                if enum_id:
                    custom_fields.append({
                        "field_id": 212083,
                        "values": [{"enum_id": enum_id}]
                    })
                    logger.info(f"  🔧 Интерес: {work_type} → enum_id={enum_id}")

        # 3. Тип сделки (field_id=212099, select: входящий=423541, исходящий=423543)
        if 212099 not in existing_fields:
            enum_id = 423541 if call_type_simple == "incoming" else 423543
            custom_fields.append({
                "field_id": 212099,
                "values": [{"enum_id": enum_id}]
            })
            logger.info(f"  📞 Тип сделки: {call_type_simple}")

        # 4. Схема оплаты (field_id=767917, text)
        if 767917 not in existing_fields:
            payment = getattr(analysis, "payment_terms", "")
            if payment and payment.lower() not in ("не обсуждали", "не указано", "не определено", ""):
                custom_fields.append({
                    "field_id": 767917,
                    "values": [{"value": payment}]
                })
                logger.info(f"  💳 Схема оплаты: {payment}")

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
                if location and location.lower() not in ("не указано", "не определено", ""):
                    name_to_set = f"{short_name} {location}"
                else:
                    name_to_set = short_name
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
    
    yield
    
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

        # 3.1. Проверяем минимальную длину транскрибации
        # Слишком короткий текст = звонок не состоялся (гудки, "алло-алло", сброс)
        transcript_text = (transcription.full_text or "").strip()
        if len(transcript_text) < 100:
            logger.info(
                f"⏭️ Транскрибация слишком короткая ({len(transcript_text)} символов) — "
                "звонок не состоялся, пропускаем обработку"
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

        # 5.5. Автозаполнение полей сделки
        if target_entity_type == "leads":
            await auto_fill_lead_fields(lead_id, analysis, call_type_simple)

        # 5.6. Обновление имени контакта (если привязан к контакту)
        normalized_check_type = entity_type.lower()
        if normalized_check_type in ["contact", "contacts"]:
            contact_id = entity_id  # entity_id = исходный ID контакта
            client_name = getattr(analysis, "client_name", "")
            if client_name and client_name.lower() not in ("клиент", "не представился", ""):
                # Проверяем, что текущее имя дефолтное (номер телефона или пустое)
                contact_data = await amocrm_service.get_contact(contact_id)
                if contact_data:
                    current_name = contact_data.get("name", "") or ""
                    # Считаем дефолтным: пустое, или начинается с цифры/+  (номер телефона)
                    is_default = (
                        not current_name
                        or current_name.lstrip("+").replace(" ", "").replace("-", "").isdigit()
                    )
                    if is_default:
                        await amocrm_service.update_contact_name(contact_id, client_name)
                        logger.info(f"  👤 Имя контакта #{contact_id}: '{current_name}' → '{client_name}'")

        # 6. Формируем примечание
        note_text = analysis_service.format_note(
            analysis,
            call_type=call_type_simple,
            duration_seconds=transcription.duration_seconds,
            manager_name=manager_name,
            stt_provider=STT_PROVIDER,
        )
        
        # 7. Сохраняем в AmoCRM (в СДЕЛКУ!)
        logger.info(f"💾 Сохраняем примечание в {target_entity_type}/{lead_id}...")
        try:
            await amocrm_service.add_note_to_entity(lead_id, note_text, target_entity_type)
            logger.info(f"✅ Примечание успешно добавлено к {target_entity_type}/{lead_id}")

            # 7.1. Второе примечание: полная расшифровка разговора
            minutes = int(transcription.duration_seconds // 60)
            seconds = int(transcription.duration_seconds % 60)
            duration_str = f"{minutes} мин {seconds} сек" if minutes else f"{seconds} сек"
            call_type_str = "Входящий" if call_type_simple == "incoming" else "Исходящий"

            full_transcript_note = (
                "📜 ПОЛНАЯ РАСШИФРОВКА ЗВОНКА\n\n"
                f"📞 {call_type_str} | {duration_str}\n\n"
                f"{formatted_transcript}"
            )
            try:
                await amocrm_service.add_note_to_entity(lead_id, full_transcript_note, target_entity_type)
                logger.info(f"✅ Полная расшифровка добавлена к {target_entity_type}/{lead_id}")
            except Exception as full_note_error:
                # Не валим обработку: анализ уже сохранён, а полный текст можно починить отдельно
                logger.error(f"❌ Ошибка добавления полной расшифровки к {target_entity_type}/{lead_id}: {full_note_error}")
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
        # НЕ отправляем ошибки в Telegram - только логируем (избегаем спама)


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
        
        # 9. Запускаем обработку в фоне
        # Начальная задержка: даём vmclouds время на обработку записи
        if call_duration and call_duration > 30:
            initial_delay = 15  # 15 секунд для записей длиннее 30с
            logger.info(f"⏳ Ждём {initial_delay}с перед скачиванием (запись {call_duration}с, vmclouds может быть не готов)")
        else:
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

        # 1.1. Проверяем минимальную длину транскрибации
        transcript_text = (transcription.full_text or "").strip()
        if len(transcript_text) < 100:
            logger.info(
                f"⏭️ Транскрибация слишком короткая ({len(transcript_text)} символов) — "
                "звонок не состоялся, пропускаем обработку"
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
            stt_provider=STT_PROVIDER,
        )
        
        # 5. Сохраняем в AmoCRM (в СДЕЛКУ!)
        logger.info(f"💾 Сохраняем примечание в leads/{lead_id}...")
        await amocrm_service.add_note_to_entity(lead_id, note_text, "leads")
        logger.info(f"✅ Примечание успешно добавлено к leads/{lead_id}")

        # 5.1. Второе примечание: полная расшифровка разговора
        minutes = int(transcription.duration_seconds // 60)
        seconds = int(transcription.duration_seconds % 60)
        duration_str = f"{minutes} мин {seconds} сек" if minutes else f"{seconds} сек"
        call_type_str = "Входящий" if call_type_simple == "incoming" else "Исходящий"

        full_transcript_note = (
            "📜 ПОЛНАЯ РАСШИФРОВКА ЗВОНКА\n\n"
            f"📞 {call_type_str} | {duration_str}\n\n"
            f"{formatted_transcript}"
        )
        try:
            await amocrm_service.add_note_to_entity(lead_id, full_transcript_note, "leads")
            logger.info(f"✅ Полная расшифровка добавлена к leads/{lead_id}")
        except Exception as full_note_error:
            logger.error(f"❌ Ошибка добавления полной расшифровки к leads/{lead_id}: {full_note_error}")
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=DEBUG
    )
