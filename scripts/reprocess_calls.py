"""
Бэкфилл звонков, не расшифрованных из-за простоя STT.

Контекст: 2026-06-29 с ~05:49 до ~07:45 UTC (08:49–10:45 МСК) OpenAI Whisper
отдавал 429 insufficient_quota — звонки скачивались, но падали на транскрибации,
анализ и примечание в AmoCRM не создавались. Вебхуки AmoCRM не повторяются,
поэтому переобрабатываем эти звонки вручную.

Безопасность:
- Окно дат строго по простою (см. START_UTC/END_UTC ниже).
- Guard идемпотентности: для каждого звонка проверяем, нет ли уже примечания
  "🎙️ АНАЛИЗ ЗВОНКА", созданного вскоре после этого звонка. Если есть — пропускаем.
  → повторный запуск скрипта ничего не дублирует, успешные звонки не трогаются.
- Фильтр размера записи (< 10000 байт = пустой/служебный звонок) — пропускаем.
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

# Корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.amocrm import amocrm_service
from main import process_call, PROCESSED_CALLS

# ---- Окно простоя (UTC) -------------------------------------------------------
# Первый сбой в логах: 2026-06-29 05:49:33Z, последний: 07:45:04Z.
# Берём с запасом по нижней границе; верхнюю ставим "сейчас минус 5 минут",
# чтобы не зацепить ещё выполняющиеся вживую звонки. Guard идемпотентности
# всё равно отсечёт всё, что уже успешно обработано.
START_UTC = datetime(2026, 6, 29, 5, 0, tzinfo=timezone.utc).timestamp()
END_UTC = (datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()

ANALYSIS_MARKER = "🎙️ АНАЛИЗ ЗВОНКА"
# Окно «примечание относится к этому звонку», сек: анализ постится вскоре после звонка.
NOTE_MATCH_BEFORE = 60       # небольшой допуск назад
NOTE_MATCH_AFTER = 30 * 60   # до 30 минут после звонка

TYPE_MAP = {
    "lead": "leads", "leads": "leads",
    "contact": "contacts", "contacts": "contacts",
    "company": "companies", "companies": "companies",
}


async def already_analyzed(entity_type: str, entity_id: int, call_created_at: int, notes_cache: dict) -> bool:
    """True, если у звонка (по времени) уже есть примечание-анализ."""
    key = (entity_type, entity_id)
    if key not in notes_cache:
        notes_cache[key] = await amocrm_service.get_recent_notes(entity_type, entity_id, limit=250)
    for note in notes_cache[key]:
        text = (note.get("params") or {}).get("text") or ""
        if ANALYSIS_MARKER not in text:
            continue
        ncreated = note.get("created_at", 0)
        if call_created_at - NOTE_MATCH_BEFORE <= ncreated <= call_created_at + NOTE_MATCH_AFTER:
            return True
    return False


async def main():
    # Очищаем in-memory кэш дублей — иначе process_call() сам всё проскипает
    PROCESSED_CALLS.clear()

    start_str = datetime.fromtimestamp(START_UTC, tz=timezone.utc) + timedelta(hours=3)
    end_str = datetime.fromtimestamp(END_UTC, tz=timezone.utc) + timedelta(hours=3)
    print(f"🗑️  Кэш дублей очищен")
    print(f"🔍 Окно простоя (МСК): {start_str:%Y-%m-%d %H:%M} – {end_str:%H:%M}")

    events = await amocrm_service.get_recent_calls(minutes=72 * 60)

    found = []
    for event in events:
        created_at = event.get("created_at", 0)
        if START_UTC <= created_at <= END_UTC:
            found.append(event)
    print(f"✅ Событий в окне: {len(found)}")

    notes_cache: dict = {}
    processed = skipped_small = skipped_dup = skipped_norec = failed = 0

    for event in found:
        call_data = await amocrm_service.process_call_event(event)
        if not call_data or not call_data.get("record_url"):
            skipped_norec += 1
            continue

        entity_type = TYPE_MAP.get(str(call_data.get("entity_type", "")).lower().strip(), "leads")
        entity_id = int(call_data["entity_id"])
        created_at = int(call_data.get("created_at") or 0)
        event_type = str(call_data.get("event_type") or "incoming_call")
        call_direction = "call_out" if "out" in event_type else "call_in"

        dt = datetime.fromtimestamp(created_at, tz=timezone.utc) + timedelta(hours=3)
        tag = f"{dt:%H:%M:%S} МСК | {call_data.get('phone','')} | сделка #{entity_id}"

        # Guard идемпотентности
        if await already_analyzed(entity_type, entity_id, created_at, notes_cache):
            print(f"⏭️  {tag} — уже есть анализ, пропуск")
            skipped_dup += 1
            continue

        try:
            audio = await amocrm_service.download_call_recording(call_data["record_url"])
            if len(audio) < 10000:
                print(f"⚠️  {tag} — запись {len(audio)} байт (пустая), пропуск")
                skipped_small += 1
                continue

            print(f"📞 {tag} | {event_type} | {len(audio)} байт → обработка...")
            await process_call(
                entity_id=entity_id,
                call_type=event_type,
                record_url=str(call_data["record_url"]),
                call_created_at=created_at,
                responsible_user_id=call_data.get("created_by"),
                phone=str(call_data.get("phone") or ""),
                entity_type=entity_type,
                call_direction=call_direction,
            )
            processed += 1
            print(f"✅ {tag} — готово")

        except Exception as e:
            failed += 1
            print(f"❌ {tag} — ошибка: {e}")

    print("\n" + "=" * 56)
    print(f"🎉 Обработано: {processed}")
    print(f"⏭️  Пропущено (уже анализ): {skipped_dup}")
    print(f"⚠️  Пропущено (пустая запись): {skipped_small}")
    print(f"🚫 Пропущено (нет записи): {skipped_norec}")
    print(f"❌ Ошибок: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
