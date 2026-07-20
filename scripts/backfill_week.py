"""
Бэкфилл нерасшифрованных звонков за текущую неделю (пн 00:00 МСК → сейчас).

Зачем: при простое STT (429 insufficient_quota) звонки скачиваются, но падают
на транскрибации. Вебхуки AmoCRM не повторяются — переобрабатываем вручную.

Порядок проверок — от дешёвых к дорогим, чтобы не жечь квоту Whisper:
  1. duration < 60 сек            → скип (прод их и так не анализирует)
  2. нет ссылки на запись         → скип
  3. звонок уже покрыт анализом   → скип (guard идемпотентности)
  4. иначе                        → скачать + process_call()

GUARD (важно). Наивная версия сверяла время примечания со временем звонка
(окно ±30 мин) и была сломана: примечания, созданные САМИМ бэкфиллом, появляются
спустя часы и сутки после звонка (напр. звонок 07.07 14:22 → примечание 08.07 13:12),
поэтому повторный запуск дублировал бы их все.

Теперь guard сопоставляет примечания и звонки ЖАДНЫМ ПАРОСОЧЕТАНИЕМ по порядку
внутри сделки: звонки сортируются по времени, примечания-анализ тоже, и каждый
звонок «забирает» самое раннее ещё не занятое примечание, созданное не раньше него.
Это корректно и для живой обработки (примечание через секунды), и для бэкфилла
(примечания создаются позже, но в том же порядке, что и звонки).

Звонок по КОНТАКТУ кладёт примечание в СДЕЛКУ, поэтому для контактов guard
читает примечания связанных сделок (read-only, ?with=leads).

Пагинация по /api/v4/events обязательна: за неделю событий больше одной страницы.
"""
import asyncio
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from services.amocrm import amocrm_service
from main import process_call, PROCESSED_CALLS

for noisy in ("httpx", "openai", "services.amocrm", "openai._base_client"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

DRY_RUN = os.getenv("DRY_RUN", "").strip() == "1"

MSK = timezone(timedelta(hours=3))
MIN_DURATION = 60                  # правило прода: звонки короче не анализируются
ANALYSIS_MARKER = "АНАЛИЗ ЗВОНКА"  # без эмодзи — устойчивее к кодировкам
CLAIM_SLACK = 60                   # допуск на рассинхрон часов
PAGE_LIMIT = 250

TYPE_MAP = {
    "lead": "leads", "leads": "leads",
    "contact": "contacts", "contacts": "contacts",
    "company": "companies", "companies": "companies",
}


def week_bounds():
    now_msk = datetime.now(MSK)
    monday = (now_msk - timedelta(days=now_msk.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return monday.timestamp(), (datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()


async def _get(client, url, params):
    r = await client.get(url, headers=amocrm_service.headers, params=params)
    if r.status_code == 204:
        return None
    r.raise_for_status()
    return r.json()


async def fetch_all_events(start_ts, end_ts) -> list:
    """Все call-события за окно, с пагинацией."""
    events, page = [], 1
    async with httpx.AsyncClient(timeout=30.0, verify=False) as c:
        while True:
            data = await _get(c, f"{amocrm_service.base_url}/events", {
                "filter[type][0]": "outgoing_call",
                "filter[type][1]": "incoming_call",
                "filter[created_at][from]": int(start_ts),
                "filter[created_at][to]": int(end_ts),
                "limit": PAGE_LIMIT, "page": page,
            })
            if not data:
                break
            batch = data.get("_embedded", {}).get("events", [])
            events.extend(batch)
            if not batch or not data.get("_links", {}).get("next"):
                break
            page += 1
    return events


async def target_lead_ids(entity_type: str, entity_id: int) -> list:
    """Сделки, куда мог лечь анализ. Для контакта — его связанные сделки (read-only)."""
    if entity_type == "leads":
        return [entity_id]
    if entity_type != "contacts":
        return []
    async with httpx.AsyncClient(timeout=30.0, verify=False) as c:
        data = await _get(c, f"{amocrm_service.base_url}/contacts/{entity_id}", {"with": "leads"})
    if not data:
        return []
    return [l["id"] for l in data.get("_embedded", {}).get("leads", []) if l.get("id")]


async def analysis_note_times(entity_type: str, entity_id: int, since_ts: float) -> list:
    """Времена примечаний-анализ по всем целевым сделкам, по возрастанию."""
    times = []
    async with httpx.AsyncClient(timeout=30.0, verify=False) as c:
        for lid in await target_lead_ids(entity_type, entity_id):
            data = await _get(c, f"{amocrm_service.base_url}/leads/{lid}/notes",
                              {"limit": PAGE_LIMIT, "order[created_at]": "desc"})
            if not data:
                continue
            for n in data.get("_embedded", {}).get("notes", []):
                text = (n.get("params") or {}).get("text") or ""
                created = n.get("created_at", 0)
                if ANALYSIS_MARKER in text and created >= since_ts:
                    times.append(created)
    return sorted(times)


def match_covered(call_times: list, note_times: list) -> set:
    """
    Жадное паросочетание: какие звонки уже покрыты примечанием.
    Звонки и примечания — по возрастанию времени.
    Returns: множество индексов покрытых звонков.
    """
    covered, used = set(), [False] * len(note_times)
    for i, ct in enumerate(call_times):
        for j, nt in enumerate(note_times):
            if not used[j] and nt >= ct - CLAIM_SLACK:
                used[j] = True
                covered.add(i)
                break
    return covered


async def main():
    PROCESSED_CALLS.clear()
    start_ts, end_ts = week_bounds()
    print(f"🔍 Окно: {datetime.fromtimestamp(start_ts, MSK):%a %d.%m %H:%M} → "
          f"{datetime.fromtimestamp(end_ts, MSK):%a %d.%m %H:%M} МСК")
    if DRY_RUN:
        print("🔸 DRY-RUN: ничего не записывается\n")

    events = await fetch_all_events(start_ts, end_ts)
    print(f"📋 Найдено call-событий за неделю: {len(events)}\n")

    stats = defaultdict(lambda: defaultdict(int))
    eligible = defaultdict(list)   # (entity_type, entity_id) -> [call_data]

    # --- Пасс 1: раскрываем события, отсеиваем короткие и без записи -----------
    for ev in sorted(events, key=lambda x: x.get("created_at", 0)):
        created = int(ev.get("created_at") or 0)
        day = datetime.fromtimestamp(created, MSK).strftime("%a %d.%m")
        stats[day]["всего"] += 1

        cd = await amocrm_service.process_call_event(ev)
        if not cd or not cd.get("record_url"):
            stats[day]["нет записи"] += 1
            continue

        duration = int((cd.get("params") or {}).get("duration") or 0)
        if duration < MIN_DURATION:
            stats[day]["короткий <60с"] += 1
            continue

        et = TYPE_MAP.get(str(cd.get("entity_type", "")).lower().strip(), "leads")
        cd["_etype"], cd["_created"], cd["_duration"], cd["_day"] = et, created, duration, day
        eligible[(et, int(cd["entity_id"]))].append(cd)

    # --- Пасс 2: guard паросочетанием, затем обработка непокрытых --------------
    to_process = []
    for (et, eid), calls in eligible.items():
        calls.sort(key=lambda c: c["_created"])
        # Отсечка — по первому звонку самой сделки: примечания от более ранних
        # звонков (вне окна) не должны быть «захвачены» паросочетанием.
        since = calls[0]["_created"] - CLAIM_SLACK
        notes = await analysis_note_times(et, eid, since)
        covered = match_covered([c["_created"] for c in calls], notes)
        for i, c in enumerate(calls):
            if i in covered:
                stats[c["_day"]]["уже разобран"] += 1
            else:
                to_process.append(c)

    to_process.sort(key=lambda c: c["_created"])
    recovered = []

    for cd in to_process:
        created, duration = cd["_created"], cd["_duration"]
        et, eid = cd["_etype"], int(cd["entity_id"])
        day = cd["_day"]
        event_type = str(cd.get("event_type") or "incoming_call")
        tstr = datetime.fromtimestamp(created, MSK).strftime("%d.%m %H:%M")

        if DRY_RUN:
            stats[day]["К ВОССТАНОВЛЕНИЮ"] += 1
            recovered.append((tstr, eid, duration))
            print(f"🔸 [dry-run] {tstr} | {et}/{eid} | {duration}с | {event_type}")
            continue

        try:
            audio = await amocrm_service.download_call_recording(cd["record_url"])
            if len(audio) < 10000:
                stats[day]["пустая запись"] += 1
                continue

            print(f"📞 {tstr} | {et}/{eid} | {duration}с | {event_type} → обработка...")
            await process_call(
                entity_id=eid,
                call_type=event_type,
                record_url=str(cd["record_url"]),
                call_created_at=created,
                responsible_user_id=cd.get("created_by"),
                phone=str(cd.get("phone") or ""),
                entity_type=et,
                call_direction="call_out" if "out" in event_type else "call_in",
                expected_duration=duration,
            )
            stats[day]["ВОССТАНОВЛЕН"] += 1
            recovered.append((tstr, eid, duration))
            print(f"✅ {tstr} | {et}/{eid} — готово\n")
        except Exception as ex:
            stats[day]["ошибка"] += 1
            print(f"❌ {tstr} | {et}/{eid} — {ex}\n")

        await asyncio.sleep(0.2)   # бережём rate-limit AmoCRM

    print("\n" + "=" * 62)
    print("ПО ДНЯМ:")
    for day in sorted(stats, key=lambda d: datetime.strptime(d.split()[1], "%d.%m")):
        row = stats[day]
        parts = [f"{k}={v}" for k, v in row.items() if k != "всего"]
        print(f"  {day}: всего {row['всего']} | " + ", ".join(parts))

    print("\n" + "=" * 62)
    label = "К ВОССТАНОВЛЕНИЮ (dry-run)" if DRY_RUN else "ВОССТАНОВЛЕНО ЗВОНКОВ"
    print(f"🎉 {label}: {len(recovered)}")
    for t, eid, d in recovered:
        print(f"   {t} | #{eid} | {d} сек")


if __name__ == "__main__":
    asyncio.run(main())
