"""
Сервис анализа разговора через Claude Sonnet 4.6 (Anthropic API).
Замена GPT-4.1-mini / Gemini → Claude Sonnet 4.6.
Интерфейс (CallAnalysis, AnalysisService, format_note) сохранён для совместимости с main.py.
"""
import anthropic
import httpx
import json
import logging
import openai
import re
import os
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    ASSEMBLYAI_API_KEY,
    ASSEMBLYAI_LLM_MODEL,
    LLM_CHAIN,
    LLM_FALLBACK_ENABLED,
    LLM_FALLBACK_RETRY_MINUTES,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_MAX_TOKENS,
    ANALYSIS_PIPELINE_VERSION,
    MAX_TRANSCRIPT_LENGTH,
    TRUNCATE_TRANSCRIPT_FOR_ANALYSIS,
)
from services.telegram import telegram_service

logger = logging.getLogger(__name__)

_anthropic_client: anthropic.AsyncAnthropic | None = None
_openai_client: "openai.AsyncOpenAI | None" = None

# Человекочитаемые названия для уведомлений
LLM_PROVIDER_TITLES = {
    "anthropic": "Anthropic (Claude)",
    "assemblyai": "AssemblyAI (Claude через их шлюз)",
    "openai": "OpenAI (GPT)",
}


def _normalize_list_field(value) -> List[str]:
    """
    Нормализует поле, которое может прийти как:
    - list[str]
    - многострочная строка с буллетами/нумерацией
    - None
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        items: List[str] = []
        for line in value.splitlines():
            s = line.strip()
            if not s:
                continue
            # убираем буллеты и нумерацию в начале строки
            s = re.sub(r"^(\s*[-•]\s+|\s*\d+\s*[).]\s+)", "", s).strip()
            if s:
                items.append(s)
        return items
    # fallback
    s = str(value).strip()
    return [s] if s else []


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    """
    Инициализируем Anthropic клиент лениво.
    """
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    api_key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY не задан (нужен для анализа звонков)")
    _anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _anthropic_client


@dataclass
class CallAnalysis:
    """Результат анализа звонка (без оценок)"""
    client_name: str  # ФИО или имя клиента
    manager_name: str  # ФИО менеджера (из разговора)
    summary: str  # Краткое резюме разговора
    # Дополнительные поля
    client_city: str  # Город клиента
    work_type: str  # Тип работы
    location: str  # Адрес/район/ориентир объекта
    cost: str  # Стоимость
    payment_terms: str  # Условия оплаты
    call_result: str  # Итог звонка
    next_contact_date: str  # Когда связаться
    next_steps: List[str]  # Следующие шаги для менеджера (0-5)
    speaker_stats: Optional["SpeakerStats"] = None  # Метрики по участникам (v2)


@dataclass
class SpeakerMetrics:
    """Метрики по отдельному спикеру."""
    label: str
    duration_seconds: float
    share_percent: float


@dataclass
class SpeakerStats:
    """Агрегированные метрики диаризации."""
    participant_count: int
    total_speech_seconds: float
    dominant_speaker: str
    suspicious_diarization: bool
    suspicious_reason: str
    speakers: List[SpeakerMetrics]


@dataclass
class FieldVerification:
    """Результат верификации одного поля."""
    field: str
    status: str  # supported | unsure | contradicted
    confidence: float
    suggested_fix: Any
    evidence: List[str]


# Значения, означающие «имя неизвестно» — их нельзя считать настоящим именем
UNKNOWN_NAME_VALUES = {
    "не представился", "не указано", "не определено", "неизвестно",
    "менеджер", "клиент", "",
}


def _is_real_name(value: Optional[str]) -> bool:
    """True, если строка похожа на настоящее имя, а не на заглушку."""
    if not value:
        return False
    v = value.strip()
    if v.lower() in UNKNOWN_NAME_VALUES:
        return False
    # «Менеджер #2629318» — техническая заглушка, когда имя не подтянулось из CRM
    return not re.match(r"^Менеджер\s*#\d+$", v, re.IGNORECASE)


def _resolve_manager_name(from_crm: str, from_llm: Optional[str]) -> str:
    """
    Выбирает имя менеджера. Приоритет — у CRM: оно приходит из
    responsible_user_id и не зависит от качества расшифровки.
    Модель используется как запасной вариант, если CRM имени не дала.
    """
    if _is_real_name(from_crm):
        return from_crm.strip()
    if _is_real_name(from_llm):
        return from_llm.strip()
    return from_crm or "Менеджер"


# Системный промпт для анализа (Агент 1)
ANALYSIS_SYSTEM_PROMPT = """Ты — ассистент для анализа телефонных разговоров геодезической компании.
Твоя задача — извлечь только факты из транскрибации и вернуть строго JSON.

1. Определение ролей и имён
- Метки [Менеджер] / [Клиент] проставлены по каналам записи (у каждого свой канал) — доверяй им.
- Менеджер компании — {manager_name}. Всё, что он говорит — позиция компании.
- Все остальные спикеры — клиенты.

Имя менеджера ("manager_name"):
- ВСЕГДА возвращай ровно "{manager_name}" — это имя уже известно из CRM и оно достоверно.
- НЕ заменяй его и НЕ пиши "Не представился", даже если менеджер в разговоре не назвался.

Имя клиента ("client_name") — ищи в репликах [Клиент]:
- прямое представление: «меня зовут X», «я X», «это X», «X беспокоит»;
- менеджер обращается к клиенту по имени.
- Имя клиента может СОВПАДАТЬ с именем менеджера — тёзки встречаются. Если клиент
  назвал имя, совпадающее с {manager_name}, всё равно запиши его в "client_name",
  а не считай это ошибкой разметки.
- Только если имя клиента действительно нигде не прозвучало — "client_name": "Не представился".

2. Инструкции по чтению транскрибации
- Читай ВСЮ транскрибацию от начала до конца. Не пропускай фрагменты.
- Длинные звонки (10+ минут) часто содержат важные детали в середине и в конце — не игнорируй их.
- Если в разговоре обсуждаются несколько объектов или задач — фиксируй каждую.
- Извлекай ТОЛЬКО то, что реально прозвучало. Не додумывай, не интерпретируй.
- КРИТИЧЕСКИ ВАЖНО: различай ПОЗИЦИЮ КОМПАНИИ и ПРОСТО УПОМИНАНИЕ.
  Если менеджер говорит «я видел объявление на Авито за 30к» — это НЕ значит, что компания оказывает эту услугу или имеет подрядчиков.
  Если менеджер говорит «у нас нет таких специалистов» — это факт, а не «вопрос остался нерешённым».
  Не приписывай компании возможности, услуги или партнёров, если менеджер прямо это НЕ подтвердил.
- Чётко разделяй: что менеджер ПОДТВЕРДИЛ как услугу компании vs что он УПОМЯНУЛ из внешних источников.

3. Правила summary
summary — это структурированная выжимка фактов. НЕ пересказ диалога.
Извлеки и запиши ТОЛЬКО следующие пункты (каждый пункт = одно предложение):
1) Кто клиент + что ему нужно (одной фразой)
2) Детали объекта: адрес/район, площадь, количество точек, тип участка, назначение
3) Нюансы/проблемы, если есть (глушится GPS, нет доступа, спор с соседями, срочность и т.д.)
4) Кто порекомендовал / откуда узнал о компании (если упоминается)
5) О чём конкретно договорились по итогу звонка
6) Если обсуждались несколько объектов или задач — перечисли каждую отдельно
Если пункт НЕ прозвучал в разговоре — пропусти его. Не выдумывай.

ВАЖНО: если в разговоре озвучивались цены/стоимость — ОБЯЗАТЕЛЬНО упомяни их в summary.

ЗАПРЕЩЕНО в summary:
- Хронологический пересказ («клиент позвонил... менеджер объяснил... клиент согласился...»)
- Дублировать информацию из полей next_steps, payment_terms (но стоимость МОЖНО и НУЖНО упоминать)
- Вводные фразы («В ходе разговора...», «Клиент обратился к менеджеру...», «Состоялся разговор...»)
- Писать то, что НЕ прозвучало в разговоре

4. Правила next_steps
Каждый шаг должен чётко указывать КТО делает ЧТО.
Правило: определяй направление действия по контексту разговора.
- Если менеджер попросил клиента прислать документ → шаг = «Получить от клиента [документ]» (мяч на стороне клиента)
- Если менеджер пообещал что-то сделать → шаг = «Менеджеру: [действие]» (мяч на стороне менеджера)
- Если договорились о встрече/созвоне → шаг = «Согласовать дату [чего именно]»

ЗАПРЕЩЕНО в next_steps:
- Размытые формулировки («Обсудить детали», «Продолжить работу»)
- Дублировать то, что уже в summary

5. Обязательные поля — правила извлечения

client_city:
  Ищи любые упоминания:
  - Прямое название города («в Ставрополе», «Кисловодск»)
  - Районы, улицы, ориентиры, шоссе — определи город по ним
  - Если город не упоминается и невозможно определить → "Не указано"

cost:
  Ищи любые упоминания цен, стоимости, расценок:
  - Точная цена: «стоит 4000 рублей» → "4 000 ₽"
  - Диапазон: «от 25 до 40 тысяч» → "от 25 000 до 40 000 ₽"
  - Несколько цен за разные услуги → перечисли все через запятую
  - Если цена не обсуждалась → "Не обсуждали"

payment_terms:
  Ищи упоминания:
  - Предоплата, постоплата, 50/50, по факту, аванс, рассрочка
  - Если не обсуждалось → "Не обсуждали"

next_contact_date:
  Ищи упоминания:
  - Конкретная дата: «в понедельник», «15 марта»
  - Относительная: «через неделю», «завтра», «после праздников»
  - Если не обсуждалось → "Не обсуждали"

call_result:
  Определи итог звонка из вариантов:
  - "Согласие" — клиент согласился на работу или следующий шаг
  - "Отказ" — клиент отказался
  - "Перезвонить" — договорились созвониться позже
  - "Думает" — клиент взял паузу на размышление
  - "Не определено" — итог неясен

work_type:
  Тип работы из контекста разговора. Примеры:
  - Межевание
  - Вынос границ
  - Техплан (дом, квартира, здание, помещение, сооружение)
  - Топографическая съёмка
  - Инженерные изыскания
  - Если несколько видов работ → перечисли через запятую
  - Если тема не попадает ни в один из перечисленных типов → "Прочие"
  - Если совсем неясно → "Не определено"

location:
  Адрес, район или ориентир объекта работ. Краткая форма для названия сделки.
  Ищи: улицу, шоссе, район, СНТ, КП, посёлок, ориентир рядом с объектом.
  Примеры: "Старомарьское шоссе", "ул. Ленина 15", "СНТ Солнечный", "п. Иноземцево", "р-н Юго-Западный"
  Пиши кратко (2-4 слова).
  Если конкретного адреса (улица, СНТ, район) нет, но назван населённый пункт, где
  находится ОБЪЕКТ работ (село, посёлок, станица, хутор) — укажи его: "с. Верхняя Татарка".
  "Не указано" — только если в разговоре нет вообще никакой географической привязки объекта.

6. Формат ответа
Верни ТОЛЬКО валидный JSON. Никакого текста до или после JSON.
{{{{
  "client_name": "string",
  "manager_name": "string",
  "summary": "string",
  "client_city": "string",
  "work_type": "string",
  "location": "string",
  "cost": "string",
  "payment_terms": "string",
  "call_result": "string",
  "next_contact_date": "string",
  "next_steps": ["string"]
}}}}"""


ANALYSIS_USER_PROMPT = """Проанализируй разговор между менеджером и клиентом.

Тип звонка: {call_type}
{call_direction_context}
Менеджер компании: {manager_name}

ПОМНИ: {manager_name} = МЕНЕДЖЕР (не клиент!)

ТРАНСКРИБАЦИЯ РАЗГОВОРА:
{transcript}"""


# Системный промпт для валидатора (Агент 2)
VALIDATOR_SYSTEM_PROMPT = """Ты — валидатор результатов анализа телефонных разговоров.

Твоя задача — проверить результат первого анализа и найти пропущенную ОБЯЗАТЕЛЬНУЮ информацию.

ОБЯЗАТЕЛЬНЫЕ ПОЛЯ (не могут быть "Не указано"):
1. client_city - населенный пункт, город, регион, адрес
2. cost - сумма договора, стоимость, цена
3. payment_terms - условия оплаты (50/50, предоплата, рассрочка)
4. next_contact_date - дата следующего контакта

ТВОЯ ЗАДАЧА:
- Перечитай транскрипцию ОЧЕНЬ ВНИМАТЕЛЬНО
- Найди пропущенную информацию для указанных полей
- Ищи синонимы и косвенные упоминания
- Если информации ДЕЙСТВИТЕЛЬНО нет в транскрипции — верни "Не указано"

Примеры того, что нужно искать:
- Город: "я из Краснодара", "участок в Ростове", "живу в пригороде", "адрес: Москва"
- Стоимость: "25 тысяч", "около 30 000", "цена будет 40 тысяч рублей"
- Оплата: "50 на 50", "половину сейчас", "предоплата 50%", "100% после"
- Дата: "перезвоню в среду", "15 января", "завтра позвоню", "через неделю"

Верни ТОЛЬКО JSON с найденными значениями:
{
  "client_city": "найденное значение или 'Не указано'",
  "cost": "найденное значение или 'Не указано'",
  "payment_terms": "найденное значение или 'Не указано'",
  "next_contact_date": "найденное значение или 'Не указано'"
}

Отвечай ТОЛЬКО JSON, без пояснений."""


VALIDATOR_USER_PROMPT = """Первый анализ пропустил обязательную информацию.

Пропущенные поля: {missing_fields}

ТРАНСКРИБАЦИЯ РАЗГОВОРА:
{transcript}

Найди пропущенную информацию для этих полей. Если действительно нет — верни "Не указано"."""


VERIFY_FIELDS = [
    "client_name",
    "manager_name",
    "client_city",
    "work_type",
    "location",
    "cost",
    "payment_terms",
    "call_result",
    "next_contact_date",
    "next_steps",
]

FIELD_DEFAULTS: Dict[str, Any] = {
    "client_name": "Клиент",
    "manager_name": "Менеджер",
    "client_city": "Не указано",
    "work_type": "Прочие",
    "location": "Не указано",
    "cost": "Не обсуждали",
    "payment_terms": "Не обсуждали",
    "call_result": "Не определено",
    "next_contact_date": "Не указано",
    "next_steps": [],
}

CHAT_ANALYSIS_SYSTEM_PROMPT = """Ты — аналитик переписки геодезической компании «Ставропольгеодезия» с клиентами в мессенджерах (WhatsApp, Авито и др.).

На входе — хронология диалога. Реплики размечены: «Клиент» и «Менеджер».
Строки вида [голосовое: ...] — расшифровка голосового сообщения, [файл: имя] и [фото] — вложения
(имена файлов информативны: «Решение суда.pdf» говорит о наличии решения суда).

Извлеки информацию для CRM. Правила:
- НЕ выдумывай. Нет информации — пиши "Не указано" (для стоимости/оплаты — "Не обсуждали").
- work_type — тип работ: Межевание, Техплан, Топосъёмка, Вынос границ, Акт обследования и т.п.;
  не попадает в известные типы → "Прочие"; неясно → "Не определено".
- location — адрес/населённый пункт ОБЪЕКТА работ, кратко (2-4 слова). Если конкретного адреса нет,
  но назван населённый пункт — укажи его.
- client_city — город/населённый пункт клиента.
- cost — стоимость, если называлась. payment_terms — условия оплаты.
- call_result — итог диалога: Согласие / Думает / Перезвонить / Отказ / Не определено.
- next_contact_date — когда договорились связаться (словами, как в диалоге: "завтра", "пятница").
- next_steps — до 5 конкретных шагов. Шаги менеджера начинай с "Менеджеру: ".
- summary — суть диалога, 2-4 предложения.
- client_name — имя клиента, если видно из диалога/подписи; иначе "Клиент".

Верни ТОЛЬКО валидный JSON:
{
  "client_name": "string",
  "summary": "string",
  "client_city": "string",
  "work_type": "string",
  "location": "string",
  "cost": "string",
  "payment_terms": "string",
  "call_result": "string",
  "next_contact_date": "string",
  "next_steps": ["string"]
}"""


FACT_VERIFIER_SYSTEM_PROMPT = """Ты — аудитор фактов по транскрибации звонка.

Твоя задача:
1) Проверить каждое поле анализа.
2) Для каждого поля вернуть status/confidence/suggested_fix/evidence.

Правила:
- status:
  - supported: значение подтверждается транскрибацией,
  - unsure: подтверждение слабое или неоднозначное,
  - contradicted: значение противоречит транскрибации.
- confidence: число от 0 до 1.
- suggested_fix: безопасная замена, если поле не подтверждено.
- evidence: 1-3 коротких точных фрагмента из транскрибации (только текст, без комментариев).
- Не выдумывай. Если данных нет, предложи безопасное значение.
- ОСОБОЕ ПРАВИЛО для пустых полей ("Не указано"/"Не обсуждали"/"Не определено"):
  если информация для такого поля в транскрибации ЕСТЬ — верни status "contradicted",
  а в suggested_fix положи НАЙДЕННОЕ значение (кратко), с evidence.
  Ищи синонимы и косвенные упоминания:
  * город: "я из Краснодара", "участок в Ростове", "живу в пригороде"
  * стоимость: "25 тысяч", "около 30 000", "цена будет 40 тысяч рублей"
  * оплата: "50 на 50", "половину сейчас", "предоплата 50%", "100% после"
  * дата: "перезвоню в среду", "15 января", "завтра позвоню", "через неделю"
  Если информации действительно нет — пустое значение корректно, status "supported".

Безопасные значения:
- city/date -> "Не указано"
- cost/payment_terms -> "Не обсуждали"
- work_type -> "Прочие"
- call_result -> "Не определено"
- next_steps -> []

Ответ строго JSON."""


FACT_VERIFIER_USER_PROMPT = """Проверь анализ на соответствие транскрибации.

Черновой анализ (JSON):
{analysis_json}

Метрики спикеров (JSON):
{speaker_stats_json}

Транскрибация:
{transcript}

Верни JSON формата:
{{
  "fields": {{
    "client_name": {{"status":"supported|unsure|contradicted","confidence":0.0,"suggested_fix":"Клиент","evidence":["..."]}},
    "manager_name": {{"status":"supported|unsure|contradicted","confidence":0.0,"suggested_fix":"Менеджер","evidence":["..."]}},
    "client_city": {{"status":"supported|unsure|contradicted","confidence":0.0,"suggested_fix":"Не указано","evidence":["..."]}},
    "work_type": {{"status":"supported|unsure|contradicted","confidence":0.0,"suggested_fix":"Консультация","evidence":["..."]}},
    "cost": {{"status":"supported|unsure|contradicted","confidence":0.0,"suggested_fix":"Не обсуждали","evidence":["..."]}},
    "payment_terms": {{"status":"supported|unsure|contradicted","confidence":0.0,"suggested_fix":"Не обсуждали","evidence":["..."]}},
    "call_result": {{"status":"supported|unsure|contradicted","confidence":0.0,"suggested_fix":"Не определено","evidence":["..."]}},
    "next_contact_date": {{"status":"supported|unsure|contradicted","confidence":0.0,"suggested_fix":"Не указано","evidence":["..."]}},
    "next_steps": {{"status":"supported|unsure|contradicted","confidence":0.0,"suggested_fix":[],"evidence":["..."]}}
  }}
}}"""


def _extract_json_from_text(text: str) -> dict:
    """
    Извлекает JSON из ответа Claude.
    Claude может обернуть JSON в ```json ... ``` или вернуть чистый JSON.
    """
    text = text.strip()

    # Попробуем распарсить как чистый JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Извлечь из markdown блока ```json ... ```
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Извлечь первый { ... } блок
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Не удалось извлечь JSON из ответа Claude", text, 0)


class AnalysisService:
    """Сервис анализа разговоров через Claude Sonnet 4.6 с валидацией"""

    def __init__(self):
        # провайдер -> unix ts, до которого он считается недоступным
        self._llm_down_until: Dict[str, float] = {}

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, v))

    @staticmethod
    def _to_short_evidence_list(value: Any) -> List[str]:
        if isinstance(value, list):
            raw = value
        elif isinstance(value, str):
            raw = [value]
        else:
            raw = []

        normalized: List[str] = []
        for item in raw:
            text = str(item).strip()
            if not text:
                continue
            normalized.append(text[:240])
            if len(normalized) == 3:
                break
        return normalized

    def _analysis_to_dict(self, analysis: CallAnalysis) -> Dict[str, Any]:
        return {
            "client_name": analysis.client_name,
            "manager_name": analysis.manager_name,
            "summary": analysis.summary,
            "client_city": analysis.client_city,
            "work_type": analysis.work_type,
            "cost": analysis.cost,
            "payment_terms": analysis.payment_terms,
            "call_result": analysis.call_result,
            "next_contact_date": analysis.next_contact_date,
            "next_steps": analysis.next_steps,
        }

    def _build_speaker_stats(self, speakers: Optional[List[Any]]) -> SpeakerStats:
        if not speakers:
            return SpeakerStats(
                participant_count=0,
                total_speech_seconds=0.0,
                dominant_speaker="-",
                suspicious_diarization=True,
                suspicious_reason="no_speakers",
                speakers=[],
            )

        durations_ms: Dict[str, int] = {}
        for item in speakers:
            label = str(getattr(item, "label", getattr(item, "speaker", "?")))
            start_ms = int(getattr(item, "start_ms", 0) or 0)
            end_ms = int(getattr(item, "end_ms", 0) or 0)
            duration_ms = max(0, end_ms - start_ms)
            if duration_ms <= 0:
                continue
            durations_ms[label] = durations_ms.get(label, 0) + duration_ms

        if not durations_ms:
            return SpeakerStats(
                participant_count=0,
                total_speech_seconds=0.0,
                dominant_speaker="-",
                suspicious_diarization=True,
                suspicious_reason="no_positive_durations",
                speakers=[],
            )

        total_ms = sum(durations_ms.values())
        metrics: List[SpeakerMetrics] = []
        for label, duration_ms in sorted(durations_ms.items(), key=lambda pair: pair[1], reverse=True):
            share_percent = (duration_ms / total_ms) * 100 if total_ms > 0 else 0.0
            metrics.append(
                SpeakerMetrics(
                    label=label,
                    duration_seconds=round(duration_ms / 1000, 1),
                    share_percent=round(share_percent, 1),
                )
            )

        participant_count = len(metrics)
        suspicious_reasons: List[str] = []
        if participant_count <= 1 and total_ms >= 45_000:
            suspicious_reasons.append("single_speaker_long_call")
        if participant_count >= 6 and total_ms <= 600_000:
            suspicious_reasons.append("too_many_speakers_for_short_call")

        return SpeakerStats(
            participant_count=participant_count,
            total_speech_seconds=round(total_ms / 1000, 1),
            dominant_speaker=metrics[0].label,
            suspicious_diarization=bool(suspicious_reasons),
            suspicious_reason=";".join(suspicious_reasons),
            speakers=metrics,
        )

    def _normalize_verification_result(self, payload: Dict[str, Any]) -> Dict[str, FieldVerification]:
        fields_payload = payload.get("fields", payload)
        result: Dict[str, FieldVerification] = {}
        for field in VERIFY_FIELDS:
            raw = fields_payload.get(field, {}) if isinstance(fields_payload, dict) else {}
            status = str(raw.get("status", "unsure")).strip().lower()
            if status not in {"supported", "unsure", "contradicted"}:
                status = "unsure"
            suggested_fix = raw.get("suggested_fix", FIELD_DEFAULTS[field])
            result[field] = FieldVerification(
                field=field,
                status=status,
                confidence=self._clamp_confidence(raw.get("confidence", 0.0)),
                suggested_fix=suggested_fix,
                evidence=self._to_short_evidence_list(raw.get("evidence", [])),
            )
        return result

    # -------------------------------------------------------------------------
    # Цепочка LLM-провайдеров: anthropic → assemblyai → openai
    # -------------------------------------------------------------------------

    def _llm_available(self, provider: str) -> bool:
        """Провайдер не в «карантине» после недавнего инфраструктурного сбоя?"""
        return time.time() >= self._llm_down_until.get(provider, 0.0)

    @staticmethod
    def _is_llm_infra_failure(exc: BaseException) -> bool:
        """Сбой, который сам не пройдёт: кончились деньги или отозван ключ."""
        # Импорт здесь: services.alerts тянет telegram → config, на уровне
        # модуля получился бы цикл импортов.
        from services import alerts

        if alerts.classify(exc) is not None:
            return True
        # LLM Gateway AssemblyAI отвечает обычным HTTP, не через SDK провайдера
        msg = str(exc).lower()
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code in (401, 402, 403):
                return True
            if exc.response.status_code == 429 and "quota" in msg:
                return True
        return any(h in msg for h in ("insufficient", "credit balance", "no credits", "billing"))

    async def _llm_mark_down(self, provider: str, exc: BaseException):
        """Помечает провайдера недоступным и уведомляет — один раз на сбой."""
        first_time = self._llm_available(provider)
        self._llm_down_until[provider] = time.time() + LLM_FALLBACK_RETRY_MINUTES * 60
        if not first_time:
            return

        nxt = [p for p in LLM_CHAIN if p != provider and self._llm_available(p)]
        target = LLM_PROVIDER_TITLES.get(nxt[0], nxt[0]) if nxt else "—"
        logger.error(f"🔁 LLM {provider} недоступен ({exc}) — переходим на {target}")
        try:
            await telegram_service.send_message(
                "🔁 <b>Переключение анализа на резерв</b>\n\n"
                f"<b>Не отвечает:</b> {LLM_PROVIDER_TITLES.get(provider, provider)}\n"
                "Вероятно, кончился баланс или отозван ключ.\n\n"
                f"<b>Анализ идёт через:</b> {target}\n"
                f"<i>Основной провайдер проверим снова через {LLM_FALLBACK_RETRY_MINUTES} мин.</i>"
            )
        except Exception as tg_err:
            logger.warning(f"⚠️ Не удалось отправить алерт о переключении LLM: {tg_err}")

    async def _llm_recovered(self, provider: str):
        """Основной провайдер ожил — снимаем карантин и сообщаем."""
        if not self._llm_down_until.pop(provider, None):
            return
        logger.info(f"✅ LLM {provider} снова доступен")
        try:
            await telegram_service.send_message(
                "✅ <b>Анализ вернулся на основной провайдер</b>\n\n"
                f"<b>Провайдер:</b> {LLM_PROVIDER_TITLES.get(provider, provider)}\n"
                "Качество сводок восстановлено."
            )
        except Exception as tg_err:
            logger.warning(f"⚠️ Не удалось отправить уведомление о возврате LLM: {tg_err}")

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2500,
    ) -> str:
        """
        Вызов LLM с автопереходом по цепочке LLM_CHAIN.

        Переключаемся только на инфраструктурных сбоях (баланс, ключ) — обычные
        ошибки (сеть, таймаут, кривой ответ) пробрасываем наверх: уводить весь
        анализ на резерв из-за одного плохого звонка неправильно.
        """
        chain = LLM_CHAIN if LLM_FALLBACK_ENABLED else LLM_CHAIN[:1]
        callers = {
            "anthropic": self._call_anthropic,
            "assemblyai": self._call_assemblyai_llm,
            "openai": self._call_openai_llm,
        }

        errors = []
        skipped = []
        for provider in chain:
            caller = callers.get(provider)
            if caller is None:
                logger.warning(f"⚠️ Неизвестный LLM-провайдер в цепочке: {provider}")
                continue
            if not self._llm_available(provider):
                skipped.append(provider)
                continue
            try:
                text = await caller(system_prompt, user_prompt, max_tokens)
                await self._llm_recovered(provider)
                return text
            except Exception as exc:
                if not self._is_llm_infra_failure(exc):
                    raise
                errors.append(f"{provider}: {str(exc)[:120]}")
                await self._llm_mark_down(provider, exc)

        # Ни один не ответил. Если кого-то пропустили по карантину — пробуем
        # их всё равно: лучше рискнуть, чем оставить звонок без анализа.
        for provider in skipped:
            try:
                text = await callers[provider](system_prompt, user_prompt, max_tokens)
                await self._llm_recovered(provider)
                return text
            except Exception as exc:
                errors.append(f"{provider} (повтор): {str(exc)[:120]}")

        raise RuntimeError("Все LLM-провайдеры недоступны — " + "; ".join(errors))

    async def _call_anthropic(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Anthropic напрямую (основной провайдер)."""
        response = await _get_anthropic_client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        usage = response.usage
        logger.info(
            f"LLM anthropic/{ANTHROPIC_MODEL}: {usage.input_tokens} in / {usage.output_tokens} out"
        )
        return response.content[0].text

    async def _call_assemblyai_llm(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """
        LLM Gateway AssemblyAI — OpenAI-совместимый API.
        Даёт ту же модель claude-sonnet-4-6, но с оплатой через AssemblyAI,
        поэтому пустой баланс Anthropic его не затрагивает.
        """
        if not ASSEMBLYAI_API_KEY:
            raise RuntimeError("ASSEMBLYAI_API_KEY не задан")

        async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
            r = await client.post(
                "https://llm-gateway.assemblyai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {ASSEMBLYAI_API_KEY}"},
                json={
                    "model": ASSEMBLYAI_LLM_MODEL,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            r.raise_for_status()
            data = r.json()

        usage = data.get("usage") or {}
        logger.info(
            f"LLM assemblyai/{ASSEMBLYAI_LLM_MODEL}: "
            f"{usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out"
        )
        return data["choices"][0]["message"]["content"]

    async def _call_openai_llm(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """OpenAI напрямую — последнее звено цепочки."""
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY не задан")

        global _openai_client
        if _openai_client is None:
            _openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

        response = await _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = response.usage
        logger.info(
            f"LLM openai/{OPENAI_MODEL}: {usage.prompt_tokens} in / {usage.completion_tokens} out"
        )
        return response.choices[0].message.content or ""

    async def _verify_with_claude(
        self,
        analysis: CallAnalysis,
        transcript: str,
        speaker_stats: SpeakerStats,
    ) -> Dict[str, FieldVerification]:
        """Fact verifier (Агент 3) через Claude."""
        try:
            prepared_transcript = self._prepare_transcript(transcript)

            user_prompt = FACT_VERIFIER_USER_PROMPT.format(
                analysis_json=json.dumps(self._analysis_to_dict(analysis), ensure_ascii=False),
                speaker_stats_json=json.dumps(
                    {
                        "participant_count": speaker_stats.participant_count,
                        "total_speech_seconds": speaker_stats.total_speech_seconds,
                        "dominant_speaker": speaker_stats.dominant_speaker,
                        "suspicious_diarization": speaker_stats.suspicious_diarization,
                    },
                    ensure_ascii=False,
                ),
                transcript=prepared_transcript,
            )

            result_text = await self._call_llm(
                system_prompt=FACT_VERIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1200,
            )
            parsed = _extract_json_from_text(result_text)
            return self._normalize_verification_result(parsed)
        except Exception as e:
            logger.error(f"Ошибка fact verifier через Claude: {e}")
            return {
                field: FieldVerification(
                    field=field,
                    status="unsure",
                    confidence=0.0,
                    suggested_fix=FIELD_DEFAULTS[field],
                    evidence=[],
                )
                for field in VERIFY_FIELDS
            }

    async def analyze_chat(self, dialog_text: str, channel_name: str = "чат") -> CallAnalysis:
        """
        Анализ переписки в мессенджере → тот же CallAnalysis, что и для звонков,
        поэтому дальше работают общие auto_fill_lead_fields и авто-задача.
        Один вызов Claude, без валидатора/верификатора: текст диалога точен,
        в отличие от расшифровки звука.
        """
        result_text = await self._call_llm(
            system_prompt=CHAT_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=f"Канал: {channel_name}\n\nПЕРЕПИСКА:\n{dialog_text}",
            max_tokens=1500,
        )
        data = _extract_json_from_text(result_text)
        next_steps = data.get("next_steps") or []
        if not isinstance(next_steps, list):
            next_steps = _normalize_list_field(next_steps)
        return CallAnalysis(
            client_name=data.get("client_name", "Клиент"),
            manager_name="Менеджер",
            summary=data.get("summary", ""),
            client_city=data.get("client_city", "Не указано"),
            work_type=data.get("work_type", "Не определено"),
            location=data.get("location", "Не указано"),
            cost=data.get("cost", "Не обсуждали"),
            payment_terms=data.get("payment_terms", "Не обсуждали"),
            call_result=data.get("call_result", "Не определено"),
            next_contact_date=data.get("next_contact_date", "Не указано"),
            next_steps=[str(x).strip() for x in next_steps if str(x).strip()][:5],
        )

    def _apply_verification(
        self,
        analysis: CallAnalysis,
        checks: Dict[str, FieldVerification],
        protected: frozenset = frozenset(),
    ) -> CallAnalysis:
        """
        Применяет вердикты верификатора. Поля из `protected` не трогаются:
        они получены не из расшифровки, а из достоверного источника (CRM),
        поэтому «проверка по тексту разговора» для них бессмысленна — менеджер
        может ни разу не назвать своё имя, и верификатор заменит его заглушкой.
        """
        for field, check in checks.items():
            if field in protected:
                logger.info(f"v2 verify: поле {field} защищено (значение из CRM), вердикт не применяем")
                continue
            logger.info(
                "v2 verify field=%s status=%s confidence=%.2f suggested_fix=%s evidence=%s",
                field,
                check.status,
                check.confidence,
                check.suggested_fix,
                check.evidence,
            )
            if check.status == "supported":
                continue

            fallback = check.suggested_fix if check.suggested_fix not in (None, "") else FIELD_DEFAULTS[field]
            if field == "next_steps":
                if isinstance(fallback, list):
                    analysis.next_steps = [str(x).strip() for x in fallback if str(x).strip()][:5]
                else:
                    analysis.next_steps = _normalize_list_field(fallback)
                continue

            setattr(analysis, field, str(fallback))
        return analysis

    async def _validate_with_claude(
        self,
        transcript: str,
        missing_fields: List[str]
    ) -> dict:
        """Валидация через Claude (Агент 2)"""
        try:
            prepared_transcript = self._prepare_transcript(transcript)

            result_text = await self._call_llm(
                system_prompt=VALIDATOR_SYSTEM_PROMPT,
                user_prompt=VALIDATOR_USER_PROMPT.format(
                    missing_fields=", ".join(missing_fields),
                    transcript=prepared_transcript,
                ),
                max_tokens=800,
            )
            return _extract_json_from_text(result_text)

        except Exception as e:
            logger.error(f"Ошибка валидации через Claude: {e}")
            return {}

    async def _validate_and_fix(
        self,
        analysis: CallAnalysis,
        transcript: str,
        manager_name: str
    ) -> CallAnalysis:
        """
        Второй агент: валидирует результат первого анализа.
        Проверяет обязательные поля и исправляет пропущенную информацию.
        """
        # Проверяем обязательные поля
        required_fields = {
            "client_city": analysis.client_city,
            "cost": analysis.cost,
            "payment_terms": analysis.payment_terms,
            "next_contact_date": analysis.next_contact_date
        }

        missing = [
            field for field, value in required_fields.items()
            if value in ["Не указано", "Не обсуждали", "Консультация", ""]
        ]

        if not missing:
            logger.info("✅ Все обязательные поля заполнены, валидация не требуется")
            return analysis

        # Запускаем валидатор (Агент 2)
        logger.warning(f"⚠️ Пропущены обязательные поля: {missing}")
        logger.info("🔍 Запускаем валидатор (Агент 2) для поиска пропущенной информации...")

        fixed_data = await self._validate_with_claude(transcript, missing)

        # Обновляем анализ найденными значениями
        updated_count = 0
        for field in missing:
            new_value = fixed_data.get(field, "")
            if new_value and new_value not in ["Не указано", "Не обсуждали", ""]:
                old_value = getattr(analysis, field)
                setattr(analysis, field, new_value)
                logger.info(f"✅ Валидатор нашёл {field}: '{old_value}' → '{new_value}'")
                updated_count += 1

        if updated_count > 0:
            logger.info(f"🎉 Валидатор исправил {updated_count} из {len(missing)} полей")
        else:
            logger.warning("⚠️ Валидатор не смог найти дополнительную информацию")

        return analysis

    def _prepare_transcript(self, transcript: str) -> str:
        """
        Подготавливает транскрипцию для анализа.
        По умолчанию НЕ обрезаем: для звонков до ~30 минут хотим анализировать весь текст.
        """
        if not TRUNCATE_TRANSCRIPT_FOR_ANALYSIS:
            return transcript

        if len(transcript) <= MAX_TRANSCRIPT_LENGTH:
            return transcript

        logger.info(f"Транскрипция длинная ({len(transcript)} символов), обрезаем до {MAX_TRANSCRIPT_LENGTH}")

        # Берём начало (первые 60%) и конец (последние 40%)
        start_length = int(MAX_TRANSCRIPT_LENGTH * 0.6)
        end_length = MAX_TRANSCRIPT_LENGTH - start_length

        start_part = transcript[:start_length]
        end_part = transcript[-end_length:]

        prepared = f"""{start_part}

[... пропущена средняя часть разговора для экономии токенов ...]

{end_part}"""

        logger.info(f"Обрезанная транскрипция: {len(prepared)} символов")
        return prepared

    async def analyze_call(
        self,
        transcript: str,
        call_type: str = "outgoing",
        manager_name: str = "Менеджер",
        speakers: Optional[List[Any]] = None,
        call_direction: str = "call_in",
    ) -> CallAnalysis:
        """
        Анализирует транскрибацию звонка и извлекает структурированные данные.
        Использует Claude Sonnet 4.6 (Anthropic API).
        """
        try:
            logger.info(f"Анализируем разговор ({len(transcript)} символов)...")

            # Подготавливаем транскрипцию (обрезаем если слишком длинная)
            prepared_transcript = self._prepare_transcript(transcript)

            # Определяем длину звонка для адаптации параметров
            is_long_call = len(transcript) > 8000  # примерно 5+ минут

            call_type_ru = "Входящий" if call_type == "incoming" else "Исходящий"

            if call_direction == "call_out":
                call_direction_context = "Это ИСХОДЯЩИЙ звонок — менеджер позвонил клиенту."
            else:
                call_direction_context = "Это ВХОДЯЩИЙ звонок — клиент позвонил в компанию."

            logger.info(f"🤖 Анализ через anthropic/{ANTHROPIC_MODEL}")

            # Адаптируем max_tokens в зависимости от длины звонка
            if is_long_call:
                max_tokens = OPENAI_MAX_TOKENS
                logger.info(f"Длинный звонок, используем увеличенные лимиты: {max_tokens} токенов")
            else:
                max_tokens = min(OPENAI_MAX_TOKENS, 1500)
                logger.info(f"Короткий звонок, используем стандартные лимиты: {max_tokens} токенов")

            system_prompt = ANALYSIS_SYSTEM_PROMPT.format(manager_name=manager_name)
            user_prompt = ANALYSIS_USER_PROMPT.format(
                transcript=prepared_transcript,
                call_type=call_type_ru,
                manager_name=manager_name,
                call_direction_context=call_direction_context,
            )

            # Retry логика (2 попытки)
            max_retries = 2
            last_error = None
            result_json = None

            for attempt in range(max_retries):
                try:
                    result_text = await self._call_llm(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=max_tokens,
                    )

                    logger.info(
                        f"🔬 Claude raw first 100 chars: {repr(result_text[:100]) if result_text else 'EMPTY'}"
                    )

                    if not result_text.strip():
                        raise ValueError("Пустой ответ от Claude")

                    result_json = _extract_json_from_text(result_text)
                    if attempt > 0:
                        logger.info(f"✅ Claude успешно ответил с попытки {attempt + 1}")
                    break

                except (json.JSONDecodeError, ValueError) as e:
                    last_error = e
                    logger.warning(f"⚠️ Claude попытка {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        logger.info("🔄 Повторяем запрос к Claude...")
                        import asyncio
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"❌ Claude не вернул валидный JSON после {max_retries} попыток")
                        raise

            if result_json is None:
                raise last_error or ValueError("Не удалось получить ответ от Claude")

            next_steps = result_json.get("next_steps") or []
            if not isinstance(next_steps, list):
                next_steps = []

            # Имя менеджера известно из CRM (responsible_user_id → имя пользователя) —
            # это достоверный источник. LLM регулярно возвращает "Не представился",
            # даже когда имя звучит в разговоре, и затирает им хорошее значение.
            # Поэтому берём имя модели ТОЛЬКО если CRM его не дала (осталась заглушка).
            resolved_manager = _resolve_manager_name(
                from_crm=manager_name,
                from_llm=result_json.get("manager_name"),
            )

            # Создаём объект результата (Агент 1)
            analysis = CallAnalysis(
                client_name=result_json.get("client_name", "Клиент"),
                manager_name=resolved_manager,
                summary=result_json.get("summary", ""),
                client_city=result_json.get("client_city", "Не указано"),
                work_type=result_json.get("work_type", "Консультация"),
                location=result_json.get("location", "Не указано"),
                cost=result_json.get("cost", "Не обсуждали"),
                payment_terms=result_json.get("payment_terms", "Не обсуждали"),
                call_result=result_json.get("call_result", "Не определено"),
                next_contact_date=result_json.get("next_contact_date", "Не указано"),
                next_steps=[str(x).strip() for x in next_steps if str(x).strip()][:5],
            )

            logger.info("✅ Агент 1 (анализ через Claude) завершил работу")

            if ANALYSIS_PIPELINE_VERSION == "v3":
                # v3: задача валидатора слита в промпт fact verifier — один вызов вместо двух
                validated_analysis = analysis
            else:
                # Запускаем валидатор (Агент 2)
                validated_analysis = await self._validate_and_fix(
                    analysis,
                    transcript,  # Используем оригинальную транскрипцию
                    manager_name
                )
                logger.info("✅ Агент 2 (валидация через Claude) завершил работу")

            # v2: детерминированные метрики спикеров + верификация фактов.
            speaker_stats = self._build_speaker_stats(speakers)
            validated_analysis.speaker_stats = speaker_stats
            logger.info(
                "v2 speaker stats: participants=%s total_speech_seconds=%.1f dominant=%s suspicious=%s reason=%s",
                speaker_stats.participant_count,
                speaker_stats.total_speech_seconds,
                speaker_stats.dominant_speaker,
                speaker_stats.suspicious_diarization,
                speaker_stats.suspicious_reason,
            )

            if ANALYSIS_PIPELINE_VERSION in ("v2", "v3"):
                logger.info(f"🚀 ANALYSIS_PIPELINE_VERSION={ANALYSIS_PIPELINE_VERSION}, запускаем fact verifier")
                checks = await self._verify_with_claude(validated_analysis, transcript, speaker_stats)
                # Имя менеджера пришло из CRM — верификатор не должен его «исправлять»
                protected = frozenset({"manager_name"}) if _is_real_name(manager_name) else frozenset()
                validated_analysis = self._apply_verification(validated_analysis, checks, protected)
                logger.info("✅ Агент 3 (fact verifier через Claude) завершил работу")
            else:
                logger.info(f"ℹ️ ANALYSIS_PIPELINE_VERSION={ANALYSIS_PIPELINE_VERSION}, fact verifier отключен")

            return validated_analysis

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON от Claude: {e}")
            raise
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e.status_code} - {e.message}")
            raise
        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            raise

    def format_note(
        self,
        analysis: CallAnalysis,
        call_type: str = "outgoing",
        duration_seconds: float = 0,
        manager_name: str = "Менеджер",
        model_used: Optional[str] = None,
        stt_provider: Optional[str] = None,
    ) -> str:
        """
        Форматирует результат анализа в примечание для AmoCRM.
        """
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        duration_str = f"{minutes} мин {seconds} сек" if minutes else f"{seconds} сек"
        call_type_str = "Исходящий" if call_type == "outgoing" else "Входящий"

        model_name = model_used or f"anthropic/{ANTHROPIC_MODEL}"

        stt_label = (stt_provider or "assemblyai").strip().lower()
        stt_display = {"whisper": "Whisper", "assemblyai": "AssemblyAI", "yandex": "Yandex"}.get(stt_label, stt_label)

        steps_block = ""
        if analysis.next_steps:
            steps_block = "\n\n✅ Следующие шаги:\n" + "\n".join([f"- {s}" for s in analysis.next_steps])

        participants_block = ""
        if analysis.speaker_stats and analysis.speaker_stats.participant_count > 0:
            participants_block = f"\n👥 Участники: {analysis.speaker_stats.participant_count}"

        note = f"""🎙️ АНАЛИЗ ЗВОНКА (AI) [{model_name} | STT: {stt_display}]

📞 {call_type_str} | {duration_str}
{participants_block}

Спикеры:
- {analysis.manager_name} (менеджер)
- {analysis.client_name} (клиент)

Суть:
{analysis.summary}

📍 Город: {analysis.client_city}
🔧 Работа: {analysis.work_type}
💰 Стоимость: {analysis.cost}
💳 Оплата: {analysis.payment_terms}
📊 Итог: {analysis.call_result}
📅 Следующий контакт: {analysis.next_contact_date}{steps_block}"""

        return note


# Синглтон
analysis_service = AnalysisService()
