"""
Сервис анализа разговора через OpenAI GPT.
Извлекает структурированную информацию из транскрибации.
"""
import openai
import json
import logging
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_MAX_TOKENS,
    GEMINI_MAX_OUTPUT_TOKENS,
    ANALYSIS_TEMPERATURE,
    ANALYSIS_PIPELINE_VERSION,
    MAX_TRANSCRIPT_LENGTH,
    TRUNCATE_TRANSCRIPT_FOR_ANALYSIS,
)

logger = logging.getLogger(__name__)

_client: openai.AsyncOpenAI | None = None
_gemini_client = None


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


def _get_client() -> openai.AsyncOpenAI:
    """
    Инициализируем OpenAI клиент лениво.

    Важно для деплоя: если OPENAI_API_KEY не задан, сервис всё равно должен стартовать
    (например, для автоматизаций без транскрибации).
    """
    global _client
    if _client is not None:
        return _client
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан (нужен для анализа звонков)")
    _client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


def _get_gemini_client():
    """
    Инициализируем Google GenAI (Gemini) клиент лениво.

    Важно: ключи не валим на старте приложения — только при попытке анализа.
    """
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY не задан (нужен для анализа звонков через Gemini)")
    # Импортируем внутри, чтобы не падать, если провайдер не используется.
    from google import genai

    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


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


# Системный промпт для анализа (Агент 1)
ANALYSIS_SYSTEM_PROMPT = """Ты — ассистент для анализа телефонных разговоров геодезической компании.
Твоя задача — извлечь только факты из транскрибации и вернуть строго JSON.

1. Определение ролей
- Менеджер компании — спикер с именем {manager_name}. Всё, что он говорит — позиция компании.
- Все остальные спикеры — клиенты.
- Если имя клиента не прозвучало — "client_name": "Не представился".
- Если менеджер обращается к клиенту по имени — это имя клиента.

2. Инструкции по чтению транскрибации
- Читай ВСЮ транскрибацию от начала до конца. Не пропускай фрагменты.
- Длинные звонки (10+ минут) часто содержат важные детали в середине и в конце — не игнорируй их.
- Если в разговоре обсуждаются несколько объектов или задач — фиксируй каждую.
- Извлекай ТОЛЬКО то, что реально прозвучало. Не додумывай, не интерпретируй.

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
  - Если неясно → "Не определено"

location:
  Адрес, район или ориентир объекта работ. Краткая форма для названия сделки.
  Ищи: улицу, шоссе, район, СНТ, КП, посёлок, ориентир рядом с объектом.
  Примеры: "Старомарьское шоссе", "ул. Ленина 15", "СНТ Солнечный", "п. Иноземцево", "р-н Юго-Западный"
  ВАЖНО: НЕ дублируй город — только адрес/район/ориентир. Пиши кратко (2-4 слова).
  Если не упоминалось → "Не указано"

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
}}}}

7. Пример

Транскрибация:
Менеджер (Игорь): Алло, добрый день!
Клиент (Давид): Здравствуйте, меня зовут Давид. Мне нужно вынести границы участка.
Менеджер: Да, подскажите, где участок находится?
Клиент: Это возле Старомарьского шоссе, рядом с автосалоном Хрополь, ну там недалеко от четырёх сезонов.
Менеджер: Понял. А площадь участка какая?
Клиент: 80 соток, участок ровный.
Менеджер: Хорошо. Вынос точек по GPS стоит 4000 рублей. Но я должен вас предупредить — в том районе GPS может глушиться. Если сигнал не ловится, нужен будет геодезический ход, это от 25 тысяч.
Клиент: Понял. Давайте сначала попробуем по GPS.
Менеджер: Договорились. Скиньте мне кадастровый номер участка в телеграм, я закажу расширенную выписку, посмотрю сколько точек, и скажу вам точную цену.
Клиент: Хорошо, скину. Только я смогу начать не раньше чем через неделю.
Клиент: А, да, мне Алексей ваш порекомендовал к вам обратиться.
Менеджер: Отлично, спасибо. Жду данные, до свидания!

Правильный ответ:
{{{{
  "client_name": "Давид",
  "manager_name": "Игорь",
  "summary": "Клиент Давид — вынос границ участка (~80 соток, ровная форма), район Старомарьского шоссе, рядом с автосалоном Хрополь / «Четыре сезона». GPS в этом районе глушится — вынос по GPS 4 000 ₽, если не ловит — геодезический ход от 25 000 ₽. Порекомендовал обратиться Алексей из компании. Договорились: Давид скинет кадастровый номер в Telegram, после чего будет определена точная цена и дата выезда.",
  "client_city": "Не указано",
  "work_type": "Вынос границ",
  "location": "Старомарьское шоссе",
  "cost": "4 000 ₽ за вынос по GPS, от 25 000 ₽ за геодезический ход",
  "payment_terms": "Не обсуждали",
  "call_result": "Согласие",
  "next_contact_date": "Через неделю",
  "next_steps": [
    "Получить от клиента кадастровый номер участка в Telegram",
    "Менеджеру: заказать расширенную выписку, определить кол-во точек и точную стоимость",
    "Менеджеру: сообщить клиенту цену и согласовать дату выезда"
  ]
}}}}

Пример ПЛОХОГО summary (так делать НЕЛЬЗЯ):
"summary": "Клиент Давид позвонил менеджеру Игорю по поводу выноса точки на участке, который находится возле Старомарьского шоссе, рядом с автосалоном Хрополь. Менеджер объяснил, что вынос по GPS стоит 4000 рублей, но если сигнал не ловится, потребуется геодезический ход, что обойдётся от 25 тысяч рублей. Клиент согласился сначала попробовать вынос по GPS. Менеджер попросил кадастровый номер участка. Клиент согласился отправить данные через Telegram."
Почему плохо: это хронологический пересказ, дублирует поля cost и next_steps, содержит вводную «позвонил менеджеру».
"""


ANALYSIS_USER_PROMPT = """Проанализируй разговор между менеджером и клиентом.

Тип звонка: {call_type}
{call_direction_context}
Менеджер компании: {manager_name}

ПОМНИ: {manager_name} = МЕНЕДЖЕР (не клиент!)

ТРАНСКРИБАЦИЯ РАЗГОВОРА:
{transcript}
"""


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

Отвечай ТОЛЬКО JSON, без пояснений.
"""


VALIDATOR_USER_PROMPT = """Первый анализ пропустил обязательную информацию.

Пропущенные поля: {missing_fields}

ТРАНСКРИБАЦИЯ РАЗГОВОРА:
{transcript}

Найди пропущенную информацию для этих полей. Если действительно нет — верни "Не указано".
"""


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
    "work_type": "Консультация",
    "location": "Не указано",
    "cost": "Не обсуждали",
    "payment_terms": "Не обсуждали",
    "call_result": "Не определено",
    "next_contact_date": "Не указано",
    "next_steps": [],
}

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

Безопасные значения:
- city/date -> "Не указано"
- cost/payment_terms -> "Не обсуждали"
- work_type -> "Консультация"
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
}}
"""


class AnalysisService:
    """Сервис анализа разговоров через GPT/Gemini с валидацией"""

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

    async def _verify_with_gemini(
        self,
        analysis: CallAnalysis,
        transcript: str,
        speaker_stats: SpeakerStats,
    ) -> Dict[str, FieldVerification]:
        try:
            prepared_transcript = self._prepare_transcript(transcript)
            gemini = _get_gemini_client()
            from google.genai import types

            schema = {
                "type": "OBJECT",
                "required": ["fields"],
                "properties": {
                    "fields": {
                        "type": "OBJECT",
                        "properties": {
                            name: {
                                "type": "OBJECT",
                                "required": ["status", "confidence", "suggested_fix", "evidence"],
                                "properties": {
                                    "status": {"type": "STRING"},
                                    "confidence": {"type": "NUMBER"},
                                    "suggested_fix": {"type": "STRING"} if name != "next_steps" else {"type": "ARRAY", "items": {"type": "STRING"}},
                                    "evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
                                },
                            }
                            for name in VERIFY_FIELDS
                        },
                    }
                },
            }

            prompt = FACT_VERIFIER_USER_PROMPT.format(
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

            response = await gemini.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{FACT_VERIFIER_SYSTEM_PROMPT}\n\n{prompt}",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1200,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            parsed = json.loads(response.text or "{}")
            return self._normalize_verification_result(parsed)
        except Exception as e:
            logger.error(f"Ошибка fact verifier через Gemini: {e}")
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

    async def _verify_with_openai(
        self,
        analysis: CallAnalysis,
        transcript: str,
        speaker_stats: SpeakerStats,
    ) -> Dict[str, FieldVerification]:
        try:
            prepared_transcript = self._prepare_transcript(transcript)
            client = _get_client()
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": FACT_VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": FACT_VERIFIER_USER_PROMPT.format(
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
                    )},
                ],
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            return self._normalize_verification_result(parsed)
        except Exception as e:
            logger.error(f"Ошибка fact verifier через OpenAI: {e}")
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

    def _apply_verification(self, analysis: CallAnalysis, checks: Dict[str, FieldVerification]) -> CallAnalysis:
        for field, check in checks.items():
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

    async def _validate_with_gemini(
        self,
        transcript: str,
        missing_fields: List[str]
    ) -> dict:
        """Валидация через Gemini (Агент 2)"""
        try:
            prepared_transcript = self._prepare_transcript(transcript)
            gemini = _get_gemini_client()
            from google.genai import types
            
            # Схема ответа для валидатора
            response_schema = {
                "type": "OBJECT",
                "required": ["client_city", "cost", "payment_terms", "next_contact_date"],
                "properties": {
                    "client_city": {"type": "STRING"},
                    "cost": {"type": "STRING"},
                    "payment_terms": {"type": "STRING"},
                    "next_contact_date": {"type": "STRING"},
                },
            }
            
            prompt = (
                f"{VALIDATOR_SYSTEM_PROMPT}\n\n"
                + VALIDATOR_USER_PROMPT.format(
                    missing_fields=", ".join(missing_fields),
                    transcript=prepared_transcript
                )
            )
            
            response = await gemini.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=800,  # Валидатору нужно меньше
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            
            result_text = response.text or "{}"
            return json.loads(result_text)
            
        except Exception as e:
            logger.error(f"Ошибка валидации через Gemini: {e}")
            return {}
    
    async def _validate_with_openai(
        self,
        transcript: str,
        missing_fields: List[str]
    ) -> dict:
        """Валидация через OpenAI (Агент 2)"""
        try:
            prepared_transcript = self._prepare_transcript(transcript)
            client = _get_client()
            
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": VALIDATOR_USER_PROMPT.format(
                        missing_fields=", ".join(missing_fields),
                        transcript=prepared_transcript
                    )}
                ],
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            return json.loads(result_text)
            
        except Exception as e:
            logger.error(f"Ошибка валидации через OpenAI: {e}")
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
        
        provider = (LLM_PROVIDER or "openai").strip().lower()
        
        if provider == "gemini":
            fixed_data = await self._validate_with_gemini(transcript, missing)
        else:
            fixed_data = await self._validate_with_openai(transcript, missing)
        
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
        Если TRUNCATE_TRANSCRIPT_FOR_ANALYSIS=true и транскрипция слишком длинная —
        берём начало и конец (где обычно ключевая информация) для экономии токенов.
        """
        if not TRUNCATE_TRANSCRIPT_FOR_ANALYSIS:
            return transcript

        if len(transcript) <= MAX_TRANSCRIPT_LENGTH:
            return transcript
        
        logger.info(f"Транскрипция длинная ({len(transcript)} символов), обрезаем до {MAX_TRANSCRIPT_LENGTH}")
        
        # Берём начало (первые 60%) и конец (последние 40%)
        # Это сохраняет представление в начале и финальные договорённости в конце
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

            provider = (LLM_PROVIDER or "openai").strip().lower()
            model_name = GEMINI_MODEL if provider == "gemini" else OPENAI_MODEL
            logger.info(f"🤖 Анализ через {provider}/{model_name}")

            # Адаптируем max_tokens в зависимости от длины звонка
            if is_long_call:
                max_tokens = OPENAI_MAX_TOKENS
                max_output_tokens = GEMINI_MAX_OUTPUT_TOKENS
                logger.info(f"Длинный звонок, используем увеличенные лимиты: {max_tokens} токенов")
            else:
                max_tokens = min(OPENAI_MAX_TOKENS, 1500)
                max_output_tokens = min(GEMINI_MAX_OUTPUT_TOKENS, 2000)
                logger.info(f"Короткий звонок, используем стандартные лимиты: {max_tokens} токенов")

            if provider == "gemini":
                gemini = _get_gemini_client()
                from google.genai import types

                # Схема ответа: строго JSON-объект с ожидаемыми полями.
                response_schema = {
                    "type": "OBJECT",
                    "required": [
                        "client_name",
                        "manager_name",
                        "summary",
                        "client_city",
                        "work_type",
                        "cost",
                        "payment_terms",
                        "call_result",
                        "next_contact_date",
                        "next_steps",
                    ],
                    "properties": {
                        "client_name": {"type": "STRING"},
                        "manager_name": {"type": "STRING"},
                        "summary": {"type": "STRING"},
                        "client_city": {"type": "STRING"},
                        "work_type": {"type": "STRING"},
                        "cost": {"type": "STRING"},
                        "payment_terms": {"type": "STRING"},
                        "call_result": {"type": "STRING"},
                        "next_contact_date": {"type": "STRING"},
                        "next_steps": {"type": "ARRAY", "items": {"type": "STRING"}},
                    },
                }

                prompt = (
                    ANALYSIS_SYSTEM_PROMPT.format(manager_name=manager_name) + "\n\n"
                    + ANALYSIS_USER_PROMPT.format(
                        transcript=prepared_transcript,
                        call_type=call_type_ru,
                        manager_name=manager_name,
                        call_direction_context=call_direction_context,
                    )
                )

                # Retry логика для Gemini (2 попытки)
                max_retries = 2
                last_error = None
                result_json = None
                
                for attempt in range(max_retries):
                    try:
                        response = await gemini.aio.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=ANALYSIS_TEMPERATURE,
                                max_output_tokens=max_output_tokens,
                                response_mime_type="application/json",
                                response_schema=response_schema,
                            ),
                        )

                        result_text = response.text or ""
                        
                        # #region agent log - DEBUG: capture raw Gemini response
                        logger.info(f"🔬 [H1] Gemini raw first 100 chars: {repr(result_text[:100]) if result_text else 'EMPTY'}")
                        logger.info(f"🔬 [H2] starts_with_brace={result_text.startswith('{') if result_text else False}, len={len(result_text) if result_text else 0}")
                        logger.info(f"🔬 [H3] has_markdown={'```' in result_text if result_text else False}")
                        if result_text and len(result_text) > 100:
                            logger.info(f"🔬 [H5] last 100 chars: {repr(result_text[-100:])}")
                        # #endregion
                        
                        if not result_text.strip():
                            raise ValueError("Пустой ответ от Gemini")
                        
                        result_json = json.loads(result_text)
                        if attempt > 0:
                            logger.info(f"✅ Gemini успешно ответил с попытки {attempt + 1}")
                        break  # Успех — выходим из цикла
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        last_error = e
                        # #region agent log - DEBUG: capture parse error
                        logger.error(f"🔬 [ERROR] JSON parse failed: {type(e).__name__}: {e}")
                        logger.error(f"🔬 [ERROR] Full raw response: {repr(result_text[:500]) if result_text else 'NO_RESPONSE'}")
                        # #endregion
                        logger.warning(f"⚠️ Gemini попытка {attempt + 1}/{max_retries}: {e}")
                        if hasattr(response, 'text') and response.text:
                            logger.warning(f"Сырой ответ ({len(response.text)} символов): {response.text[:500]}...")
                        
                        if attempt < max_retries - 1:
                            logger.info("🔄 Повторяем запрос к Gemini...")
                            import asyncio
                            await asyncio.sleep(1)  # Небольшая пауза перед retry
                        else:
                            logger.error(f"❌ Gemini не вернул валидный JSON после {max_retries} попыток")
                            raise
                
                if result_json is None:
                    raise last_error or ValueError("Не удалось получить ответ от Gemini")

            else:
                client = _get_client()
                response = await client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT.format(manager_name=manager_name)},
                        {"role": "user", "content": ANALYSIS_USER_PROMPT.format(
                            transcript=prepared_transcript,
                            call_type=call_type_ru,
                            manager_name=manager_name,
                            call_direction_context=call_direction_context,
                        )}
                    ],
                    temperature=ANALYSIS_TEMPERATURE,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )

                result_text = response.choices[0].message.content
                result_json = json.loads(result_text)

            next_steps = result_json.get("next_steps") or []
            if not isinstance(next_steps, list):
                next_steps = []
            
            # Создаём объект результата (Агент 1)
            analysis = CallAnalysis(
                client_name=result_json.get("client_name", "Клиент"),
                manager_name=result_json.get("manager_name", manager_name),
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
            
            logger.info("✅ Агент 1 (анализ) завершил работу")
            
            # Запускаем валидатор (Агент 2)
            validated_analysis = await self._validate_and_fix(
                analysis,
                transcript,  # Используем оригинальную транскрипцию
                manager_name
            )

            logger.info("✅ Агент 2 (валидация) завершил работу")

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

            if ANALYSIS_PIPELINE_VERSION == "v2":
                logger.info("🚀 ANALYSIS_PIPELINE_VERSION=v2, запускаем fact verifier (Агент 3)")
                provider = (LLM_PROVIDER or "openai").strip().lower()
                if provider == "gemini":
                    checks = await self._verify_with_gemini(validated_analysis, transcript, speaker_stats)
                else:
                    checks = await self._verify_with_openai(validated_analysis, transcript, speaker_stats)
                validated_analysis = self._apply_verification(validated_analysis, checks)
                logger.info("✅ Агент 3 (fact verifier) завершил работу")
            else:
                logger.info("ℹ️ ANALYSIS_PIPELINE_VERSION=v1, fact verifier отключен")

            return validated_analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON от GPT: {e}")
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
        model_used: например "openai/gpt-4o-mini" или "gemini/gemini-2.0-flash-001"
        stt_provider: например "whisper", "assemblyai", "yandex"
        """
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        duration_str = f"{minutes} мин {seconds} сек" if minutes else f"{seconds} сек"
        call_type_str = "Исходящий" if call_type == "outgoing" else "Входящий"

        provider = (LLM_PROVIDER or "openai").strip().lower()
        model_name = model_used or (f"{provider}/{GEMINI_MODEL}" if provider == "gemini" else f"{provider}/{OPENAI_MODEL}")

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
