"""
Сервис анализа разговора через OpenAI GPT.
Извлекает структурированную информацию из транскрибации.
"""
import openai
import json
import logging
import re
from typing import List
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
    cost: str  # Стоимость
    payment_terms: str  # Условия оплаты
    call_result: str  # Итог звонка
    next_contact_date: str  # Когда связаться
    next_steps: List[str]  # Следующие шаги для менеджера (0-5)


# Системный промпт для анализа (Агент 1)
ANALYSIS_SYSTEM_PROMPT = """Ты — ассистент для анализа телефонных разговоров геодезической компании.

Твоя задача — извлечь ТОЛЬКО ФАКТЫ из транскрибации и вернуть структурированный JSON.
НЕЛЬЗЯ выдумывать. Если данных нет в тексте — пиши "Не указано" или "Не обсуждали" (как указано ниже).

ВАЖНО ПРО РОЛИ (читай внимательно!):
- Менеджер компании: {manager_name}
- {manager_name} = ВСЕГДА менеджер (это сотрудник компании!)
- Если кто-то представляется от имени компании → менеджер
- НЕ путай роли! Если {manager_name} говорит "мне нужно отправить договор" — это МЕНЕДЖЕР говорит о своих задачах, НЕ клиент!
- Остальные участники разговора → клиенты

ВАЖНО для длинных звонков (5+ минут):
- Внимательно прочитай ВСЮ транскрибацию от начала до конца
- Ключевая информация может быть в ЛЮБОЙ части разговора (начало, середина, конец)
- Не пропускай детали: стоимость, условия оплаты, даты, адреса могут упоминаться несколько раз
- Если информация повторяется — используй самую полную версию
- Обязательно проверь конец разговора — там часто финальные договорённости

ОБЯЗАТЕЛЬНЫЕ ПОЛЯ (ищи особенно внимательно!):
client_city (населенный пункт):
  - Ищи: город, регион, населенный пункт, адрес, "я из...", "живу в...", "участок в..."
  - Примеры: "Краснодар", "Москва", "Ростов", "пригород Ростова", "СНТ Солнечный"

cost (сумма договора):
  - Ищи: цена, стоимость, сумма, "тысяч", "рублей", "₽", "стоит", "обойдется"
  - Примеры: "25 000 ₽", "двадцать тысяч", "от 20 000", "около 30 тысяч"
  - ВАЖНО: Даже если говорят "примерно" или "около" — это СТОИМОСТЬ!

payment_terms (условия оплаты):
  - Ищи: предоплата, аванс, 50/50, 100%, рассрочка, "сначала", "потом", "после выполнения"
  - Примеры: "50% предоплата", "оплата после выполнения", "половину сейчас, половину потом"

next_contact_date (дата следующего контакта):
  - Ищи: дата, день недели, "позвоню", "свяжусь", "перезвоню", "когда связаться"
  - Примеры: "среда", "15 января", "завтра в 14:00", "через неделю", "в понедельник"

Верни JSON со следующими полями:
{
  "client_name": "Имя клиента (как представился) или 'Клиент'. Ищи в начале разговора.",
  "manager_name": "Имя/ФИО менеджера из разговора или то, что передали в поле manager_name",
  "summary": "Развёрнутая суть разговора: для коротких звонков 8–12 предложений, для длинных (5+ мин) до 20 предложений. ОБЯЗАТЕЛЬНО охвати: начало (представление, запрос клиента), середину (обсуждение деталей, стоимости, условий), конец (финальные договорённости, итог, следующий шаг). Без воды и повторов.",
  "client_city": "Город/регион или 'Не указано'. Ищи упоминания городов, адресов, регионов.",
  "work_type": "Тип работ (например: 'Топографическая съёмка 1:500', 'Межевание участка', 'Вынос границ') или 'Консультация'. Ищи конкретные формулировки типа работ.",
  "cost": "Стоимость (например: '25 000 ₽', '18 тысяч рублей', 'от 20 000') или 'Не обсуждали'. Ищи ВСЕ упоминания цен, сумм, стоимости.",
  "payment_terms": "Условия оплаты (например: '50/50', 'предоплата 50%', '100% после выполнения') или 'Не обсуждали'. Ищи упоминания предоплаты, рассрочки, условий оплаты.",
  "call_result": "Итог: 'Договорились'/'Клиент думает'/'Отказ'/'Перезвонить'/'Отправить договор' и т.п. ОБЯЗАТЕЛЬНО проверь конец разговора. Если непонятно — 'Не определено'",
  "next_contact_date": "Когда связаться (если было) иначе 'Не указано'. Ищи конкретные даты, дни недели, время.",
  "next_steps": ["1-5 конкретных следующих шагов для менеджера по итогам разговора. Извлекай из концовки разговора. Если нет — пустой массив []"]
}

Правила качества summary:
- Пиши человеческим языком, без канцелярита и без 'возможно/наверное'
- Не вставляй мусорные слова, не повторяй одну мысль
- Не обрезай концовку: в summary должен быть финал разговора и чем закончили
- Для длинных звонков: обязательно упомяни ключевые моменты из начала, середины И конца
- Если в транскрибации каша/обрывки — формулируй только то, что точно ясно; остальное не додумывай
- Если разговор длинный — summary может быть до 20 предложений, но без воды

Правила извлечения информации:
- Ищи информацию во ВСЕЙ транскрибации, не только в начале
- Если информация упоминается несколько раз — используй самую полную/точную версию
- Особое внимание к концу разговора — там часто финальные договорённости
- Не пиши "Не определено" если информация есть в тексте — найди её!

Отвечай ТОЛЬКО JSON, без пояснений и без Markdown.
"""


ANALYSIS_USER_PROMPT = """Проанализируй разговор между менеджером и клиентом.

Тип звонка: {call_type}
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


class AnalysisService:
    """Сервис анализа разговоров через GPT/Gemini с валидацией"""
    
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
        manager_name: str = "Менеджер"
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

            provider = (LLM_PROVIDER or "openai").strip().lower()
            
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
                            manager_name=manager_name
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
        manager_name: str = "Менеджер"
    ) -> str:
        """
        Форматирует результат анализа в примечание для AmoCRM.
        """
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        duration_str = f"{minutes} мин {seconds} сек" if minutes else f"{seconds} сек"
        call_type_str = "Исходящий" if call_type == "outgoing" else "Входящий"

        steps_block = ""
        if analysis.next_steps:
            steps_block = "\n\n✅ Следующие шаги:\n" + "\n".join([f"- {s}" for s in analysis.next_steps])
        
        note = f"""🎙️ АНАЛИЗ ЗВОНКА (AI)

📞 {call_type_str} | {duration_str}

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
