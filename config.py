"""
Конфигурация приложения.
Все секретные ключи берутся из переменных окружения Railway.
"""
import os
from dotenv import load_dotenv

# Загружаем .env файл для локальной разработки
load_dotenv()

# ============== AmoCRM ==============
AMOCRM_DOMAIN = os.getenv("AMOCRM_DOMAIN")  # например: stavgeo26.amocrm.ru
AMOCRM_ACCESS_TOKEN = os.getenv("AMOCRM_ACCESS_TOKEN")
AMOCRM_REFRESH_TOKEN = os.getenv("AMOCRM_REFRESH_TOKEN")
AMOCRM_CLIENT_ID = os.getenv("AMOCRM_CLIENT_ID")
AMOCRM_CLIENT_SECRET = os.getenv("AMOCRM_CLIENT_SECRET")
AMOCRM_VERIFY_SSL = os.getenv("AMOCRM_VERIFY_SSL", "true").strip().lower() == "true"

# ============== STT Provider ==============
# whisper | assemblyai | yandex
STT_PROVIDER = os.getenv("STT_PROVIDER", "whisper").strip().lower()

# ============== AssemblyAI (резервный провайдер) ==============
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Автопереключение на AssemblyAI, когда основной провайдер отвалился
# по инфраструктурной причине (кончился баланс / отозван ключ).
# Обычные ошибки (сеть, битое аудио) переключение НЕ вызывают.
STT_FALLBACK_ENABLED = os.getenv("STT_FALLBACK_ENABLED", "true").strip().lower() == "true"

# Через сколько минут в режиме резерва пробовать основной провайдер снова
STT_FALLBACK_RETRY_MINUTES = int(os.getenv("STT_FALLBACK_RETRY_MINUTES", "30"))

# Модель распознавания. Пусто = дефолт AssemblyAI (проверено на русских
# звонках: дефолт, best и universal дают одинаковый результат, nano падает).
# Прежнее значение universal-2 невалидно для текущего SDK — оставляем пустым.
ASSEMBLYAI_SPEECH_MODEL = os.getenv("ASSEMBLYAI_SPEECH_MODEL", "").strip()

# Ожидаемое кол-во спикеров (для телефонных звонков = 2)
ASSEMBLYAI_SPEAKERS_EXPECTED = int(os.getenv("ASSEMBLYAI_SPEAKERS_EXPECTED", "2"))

# Multichannel не используется: каналы стерео разделяются нашим ffmpeg,
# в AssemblyAI уходит уже моно-канал — так работает общий пайплайн
# (энергетический гейт, дедуп, роли) независимо от провайдера.
ASSEMBLYAI_MULTICHANNEL = os.getenv("ASSEMBLYAI_MULTICHANNEL", "false").strip().lower() == "true"

# ============== OpenAI ==============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============== Anthropic (Claude) ==============
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ============== Цепочка провайдеров анализа (LLM) ==============
# Анализ звонка идёт по цепочке: если у провайдера кончились деньги или отозван
# ключ — автоматически переходим к следующему, уведомляя в Telegram. Через
# LLM_FALLBACK_RETRY_MINUTES пробуем вернуться на основной.
#
# Порядок по умолчанию: anthropic → assemblyai → openai.
# assemblyai здесь — не распознавание речи, а LLM Gateway AssemblyAI
# (llm-gateway.assemblyai.com), где доступна та же модель claude-sonnet-4-6,
# что и у Anthropic напрямую. Поэтому качество сводок при переключении
# не падает, а оплата идёт с баланса AssemblyAI.
LLM_CHAIN = [
    p.strip().lower()
    for p in os.getenv("LLM_CHAIN", "anthropic,assemblyai,openai").split(",")
    if p.strip()
]

# Автопереключение по цепочке. false — работает только первый провайдер.
LLM_FALLBACK_ENABLED = os.getenv("LLM_FALLBACK_ENABLED", "true").strip().lower() == "true"

# Через сколько минут пробовать вернуться на основной провайдер
LLM_FALLBACK_RETRY_MINUTES = int(os.getenv("LLM_FALLBACK_RETRY_MINUTES", "30"))

# Модель в LLM Gateway AssemblyAI (список: GET llm-gateway.assemblyai.com/v1/models)
ASSEMBLYAI_LLM_MODEL = os.getenv("ASSEMBLYAI_LLM_MODEL", "claude-sonnet-4-6")

# Устаревшая настройка: в рабочем коде не используется, оставлена для скриптов.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()

# Модели (можно переопределить в Railway Variables)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Google Gemini (google-genai SDK)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")

# ============== AI Analysis Settings ==============
# Максимальное количество токенов для ответа (увеличено для длинных звонков)
# Для коротких звонков (< 3 мин): 1200 токенов достаточно
# Для длинных звонков (5+ мин): нужно 2500-3000 токенов
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2500"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "3000"))

# Температура для анализа (низкая = более точные факты)
ANALYSIS_TEMPERATURE = float(os.getenv("ANALYSIS_TEMPERATURE", "0.1"))

# Версия конвейера анализа:
# v1 — базовый (агент анализа + валидатор пропущенных полей)
# v2 — усиленный (speaker stats + fact verifier)
#
# Дефолт v2 = то, что стоит в проде. Раньше здесь был v1, и расхождение
# скрывало баги: fact verifier есть только в v2, поэтому локальный прогон
# вёл себя иначе, чем прод (так был упущен случай, когда верификатор
# заменял имя менеджера из CRM на заглушку).
# Минимальная длительность звонка для анализа, сек (короче — сброс/автоответчик)
MIN_CALL_SECONDS = int(os.getenv("MIN_CALL_SECONDS", "25"))

ANALYSIS_PIPELINE_VERSION = os.getenv("ANALYSIS_PIPELINE_VERSION", "v3").strip().lower()

# Максимальная длина транскрипции для анализа (символов)
# Если транскрипция длиннее - берём начало и конец (где обычно ключевая информация)
MAX_TRANSCRIPT_LENGTH = int(os.getenv("MAX_TRANSCRIPT_LENGTH", "15000"))

# Обрезать транскрипцию для анализа (экономия токенов).
# По умолчанию ВЫКЛЮЧЕНО: для звонков до ~30 минут хотим анализировать весь текст без потерь.
TRUNCATE_TRANSCRIPT_FOR_ANALYSIS = os.getenv("TRUNCATE_TRANSCRIPT_FOR_ANALYSIS", "false").strip().lower() == "true"

# ============== Yandex SpeechKit ==============
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_STT_LANGUAGE = os.getenv("YANDEX_STT_LANGUAGE", "ru-RU")

# ============== Telegram ==============
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # ID чата для уведомлений об ошибках

# ============== Стерео-транскрибация ==============
# Откуда брать ТЕКСТ на стерео-записях.
# true  — распознаём моно-смесь каналов (один запрос вместо двух), а роли
#         раздаём по энергии каналов. Точнее: на отдельном канале тихие быстрые
#         фразы разваливаются, и Whisper подставляет слова из соседних реплик
#         (реальный случай: «меня Эдуард» → «я Игорь»). В сумме каналов фраза
#         читается. Побочный эффект — вдвое меньше вызовов STT.
# false — прежнее поведение: Whisper отдельно на каждый канал.
STEREO_TEXT_FROM_MONO = os.getenv("STEREO_TEXT_FROM_MONO", "true").strip().lower() == "true"

# ============== Инфраструктурные алерты ==============
# Алертим только на сбои, останавливающие конвейер (кончился баланс / отозван ключ).
# Обычные ошибки звонков по-прежнему только логируются — иначе один сбой
# провайдера даёт десятки сообщений подряд.
ALERTS_ENABLED = os.getenv("ALERTS_ENABLED", "true").strip().lower() == "true"

# Минимальный интервал между повторными алертами одного класса, минут
ALERT_COOLDOWN_MINUTES = int(os.getenv("ALERT_COOLDOWN_MINUTES", "30"))

# ============== Приложение ==============
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8000))

# ============== Список менеджеров ==============
# Формат: {"user_id_в_amocrm": "Имя"}
# Заполни ID своих менеджеров из AmoCRM
MANAGERS = {
    # "12345": "Елена",
    # "12346": "Дмитрий",
    # "12347": "Александр",
}

def validate_config():
    """
    Проверяет конфигурацию.

    Возвращает список отсутствующих переменных (пустой список = всё ок).
    """
    required = [
        ("AMOCRM_DOMAIN", AMOCRM_DOMAIN),
        ("AMOCRM_ACCESS_TOKEN", AMOCRM_ACCESS_TOKEN),
    ]

    optional_groups = [
        ("ASSEMBLYAI_API_KEY", ASSEMBLYAI_API_KEY),
        ("OPENAI_API_KEY", OPENAI_API_KEY),
    ]

    missing_required = [name for name, value in required if not value]
    missing_optional = [name for name, value in optional_groups if not value]

    return missing_required + missing_optional
