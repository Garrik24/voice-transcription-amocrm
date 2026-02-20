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

# ============== AssemblyAI ==============
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Модель распознавания (Universal-3-Pro — лучшее качество, Universal-2 — fallback)
ASSEMBLYAI_SPEECH_MODEL = os.getenv("ASSEMBLYAI_SPEECH_MODEL", "universal-3-pro")
ASSEMBLYAI_FALLBACK_MODEL = os.getenv("ASSEMBLYAI_FALLBACK_MODEL", "universal-2")

# Ожидаемое кол-во спикеров (для телефонных звонков = 2)
ASSEMBLYAI_SPEAKERS_EXPECTED = int(os.getenv("ASSEMBLYAI_SPEAKERS_EXPECTED", "2"))

# Multichannel: если телефония пишет стерео (каждый спикер на своём канале).
# Временно включено по умолчанию для проверки качества диаризации.
ASSEMBLYAI_MULTICHANNEL = os.getenv("ASSEMBLYAI_MULTICHANNEL", "true").strip().lower() == "true"

# ============== OpenAI ==============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============== LLM provider switch (OpenAI / Gemini) ==============
# openai | gemini
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

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
# v1 — текущий (агент анализа + валидатор пропущенных полей)
# v2 — усиленный (speaker stats + fact verifier)
ANALYSIS_PIPELINE_VERSION = os.getenv("ANALYSIS_PIPELINE_VERSION", "v1").strip().lower()

# Максимальная длина транскрипции для анализа (символов)
# Если транскрипция длиннее - берём начало и конец (где обычно ключевая информация)
MAX_TRANSCRIPT_LENGTH = int(os.getenv("MAX_TRANSCRIPT_LENGTH", "15000"))

# Обрезать транскрипцию для анализа (экономия токенов).
# По умолчанию ВЫКЛЮЧЕНО: для звонков до ~30 минут хотим анализировать весь текст без потерь.
TRUNCATE_TRANSCRIPT_FOR_ANALYSIS = os.getenv("TRUNCATE_TRANSCRIPT_FOR_ANALYSIS", "false").strip().lower() == "true"

# ============== Telegram ==============
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # ID чата для уведомлений об ошибках

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
