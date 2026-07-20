"""
Инфраструктурные алерты в Telegram.

Обычные ошибки обработки звонка сюда НЕ попадают — они по-прежнему только
логируются. Иначе один сбой провайдера превращается в лавину сообщений:
08.07.2026 за время простоя было 26 падений подряд.

Алертим только на то, что требует ручного вмешательства и останавливает
конвейер целиком:
  • закончился баланс (OpenAI / Anthropic)
  • ключ недействителен или отозван

Плюс троттлинг: повторный алерт того же класса — не чаще, чем раз в
ALERT_COOLDOWN_MINUTES. И отдельное уведомление, когда всё снова заработало.
"""
import logging
import time
from typing import Optional, Tuple

import anthropic
import openai

from config import ALERT_COOLDOWN_MINUTES, ALERTS_ENABLED
from services.telegram import telegram_service

logger = logging.getLogger(__name__)

KIND_QUOTA = "quota"
KIND_AUTH = "auth"

# Когда последний раз слали алерт по ключу "Провайдер:вид"
_last_sent: dict[str, float] = {}
# Какие сбои сейчас считаются активными (для уведомления о восстановлении)
_active: set[str] = set()


def classify(exc: BaseException) -> Optional[Tuple[str, str, str]]:
    """
    Определяет, является ли ошибка инфраструктурной.

    Returns:
        (kind, provider, human_text) либо None — если ошибка обычная
        (сеть, битое аудио, таймаут и т.п.) и алерт не нужен.
    """
    # --- OpenAI (Whisper STT) ---
    # openai.APIError заполняет .code из тела ответа: 'insufficient_quota',
    # 'invalid_api_key' и т.п. RateLimitError/AuthenticationError — подклассы
    # APIStatusError, поэтому одной проверки достаточно.
    if isinstance(exc, openai.APIStatusError):
        code = getattr(exc, "code", None)
        status = getattr(exc, "status_code", None)
        if code == "insufficient_quota":
            return (
                KIND_QUOTA,
                "OpenAI",
                "Закончился баланс OpenAI — Whisper не расшифровывает звонки.",
            )
        if status == 401 or code in ("invalid_api_key", "account_deactivated"):
            return (
                KIND_AUTH,
                "OpenAI",
                "Ключ OpenAI недействителен или отозван.",
            )
        # 429 без insufficient_quota — обычный rate limit, SDK сам ретраит. Не алертим.

    # --- Anthropic (анализ звонка) ---
    # anthropic.APIError НЕ заполняет .code, поэтому смотрим статус и текст.
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", None)
        msg = (getattr(exc, "message", "") or str(exc)).lower()
        if "credit balance" in msg or "billing" in msg:
            return (
                KIND_QUOTA,
                "Anthropic",
                "Закончился баланс Anthropic — анализ звонков не выполняется.",
            )
        if status == 401:
            return (
                KIND_AUTH,
                "Anthropic",
                "Ключ Anthropic недействителен или отозван.",
            )

    return None


def _cooldown_passed(key: str) -> bool:
    """True, если по этому ключу можно слать алерт (и отмечает отправку)."""
    now = time.time()
    if now - _last_sent.get(key, 0.0) < ALERT_COOLDOWN_MINUTES * 60:
        return False
    _last_sent[key] = now
    return True


async def maybe_alert(exc: BaseException, lead_id: Optional[int] = None) -> bool:
    """
    Шлёт алерт, если ошибка инфраструктурная и не подавлена троттлингом.

    Returns:
        True — сообщение ушло в Telegram.
    """
    verdict = classify(exc)
    if not verdict:
        return False

    kind, provider, human = verdict
    key = f"{provider}:{kind}"
    _active.add(key)

    if not ALERTS_ENABLED:
        return False

    if not _cooldown_passed(key):
        logger.info(f"🔕 Алерт [{key}] подавлен троттлингом (не чаще раза в {ALERT_COOLDOWN_MINUTES} мин)")
        return False

    icon = "💳" if kind == KIND_QUOTA else "🔑"
    action = (
        "Пополните баланс в биллинге провайдера."
        if kind == KIND_QUOTA
        else "Обновите ключ в переменных Railway."
    )

    text = (
        f"{icon} <b>СБОЙ ТРАНСКРИБАЦИИ</b>\n\n"
        f"<b>Провайдер:</b> {provider}\n"
        f"<b>Причина:</b> {human}\n"
    )
    if lead_id:
        text += f"<b>Первая пострадавшая сделка:</b> #{lead_id}\n"
    text += (
        f"\n⚠️ Звонки не расшифровываются и не попадают в CRM.\n"
        f"👉 {action}\n\n"
        f"<i>Повтор этого алерта — не раньше чем через {ALERT_COOLDOWN_MINUTES} мин.</i>"
    )

    logger.error(f"🚨 Инфраструктурный сбой [{key}] — шлём алерт в Telegram")
    return await telegram_service.send_message(text)


async def notify_recovered(provider: str = "OpenAI") -> bool:
    """
    Уведомляет, что провайдер снова работает — но только если до этого
    был зафиксирован активный сбой. Вызывается на успешном пути, поэтому
    обязана быть дешёвой и молчаливой в обычной ситуации.
    """
    keys = [k for k in _active if k.startswith(f"{provider}:")]
    if not keys:
        return False

    for k in keys:
        _active.discard(k)
        _last_sent.pop(k, None)

    if not ALERTS_ENABLED:
        return False

    logger.info(f"✅ {provider} снова доступен — шлём уведомление о восстановлении")
    return await telegram_service.send_message(
        f"✅ <b>Транскрибация восстановлена</b>\n\n"
        f"<b>Провайдер:</b> {provider}\n"
        f"Звонки снова расшифровываются и попадают в CRM.\n\n"
        f"<i>Звонки, пропущенные за время сбоя, автоматически не переобрабатываются.</i>"
    )


def reset_state() -> None:
    """Сбрасывает троттлинг и активные сбои (для тестов)."""
    _last_sent.clear()
    _active.clear()
