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


# Признаки «кончились деньги». Провайдеры меняют формулировки и коды на ходу:
# 28.08.2026 OpenAI переименовал code 'insufficient_quota' → 'credit_balance_exhausted',
# и привязка к одному коду обезоружила сразу и алерты, и фолбэк на резервный STT.
# Поэтому смотрим три независимых сигнала: code, type и текст сообщения.
QUOTA_CODES = {
    "insufficient_quota",
    "credit_balance_exhausted",
    "billing_hard_limit_reached",
    "billing_not_active",
}
QUOTA_HINTS = (
    "no credits remaining",
    "credit balance",
    "exceeded your current quota",
    "billing details",
    "billing hard limit",
    "quota",
)
AUTH_CODES = {"invalid_api_key", "account_deactivated", "invalid_authentication"}


def _is_quota(code: Optional[str], type_: Optional[str], message: str) -> bool:
    """Кончились деньги? Любого из сигналов достаточно."""
    if code in QUOTA_CODES or type_ in QUOTA_CODES:
        return True
    return any(hint in message for hint in QUOTA_HINTS)


def classify(exc: BaseException) -> Optional[Tuple[str, str, str]]:
    """
    Определяет, является ли ошибка инфраструктурной.

    Returns:
        (kind, provider, human_text) либо None — если ошибка обычная
        (сеть, битое аудио, таймаут, обычный rate limit) и алерт не нужен.
    """
    for lib, provider, quota_text, auth_text in (
        (
            openai,
            "OpenAI",
            "Закончился баланс OpenAI — Whisper не расшифровывает звонки.",
            "Ключ OpenAI недействителен или отозван.",
        ),
        (
            anthropic,
            "Anthropic",
            "Закончился баланс Anthropic — анализ звонков не выполняется.",
            "Ключ Anthropic недействителен или отозван.",
        ),
    ):
        if not isinstance(exc, lib.APIStatusError):
            continue

        status = getattr(exc, "status_code", None)
        code = getattr(exc, "code", None)
        type_ = getattr(exc, "type", None)
        message = (getattr(exc, "message", "") or str(exc)).lower()

        if status == 401 or code in AUTH_CODES:
            return (KIND_AUTH, provider, auth_text)
        if _is_quota(code, type_, message):
            return (KIND_QUOTA, provider, quota_text)
        # Прочие 429 — обычный rate limit, SDK ретраит сам. Не алертим.

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
