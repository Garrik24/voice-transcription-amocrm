"""
Сервис уведомлений через Telegram.
Отправляет уведомления об ошибках и статусах обработки.
"""
import httpx
import logging
from typing import Optional, List
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


class TelegramService:
    """Сервис отправки уведомлений в Telegram"""
    
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    @property
    def is_configured(self) -> bool:
        """Проверяет, настроен ли Telegram"""
        return bool(self.bot_token and self.chat_id)
    
    async def send_message(
        self, 
        text: str, 
        parse_mode: str = "HTML",
        disable_notification: bool = False
    ) -> bool:
        """
        Отправляет сообщение в Telegram.
        
        Args:
            text: Текст сообщения
            parse_mode: Режим парсинга (HTML, Markdown)
            disable_notification: Отключить звук уведомления
            
        Returns:
            True если успешно
        """
        if not self.is_configured:
            logger.warning("Telegram не настроен, пропускаем отправку")
            return False
        
        try:
            async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_notification": disable_notification
                    }
                )
                response.raise_for_status()
                logger.info("Сообщение отправлено в Telegram")
                return True
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Ошибка отправки в Telegram: HTTP {e.response.status_code} — {e.response.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"Ошибка отправки в Telegram: {e}")
            return False
    
    async def send_error(
        self, 
        error_type: str,
        error_message: str,
        lead_id: Optional[int] = None,
        details: Optional[str] = None
    ) -> bool:
        """
        Отправляет уведомление об ошибке.
        
        Args:
            error_type: Тип ошибки
            error_message: Текст ошибки
            lead_id: ID сделки (если есть)
            details: Дополнительные детали
            
        Returns:
            True если успешно
        """
        text = f"""🚨 <b>ОШИБКА ТРАНСКРИБАЦИИ</b>

<b>Тип:</b> {error_type}
<b>Ошибка:</b> {error_message}"""
        
        if lead_id:
            text += f"\n<b>Сделка:</b> #{lead_id}"
        
        if details:
            text += f"\n\n<b>Детали:</b>\n<code>{details[:500]}</code>"
        
        return await self.send_message(text)
    
    async def send_call_analysis(
        self,
        call_datetime: str,
        call_type: str,
        phone: str,
        manager_name: str,
        client_name: str,
        summary: str,
        amocrm_url: str,
        record_url: str = "",
        client_city: str = "Не указано",
        work_type: str = "Консультация",
        cost: str = "Не обсуждали",
        payment_terms: str = "Не обсуждали",
        call_result: str = "Не определено",
        next_contact_date: str = "Не указано",
        next_steps: Optional[List[str]] = None,
    ) -> bool:
        """
        Отправляет красивый анализ звонка в Telegram.
        Формат как в Make.com автоматизации.
        """
        call_type_str = "Входящий" if call_type == "incoming" else "Исходящий"

        steps_block = ""
        if next_steps:
            steps_block = "\n\n✅ <b>Следующие шаги:</b>\n" + "\n".join([f"- {s}" for s in next_steps])
        
        text = f"""📊 <b>АНАЛИЗ ЗВОНКА</b>

📅 {call_datetime}
📞 {call_type_str}
☎️ Тел: {phone}

<b>Спикеры:</b>
- {manager_name} (менеджер)
- {client_name} (клиент)

<b>Суть:</b>
{summary}

<b>Факты:</b>
📍 {client_city}
🔧 {work_type}
💰 {cost}
💳 {payment_terms}
📊 {call_result}
📅 {next_contact_date}{steps_block}

🔗 <a href="{amocrm_url}">AmoCRM</a>"""
        
        if record_url:
            text += f"\n🎧 <a href=\"{record_url}\">Запись звонка</a>"
        
        return await self.send_message(text, disable_notification=False)
    
    async def send_success(
        self,
        lead_id: int,
        client_name: str,
        call_result: str,
        duration_seconds: float
    ) -> bool:
        """Простое уведомление об успехе (для обратной совместимости)"""
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        
        text = f"""✅ <b>Звонок обработан</b>

<b>Сделка:</b> #{lead_id}
<b>Клиент:</b> {client_name}
<b>Длительность:</b> {minutes}:{seconds:02d}
<b>Итог:</b> {call_result}"""
        
        return await self.send_message(text, disable_notification=True)
    
    async def send_startup(self) -> bool:
        """Отправляет уведомление о запуске сервера"""
        text = """🟢 <b>Сервер транскрибации запущен</b>

Готов принимать webhook от AmoCRM."""
        
        return await self.send_message(text)
    
    async def send_shutdown(self, reason: str = "Плановая остановка") -> bool:
        """Отправляет уведомление об остановке сервера"""
        text = f"""🔴 <b>Сервер транскрибации остановлен</b>

<b>Причина:</b> {reason}"""
        
        return await self.send_message(text)


# Синглтон
telegram_service = TelegramService()
