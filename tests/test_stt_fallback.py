"""
Автопереключение STT: Whisper → AssemblyAI при инфраструктурном сбое.

Ключевое требование: обычные ошибки (сеть, битое аудио) НЕ должны уводить
конвейер на резервный провайдер — переключаемся только когда основной
не восстановится сам (кончился баланс, отозван ключ).
"""
import os
import time
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import httpx
import openai

from services.transcription import TranscriptionService

WHISPER_OUT = ("текст от whisper", [{"text": "текст от whisper", "start": 0.0, "end": 1.0}])
AAI_OUT = ("текст от assemblyai", [{"text": "текст от assemblyai", "start": 0.0, "end": 1.0}])


def _quota_error() -> openai.APIStatusError:
    """Ошибка «кончился баланс», как её отдаёт OpenAI SDK."""
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    response = httpx.Response(429, request=request, json={
        "error": {"message": "You exceeded your current quota",
                  "type": "insufficient_quota", "code": "insufficient_quota"}
    })
    return openai.RateLimitError(
        "quota", response=response,
        body={"code": "insufficient_quota", "message": "You exceeded your current quota"},
    )


class TestSttFallback(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.svc = TranscriptionService()
        # На проде ключ приходит из Railway; в тестах подставляем свой,
        # иначе фолбэк корректно не активируется (нечем подменять Whisper).
        self._key_patch = patch("services.transcription.ASSEMBLYAI_API_KEY", "test-aai-key")
        self._key_patch.start()
        self.addCleanup(self._key_patch.stop)

    async def test_normal_path_uses_whisper(self):
        with patch.object(self.svc, "_whisper_with_segments", new=AsyncMock(return_value=WHISPER_OUT)), \
             patch.object(self.svc, "_assemblyai_with_segments", new=AsyncMock()) as aai:
            result = await self.svc._stt_with_segments(b"audio", "left")
        self.assertEqual(result, WHISPER_OUT)
        aai.assert_not_awaited()
        self.assertFalse(self.svc._fallback_active)

    async def test_quota_error_switches_to_assemblyai(self):
        with patch.object(self.svc, "_whisper_with_segments", new=AsyncMock(side_effect=_quota_error())), \
             patch.object(self.svc, "_assemblyai_with_segments", new=AsyncMock(return_value=AAI_OUT)), \
             patch("services.telegram.telegram_service.send_message", new=AsyncMock()) as tg:
            result = await self.svc._stt_with_segments(b"audio", "left")
        self.assertEqual(result, AAI_OUT)
        self.assertTrue(self.svc._fallback_active)
        tg.assert_awaited_once()  # владелец уведомлён о работе на резерве

    async def test_ordinary_error_does_not_switch(self):
        # Битое аудио / сеть — резерв не поможет, ошибка должна всплыть наверх
        with patch.object(self.svc, "_whisper_with_segments", new=AsyncMock(side_effect=ValueError("битый файл"))), \
             patch.object(self.svc, "_assemblyai_with_segments", new=AsyncMock()) as aai:
            with self.assertRaises(ValueError):
                await self.svc._stt_with_segments(b"audio", "left")
        aai.assert_not_awaited()
        self.assertFalse(self.svc._fallback_active)

    async def test_second_call_goes_straight_to_fallback(self):
        # Пока резерв активен, основной провайдер не дёргаем вообще
        self.svc._fallback_until = time.time() + 600
        with patch.object(self.svc, "_whisper_with_segments", new=AsyncMock()) as whisper, \
             patch.object(self.svc, "_assemblyai_with_segments", new=AsyncMock(return_value=AAI_OUT)):
            result = await self.svc._stt_with_segments(b"audio", "right")
        self.assertEqual(result, AAI_OUT)
        whisper.assert_not_awaited()

    async def test_returns_to_whisper_after_window(self):
        # Окно резерва истекло — пробуем основной снова и возвращаемся на него
        self.svc._fallback_until = time.time() - 1
        with patch.object(self.svc, "_whisper_with_segments", new=AsyncMock(return_value=WHISPER_OUT)), \
             patch.object(self.svc, "_assemblyai_with_segments", new=AsyncMock()) as aai:
            result = await self.svc._stt_with_segments(b"audio", "left")
        self.assertEqual(result, WHISPER_OUT)
        aai.assert_not_awaited()
        self.assertEqual(self.svc._fallback_until, 0.0)

    async def test_single_alert_for_both_channels(self):
        # Стерео: два канала подряд — уведомление уходит один раз
        with patch.object(self.svc, "_whisper_with_segments", new=AsyncMock(side_effect=_quota_error())), \
             patch.object(self.svc, "_assemblyai_with_segments", new=AsyncMock(return_value=AAI_OUT)), \
             patch("services.telegram.telegram_service.send_message", new=AsyncMock()) as tg:
            await self.svc._stt_with_segments(b"audio", "left")
            await self.svc._stt_with_segments(b"audio", "right")
        tg.assert_awaited_once()

    async def test_infra_failure_detection(self):
        self.assertTrue(self.svc._is_infra_failure(_quota_error()))
        self.assertFalse(self.svc._is_infra_failure(ValueError("битый файл")))
        self.assertFalse(self.svc._is_infra_failure(TimeoutError()))

    async def test_no_key_means_no_fallback(self):
        # Без ключа AssemblyAI переключаться некуда — ошибка должна всплыть,
        # а не превратиться в тихий отказ транскрибации
        with patch("services.transcription.ASSEMBLYAI_API_KEY", None), \
             patch.object(self.svc, "_whisper_with_segments", new=AsyncMock(side_effect=_quota_error())), \
             patch.object(self.svc, "_assemblyai_with_segments", new=AsyncMock()) as aai:
            with self.assertRaises(openai.RateLimitError):
                await self.svc._stt_with_segments(b"audio", "left")
        aai.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
