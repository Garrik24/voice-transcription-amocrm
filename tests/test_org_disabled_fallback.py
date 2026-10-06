import json
import unittest
from unittest.mock import AsyncMock, patch

import anthropic
import httpx

ORG_DISABLED = (
    "This organization has been disabled. An organization admin can appeal at "
    "https://console.anthropic.com/appeal"
)


def anthropic_error(status: int, message: str, error_code: str | None = None) -> anthropic.APIStatusError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = {"type": "invalid_request_error", "message": message}
    if error_code:
        err["details"] = {"error_code": error_code}
    resp = httpx.Response(status, request=req, json={"type": "error", "error": err})
    return anthropic.APIStatusError(message, response=resp, body={"type": "error", "error": err})


class OrgDisabledClassification(unittest.TestCase):
    def test_org_disabled_is_provider_failure(self):
        from services import alerts

        verdict = alerts.classify(anthropic_error(400, ORG_DISABLED, "organization_on_hold"))
        self.assertIsNotNone(verdict)
        kind, provider, text = verdict
        self.assertEqual(kind, alerts.KIND_AUTH)
        self.assertEqual(provider, "Anthropic")
        self.assertIn("отключена", text)

    def test_ordinary_bad_request_is_not(self):
        from services import alerts

        self.assertIsNone(alerts.classify(anthropic_error(400, "max_tokens: must be greater than 0")))

    def test_llm_infra_failure_recognizes_org_disabled(self):
        from services.analysis import AnalysisService

        self.assertTrue(AnalysisService._is_llm_infra_failure(anthropic_error(400, ORG_DISABLED, "organization_on_hold")))
        self.assertFalse(AnalysisService._is_llm_infra_failure(anthropic_error(400, "max_tokens: bad")))


class ChainFallsThroughToGemini(unittest.IsolatedAsyncioTestCase):
    async def test_anthropic_and_assemblyai_down_gemini_answers(self):
        from services import analysis as mod

        svc = mod.AnalysisService()
        svc._llm_mark_down = AsyncMock()
        svc._llm_recovered = AsyncMock()
        svc._call_anthropic = AsyncMock(side_effect=anthropic_error(400, ORG_DISABLED, "organization_on_hold"))
        svc._call_assemblyai_llm = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "401", request=httpx.Request("POST", "https://x"), response=httpx.Response(401)
            )
        )
        svc._call_gemini_llm = AsyncMock(return_value='{"ok": true}')
        svc._call_openai_llm = AsyncMock(side_effect=AssertionError("до OpenAI дойти не должно"))

        with patch.object(mod, "LLM_CHAIN", ["anthropic", "assemblyai", "gemini", "openai"]), patch.object(
            mod, "LLM_FALLBACK_ENABLED", True
        ):
            self.assertEqual(await svc._call_llm("s", "u"), '{"ok": true}')
            note = svc.format_note(
                mod.CallAnalysis(
                    client_name="К", manager_name="М", summary="с", client_city="-", location="-",
                    work_type="-", cost="-", payment_terms="-", call_result="-",
                    next_contact_date="-", next_steps=[],
                )
            )
        self.assertIn(f"[gemini/{mod.GEMINI_LLM_MODEL} |", note)  # в заметке реальный провайдер
        svc._call_openai_llm.assert_not_called()


class GeminiCall(unittest.IsolatedAsyncioTestCase):
    def _client(self, payload, status=200):
        response = httpx.Response(status, json=payload, request=httpx.Request("POST", "https://g"))
        client = AsyncMock()
        client.post = AsyncMock(return_value=response)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        return client

    async def test_returns_text_without_thoughts_and_sends_expected_body(self):
        from services import analysis as mod

        payload = {
            "candidates": [{"finishReason": "STOP", "content": {"parts": [
                {"text": "скрытое размышление", "thought": True}, {"text": '{"a": 1}'}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7},
        }
        client = self._client(payload)
        with patch.object(mod, "GEMINI_API_KEY", "key"), patch.object(mod.httpx, "AsyncClient", return_value=client):
            text = await mod.AnalysisService()._call_gemini_llm("sys", "usr", 1500)
        self.assertEqual(text, '{"a": 1}')

        kwargs = client.post.call_args.kwargs
        body = kwargs["json"]
        self.assertEqual(body["system_instruction"]["parts"][0]["text"], "sys")
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], 5500)  # запас под размышления
        self.assertEqual(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "low")
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "key")

    async def test_truncated_answer_is_error(self):
        from services import analysis as mod

        payload = {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "обрезок"}]}}]}
        with patch.object(mod, "GEMINI_API_KEY", "key"), patch.object(mod.httpx, "AsyncClient", return_value=self._client(payload)):
            with self.assertRaisesRegex(ValueError, "обрезан"):
                await mod.AnalysisService()._call_gemini_llm("s", "u", 100)

    async def test_no_key_is_error(self):
        from services import analysis as mod

        with patch.object(mod, "GEMINI_API_KEY", ""):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                await mod.AnalysisService()._call_gemini_llm("s", "u", 100)


if __name__ == "__main__":
    unittest.main()


class QuietFallbackNotifications(unittest.IsolatedAsyncioTestCase):
    async def test_switch_message_not_sent_by_default(self):
        from services import analysis as mod

        svc = mod.AnalysisService()
        with patch.object(mod, "FALLBACK_NOTIFICATIONS", False), patch.object(
            mod.telegram_service, "send_message", new=AsyncMock()
        ) as send:
            await svc._llm_mark_down("anthropic", RuntimeError("down"))
            await svc._llm_recovered("anthropic")
        send.assert_not_called()

    async def test_switch_message_sent_when_enabled(self):
        from services import analysis as mod

        svc = mod.AnalysisService()
        with patch.object(mod, "FALLBACK_NOTIFICATIONS", True), patch.object(
            mod.telegram_service, "send_message", new=AsyncMock()
        ) as send:
            await svc._llm_mark_down("anthropic", RuntimeError("down"))
        send.assert_called_once()

    async def test_stt_switch_message_not_sent_by_default(self):
        from services import transcription as mod

        svc = mod.TranscriptionService()
        with patch.object(mod, "FALLBACK_NOTIFICATIONS", False), patch.object(
            mod.telegram_service, "send_message", new=AsyncMock()
        ) as send:
            await svc._activate_fallback(RuntimeError("401"))
            await svc._deactivate_fallback()
        send.assert_not_called()


class EngineNote(unittest.IsolatedAsyncioTestCase):
    async def test_note_names_actual_providers(self):
        from services import analysis as mod

        svc = mod.AnalysisService()
        svc._llm_mark_down = AsyncMock()
        svc._llm_recovered = AsyncMock()
        svc._call_anthropic = AsyncMock(side_effect=anthropic_error(400, ORG_DISABLED, "organization_on_hold"))
        svc._call_assemblyai_llm = AsyncMock(return_value="ok")
        with patch.object(mod, "LLM_CHAIN", ["anthropic", "assemblyai"]), patch.object(mod, "LLM_FALLBACK_ENABLED", True):
            await svc._call_llm("s", "u")
            self.assertEqual(svc.engine_note("assemblyai"), "анализ: AssemblyAI · речь: AssemblyAI")
            self.assertEqual(svc.engine_note("whisper"), "анализ: AssemblyAI · речь: Whisper")

    async def test_telegram_summary_header_contains_note(self):
        from services.telegram import TelegramService

        tg = TelegramService()
        tg.send_message = AsyncMock(return_value=True)
        await tg.send_call_analysis(
            call_datetime="06.10.2026 10:00", call_type="incoming", phone="+7", manager_name="М",
            client_name="К", summary="с", amocrm_url="https://x", engine_note="анализ: AssemblyAI · речь: AssemblyAI",
        )
        text = tg.send_message.call_args[0][0]
        self.assertIn("АНАЛИЗ ЗВОНКА</b> <i>(анализ: AssemblyAI · речь: AssemblyAI)</i>", text)

    async def test_telegram_summary_without_note_unchanged(self):
        from services.telegram import TelegramService

        tg = TelegramService()
        tg.send_message = AsyncMock(return_value=True)
        await tg.send_call_analysis(
            call_datetime="d", call_type="incoming", phone="+7", manager_name="М",
            client_name="К", summary="с", amocrm_url="https://x",
        )
        self.assertTrue(tg.send_message.call_args[0][0].startswith("📊 <b>АНАЛИЗ ЗВОНКА</b>\n"))
