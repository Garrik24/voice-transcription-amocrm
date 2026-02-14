import unittest
from types import SimpleNamespace


class TestAnalysisParsing(unittest.TestCase):
    def test_normalize_list_field_from_list(self):
        from services.analysis import _normalize_list_field

        self.assertEqual(_normalize_list_field([" a ", "b", "", "  "]), ["a", "b"])

    def test_normalize_list_field_from_string(self):
        from services.analysis import _normalize_list_field

        s = "- один\n- два\n\n• три\n1) четыре\n2. пять"
        self.assertEqual(_normalize_list_field(s), ["один", "два", "три", "четыре", "пять"])

    def test_normalize_list_field_from_none(self):
        from services.analysis import _normalize_list_field

        self.assertEqual(_normalize_list_field(None), [])


class TestNoteFormatting(unittest.TestCase):
    def test_format_note_includes_sections(self):
        from services.analysis import CallAnalysis, AnalysisService

        analysis = CallAnalysis(
            client_name="Клиент",
            manager_name="Менеджер",
            summary="Обсудили задачу. Договорились о следующем.",
            client_city="Не указано",
            work_type="Консультация",
            cost="Не обсуждали",
            payment_terms="Не обсуждали",
            call_result="Перезвонить",
            next_contact_date="Не указано",
            next_steps=["Отправить КП", "Договориться о времени выезда"],
        )

        service = AnalysisService()
        note = service.format_note(analysis)

        self.assertIn("🎙️ АНАЛИЗ ЗВОНКА", note)
        self.assertIn("Суть:", note)
        self.assertIn("📍 Город:", note)
        self.assertIn("📊 Итог:", note)
        self.assertIn("✅ Следующие шаги:", note)

    def test_format_note_includes_participant_count_only(self):
        from services.analysis import CallAnalysis, AnalysisService, SpeakerStats, SpeakerMetrics

        analysis = CallAnalysis(
            client_name="Клиент",
            manager_name="Менеджер",
            summary="Короткая проверка.",
            client_city="Не указано",
            work_type="Консультация",
            cost="Не обсуждали",
            payment_terms="Не обсуждали",
            call_result="Не определено",
            next_contact_date="Не указано",
            next_steps=[],
            speaker_stats=SpeakerStats(
                participant_count=2,
                total_speech_seconds=85.0,
                dominant_speaker="A",
                suspicious_diarization=False,
                suspicious_reason="",
                speakers=[
                    SpeakerMetrics(label="A", duration_seconds=50.0, share_percent=58.8),
                    SpeakerMetrics(label="B", duration_seconds=35.0, share_percent=41.2),
                ],
            ),
        )

        note = AnalysisService().format_note(analysis)
        self.assertIn("👥 Участники: 2", note)
        self.assertNotIn("evidence", note.lower())
        self.assertNotIn("status", note.lower())


class TestV2Helpers(unittest.TestCase):
    def test_build_speaker_stats(self):
        from services.analysis import AnalysisService

        speakers = [
            SimpleNamespace(label="A", start_ms=0, end_ms=30000),
            SimpleNamespace(label="B", start_ms=30000, end_ms=50000),
            SimpleNamespace(label="A", start_ms=50000, end_ms=80000),
        ]
        stats = AnalysisService()._build_speaker_stats(speakers)

        self.assertEqual(stats.participant_count, 2)
        self.assertAlmostEqual(stats.total_speech_seconds, 80.0, places=1)
        self.assertEqual(stats.dominant_speaker, "A")
        self.assertFalse(stats.suspicious_diarization)
        self.assertEqual(len(stats.speakers), 2)

    def test_apply_verification_degrades_unconfirmed_fields(self):
        from services.analysis import AnalysisService, CallAnalysis, FieldVerification

        service = AnalysisService()
        analysis = CallAnalysis(
            client_name="Иван",
            manager_name="Елена",
            summary="Разговор",
            client_city="Краснодар",
            work_type="Топосъемка",
            cost="40 000 ₽",
            payment_terms="50% предоплата",
            call_result="Договорились",
            next_contact_date="Пятница",
            next_steps=["Отправить КП"],
        )

        checks = {
            "cost": FieldVerification("cost", "contradicted", 0.2, "Не обсуждали", ["в разговоре нет суммы"]),
            "next_steps": FieldVerification("next_steps", "unsure", 0.3, [], ["задачи не названы явно"]),
        }
        updated = service._apply_verification(analysis, checks)

        self.assertEqual(updated.cost, "Не обсуждали")
        self.assertEqual(updated.next_steps, [])

if __name__ == "__main__":
    unittest.main()

