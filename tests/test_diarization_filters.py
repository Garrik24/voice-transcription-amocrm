"""
Тесты фильтрации галлюцинаций, дедупликации и доставки имени в контакт.
Кейсы взяты из реального звонка (сделка 34810833), на котором прод
схлопнул все 145 реплик в одну роль.
"""
import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import main
from services.transcription import TranscriptionService, SPEECH_GATE_DB


svc = TranscriptionService()


class TestTextsSimilar(unittest.TestCase):
    def test_short_exact_match(self):
        self.assertTrue(svc._texts_similar("Да.", "да"))

    def test_short_different_not_similar(self):
        self.assertFalse(svc._texts_similar("Да.", "Нда."))

    def test_echo_containment(self):
        # Эхо — обрезок исходной фразы (реальный кейс «20 соток»)
        self.assertTrue(svc._texts_similar("20 соток.", "Это 20 соток."))

    def test_address_containment(self):
        self.assertTrue(svc._texts_similar(
            "Советская улица...", "Станица Советская. Улица дом 63."
        ))

    def test_long_similar(self):
        self.assertTrue(svc._texts_similar(
            "Мы можем, но это будет стоить за 20-30 тысяч",
            "Мы можем, но это будет стоить там за 20-30 тысяч",
        ))

    def test_unrelated_not_similar(self):
        self.assertFalse(svc._texts_similar(
            "Этот земельный участок какой площади?",
            "Граница от, допустим, дома",
        ))


class TestRepetitive(unittest.TestCase):
    def test_whisper_loop(self):
        self.assertTrue(svc._is_repetitive("Да, да, да, да, да, да, да, да, да"))

    def test_normal_speech(self):
        self.assertFalse(svc._is_repetitive("Участок двадцать соток, жилой дом, межевание проведено"))


class TestSoundTag(unittest.TestCase):
    def test_sound_events_detected(self):
        self.assertTrue(svc._is_sound_tag("ТЕЛЕФОННЫЙ ЗВОНОК"))
        self.assertTrue(svc._is_sound_tag("ТРЕВОЖНАЯ МУЗЫКА"))

    def test_abbreviations_kept(self):
        self.assertFalse(svc._is_sound_tag("ЗОИТ"))
        self.assertFalse(svc._is_sound_tag("ЕГРН"))

    def test_real_speech_kept(self):
        self.assertFalse(svc._is_sound_tag("Это 20 соток."))
        self.assertFalse(svc._is_sound_tag("Управление Росреестра, Александр"))


class TestSegmentEnergy(unittest.TestCase):
    def test_p95_ignores_single_click(self):
        # 20 окон тишины + одиночный щелчок: p95 не должен на него купиться
        profile = [(i * 0.1, -168.0) for i in range(20)]
        profile[10] = (1.0, -20.0)
        self.assertLess(svc._segment_energy(profile, 0.0, 2.0), SPEECH_GATE_DB)

    def test_real_speech_passes(self):
        profile = [(i * 0.1, -35.0) for i in range(20)]
        self.assertGreater(svc._segment_energy(profile, 0.0, 2.0), SPEECH_GATE_DB)

    def test_empty_window_is_silence(self):
        self.assertEqual(svc._segment_energy([], 0.0, 1.0), -168.0)


class TestFilterChannelSegments(unittest.TestCase):
    def test_drops_hallucination_on_silence(self):
        silence = [(i * 0.1, -168.0) for i in range(100)]
        segs = [{"text": "ТЕЛЕФОННЫЙ ЗВОНОК", "start": 1.0, "end": 3.0,
                 "no_speech_prob": 0.9, "avg_logprob": -1.2, "compression_ratio": 1.0}]
        self.assertEqual(svc._filter_channel_segments(segs, silence, "тест"), [])

    def test_keeps_real_backchannel(self):
        speech = [(i * 0.1, -45.0) for i in range(100)]
        segs = [{"text": "Угу.", "start": 1.0, "end": 1.5,
                 "no_speech_prob": 0.2, "avg_logprob": -0.3, "compression_ratio": 1.0}]
        self.assertEqual(len(svc._filter_channel_segments(segs, speech, "тест")), 1)


class TestScoreChannels(unittest.TestCase):
    def test_detects_inverted_channels(self):
        # Реальный кейс: менеджер в ПРАВОМ канале (код раньше жёстко считал левый)
        left = [{"text": "Я ищу в интернете геодезию", "start": 0, "end": 2},
                {"text": "Мне просто отстрелять надо участок", "start": 3, "end": 5},
                {"text": "Сколько стоит у вас?", "start": 6, "end": 8}]
        right = [{"text": "Мы занимаемся межеванием, будет стоить порядка 30 тысяч рублей", "start": 0, "end": 3},
                 {"text": "Этот земельный участок какой площади?", "start": 4, "end": 6},
                 {"text": "Скажите адрес, посмотрю по базе", "start": 7, "end": 9}]
        decision = svc._score_channels(left, right)
        self.assertEqual(decision.manager_channel, "right")
        self.assertFalse(decision.uncertain)

    def test_uncertain_when_no_markers(self):
        left = [{"text": "Алло", "start": 0, "end": 1}]
        right = [{"text": "Да", "start": 0, "end": 1}]
        self.assertIsNone(svc._score_channels(left, right).manager_channel)


class TestCrossChannelDedup(unittest.TestCase):
    def test_real_repeat_at_different_times_kept(self):
        # Переспрос: клиент сказал «Это 20 соток», менеджер следом повторил
        # «20 соток». Таймфреймы НЕ пересекаются → это не эхо, оба остаются.
        prof = [(i * 0.1, -30.0) for i in range(1200)]
        left = [{"text": "Это 20 соток.", "start": 108.0, "end": 109.5, "no_speech_prob": 0.19}]
        right = [{"text": "20 соток.", "start": 107.0, "end": 108.0, "no_speech_prob": 0.0}]
        l2, r2 = svc._dedupe_cross_channel(left, right, prof, prof)
        self.assertEqual((len(l2), len(r2)), (1, 1))

    def test_simultaneous_echo_dropped_by_energy(self):
        # Эхо: одновременные таймфреймы, источник громче на >12 dB → эхо удаляется
        prof_loud = [(i * 0.1, -30.0) for i in range(200)]
        prof_quiet = [(i * 0.1, -50.0) for i in range(200)]
        left = [{"text": "Скажите адрес, посмотрю по базе", "start": 10.0, "end": 12.0, "no_speech_prob": 0.1}]
        right = [{"text": "Скажите адрес, посмотрю по базе", "start": 10.2, "end": 12.1, "no_speech_prob": 0.5}]
        l2, r2 = svc._dedupe_cross_channel(left, right, prof_loud, prof_quiet)
        self.assertEqual((len(l2), len(r2)), (1, 0))


class TestContactNameHelpers(unittest.TestCase):
    def test_clean_strips_organization(self):
        self.assertEqual(
            main._clean_client_name("Александр (Управление Росреестра)"), "Александр"
        )

    def test_clean_rejects_stopwords(self):
        self.assertIsNone(main._clean_client_name("Не представился"))
        self.assertIsNone(main._clean_client_name("Клиент"))
        self.assertIsNone(main._clean_client_name(""))
        self.assertIsNone(main._clean_client_name(None))

    def test_default_amocrm_autoname(self):
        # Реальное автоимя amoCRM — раньше не распознавалось и блокировало запись
        self.assertTrue(main._is_default_contact_name(
            "Входящий +78652951729 (9280058522 - Менеджер)"
        ))

    def test_default_phone_number(self):
        self.assertTrue(main._is_default_contact_name("+7 865 295-17-29"))
        self.assertTrue(main._is_default_contact_name(""))

    def test_manual_name_not_default(self):
        self.assertFalse(main._is_default_contact_name("Иванов Сергей"))
        self.assertFalse(main._is_default_contact_name("РОСРЕЕСТР"))


if __name__ == "__main__":
    unittest.main()
