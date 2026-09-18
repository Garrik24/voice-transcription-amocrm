"""
Отсев примечаний, которые не звонки, до запроса в amoCRM.

18.09.2026 amoCRM подтянул старую переписку в новые контакты: десятки
примечаний-писем (тип 15) разом. Сервис запрашивал каждое, упёрся в лимит,
и amoCRM заблокировал его IP — расшифровка звонков встала. Здесь закреплено:
звонки (10, 11) проходят, всё остальное отсекается без запроса, а если тип не
пришёл — проверяем как раньше, чтобы не потерять звонок.
"""
import os
import unittest

os.environ.setdefault("AMOCRM_ACCESS_TOKEN", "test")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from main import is_call_note_type  # noqa: E402


class TestCallNoteType(unittest.TestCase):
    def test_calls_pass(self):
        for value in ("10", "11", 10, 11, "call_in", "call_out", " 10 "):
            with self.subTest(value=value):
                self.assertTrue(is_call_note_type(value))

    def test_mail_and_common_are_skipped(self):
        for value in ("15", 15, "4", "25", "amomail_message", "common"):
            with self.subTest(value=value):
                self.assertFalse(is_call_note_type(value))

    def test_unknown_type_still_checked(self):
        """Без типа в webhook — проверяем по-старому: лишний запрос лучше потерянного звонка."""
        self.assertTrue(is_call_note_type(None))
        self.assertTrue(is_call_note_type(""))


if __name__ == "__main__":
    unittest.main()
