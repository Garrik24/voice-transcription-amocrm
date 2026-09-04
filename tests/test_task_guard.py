"""
Дедупликация автозадач amoCRM.

04.09.2026 в CRM висело 200 открытых задач, 177 просроченных, 197 создала автоматика —
одна сделка тащила 15 штук. Здесь закреплены три правила, которые это чинят:
на сделке одна открытая задача, задача по звонку только при договорённости,
срок не падает в выходные.
"""
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import httpx

import main
from services.amocrm import AmoCRMService
from services.analysis import AnalysisService, DEFAULT_TASK_DUE_HOURS, _parse_task_decision

MSK = timezone(timedelta(hours=3))


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self.text = json.dumps(self._payload, ensure_ascii=False)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("GET", "https://test.amocrm.ru"),
                response=self,
            )


class FakeAmo:
    """Подменяет httpx.AsyncClient: пишет все запросы и отдаёт заданные ответы."""

    def __init__(self, open_tasks=None, created_id=999, tasks_status=200):
        self.open_tasks = list(open_tasks or [])
        self.created_id = created_id
        self.tasks_status = tasks_status
        self.requests = []

    # httpx.AsyncClient(...) → сам себе контекст-менеджер
    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kwargs):
        self.requests.append(("GET", url, kwargs))
        assert url.endswith("/tasks"), f"неожиданный GET {url}"
        if self.tasks_status >= 400:
            return FakeResponse(self.tasks_status, {"error": "boom"})
        if not self.open_tasks:
            return FakeResponse(204)  # amoCRM отдаёт 204 на пустую выборку
        return FakeResponse(200, {"_embedded": {"tasks": self.open_tasks}})

    async def post(self, url, **kwargs):
        self.requests.append(("POST", url, kwargs))
        assert url.endswith("/tasks"), f"неожиданный POST {url}"
        return FakeResponse(200, {"_embedded": {"tasks": [{"id": self.created_id}]}})

    async def patch(self, url, **kwargs):
        self.requests.append(("PATCH", url, kwargs))
        return FakeResponse(200, {})

    def by_method(self, method):
        return [(url, kw) for m, url, kw in self.requests if m == method]


AUTO_TEXT = "По итогам звонка (AI): отправить старое КП"
MANUAL_TEXT = "Позвонить Петровичу насчёт бетона"


def _task(task_id, complete_till, text=AUTO_TEXT):
    return {
        "id": task_id,
        "text": text,
        "complete_till": complete_till,
        "responsible_user_id": 111,
    }


class TestEnsureTask(unittest.IsolatedAsyncioTestCase):
    """Правило 1: на сделке ровно одна открытая задача."""

    async def test_no_open_tasks_creates(self):
        fake = FakeAmo(open_tasks=[])
        with patch("services.amocrm.httpx.AsyncClient", fake):
            result = await AmoCRMService().ensure_task(
                lead_id=1, text="Отправить КП по межеванию", complete_till=1_800_000_000,
                responsible_user_id=222,
            )

        self.assertEqual(result, {"action": "created", "task_id": 999})
        self.assertEqual(len(fake.by_method("POST")), 1)
        self.assertEqual(fake.by_method("PATCH"), [])

        payload = fake.by_method("POST")[0][1]["json"][0]
        self.assertEqual(payload["text"], "Отправить КП по межеванию")
        self.assertEqual(payload["entity_id"], 1)
        self.assertEqual(payload["responsible_user_id"], 222)

    async def test_one_open_task_updates_without_creating(self):
        fake = FakeAmo(open_tasks=[_task(10, 1_700_000_000)])
        with patch("services.amocrm.httpx.AsyncClient", fake):
            result = await AmoCRMService().ensure_task(
                lead_id=1, text="Перезвонить по геологии после 15:00",
                complete_till=1_800_000_000, responsible_user_id=222,
            )

        self.assertEqual(result, {"action": "updated", "task_id": 10})
        self.assertEqual(fake.by_method("POST"), [], "вторая задача создаваться не должна")

        patches = fake.by_method("PATCH")
        self.assertEqual(len(patches), 1)
        url, kwargs = patches[0]
        self.assertTrue(url.endswith("/tasks/10"))
        self.assertEqual(kwargs["json"]["text"], "Перезвонить по геологии после 15:00")
        self.assertEqual(kwargs["json"]["complete_till"], 1_800_000_000)

    async def test_three_open_tasks_update_earliest_and_close_dupes(self):
        fake = FakeAmo(open_tasks=[
            _task(30, 1_700_000_300),
            _task(10, 1_700_000_100),  # самая ранняя по сроку — её оставляем
            _task(20, 1_700_000_200),
        ])
        with patch("services.amocrm.httpx.AsyncClient", fake):
            result = await AmoCRMService().ensure_task(
                lead_id=1, text="Выехать на объект в четверг", complete_till=1_800_000_000,
            )

        self.assertEqual(result, {"action": "updated", "task_id": 10})
        self.assertEqual(fake.by_method("POST"), [])

        patches = fake.by_method("PATCH")
        self.assertEqual(len(patches), 3, "1 обновление + 2 закрытия дублей")

        # Каждая задача патчится отдельным запросом — батч по /tasks не используем
        self.assertTrue(patches[0][0].endswith("/tasks/10"))
        self.assertNotIn("is_completed", patches[0][1]["json"])

        closed = {}
        for url, kwargs in patches[1:]:
            closed[url.rsplit("/", 1)[-1]] = kwargs["json"]
        self.assertEqual(set(closed), {"20", "30"})
        for body in closed.values():
            self.assertIs(body["is_completed"], True)
            self.assertIn("дубль", body["result"]["text"])

    async def test_read_failure_does_not_create_task(self):
        """Список задач не прочитался → молчим. Иначе на каждом сбое API — новый дубль."""
        fake = FakeAmo(tasks_status=500)
        with patch("services.amocrm.httpx.AsyncClient", fake):
            result = await AmoCRMService().ensure_task(
                lead_id=1, text="Отправить договор", complete_till=1_800_000_000,
            )

        self.assertIsNone(result)
        self.assertEqual(fake.by_method("POST"), [])
        self.assertEqual(fake.by_method("PATCH"), [])


class TestManualTasksUntouched(unittest.IsolatedAsyncioTestCase):
    """Задачу, заведённую менеджером руками, сервис не переписывает и не закрывает."""

    async def test_manual_task_blocks_everything(self):
        fake = FakeAmo(open_tasks=[_task(10, 1_700_000_000, MANUAL_TEXT)])
        with patch("services.amocrm.httpx.AsyncClient", fake):
            result = await AmoCRMService().ensure_task(
                lead_id=1, text="Отправить КП по межеванию", complete_till=1_800_000_000,
            )

        self.assertEqual(result, {"action": "skipped", "task_id": 10})
        self.assertEqual(fake.by_method("POST"), [], "новая задача создаваться не должна")
        self.assertEqual(fake.by_method("PATCH"), [], "чужую задачу не трогаем")

    async def test_manual_task_protects_auto_dupes_from_closing(self):
        """Смешанный случай: раз есть ручная задача — не трогаем и свои, чтобы не гадать."""
        fake = FakeAmo(open_tasks=[
            _task(10, 1_700_000_100, AUTO_TEXT),
            _task(20, 1_700_000_200, MANUAL_TEXT),
        ])
        with patch("services.amocrm.httpx.AsyncClient", fake):
            result = await AmoCRMService().ensure_task(
                lead_id=1, text="Отправить КП", complete_till=1_800_000_000,
            )

        self.assertEqual(result["action"], "skipped")
        self.assertEqual(fake.by_method("PATCH"), [])
        self.assertEqual(fake.by_method("POST"), [])

    async def test_missed_call_task_counts_as_auto(self):
        missed = "Перезвонить: пропущенный звонок с +79181234567 в 14:05, никто не перезвонил"
        fake = FakeAmo(open_tasks=[_task(10, 1_700_000_000, missed)])
        with patch("services.amocrm.httpx.AsyncClient", fake):
            result = await AmoCRMService().ensure_task(
                lead_id=1, text="Отправить КП", complete_till=1_800_000_000,
            )

        self.assertEqual(result, {"action": "updated", "task_id": 10})
        self.assertEqual(len(fake.by_method("PATCH")), 1)

    async def test_legacy_placeholder_counts_as_auto(self):
        """Заглушки старого кода должны подхватываться, а не считаться ручными."""
        fake = FakeAmo(open_tasks=[
            _task(10, 1_700_000_000, "Связаться с клиентом по итогам звонка (AI)")
        ])
        with patch("services.amocrm.httpx.AsyncClient", fake):
            result = await AmoCRMService().ensure_task(
                lead_id=1, text="Отправить КП", complete_till=1_800_000_000,
            )

        self.assertEqual(result, {"action": "updated", "task_id": 10})


class TestFollowupTaskDecision(unittest.IsolatedAsyncioTestCase):
    """Правило 2: задача по звонку — только при договорённости."""

    @staticmethod
    def _analysis(**kwargs):
        base = dict(
            needs_task=False, task_text="", due_in_hours=DEFAULT_TASK_DUE_HOURS,
            next_contact_date="Не обсуждали", next_steps=[],
        )
        base.update(kwargs)
        return SimpleNamespace(**base)

    async def test_needs_task_false_makes_no_requests(self):
        fake = FakeAmo()
        with patch("services.amocrm.httpx.AsyncClient", fake):
            await main._create_followup_task(1, self._analysis(needs_task=False), 222)
        self.assertEqual(fake.requests, [], "к /tasks не должно быть ни одного запроса")

    async def test_needs_task_true_without_text_makes_no_requests(self):
        fake = FakeAmo()
        with patch("services.amocrm.httpx.AsyncClient", fake):
            await main._create_followup_task(
                1, self._analysis(needs_task=True, task_text="   "), 222
            )
        self.assertEqual(fake.requests, [])

    async def test_needs_task_true_creates_task_with_llm_text(self):
        fake = FakeAmo(open_tasks=[])
        with patch("services.amocrm.httpx.AsyncClient", fake):
            await main._create_followup_task(
                1,
                self._analysis(
                    needs_task=True,
                    task_text="Отправить КП по межеванию",
                    due_in_hours=4,
                    next_contact_date="Не обсуждали",
                ),
                222,
            )

        created = fake.by_method("POST")
        self.assertEqual(len(created), 1)
        payload = created[0][1]["json"][0]
        self.assertIn("Отправить КП по межеванию", payload["text"])
        # Фраза «Не обсуждали» — не договорённость, в текст задачи не подмешивается
        self.assertNotIn("След. контакт", payload["text"])

    async def test_recognized_date_wins_over_due_in_hours(self):
        fake = FakeAmo(open_tasks=[])
        with patch("services.amocrm.httpx.AsyncClient", fake):
            await main._create_followup_task(
                1,
                self._analysis(
                    needs_task=True, task_text="Перезвонить по геологии",
                    due_in_hours=1, next_contact_date="завтра",
                ),
                222,
            )

        payload = fake.by_method("POST")[0][1]["json"][0]
        self.assertIn("След. контакт: завтра", payload["text"])
        due = datetime.fromtimestamp(payload["complete_till"], MSK)
        expected = (datetime.now(MSK) + timedelta(days=1)).date()
        self.assertEqual(due.date(), expected)
        self.assertEqual(due.hour, 10)


class TestParseTaskDecision(unittest.TestCase):
    """Разбор блока решения о задаче — fail-closed на любой кривизне."""

    def test_missing_field_is_undecided_not_false(self):
        """Блока нет → None, чтобы включилось запасное правило по next_steps."""
        self.assertEqual(_parse_task_decision({}), (None, "", DEFAULT_TASK_DUE_HOURS))

    def test_explicit_false_means_no_task(self):
        self.assertEqual(
            _parse_task_decision({"needs_task": False}), (False, "", DEFAULT_TASK_DUE_HOURS)
        )

    def test_true_without_text_means_no_task(self):
        self.assertEqual(
            _parse_task_decision({"needs_task": True, "task_text": "  "}),
            (False, "", DEFAULT_TASK_DUE_HOURS),
        )

    def test_string_true_is_accepted(self):
        needs, text, hours = _parse_task_decision(
            {"needs_task": "true", "task_text": "Отправить договор", "due_in_hours": 48}
        )
        self.assertTrue(needs)
        self.assertEqual(text, "Отправить договор")
        self.assertEqual(hours, 48)

    def test_broken_hours_fall_back_to_default(self):
        for raw in ("завтра", None, 0, -5, 100_000):
            with self.subTest(raw=raw):
                _, _, hours = _parse_task_decision(
                    {"needs_task": True, "task_text": "Отправить КП", "due_in_hours": raw}
                )
                self.assertEqual(hours, DEFAULT_TASK_DUE_HOURS)


class TestFollowupFallbackRule(unittest.IsolatedAsyncioTestCase):
    """
    Запасное правило: LLM не вернул needs_task → решаем по next_steps.
    Четыре сценария из задания, сквозь _create_followup_task.
    """

    @staticmethod
    def _analysis(next_steps=None, next_contact="Не обсуждали"):
        # needs_task=None — блока решения в ответе LLM не было
        return SimpleNamespace(
            needs_task=None, task_text="", due_in_hours=DEFAULT_TASK_DUE_HOURS,
            next_contact_date=next_contact, next_steps=next_steps or [],
        )

    async def _run(self, analysis, open_tasks=None):
        fake = FakeAmo(open_tasks=open_tasks or [])
        spies = {}
        with patch("services.amocrm.httpx.AsyncClient", fake), \
                patch.object(main.amocrm_service, "create_task",
                             wraps=main.amocrm_service.create_task) as create_spy, \
                patch.object(main.amocrm_service, "update_task",
                             wraps=main.amocrm_service.update_task) as update_spy:
            await main._create_followup_task(1, analysis, 222)
            spies["create"] = create_spy
            spies["update"] = update_spy
        return fake, spies

    async def test_no_manager_steps_and_no_contact_creates_nothing(self):
        analysis = self._analysis(next_steps=["Получить от клиента выписку"])
        fake, spies = await self._run(analysis)

        spies["create"].assert_not_called()
        spies["update"].assert_not_called()
        self.assertEqual(fake.requests, [], "к amoCRM не должно быть ни одного запроса")

    async def test_manager_step_creates_task(self):
        analysis = self._analysis(next_steps=["Менеджеру: отправить КП"])
        fake, spies = await self._run(analysis)

        spies["create"].assert_called_once()
        spies["update"].assert_not_called()
        payload = fake.by_method("POST")[0][1]["json"][0]
        self.assertIn("отправить КП", payload["text"])
        self.assertTrue(
            payload["text"].startswith(main.AUTO_TASK_MARKER),
            "без маркера авторства сервис потом примет свою задачу за ручную",
        )

    async def test_named_contact_date_alone_creates_task(self):
        """Договорённость может быть только о времени звонка — это тоже задача."""
        analysis = self._analysis(next_steps=[], next_contact="в пятницу")
        fake, spies = await self._run(analysis)

        spies["create"].assert_called_once()
        payload = fake.by_method("POST")[0][1]["json"][0]
        self.assertIn("перезвонить в пятницу", payload["text"])

    async def test_existing_auto_task_is_updated_not_duplicated(self):
        analysis = self._analysis(next_steps=["Менеджеру: отправить КП"])
        fake, spies = await self._run(analysis, open_tasks=[_task(10, 1_700_000_000, AUTO_TEXT)])

        spies["update"].assert_called_once()
        spies["create"].assert_not_called()
        self.assertEqual(fake.by_method("POST"), [])
        self.assertTrue(fake.by_method("PATCH")[0][0].endswith("/tasks/10"))

    async def test_existing_manual_task_stops_everything(self):
        analysis = self._analysis(next_steps=["Менеджеру: отправить КП"])
        fake, spies = await self._run(analysis, open_tasks=[_task(10, 1_700_000_000, MANUAL_TEXT)])

        spies["create"].assert_not_called()
        spies["update"].assert_not_called()
        self.assertEqual(fake.by_method("POST"), [])
        self.assertEqual(fake.by_method("PATCH"), [])


class TestAnalysisWiring(unittest.IsolatedAsyncioTestCase):
    """Блок решения о задаче доезжает из ответа LLM до CallAnalysis."""

    async def _analyze_chat(self, llm_json: str):
        service = AnalysisService()
        with patch.object(AnalysisService, "_call_llm", new=AsyncMock(return_value=llm_json)):
            return await service.analyze_chat("Клиент: здравствуйте", channel_name="WhatsApp")

    async def test_task_fields_reach_analysis(self):
        analysis = await self._analyze_chat(json.dumps({
            "client_name": "Иван", "summary": "нужен техплан",
            "client_city": "Ставрополь", "work_type": "Техплан", "location": "ул. Мира 1",
            "cost": "25 000 ₽", "payment_terms": "Не обсуждали", "call_result": "Согласие",
            "next_contact_date": "завтра", "next_steps": ["Менеджеру: отправить КП"],
            "needs_task": True, "task_text": "Отправить КП по техплану", "due_in_hours": 4,
        }, ensure_ascii=False))

        self.assertTrue(analysis.needs_task)
        self.assertEqual(analysis.task_text, "Отправить КП по техплану")
        self.assertEqual(analysis.due_in_hours, 4)

    async def test_missing_block_degrades_to_no_task(self):
        """Старый формат ответа (без needs_task) не роняет анализ — просто нет задачи."""
        analysis = await self._analyze_chat(json.dumps({
            "client_name": "Иван", "summary": "консультация",
            "client_city": "Ставрополь", "work_type": "Прочие", "location": "Не указано",
            "cost": "Не обсуждали", "payment_terms": "Не обсуждали",
            "call_result": "Не определено", "next_contact_date": "Не указано", "next_steps": [],
        }, ensure_ascii=False))

        self.assertEqual(analysis.summary, "консультация")
        self.assertIsNone(analysis.needs_task, "должно быть «не высказался», а не «нет»")
        self.assertEqual(analysis.task_text, "")


class TestSnapToWorkHours(unittest.TestCase):
    """Правило 3: срок не должен падать в выходные и в нерабочее время."""

    @staticmethod
    def _ts(y, m, d, hh, mm=0):
        return int(datetime(y, m, d, hh, mm, tzinfo=MSK).timestamp())

    def _snapped(self, *args):
        return datetime.fromtimestamp(main._snap_to_work_hours(self._ts(*args)), MSK)

    def test_friday_evening_moves_to_monday_morning(self):
        got = self._snapped(2026, 9, 4, 19)  # пятница 19:00
        self.assertEqual((got.year, got.month, got.day), (2026, 9, 7))  # понедельник
        self.assertEqual((got.hour, got.minute, got.second), (10, 0, 0))

    def test_saturday_moves_to_monday_morning(self):
        got = self._snapped(2026, 9, 5, 12)  # суббота
        self.assertEqual((got.day, got.hour), (7, 10))

    def test_early_morning_moves_to_ten(self):
        got = self._snapped(2026, 9, 9, 6)  # среда 06:00
        self.assertEqual((got.day, got.hour), (9, 10))

    def test_work_hours_left_intact(self):
        ts = self._ts(2026, 9, 9, 14, 30)  # среда 14:30
        self.assertEqual(main._snap_to_work_hours(ts), ts)


if __name__ == "__main__":
    unittest.main()
