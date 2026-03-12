import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


# Нужен для инициализации AsyncOpenAI внутри services/transcription при импорте main.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import main


class TestAutofillHelpers(unittest.TestCase):
    def test_has_custom_field_value_accepts_enum_id(self):
        custom_field = {
            "field_id": 212083,
            "values": [{"enum_id": 423479}],
        }
        self.assertTrue(main._has_custom_field_value(custom_field))

    def test_parse_price_with_bounds(self):
        self.assertEqual(main._parse_price("25 000 ₽"), 25000)
        self.assertIsNone(main._parse_price("500 ₽"))
        self.assertIsNone(main._parse_price("номер 8 918 123 45 67"))


class TestAutoFillLeadFields(unittest.IsolatedAsyncioTestCase):
    async def test_does_not_override_existing_fields(self):
        lead_data = {
            "price": 35000,
            "custom_fields_values": [
                {"field_id": 212029, "values": [{"value": "Ставрополь"}]},
                {"field_id": 212083, "values": [{"enum_id": 423479}]},
                {"field_id": 212099, "values": [{"enum_id": 423541}]},
                {"field_id": 767917, "values": [{"value": "50% аванс"}]},
            ],
        }
        analysis = SimpleNamespace(
            client_city="Краснодар",
            work_type="топосъемка",
            payment_terms="безнал",
            cost="25 000 ₽",
        )

        with patch.object(main.amocrm_service, "get_lead", new=AsyncMock(return_value=lead_data)), \
             patch.object(main.amocrm_service, "update_lead_fields", new=AsyncMock(return_value=True)) as update_mock:
            await main.auto_fill_lead_fields(lead_id=1001, analysis=analysis, call_type_simple="incoming")
            update_mock.assert_not_awaited()

    async def test_updates_only_empty_fields(self):
        lead_data = {
            "price": 0,
            "custom_fields_values": [],
        }
        analysis = SimpleNamespace(
            client_city="Ставрополь",
            work_type="топосъемка",
            payment_terms="50% аванс",
            cost="25 000 ₽",
        )

        with patch.object(main.amocrm_service, "get_lead", new=AsyncMock(return_value=lead_data)), \
             patch.object(main.amocrm_service, "update_lead_fields", new=AsyncMock(return_value=True)) as update_mock:
            await main.auto_fill_lead_fields(lead_id=1002, analysis=analysis, call_type_simple="incoming")

            update_mock.assert_awaited_once()
            kwargs = update_mock.await_args.kwargs
            self.assertEqual(kwargs["lead_id"], 1002)
            self.assertEqual(kwargs["price"], 25000)

            field_ids = {f["field_id"] for f in kwargs["custom_fields_values"]}
            self.assertEqual(field_ids, {212029, 212083, 212099, 767917})

