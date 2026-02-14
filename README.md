# 🎙️ Voice Transcription Service

Сервис автоматической транскрибации звонков из AmoCRM с диаризацией (разделением по говорящим) и AI-анализом.

## 🚀 Возможности

- ✅ **Диаризация** — автоматическое разделение по говорящим (Менеджер/Клиент)
- ✅ **Высокое качество** — AssemblyAI для русского языка
- ✅ **AI-анализ** — извлечение ключевой информации через GPT
- ✅ **Автоматизация** — webhook от AmoCRM при завершении звонка
- ✅ **Уведомления** — Telegram при ошибках

## 📋 Извлекаемая информация

- ФИО клиента
- Город
- Тип работ
- Стоимость
- Условия оплаты
- Краткое резюме разговора
- Задачи для менеджера
- Итог звонка
- Дата следующего контакта

## 🏗️ Структура проекта

```
voice-transcription/
├── main.py              # FastAPI сервер + webhook
├── config.py            # Конфигурация
├── services/
│   ├── amocrm.py        # Работа с AmoCRM API
│   ├── transcription.py # AssemblyAI транскрибация
│   ├── analysis.py      # GPT анализ
│   └── telegram.py      # Уведомления
├── requirements.txt     # Зависимости
├── Procfile            # Для Railway
└── railway.toml        # Конфиг Railway
```

## ⚙️ Установка на Railway

### 1. Создай новый проект в Railway

1. Зайди на [railway.app](https://railway.app)
2. New Project → Deploy from GitHub repo
3. Выбери репозиторий с этим кодом

### 2. Добавь переменные окружения

В Railway Dashboard → Variables добавь:

| Переменная | Значение |
|------------|----------|
| `AMOCRM_DOMAIN` | твой-домен.amocrm.ru |
| `AMOCRM_ACCESS_TOKEN` | токен из AmoCRM |
| `ASSEMBLYAI_API_KEY` | ключ из AssemblyAI |
| `LLM_PROVIDER` | `openai` или `gemini` (по умолчанию `openai`) |
| `OPENAI_API_KEY` | ключ из OpenAI (нужен если `LLM_PROVIDER=openai`) |
| `OPENAI_MODEL` | модель OpenAI (опционально, по умолчанию `gpt-4o-mini`) |
| `GEMINI_API_KEY` | ключ из Google Gemini (нужен если `LLM_PROVIDER=gemini`) |
| `GEMINI_MODEL` | модель Gemini (опционально, по умолчанию `gemini-2.0-flash-001`) |
| `ANALYSIS_PIPELINE_VERSION` | `v1` или `v2` (по умолчанию `v1`, `v2` включает speaker stats + fact verifier) |
| `TELEGRAM_BOT_TOKEN` | токен бота (опционально) |
| `TELEGRAM_CHAT_ID` | ID чата (опционально) |
| `APP_TIMEZONE` | таймзона для отображения времени (опционально, напр. `Europe/Moscow`) |

### 3. Получи URL сервиса

После деплоя Railway даст URL типа:
`https://your-service.up.railway.app`

### 4. Настрой webhook в AmoCRM

1. Зайди в AmoCRM → Настройки → API и Интеграции → Webhooks
2. Добавь новый webhook:
   - **URL**: `https://your-service.up.railway.app/webhook/amocrm`
   - **События**: Добавление примечания (note_add)

## 🔧 Локальная разработка

```bash
# Клонируй репозиторий
git clone <repo-url>
cd voice-transcription

# Создай виртуальное окружение
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows

# Установи зависимости
pip install -r requirements.txt

# Создай .env файл (скопируй из ENV_EXAMPLE.txt)
cp ENV_EXAMPLE.txt .env
# Заполни значения в .env

# Запусти сервер
python main.py
```

## 📡 API Endpoints

| Метод | URL | Описание |
|-------|-----|----------|
| GET | `/` | Информация о сервисе |
| GET | `/health` | Health check |
| POST | `/webhook/amocrm` | Webhook для AmoCRM |
| POST | `/webhook/amocrm/geodesist-assigned` | Webhook: уведомление геодезиста (MAX/Wappi) |
| POST | `/webhook/test` | Тестовый endpoint |

### Тестовый запрос

```bash
curl -X POST https://your-service.up.railway.app/webhook/test \
  -H "Content-Type: application/json" \
  -d '{
    "lead_id": 12345,
    "record_url": "https://url-to-audio-file.mp3",
    "call_type": "outgoing_call",
    "responsible_user_id": 123
  }'
```

## 📝 Формат примечания в AmoCRM

```
🎙️ РАСШИФРОВКА ЗВОНКА (AI)

📞 Звонок: Исходящий | 5 мин 32 сек
👤 Клиент: Иванов Сергей Петрович
📍 Город: Краснодар
👨‍💼 Менеджер: Елена

📝 СУТЬ РАЗГОВОРА:
Клиент интересуется геодезической съёмкой участка 
под ИЖС. Участок 12 соток в пригороде Краснодара.

🔧 Тип работы: Топографическая съёмка 1:500
💰 Стоимость: 18 000 ₽
💳 Оплата: предоплата 50%

📊 Итог: Договорились о работе
📅 Следующий контакт: Среда, 15 января

✅ ЗАДАЧИ МЕНЕДЖЕРУ:
• Отправить договор на email до 15.01
• Согласовать дату выезда на следующую неделю
• Перезвонить в среду для подтверждения
```

## 💰 Стоимость

| Сервис | Стоимость |
|--------|-----------|
| Railway | $5/мес (уже оплачен) |
| AssemblyAI | ~$0.01/мин аудио |
| OpenAI GPT-4o-mini | ~$0.0001/запрос |

**Пример**: 100 звонков по 5 минут = ~$5-7/мес

## ❓ Troubleshooting

### Нет записи звонка
- Проверь, что в AmoCRM настроена запись звонков
- Проверь, что webhook настроен правильно

### Ошибка транскрибации
- Проверь ASSEMBLYAI_API_KEY
- Проверь, что файл не слишком маленький (<10KB)

### Ошибка AmoCRM
- Проверь AMOCRM_ACCESS_TOKEN (может истечь)
- Обнови токен через refresh_token

---

Made with ❤️ for геодезия
