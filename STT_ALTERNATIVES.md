# Альтернативы AssemblyAI для русского языка

Сравнение сервисов распознавания речи (STT) для замены AssemblyAI.  
Критерии: качество русского, диаризация, Railway-совместимость.

---

## 1. Yandex SpeechKit (Яндекс Облако)

**Плюсы:**
- Русский — родной язык, отличное качество
- REST API, работает из любого региона
- Поддержка диаризации (разделение спикеров)
- Модели: `general`, `general:rc` (улучшенная), `telephony` (для звонков)
- Цена: ~1.5₽/мин для `general`, ~3₽/мин для `telephony`

**Минусы:**
- Нужен аккаунт Yandex Cloud
- OAuth или API-ключ (IAM)

**Интеграция:**
- REST API: `https://stt.api.cloud.yandex.net/speech/v1/stt:recognize`
- Или `yandex-speechkit` (неофициальный Python SDK)
- Документация: https://cloud.yandex.ru/docs/speechkit/stt/

**Рекомендация:** Лучший вариант для русского. Модель `telephony` заточена под телефонные звонки.

---

## 2. Google Cloud Speech-to-Text

**Плюсы:**
- Отличная поддержка русского (`ru-RU`)
- Диаризация (Speaker Diarization) — встроена
- REST API, `google-cloud-speech` SDK
- $300 бесплатно при регистрации
- Работает с Railway (через `GOOGLE_APPLICATION_CREDENTIALS` или Workload Identity)

**Минусы:**
- Нужен GCP-проект, сервисный аккаунт
- Цена: ~$0.024/мин (диаризация дороже)

**Интеграция:**
```python
from google.cloud import speech
client = speech.SpeechClient()
config = speech.RecognitionConfig(
    language_code="ru-RU",
    diarization_config=speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=2,
    ),
)
```

**Рекомендация:** Надёжный вариант, хороший русский. Нужен JSON-ключ сервисного аккаунта в Railway Variables.

---

## 3. OpenAI Whisper API

**Плюсы:**
- Очень хорошее качество для русского
- Простая интеграция (уже есть `openai` в проекте)
- Нет диаризации — только один поток текста
- $0.006/мин (дёшево)

**Минусы:**
- **Нет диаризации** — не разделяет Менеджер/Клиент
- Придётся использовать эвристику или отдельный сервис для разделения спикеров

**Интеграция:**
```python
from openai import OpenAI
client = OpenAI()
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        language="ru",
    )
```

**Рекомендация:** Если диаризация не критична — самый простой переход. Иначе — не подходит.

---

## 4. Anthropic

**Anthropic не предоставляет STT.** Claude — это LLM для текста.  
Использовать можно только для этапа анализа (вместо GPT), но не для транскрибации.

---

## Сравнительная таблица

| Сервис              | Русский | Диаризация | Railway | Сложность |
|---------------------|---------|------------|---------|-----------|
| Yandex SpeechKit    | ⭐⭐⭐    | ✅         | ✅      | Средняя   |
| Google Speech-to-Text| ⭐⭐⭐   | ✅         | ✅      | Средняя   |
| OpenAI Whisper      | ⭐⭐⭐    | ❌         | ✅      | Низкая    |
| AssemblyAI (текущий)| ⭐⭐     | ✅         | ✅      | —         |

---

## Рекомендация

1. **Yandex SpeechKit** — приоритет для русского, особенно для телефонных звонков.
2. **Google Speech-to-Text** — запасной вариант, если Yandex не подходит.
3. **Whisper** — только если можно обойтись без диаризации.

Следующий шаг: реализовать провайдер `transcription_yandex.py` или `transcription_google.py` с тем же интерфейсом, что и `TranscriptionService`, и переключать через `TRANSCRIPTION_PROVIDER=assemblyai|yandex|google` в config.
