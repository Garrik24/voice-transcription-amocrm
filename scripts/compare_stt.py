#!/usr/bin/env python3
"""
Скрипт сравнения качества транскрибации разных провайдеров STT.

Использование:
  python scripts/compare_stt.py <путь_к_файлу.mp3>
  python scripts/compare_stt.py <URL_записи_звонка>

Сейчас поддерживаются: AssemblyAI, OpenAI Whisper.
После подключения Yandex SpeechKit — добавить transcribe_yandex() и вызов в main().

Перед запуском: source RAILWAY_SECRETS.local.txt (или .env)
"""
import asyncio
import sys
import os
import tempfile
import httpx
import ssl

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


async def load_audio(source: str) -> bytes:
    """Загружает аудио из файла или URL."""
    if source.startswith("http://") or source.startswith("https://"):
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        async with httpx.AsyncClient(follow_redirects=True, timeout=120.0, verify=ssl_ctx) as client:
            r = await client.get(source)
            r.raise_for_status()
            return r.content
    else:
        with open(source, "rb") as f:
            return f.read()


async def transcribe_assemblyai(audio_data: bytes) -> dict:
    """Транскрибация через AssemblyAI (с диаризацией)."""
    from services.transcription import transcription_service

    result = await transcription_service.transcribe_audio(audio_data, language_code="ru", speaker_labels=True)
    return {
        "provider": "AssemblyAI",
        "text": result.formatted_text or result.full_text,
        "raw": result.full_text,
        "chars": len(result.full_text),
        "speakers": len(result.speakers),
        "duration_sec": result.duration_seconds,
    }


async def transcribe_whisper(audio_data: bytes) -> dict:
    """Транскрибация через OpenAI Whisper (без диаризации)."""
    from openai import OpenAI

    client = OpenAI()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ru",
            )
    finally:
        os.unlink(tmp_path)
    text = transcript.text or ""
    return {
        "provider": "OpenAI Whisper",
        "text": text,
        "raw": text,
        "chars": len(text),
        "speakers": 0,  # нет диаризации
        "duration_sec": 0,
    }


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nПример:")
        print("  python scripts/compare_stt.py ./sample.mp3")
        print("  python scripts/compare_stt.py https://amo.vmclouds.ru/.../recording.mp3")
        sys.exit(1)

    source = sys.argv[1]
    print(f"📁 Загрузка: {source[:80]}...")
    audio_data = await load_audio(source)
    print(f"   Размер: {len(audio_data)} байт\n")

    results = []

    # AssemblyAI
    if os.getenv("ASSEMBLYAI_API_KEY"):
        try:
            print("🎙️ AssemblyAI...")
            r = await transcribe_assemblyai(audio_data)
            results.append(r)
            print(f"   ✅ {r['chars']} символов, {r['speakers']} фрагментов\n")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}\n")
    else:
        print("⏭️ AssemblyAI: нет ASSEMBLYAI_API_KEY\n")

    # Whisper
    if os.getenv("OPENAI_API_KEY"):
        try:
            print("🎙️ OpenAI Whisper...")
            r = await transcribe_whisper(audio_data)
            results.append(r)
            print(f"   ✅ {r['chars']} символов\n")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}\n")
    else:
        print("⏭️ Whisper: нет OPENAI_API_KEY\n")

    if not results:
        print("Нет доступных провайдеров. Проверьте ASSEMBLYAI_API_KEY и OPENAI_API_KEY.")
        sys.exit(1)

    # Вывод сравнения
    print("=" * 60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    for r in results:
        print(f"\n--- {r['provider']} ({r['chars']} символов) ---\n")
        print(r["text"])
        print()


if __name__ == "__main__":
    asyncio.run(main())
