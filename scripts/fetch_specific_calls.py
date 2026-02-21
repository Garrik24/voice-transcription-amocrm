import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.amocrm import amocrm_service
from services.transcription import transcription_service
from config import ASSEMBLYAI_SPEECH_MODEL, LLM_PROVIDER, OPENAI_MODEL

# Время поиска: 20 февраля 10:40 - 11:00 МСК
# МСК = UTC+3. Значит ищем 07:40 - 08:00 UTC
START_UTC = datetime(2026, 2, 20, 7, 0, tzinfo=timezone.utc).timestamp()
END_UTC = datetime(2026, 2, 20, 9, 0, tzinfo=timezone.utc).timestamp()

async def main():
    print(f"🔍 Ищем звонки за 20.02 между 10:00 и 12:00 МСК...")
    print(f"⚙️ Модели в конфиге: STT={ASSEMBLYAI_SPEECH_MODEL}, LLM={LLM_PROVIDER}/{OPENAI_MODEL}")
    
    # Берём с запасом (последние 48 часов, фильтруем вручную)
    events = await amocrm_service.get_recent_calls(minutes=48*60)
    
    found_calls = []
    for event in events:
        created_at = event.get("created_at", 0)
        if START_UTC <= created_at <= END_UTC:
            found_calls.append(event)
            
    print(f"✅ Найдено событий в интервале: {len(found_calls)}")
    
    processed_count = 0
    for event in found_calls:
        call_data = await amocrm_service.process_call_event(event)
        if not call_data or not call_data.get("record_url"):
            continue
            
        dt = datetime.fromtimestamp(call_data["created_at"], tz=timezone.utc) + timedelta(hours=3)
        time_str = dt.strftime("%H:%M:%S")
        
        print(f"\n📞 ЗВОНОК в {time_str} (МСК) | {call_data['phone']}")
        print(f"🔗 URL: {call_data['record_url']}")
        
        # Скачиваем и расшифровываем
        try:
            audio = await amocrm_service.download_call_recording(call_data['record_url'])
            if len(audio) < 1000:
                print("⚠️ Файл слишком маленький, пропускаем.")
                continue
                
            print("🎙️ Расшифровка (AssemblyAI)...")
            transcription = await transcription_service.transcribe_audio(audio, speaker_labels=True)
            
            print("\n" + "="*40)
            print(f"📄 РАСШИФРОВКА ({time_str})")
            print("="*40)
            print(transcription.formatted_text)
            print("="*40 + "\n")
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Ошибка обработки: {e}")

    if processed_count == 0:
        print("❌ Не удалось получить расшифровку (возможно, нет ссылок на записи или файлы недоступны).")

if __name__ == "__main__":
    asyncio.run(main())
