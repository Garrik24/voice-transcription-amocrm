"""
Сервис транскрибации v2. Стерео-диаризация через разделение каналов + OpenAI Whisper.

Ключевые улучшения:
1. Стерео записи: ffmpeg разделяет каналы → Whisper на каждый → склейка по таймстемпам
2. Длинные файлы: конвертация в mp3 64kbps если > 20 МБ
3. Fallback: если не стерео → Whisper на весь файл + GPT определяет роли
"""
import asyncio
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

KNOWN_ROLES = {"Менеджер", "Клиент"}
WHISPER_FILE_LIMIT = 24 * 1024 * 1024  # 24 MB (оставляем запас от 25 MB лимита)


@dataclass
class Speaker:
    """Информация о говорящем"""
    label: str
    text: str
    start_ms: int
    end_ms: int


@dataclass
class TranscriptionResult:
    """Результат транскрибации"""
    full_text: str
    speakers: List[Speaker]
    formatted_text: str
    duration_seconds: float
    confidence: float
    language: str
    roles_from_ai: bool = False


class TranscriptionService:
    """Сервис транскрибации. Стерео → разделение каналов, моно → Whisper + GPT роли."""

    def __init__(self):
        self._openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def transcribe_audio(
        self,
        audio_data: bytes,
        language_code: str = "ru",
        speaker_labels: bool = True,
    ) -> TranscriptionResult:
        """
        Главная точка входа. Определяет стерео/моно и выбирает стратегию.
        """
        logger.info(f"🎙️ Начинаем транскрибацию, размер: {len(audio_data)} байт")

        # Сохраняем во временный файл для ffprobe/ffmpeg
        suffix = self._detect_suffix(audio_data)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            input_path = f.name

        try:
            # Проверяем количество каналов
            channels = await self._get_channel_count(input_path)
            duration = await self._get_duration(input_path)
            logger.info(f"📁 Формат: {suffix}, каналы: {channels}, длительность: {duration:.1f} сек")

            if channels >= 2 and speaker_labels:
                # СТЕРЕО → идеальная диаризация через разделение каналов
                logger.info("🎧 Стерео запись → разделяем каналы для диаризации")
                return await self._transcribe_stereo(input_path, duration)
            else:
                # МОНО → обычная транскрибация Whisper
                logger.info("🔈 Моно запись → транскрибация без диаризации")
                optimized_data = await self._optimize_for_whisper(input_path, audio_data)
                return await self._transcribe_whisper(optimized_data)

        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)

    # -------------------------------------------------------------------------
    # Стерео: разделение каналов + Whisper на каждый
    # -------------------------------------------------------------------------

    async def _transcribe_stereo(self, input_path: str, duration: float) -> TranscriptionResult:
        """
        Разделяет стерео на 2 канала, транскрибирует каждый через Whisper,
        склеивает по таймстемпам.

        Левый канал (0) = Менеджер
        Правый канал (1) = Клиент
        """
        left_path = None
        right_path = None

        try:
            # 1. Разделяем каналы через ffmpeg
            left_path, right_path = await self._split_channels(input_path)

            # 2. Оптимизируем размер каждого канала для Whisper
            left_data = await self._read_and_optimize(left_path)
            right_data = await self._read_and_optimize(right_path)

            logger.info(
                f"📊 Левый (менеджер): {len(left_data)} байт, "
                f"Правый (клиент): {len(right_data)} байт"
            )

            # 3. Транскрибируем оба канала параллельно
            logger.info("🎙️ Транскрибируем оба канала параллельно...")
            left_result, right_result = await asyncio.gather(
                self._whisper_with_segments(left_data, "manager"),
                self._whisper_with_segments(right_data, "client"),
            )

            left_text, left_segments = left_result
            right_text, right_segments = right_result

            logger.info(
                f"📝 Менеджер: {len(left_text)} символов, {len(left_segments)} сегментов | "
                f"Клиент: {len(right_text)} символов, {len(right_segments)} сегментов"
            )

            # 4. Склеиваем сегменты по таймстемпам
            speakers = self._merge_segments(left_segments, right_segments)

            # 5. Формируем результат
            formatted_lines = []
            for s in speakers:
                formatted_lines.append(f"[{s.label}]: {s.text}")

            full_text = " ".join(s.text for s in speakers)
            formatted_text = "\n".join(formatted_lines)

            logger.info(
                f"✅ Стерео транскрибация завершена: {len(full_text)} символов, "
                f"{len(speakers)} реплик, {duration:.1f} сек"
            )

            return TranscriptionResult(
                full_text=full_text,
                speakers=speakers,
                formatted_text=formatted_text,
                duration_seconds=duration,
                confidence=1.0,
                language="ru",
                roles_from_ai=True,
            )

        finally:
            for path in [left_path, right_path]:
                if path and os.path.exists(path):
                    os.unlink(path)

    async def _split_channels(self, input_path: str) -> Tuple[str, str]:
        """Разделяет стерео файл на два моно MP3 через ffmpeg."""
        left_path = input_path + "_left.mp3"
        right_path = input_path + "_right.mp3"

        cmd_left = [
            "ffmpeg", "-y", "-i", input_path,
            "-map_channel", "0.0.0",
            "-ac", "1", "-ab", "64k", "-ar", "16000",
            left_path
        ]

        cmd_right = [
            "ffmpeg", "-y", "-i", input_path,
            "-map_channel", "0.0.1",
            "-ac", "1", "-ab", "64k", "-ar", "16000",
            right_path
        ]

        logger.info("✂️ Разделяем каналы через ffmpeg...")

        proc_left = await asyncio.create_subprocess_exec(
            *cmd_left,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        proc_right = await asyncio.create_subprocess_exec(
            *cmd_right,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        (_, stderr_left), (_, stderr_right) = await asyncio.gather(
            proc_left.communicate(),
            proc_right.communicate(),
        )

        if proc_left.returncode != 0:
            logger.error(f"❌ ffmpeg left channel error: {stderr_left.decode()[-500:]}")
            raise RuntimeError("Ошибка разделения левого канала")

        if proc_right.returncode != 0:
            logger.error(f"❌ ffmpeg right channel error: {stderr_right.decode()[-500:]}")
            raise RuntimeError("Ошибка разделения правого канала")

        left_size = os.path.getsize(left_path)
        right_size = os.path.getsize(right_path)
        logger.info(f"✅ Каналы разделены: L={left_size} байт, R={right_size} байт")

        return left_path, right_path

    async def _whisper_with_segments(
        self, audio_data: bytes, channel_name: str
    ) -> Tuple[str, List[dict]]:
        """
        Транскрибирует через Whisper API и возвращает текст + сегменты с таймстемпами.
        """
        suffix = self._detect_suffix(audio_data)
        filename = f"{channel_name}{suffix}"
        mime = self._get_mime(suffix)

        try:
            response = await self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=(filename, audio_data, mime),
                language="ru",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

            full_text = response.text or ""
            segments = []

            raw_segments = getattr(response, "segments", []) or []
            for seg in raw_segments:
                if isinstance(seg, dict):
                    text = seg.get("text", "").strip()
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                else:
                    text = getattr(seg, "text", "").strip()
                    start = getattr(seg, "start", 0)
                    end = getattr(seg, "end", 0)

                if text:
                    segments.append({
                        "text": text,
                        "start": float(start),
                        "end": float(end),
                    })

            return full_text, segments

        except Exception as e:
            logger.error(f"❌ Whisper ошибка ({channel_name}): {e}")
            raise

    def _merge_segments(
        self,
        left_segments: List[dict],
        right_segments: List[dict],
    ) -> List[Speaker]:
        """
        Склеивает сегменты двух каналов по таймстемпам в единый поток.
        Левый = Менеджер, Правый = Клиент.
        """
        speakers = []

        for seg in left_segments:
            speakers.append(Speaker(
                label="Менеджер",
                text=seg["text"],
                start_ms=int(seg["start"] * 1000),
                end_ms=int(seg["end"] * 1000),
            ))

        for seg in right_segments:
            speakers.append(Speaker(
                label="Клиент",
                text=seg["text"],
                start_ms=int(seg["start"] * 1000),
                end_ms=int(seg["end"] * 1000),
            ))

        speakers.sort(key=lambda s: s.start_ms)
        speakers = [s for s in speakers if len(s.text.strip()) > 2]

        return speakers

    # -------------------------------------------------------------------------
    # Моно: обычная транскрибация Whisper (без диаризации)
    # -------------------------------------------------------------------------

    async def _transcribe_whisper(self, audio_data: bytes) -> TranscriptionResult:
        """
        Транскрибация через OpenAI Whisper API.
        Для моно записей — без диаризации, просто текст.
        """
        logger.info(f"📁 Размер аудио для Whisper: {len(audio_data)} байт")

        suffix = self._detect_suffix(audio_data)
        mime = self._get_mime(suffix)
        filename = f"audio{suffix}"

        logger.info("🎙️ Отправляем в OpenAI Whisper (whisper-1)...")

        response = await self._openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, audio_data, mime),
            language="ru",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

        full_text = response.text or ""
        segments = getattr(response, "segments", []) or []
        duration_seconds = 0.0
        if segments:
            last = segments[-1]
            if isinstance(last, dict):
                duration_seconds = float(last.get("end", 0))
            else:
                duration_seconds = float(getattr(last, "end", 0))

        logger.info(
            f"✅ Транскрибация завершена (Whisper моно): {len(full_text)} символов, "
            f"{len(segments)} сегментов, {duration_seconds:.1f} сек"
        )

        return TranscriptionResult(
            full_text=full_text,
            speakers=[],
            formatted_text=full_text,
            duration_seconds=duration_seconds,
            confidence=1.0,
            language="ru",
            roles_from_ai=False,
        )

    # -------------------------------------------------------------------------
    # Утилиты: ffprobe, оптимизация размера, определение форматов
    # -------------------------------------------------------------------------

    async def _get_channel_count(self, path: str) -> int:
        """Определяет количество каналов через ffprobe."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "quiet",
                "-show_entries", "stream=channels",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            channels = int(stdout.decode().strip().split('\n')[0])
            return channels
        except Exception as e:
            logger.warning(f"⚠️ ffprobe channels ошибка: {e}, считаем моно")
            return 1

    async def _get_duration(self, path: str) -> float:
        """Определяет длительность аудио через ffprobe."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return float(stdout.decode().strip())
        except Exception as e:
            logger.warning(f"⚠️ ffprobe duration ошибка: {e}")
            return 0.0

    async def _optimize_for_whisper(self, input_path: str, original_data: bytes) -> bytes:
        """
        Если файл > 24 МБ, конвертирует в mp3 64kbps моно 16kHz.
        Это уменьшает 8-минутный WAV с 80 МБ до ~4 МБ.
        """
        if len(original_data) <= WHISPER_FILE_LIMIT:
            return original_data

        logger.info(
            f"⚠️ Файл слишком большой ({len(original_data) / 1024 / 1024:.1f} МБ), "
            "конвертируем в mp3 64kbps..."
        )

        output_path = input_path + "_optimized.mp3"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", input_path,
                "-ac", "1", "-ab", "64k", "-ar", "16000",
                output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error(f"❌ ffmpeg optimize error: {stderr.decode()[-500:]}")
                return original_data

            with open(output_path, "rb") as f:
                optimized = f.read()

            logger.info(
                f"✅ Оптимизировано: {len(original_data) / 1024 / 1024:.1f} МБ → "
                f"{len(optimized) / 1024 / 1024:.1f} МБ"
            )
            return optimized

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    async def _read_and_optimize(self, path: str) -> bytes:
        """Читает файл и оптимизирует если нужно."""
        with open(path, "rb") as f:
            data = f.read()

        if len(data) > WHISPER_FILE_LIMIT:
            return await self._optimize_for_whisper(path, data)
        return data

    @staticmethod
    def _detect_suffix(audio_data: bytes) -> str:
        """Определяет формат аудио по magic bytes."""
        if audio_data[:4] == b'RIFF':
            return ".wav"
        elif audio_data[:4] == b'OggS':
            return ".ogg"
        elif audio_data[:4] == b'fLaC':
            return ".flac"
        else:
            return ".mp3"

    @staticmethod
    def _get_mime(suffix: str) -> str:
        """Возвращает MIME-тип по расширению."""
        return {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
        }.get(suffix, "audio/mpeg")

    # -------------------------------------------------------------------------
    # Legacy API: identify_roles / format_with_roles
    # Сохраняем для обратной совместимости с main.py
    # -------------------------------------------------------------------------

    def identify_roles(self, speakers: List[Speaker]) -> Dict[str, str]:
        """Возвращает маппинг label → роль. Для стерео уже определено."""
        if not speakers:
            return {}
        unique_labels = {s.label for s in speakers}
        if unique_labels & KNOWN_ROLES:
            return {label: label for label in unique_labels}
        return self._identify_roles_heuristic(speakers)

    def format_with_roles(self, speakers: List[Speaker], roles: Dict[str, str]) -> str:
        """Форматирует текст с ролями."""
        lines = []
        for speaker in speakers:
            role = roles.get(speaker.label, f"Говорящий {speaker.label}")
            lines.append(f"[{role}]: {speaker.text}")
        return "\n".join(lines)

    def _identify_roles_heuristic(self, speakers: List[Speaker]) -> Dict[str, str]:
        """Эвристика определения ролей по ключевым словам (для моно)."""
        roles = {}
        speaker_texts: Dict[str, List[str]] = {}
        for speaker in speakers:
            if speaker.label not in speaker_texts:
                speaker_texts[speaker.label] = []
            speaker_texts[speaker.label].append(speaker.text.lower())

        manager_indicators = [
            "добрый день", "здравствуйте", "компания", "меня зовут",
            "чем могу помочь", "ставрополь", "геодезия", "стоимость",
        ]
        client_indicators = [
            "мне нужно", "хочу", "интересует", "сколько стоит",
            "какая цена", "можете сделать", "участок", "дом",
        ]

        for label, texts in speaker_texts.items():
            full_text = " ".join(texts)
            manager_score = sum(1 for ind in manager_indicators if ind in full_text)
            client_score = sum(1 for ind in client_indicators if ind in full_text)
            roles[label] = "Менеджер" if manager_score > client_score else "Клиент"

        if len(roles) == 2 and list(roles.values()).count("Менеджер") != 1:
            labels = sorted(roles.keys())
            roles[labels[0]] = "Менеджер"
            roles[labels[1]] = "Клиент"

        return roles


transcription_service = TranscriptionService()
