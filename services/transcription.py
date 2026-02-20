"""
Сервис транскрибации через AssemblyAI.
Поддерживает диаризацию (разделение по говорящим) и Speaker Identification по ролям.
"""
import assemblyai as aai
import logging
import tempfile
import os
from typing import Optional, List, Dict
from dataclasses import dataclass
from config import (
    ASSEMBLYAI_API_KEY,
    ASSEMBLYAI_SPEECH_MODEL,
    ASSEMBLYAI_FALLBACK_MODEL,
    ASSEMBLYAI_SPEAKERS_EXPECTED,
    ASSEMBLYAI_MULTICHANNEL,
)

logger = logging.getLogger(__name__)

aai.settings.api_key = ASSEMBLYAI_API_KEY

KNOWN_ROLES = {"Менеджер", "Клиент"}


@dataclass
class Speaker:
    """Информация о говорящем"""
    label: str  # A, B, C, ... или роль (Менеджер, Клиент)
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
    """Сервис транскрибации с диаризацией"""

    def __init__(self):
        self.transcriber = aai.Transcriber()

    def _build_config(
        self,
        language_code: str,
        speaker_labels: bool,
    ) -> aai.TranscriptionConfig:
        speech_models = [ASSEMBLYAI_SPEECH_MODEL]
        if ASSEMBLYAI_FALLBACK_MODEL and ASSEMBLYAI_FALLBACK_MODEL != ASSEMBLYAI_SPEECH_MODEL:
            speech_models.append(ASSEMBLYAI_FALLBACK_MODEL)

        logger.info(
            f"AssemblyAI config: models={speech_models}, "
            f"speakers_expected={ASSEMBLYAI_SPEAKERS_EXPECTED}, "
            f"multichannel={ASSEMBLYAI_MULTICHANNEL}, "
            f"speaker_labels={speaker_labels}"
        )

        kwargs: dict = {
            "speech_models": speech_models,
            "language_code": language_code,
            "punctuate": True,
            "format_text": True,
        }

        if ASSEMBLYAI_MULTICHANNEL:
            kwargs["multichannel"] = True
        elif speaker_labels:
            kwargs["speaker_labels"] = True
            kwargs["speakers_expected"] = ASSEMBLYAI_SPEAKERS_EXPECTED
            kwargs["speech_understanding"] = {
                "request": {
                    "speaker_identification": {
                        "speaker_type": "role",
                        "known_values": ["Менеджер", "Клиент"],
                    }
                }
            }
        else:
            kwargs["speaker_labels"] = False

        return aai.TranscriptionConfig(**kwargs)

    @staticmethod
    def _has_ai_roles(utterances) -> bool:
        """Проверяет, вернул ли AssemblyAI именованные роли вместо букв."""
        if not utterances:
            return False
        labels = {u.speaker for u in utterances}
        return bool(labels & KNOWN_ROLES)

    async def transcribe_audio(
        self,
        audio_data: bytes,
        language_code: str = "ru",
        speaker_labels: bool = True,
    ) -> TranscriptionResult:
        """
        Транскрибирует аудио (опционально с диаризацией).

        Args:
            audio_data: Бинарные данные аудиофайла
            language_code: Код языка (ru, en, etc.)
            speaker_labels: Включить диаризацию (speaker labels)

        Returns:
            Результат транскрибации с разделением по говорящим
        """
        try:
            logger.info(f"📁 Размер аудио: {len(audio_data)} байт")

            suffix = ".mp3"
            if audio_data[:4] == b'RIFF':
                suffix = ".wav"
            elif audio_data[:3] == b'ID3' or audio_data[:2] == b'\xff\xfb':
                suffix = ".mp3"
            elif audio_data[:4] == b'OggS':
                suffix = ".ogg"
            elif audio_data[:4] == b'fLaC':
                suffix = ".flac"

            logger.info(f"📁 Определён формат: {suffix}")

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            logger.info(f"📁 Временный файл: {temp_path}")

            try:
                config = self._build_config(language_code, speaker_labels)

                if speaker_labels:
                    logger.info("🎙️ Начинаем транскрибацию с диаризацией (Universal-3-Pro + Speaker ID)...")
                else:
                    logger.info("🎙️ Начинаем транскрибацию без диаризации...")

                transcript = self.transcriber.transcribe(temp_path, config)

                logger.info(f"📝 Статус транскрибации: {transcript.status}")

                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(f"Ошибка транскрибации: {transcript.error}")

                speakers = []
                formatted_lines = []
                roles_from_ai = False

                if transcript.utterances:
                    roles_from_ai = self._has_ai_roles(transcript.utterances)
                    if roles_from_ai:
                        logger.info("✅ AssemblyAI вернул именованные роли (Speaker Identification)")
                    else:
                        unique_labels = {u.speaker for u in transcript.utterances}
                        logger.info(f"ℹ️ AssemblyAI вернул буквенные метки: {unique_labels}")

                    for utterance in transcript.utterances:
                        speaker = Speaker(
                            label=utterance.speaker,
                            text=utterance.text,
                            start_ms=utterance.start,
                            end_ms=utterance.end,
                        )
                        speakers.append(speaker)
                        formatted_lines.append(f"[{utterance.speaker}]: {utterance.text}")
                else:
                    formatted_lines.append(transcript.text or "")

                duration_seconds = 0
                if transcript.audio_duration:
                    duration_seconds = transcript.audio_duration
                elif speakers:
                    duration_seconds = speakers[-1].end_ms / 1000

                result = TranscriptionResult(
                    full_text=transcript.text or "",
                    speakers=speakers,
                    formatted_text="\n".join(formatted_lines),
                    duration_seconds=duration_seconds,
                    confidence=transcript.confidence or 0.0,
                    language=language_code,
                    roles_from_ai=roles_from_ai,
                )

                logger.info(
                    f"Транскрибация завершена: {len(result.full_text)} символов, "
                    f"{len(speakers)} фрагментов, {duration_seconds:.1f} сек, "
                    f"roles_from_ai={roles_from_ai}"
                )

                return result

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}")
            raise

    def identify_roles(self, speakers: List[Speaker]) -> Dict[str, str]:
        """
        Определяет роли (менеджер/клиент).

        Приоритет 1: Если AssemblyAI Speaker Identification уже вернул роли
        (label == "Менеджер" или "Клиент"), используем их напрямую.
        Приоритет 2: Эвристика по ключевым словам (fallback).
        """
        if not speakers:
            return {}

        unique_labels = {s.label for s in speakers}

        if unique_labels & KNOWN_ROLES:
            logger.info("identify_roles: используем роли от AssemblyAI Speaker ID")
            roles = {}
            for label in unique_labels:
                if label in KNOWN_ROLES:
                    roles[label] = label
                else:
                    roles[label] = f"Говорящий {label}"
            return roles

        logger.info("identify_roles: fallback на эвристику по ключевым словам")
        return self._identify_roles_heuristic(speakers)

    def _identify_roles_heuristic(self, speakers: List[Speaker]) -> Dict[str, str]:
        """Эвристика определения ролей по ключевым словам."""
        roles = {}

        speaker_texts: Dict[str, List[str]] = {}
        for speaker in speakers:
            if speaker.label not in speaker_texts:
                speaker_texts[speaker.label] = []
            speaker_texts[speaker.label].append(speaker.text.lower())

        manager_indicators = [
            "добрый день", "здравствуйте", "компания", "меня зовут",
            "чем могу помочь", "по поводу вашей заявки", "вы оставляли",
            "давайте", "предлагаю", "стоимость", "цена будет",
        ]

        client_indicators = [
            "мне нужно", "хочу", "интересует", "сколько стоит",
            "какая цена", "можете сделать", "когда сможете",
        ]

        for label, texts in speaker_texts.items():
            full_text = " ".join(texts)

            manager_score = sum(1 for ind in manager_indicators if ind in full_text)
            client_score = sum(1 for ind in client_indicators if ind in full_text)

            if manager_score > client_score:
                roles[label] = "Менеджер"
            else:
                roles[label] = "Клиент"

        if len(roles) == 2 and list(roles.values()).count("Менеджер") != 1:
            labels = sorted(roles.keys())
            roles[labels[0]] = "Менеджер"
            roles[labels[1]] = "Клиент"

        return roles

    def format_with_roles(
        self,
        speakers: List[Speaker],
        roles: Dict[str, str],
    ) -> str:
        """
        Форматирует текст с ролями вместо Speaker A/B.
        """
        lines = []
        for speaker in speakers:
            role = roles.get(speaker.label, f"Говорящий {speaker.label}")
            lines.append(f"[{role}]: {speaker.text}")

        return "\n".join(lines)


transcription_service = TranscriptionService()
