"""
Сервис транскрибации v3. Стерео-диаризация через разделение каналов + OpenAI Whisper.

Ключевые механики:
1. Стерео записи: ffmpeg разделяет каналы → Whisper на каждый → склейка по таймстемпам
2. Фильтрация галлюцинаций Whisper по энергии канала (первичный критерий)
   + метрикам Whisper (no_speech_prob / compression_ratio — усилитель решения)
3. Дедупликация кросс-тока между каналами и петель внутри канала
4. Роли: одно решение на звонок «какой канал — менеджер» (эвристика → LLM-тайбрейк),
   никаких построчных правок. Не определили уверенно → roles_uncertain=True,
   и автозаполнение CRM по такому звонку блокируется (fail-closed).
5. Длинные файлы: конвертация в mp3 64kbps если > 20 МБ
6. Fallback: если не стерео → Whisper на весь файл + GPT определяет роли
"""
import asyncio
import difflib
import logging
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from config import OPENAI_API_KEY
from services.telegram import telegram_service

logger = logging.getLogger(__name__)

KNOWN_ROLES = {"Менеджер", "Клиент"}
WHISPER_FILE_LIMIT = 24 * 1024 * 1024  # 24 MB (оставляем запас от 25 MB лимита)

# ── Пороги фильтрации сегментов ──────────────────────────────────────────────
# Подобраны по измерениям реальных записей onlinePBX: речь в полосе 300–3400 Гц
# даёт ≥ −52 dBFS, молчащий канал — цифровую тишину −168 dBFS (фон −73 dB —
# DC-гул 0–50 Гц, полосовой фильтр его отсекает). Запас между классами > 100 dB.
SPEECH_GATE_DB = -60.0        # ниже — в канале не было речи, сегмент = галлюцинация
CROSSTALK_MARGIN_DB = 12.0    # межканальный дубль: дропаем канал слабее на столько dB
ENERGY_WINDOW_SEC = 0.1       # окно энергетического профиля (100 мс)
SEGMENT_PADDING_SEC = 0.15    # допуск на неточность таймстемпов Whisper
NO_SPEECH_PROB_MAX = 0.6      # метрика Whisper: подозрение «речи не было»
COMPRESSION_RATIO_MAX = 2.4   # метрика Whisper: зацикливание (повторы фраз)

# ── Маркеры ролей для детерминированного скоринга каналов ────────────────────
# Матчатся по тексту с нормализованной пунктуацией (запятые/точки → пробел):
# Whisper нестабилен в пунктуации, «Ставрополь, Геодезия» и «Ставрополь Геодезия»
# должны матчиться одинаково. «ставропольгеодези» — основа, ловит склонения.
MANAGER_MARKERS = (
    "ставропольгеодези", "ставрополь геодезия", "геодезия добрый",
    "чем могу помочь", "чем могу быть полезен", "слушаю вас",
    "мы занимаемся", "мы не занимаемся", "мы делаем", "мы можем", "мы выезжаем",
    "будет стоить", "стоимость составит", "тысяч рублей", "это порядка",
    "посмотрю по базе", "посмотрю по спутник", "скажите адрес", "какой площади",
    "оставьте номер", "наш инженер", "наш специалист", "кадастровый инженер",
)
CLIENT_MARKERS = (
    "я ищу", "мне нужно", "мне надо", "я хотел", "я хочу", "мне сказали",
    "сколько стоит", "сколько будет", "какая цена", "а дорого",
    "вы занимаетесь", "вы делаете", "у вас есть", "а вы можете", "вы сами",
    "подскажите", "мой участок", "у меня участок", "мои соседи", "мой дом",
    "у меня дом", "я звоню",
)


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
    # Роли каналов не определились уверенно → автозаполнение CRM блокируется
    roles_uncertain: bool = False


@dataclass
class RoleDecision:
    """Решение «какой канал принадлежит менеджеру» — одно на весь звонок."""
    manager_channel: Optional[str]  # "left" | "right" | None (не определили)
    source: str                     # "heuristic" | "llm" | "prior"
    uncertain: bool


class TranscriptionService:
    """Сервис транскрибации. Стерео → разделение каналов, моно → Whisper + GPT роли."""

    def __init__(self):
        self._openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def transcribe_audio(
        self,
        audio_data: bytes,
        language_code: str = "ru",
        speaker_labels: bool = True,
        call_direction: str = "call_in",
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
                return await self._transcribe_stereo(input_path, duration, call_direction=call_direction)
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

    async def _transcribe_stereo(self, input_path: str, duration: float, call_direction: str = "call_in") -> TranscriptionResult:
        """
        Стерео-диаризация: ffmpeg делит каналы → Whisper на каждый →
        фильтрация галлюцинаций по энергии канала → дедупликация →
        одно решение о ролях на весь звонок → склейка по таймстемпам.

        Допущение «левый канал = менеджер» НЕ жёсткое: реальные записи onlinePBX
        встречаются с инвертированными каналами, поэтому роль канала определяется
        по содержанию (эвристика → LLM-тайбрейк), а при неуверенности звонок
        помечается roles_uncertain и автозаполнение CRM блокируется.
        """
        left_path = None
        right_path = None

        try:
            # 1. Разделяем каналы через ffmpeg
            left_path, right_path = await self._split_channels(input_path)

            # 2. Оптимизируем размер каждого канала для Whisper
            left_data = await self._read_and_optimize(left_path)
            right_data = await self._read_and_optimize(right_path)

            logger.info(f"📊 Левый: {len(left_data)} байт, Правый: {len(right_data)} байт")

            # 3. Whisper на оба канала + энергетический профиль исходника — параллельно
            logger.info("🎙️ Транскрибируем оба канала параллельно...")
            whisper_results, (left_profile, right_profile) = await asyncio.gather(
                asyncio.gather(
                    self._whisper_with_segments(left_data, "left"),
                    self._whisper_with_segments(right_data, "right"),
                ),
                self._band_energy_profiles(input_path),
            )
            (_, left_segments), (_, right_segments) = whisper_results

            logger.info(
                f"📝 Whisper: левый {len(left_segments)} сегм., "
                f"правый {len(right_segments)} сегм."
            )

            # 3.5 Фильтрация галлюцинаций (по энергии) и дублей (кросс-ток, петли)
            left_segments = self._filter_channel_segments(left_segments, left_profile, "левый")
            right_segments = self._filter_channel_segments(right_segments, right_profile, "правый")
            left_segments, right_segments = self._dedupe_cross_channel(
                left_segments, right_segments, left_profile, right_profile
            )
            left_segments = self._dedupe_within_channel(left_segments, "левый")
            right_segments = self._dedupe_within_channel(right_segments, "правый")

            # 4. Решение о ролях: детерминированная эвристика → LLM-тайбрейк → приор
            decision = self._score_channels(left_segments, right_segments)
            if decision.manager_channel is None:
                decision = await self._llm_manager_channel(left_segments, right_segments, call_direction)
            if decision.manager_channel is None:
                # Не определили: берём приор (левый = менеджер), но помечаем
                # неуверенность — автозаполнение по такому звонку не выполняется.
                decision = RoleDecision("left", "prior", True)

            # На совсем коротких звонках сигнала мало, а LLM склонен отвечать
            # «уверен» и на пустом основании — форсируем fail-closed.
            total_replies = len(left_segments) + len(right_segments)
            if total_replies < 8 and not decision.uncertain:
                logger.info(f"🎭 Мало реплик ({total_replies}) — решение о ролях помечено неуверенным")
                decision = RoleDecision(decision.manager_channel, decision.source, True)

            logger.info(
                f"🎭 Менеджер = {'левый' if decision.manager_channel == 'left' else 'правый'} канал "
                f"(источник: {decision.source}, uncertain={decision.uncertain})"
            )
            if decision.manager_channel == "left":
                left_label, right_label = "Менеджер", "Клиент"
            else:
                left_label, right_label = "Клиент", "Менеджер"

            # 5. Склеиваем сегменты по таймстемпам
            speakers = self._merge_segments(left_segments, right_segments, left_label, right_label)

            formatted_lines = [f"[{s.label}]: {s.text}" for s in speakers]
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
                roles_uncertain=decision.uncertain,
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
            "-af", "pan=mono|c0=c0",
            "-ab", "64k", "-ar", "16000",
            left_path
        ]

        cmd_right = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", "pan=mono|c0=c1",
            "-ab", "64k", "-ar", "16000",
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

            def _attr(seg, name, default=0):
                if isinstance(seg, dict):
                    return seg.get(name, default)
                return getattr(seg, name, default)

            raw_segments = getattr(response, "segments", []) or []
            for seg in raw_segments:
                text = (_attr(seg, "text", "") or "").strip()
                if text:
                    segments.append({
                        "text": text,
                        "start": float(_attr(seg, "start")),
                        "end": float(_attr(seg, "end")),
                        # Метрики Whisper — вторичные сигналы галлюцинаций
                        "no_speech_prob": float(_attr(seg, "no_speech_prob") or 0),
                        "avg_logprob": float(_attr(seg, "avg_logprob") or 0),
                        "compression_ratio": float(_attr(seg, "compression_ratio") or 0),
                    })

            return full_text, segments

        except Exception as e:
            logger.error(f"❌ Whisper ошибка ({channel_name}): {e}")
            raise

    def _merge_segments(
        self,
        left_segments: List[dict],
        right_segments: List[dict],
        left_label: str = "Менеджер",
        right_label: str = "Клиент",
    ) -> List[Speaker]:
        """
        Склеивает сегменты двух каналов по таймстемпам в единый поток.
        Метки определяются вызывающим кодом в зависимости от направления звонка.
        """
        speakers = []

        for seg in left_segments:
            speakers.append(Speaker(
                label=left_label,
                text=seg["text"],
                start_ms=int(seg["start"] * 1000),
                end_ms=int(seg["end"] * 1000),
            ))

        for seg in right_segments:
            speakers.append(Speaker(
                label=right_label,
                text=seg["text"],
                start_ms=int(seg["start"] * 1000),
                end_ms=int(seg["end"] * 1000),
            ))

        speakers.sort(key=lambda s: s.start_ms)
        # Убираем только пустые: «Да», «Ну», «Угу» — легитимные реплики диалога
        speakers = [s for s in speakers if s.text.strip()]

        return speakers

    # -------------------------------------------------------------------------
    # Энергетический профиль и фильтрация галлюцинаций
    # -------------------------------------------------------------------------

    async def _band_energy_profiles(
        self, input_path: str
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Один проход ffmpeg по ИСХОДНОМУ стерео (до сплита и ресемпла):
        RMS-профиль каждого канала в речевой полосе 300–3400 Гц
        с шагом ENERGY_WINDOW_SEC. Возвращает (левый, правый)
        как списки (время_сек, rms_db).
        """
        out_paths = []
        procs = []
        for ch in (0, 1):
            out = tempfile.NamedTemporaryFile(suffix=f"_astats_c{ch}.txt", delete=False)
            out.close()
            out_paths.append(out.name)
            cmd = [
                "ffmpeg", "-y", "-v", "quiet", "-i", input_path,
                "-af",
                (
                    f"pan=mono|c0=c{ch},highpass=f=300,lowpass=f=3400,"
                    f"astats=metadata=1:reset=1:length={ENERGY_WINDOW_SEC},"
                    f"ametadata=print:key=lavfi.astats.Overall.RMS_level:file={out.name}"
                ),
                "-f", "null", "-",
            ]
            procs.append(await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ))

        for proc in procs:
            await proc.communicate()

        profiles: List[List[Tuple[float, float]]] = []
        try:
            for path in out_paths:
                profile: List[Tuple[float, float]] = []
                current_time: Optional[float] = None
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("frame:"):
                            m = re.search(r"pts_time:([\d.]+)", line)
                            current_time = float(m.group(1)) if m else None
                        elif line.startswith("lavfi.astats.Overall.RMS_level="):
                            raw = line.split("=", 1)[1]
                            try:
                                db = float(raw)
                            except ValueError:
                                db = -168.0
                            # float("-inf")/float("nan") парсятся УСПЕШНО — ловим
                            # отдельно: nan в профиле отравил бы p95 и все сравнения
                            if not math.isfinite(db):
                                db = -168.0
                            t = current_time if current_time is not None else len(profile) * ENERGY_WINDOW_SEC
                            profile.append((t, db))
                profiles.append(profile)
        finally:
            for path in out_paths:
                if os.path.exists(path):
                    os.unlink(path)

        logger.info(
            f"📈 Энергопрофиль: левый {len(profiles[0])} окон, правый {len(profiles[1])} окон"
        )
        return profiles[0], profiles[1]

    @staticmethod
    def _segment_energy(profile: List[Tuple[float, float]], start: float, end: float) -> float:
        """
        p95 RMS канала в таймфрейме сегмента (± допуск): близко к максимуму,
        но устойчиво к одиночным щелчкам. Whisper часто растягивает таймстемпы
        на паузы, поэтому берём верхний перцентиль, а не среднее.
        Пустое окно = тишина.
        """
        lo, hi = start - SEGMENT_PADDING_SEC, end + SEGMENT_PADDING_SEC
        vals = sorted(db for t, db in profile if lo <= t <= hi)
        if not vals:
            return -168.0
        # (n-1)*0.95, а не n*0.95: иначе на малых окнах индекс упирается
        # в максимум и одиночный щелчок сходит за речь
        return vals[int((len(vals) - 1) * 0.95)]

    def _filter_channel_segments(
        self,
        segments: List[dict],
        own_profile: List[Tuple[float, float]],
        channel_name: str,
    ) -> List[dict]:
        """
        Отсекает галлюцинации Whisper. Первичный критерий — энергия собственного
        канала (абсолютный порог: была ли там речь вообще); метрики Whisper —
        только усилитель решения, чтобы не резать реальные короткие реплики.

        ВАЖНО: nsp сам по себе НЕ критерий — на телефонии 8kHz Whisper даёт
        no_speech_prob до 0.97 на реальной внятной речи (проверено на записях
        onlinePBX), безусловный дроп по nsp удалял бы живые реплики.
        """
        if not own_profile:
            # ffmpeg не дал профиль — без энергии гейтить нельзя: иначе
            # -168 у всех сегментов выкосил бы расшифровку целиком
            logger.warning(f"⚠️ {channel_name}: пустой энергопрофиль — гейт пропущен")
            for seg in segments:
                seg["energy_db"] = 0.0
            return segments

        kept, dropped = [], []
        for seg in segments:
            energy = self._segment_energy(own_profile, seg["start"], seg["end"])
            seg["energy_db"] = energy
            reason = None
            if self._is_sound_tag(seg["text"]):
                # «ТЕЛЕФОННЫЙ ЗВОНОК», «ТРЕВОЖНАЯ МУЗЫКА» — звуковые события;
                # энергия у них есть (гудки, музыка), гейт их не берёт
                reason = "звуковой тег"
            elif energy < SPEECH_GATE_DB:
                reason = f"тишина {energy:.0f}dB"
            elif energy < SPEECH_GATE_DB + 8 and seg.get("no_speech_prob", 0) > NO_SPEECH_PROB_MAX:
                reason = f"no_speech p={seg['no_speech_prob']:.2f} при {energy:.0f}dB"
            elif seg.get("compression_ratio", 0) > COMPRESSION_RATIO_MAX and self._is_repetitive(seg["text"]):
                reason = f"петля cr={seg['compression_ratio']:.1f}"

            if reason:
                dropped.append((reason, seg["text"][:40]))
            else:
                kept.append(seg)

        if dropped:
            details = "; ".join(f"[{r}] {t!r}" for r, t in dropped[:8])
            more = f" (+{len(dropped) - 8})" if len(dropped) > 8 else ""
            logger.info(f"🧹 {channel_name}: отброшено {len(dropped)} сегм.: {details}{more}")
        return kept

    @staticmethod
    def _is_repetitive(text: str) -> bool:
        """Зацикленный текст: много слов, но почти все — повторы («Да, да, да...»)."""
        words = re.findall(r"\w+", text.lower())
        return len(words) >= 6 and len(set(words)) / len(words) < 0.34

    @staticmethod
    def _is_sound_tag(text: str) -> bool:
        """
        Whisper описывает звуковые события КАПСОМ («ТЕЛЕФОННЫЙ ЗВОНОК»,
        «ТРЕВОЖНАЯ МУЗЫКА») — реальную речь он капсом не пишет.
        Одиночные аббревиатуры (ЕГРН, ЗОИТ, МФЦ) не трогаем: только фразы 2+ слов.
        """
        words = text.split()
        return len(words) >= 2 and text.upper() == text and any(c.isalpha() for c in text)

    # -------------------------------------------------------------------------
    # Дедупликация
    # -------------------------------------------------------------------------

    @staticmethod
    def _norm_text(text: str) -> str:
        return re.sub(r"[^\wё]+", "", text.lower())

    @classmethod
    def _texts_similar(cls, a: str, b: str) -> bool:
        """
        Порог схожести зависит от длины: короткие фразы — точное совпадение
        или вхождение подстроки (эхо в чужом канале — всегда обрезок исходной
        фразы: «20 соток» внутри «Это 20 соток»), длинные — по коэффициенту.
        """
        na, nb = cls._norm_text(a), cls._norm_text(b)
        if not na or not nb:
            return False
        min_len = min(len(na), len(nb))
        if min_len >= 5 and (na in nb or nb in na):
            return True
        if min_len < 10:
            return na == nb
        ratio = difflib.SequenceMatcher(None, na, nb).ratio()
        return ratio >= (0.85 if min_len < 30 else 0.75)

    def _dedupe_cross_channel(
        self,
        left_segments: List[dict],
        right_segments: List[dict],
        left_profile: List[Tuple[float, float]],
        right_profile: List[Tuple[float, float]],
    ) -> Tuple[List[dict], List[dict]]:
        """
        Межканальные дубли (кросс-ток / эхо): похожий текст с пересекающимся
        временем остаётся ТОЛЬКО у более энергичного канала.

        Эхо звучит ОДНОВРЕМЕННО с источником, поэтому энергии сравниваем в одном
        и том же окне — пересечении таймфреймов пары (собственные таймстемпы
        Whisper неточны на ±2-3 сек). Если энергии сопоставимы — тай-брейк по
        no_speech_prob; иначе оставляем оба (могли реально сказать одно и то же).
        """
        left_segments = sorted(left_segments, key=lambda s: s["start"])
        right_segments = sorted(right_segments, key=lambda s: s["start"])
        drop_left: set = set()
        drop_right: set = set()

        for i, ls in enumerate(left_segments):
            for j, rs in enumerate(right_segments):
                if j in drop_right:
                    continue
                if rs["start"] > ls["end"] + 1.0:
                    break  # правые отсортированы — дальше пересечений не будет
                if ls["start"] > rs["end"] + 1.0:
                    continue
                if not self._texts_similar(ls["text"], rs["text"]):
                    continue

                # Окно одновременности: пересечение таймфреймов пары.
                # Эхо по физике ОДНОВРЕМЕННО с источником. Если пересечения нет
                # (пара сматчилась только за счёт допуска ±1с) — это две реальные
                # реплики в разное время (переспрос), пару НЕ трогаем.
                # Объединение таймфреймов брать нельзя: в широком окне доминирует
                # посторонняя речь и удаляется настоящая реплика (проверено).
                lo = max(ls["start"], rs["start"])
                hi = min(ls["end"], rs["end"])
                if hi <= lo:
                    continue
                le = self._segment_energy(left_profile, lo, hi)
                re_db = self._segment_energy(right_profile, lo, hi)
                delta = le - re_db

                if delta >= CROSSTALK_MARGIN_DB:
                    drop_right.add(j)
                elif delta <= -CROSSTALK_MARGIN_DB:
                    drop_left.add(i)
                    break
                else:
                    # Энергии сопоставимы — решает метрика Whisper «была ли речь»
                    lnsp = ls.get("no_speech_prob", 0)
                    rnsp = rs.get("no_speech_prob", 0)
                    if rnsp > NO_SPEECH_PROB_MAX and lnsp < 0.3:
                        drop_right.add(j)
                    elif lnsp > NO_SPEECH_PROB_MAX and rnsp < 0.3:
                        drop_left.add(i)
                        break

        if drop_left or drop_right:
            logger.info(
                f"🧹 Кросс-ток: убрано дублей — левый {len(drop_left)}, правый {len(drop_right)}"
            )
        return (
            [s for i, s in enumerate(left_segments) if i not in drop_left],
            [s for j, s in enumerate(right_segments) if j not in drop_right],
        )

    def _dedupe_within_channel(self, segments: List[dict], channel_name: str) -> List[dict]:
        """
        Схлопывает смежные повторы одного канала (петли Whisper): точное
        совпадение или длинное (>= 12 симв.) вхождение хвоста предыдущего
        сегмента — артефакт перекрытия окон Whisper.
        """
        result: List[dict] = []
        dropped = 0
        for seg in segments:
            prev = result[-1] if result else None
            if prev is not None and seg["start"] - prev["end"] < 2.0:
                ns, np_ = self._norm_text(seg["text"]), self._norm_text(prev["text"])
                exact = ns == np_
                contained = min(len(ns), len(np_)) >= 12 and (ns in np_ or np_ in ns)
                if exact or contained:
                    prev["end"] = max(prev["end"], seg["end"])
                    dropped += 1
                    continue
            result.append(seg)
        if dropped:
            logger.info(f"🧹 {channel_name}: схлопнуто {dropped} смежных повторов")
        return result

    # -------------------------------------------------------------------------
    # Роли каналов: одно решение на звонок
    # -------------------------------------------------------------------------

    def _score_channels(
        self, left_segments: List[dict], right_segments: List[dict]
    ) -> RoleDecision:
        """
        Детерминированный скоринг «какой канал — менеджер» по маркерам речи.
        Уверенное решение при разрыве >= 3, иначе — на LLM-тайбрейк.
        """
        def score(segs: List[dict]) -> int:
            # Нормализуем пунктуацию: запятые/точки → пробел, схлопываем пробелы
            text = re.sub(r"[^\wё]+", " ", " ".join(s["text"].lower() for s in segs))
            managers = sum(text.count(k) for k in MANAGER_MARKERS)
            clients = sum(text.count(k) for k in CLIENT_MARKERS)
            return managers - clients

        ls, rs = score(left_segments), score(right_segments)
        logger.info(f"🎭 Роль-скоринг каналов: левый={ls}, правый={rs}")
        if ls - rs >= 3:
            return RoleDecision("left", "heuristic", False)
        if rs - ls >= 3:
            return RoleDecision("right", "heuristic", False)
        return RoleDecision(None, "heuristic", True)

    async def _llm_manager_channel(
        self,
        left_segments: List[dict],
        right_segments: List[dict],
        call_direction: str,
    ) -> RoleDecision:
        """
        LLM-тайбрейк: ОДНО решение на звонок — какой канал принадлежит менеджеру.
        Никаких построчных правок: swap канала целиком или ничего. Прошлая схема
        (построчные флипы) на реальном звонке схлопнула все 145 реплик в одну роль.
        """
        merged = sorted(
            [{"ch": "ЛЕВЫЙ", **s} for s in left_segments]
            + [{"ch": "ПРАВЫЙ", **s} for s in right_segments],
            key=lambda s: s["start"],
        )
        if not merged:
            return RoleDecision(None, "llm", True)

        transcript_text = "\n".join(f"[{s['ch']}]: {s['text']}" for s in merged[:120])
        direction_note = (
            "Это ИСХОДЯЩИЙ звонок: менеджер сам набрал клиента."
            if call_direction == "call_out"
            else "Это ВХОДЯЩИЙ звонок: менеджер отвечает и представляется от имени компании."
        )

        prompt = f"""Ты определяешь роли в записи телефонного разговора геодезической компании.
Запись стерео: каждая реплика помечена каналом ЛЕВЫЙ или ПРАВЫЙ.

{direction_note}

Признаки МЕНЕДЖЕРА: говорит от лица компании («мы делаем», «мы не занимаемся»), называет цены и сроки, спрашивает адрес и площадь объекта, смотрит по базе/спутнику.
Признаки КЛИЕНТА: описывает свою задачу/участок/дом, спрашивает «вы занимаетесь...?», «сколько стоит...?», ищет услугу.

Ответь СТРОГО двумя строками:
МЕНЕДЖЕР=ЛЕВЫЙ или МЕНЕДЖЕР=ПРАВЫЙ
УВЕРЕННОСТЬ=ВЫСОКАЯ или УВЕРЕННОСТЬ=НИЗКАЯ

Разговор:
{transcript_text}"""

        try:
            response = await self._openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                # Кириллица токенизится дорого: «МЕНЕДЖЕР=ЛЕВЫЙ\nУВЕРЕННОСТЬ=ВЫСОКАЯ»
                # не влезала в 20 токенов — ответ обрезался и уверенность терялась
                max_tokens=60,
            )
            answer = (response.choices[0].message.content or "").strip().upper().replace(" ", "")
            logger.info(f"🎭 LLM-тайбрейк ролей: {answer!r}")

            if "МЕНЕДЖЕР=ЛЕВЫЙ" in answer:
                channel = "left"
            elif "МЕНЕДЖЕР=ПРАВЫЙ" in answer:
                channel = "right"
            else:
                return RoleDecision(None, "llm", True)

            confident = "УВЕРЕННОСТЬ=ВЫСОКАЯ" in answer
            return RoleDecision(channel, "llm", not confident)

        except Exception as e:
            logger.error(f"❌ LLM-тайбрейк ролей не удался: {e}")
            return RoleDecision(None, "llm", True)

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
            # Тихий фолбэк в моно раньше незаметно отключал диаризацию целиком —
            # теперь орём в Telegram (с троттлингом), но пайплайн не валим.
            logger.error(f"❌ ffprobe не определил каналы: {e} — обрабатываем как моно, диаризация потеряна")
            await self._alert_ffprobe_failure(e)
            return 1

    _ffprobe_alert_last = 0.0

    async def _alert_ffprobe_failure(self, exc: Exception):
        """Алерт о потере диаризации из-за ffprobe, не чаще раза в 30 минут."""
        now = time.time()
        if now - TranscriptionService._ffprobe_alert_last < 1800:
            return
        TranscriptionService._ffprobe_alert_last = now
        try:
            await telegram_service.send_message(
                "⚠️ <b>ffprobe не определил каналы записи</b>\n\n"
                f"Ошибка: {exc}\n\n"
                "Звонок обработан как моно — БЕЗ разделения ролей. Проверьте логи Railway."
            )
        except Exception as tg_err:
            logger.warning(f"⚠️ Не удалось отправить алерт о ffprobe: {tg_err}")

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
