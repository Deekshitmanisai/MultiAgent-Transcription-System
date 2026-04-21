from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from gemini_service import generate_text
from speaker_diarization import _extract_segment_features, _turn_taking_speakers
from speech_to_text import SpeechToTextAgent


@dataclass
class LiveChunkResult:
    english: str
    timestamp: str
    start_seconds: float
    end_seconds: float


class OnlineSpeakerTracker:
    def __init__(self, *, sample_rate=16000, max_speakers=6, distance_threshold=115.0):
        self.sample_rate = sample_rate
        self.max_speakers = max_speakers
        self.distance_threshold = float(distance_threshold)
        self._profiles = []

    def assign(self, audio_float32, segments):
        if not segments:
            return []

        try:
            import librosa
        except Exception:
            return _turn_taking_speakers(segments)

        assigned_segments = []
        last_end = None

        for segment in segments:
            enriched = dict(segment)
            start = float(segment.get("start", 0.0))
            pause_before = max(0.0, start - last_end) if last_end is not None else 0.0
            enriched["_pause_before"] = pause_before
            last_end = float(segment.get("end", start))

            feature_vector = _extract_segment_features(audio_float32, self.sample_rate, enriched, librosa)
            speaker_id = self._match_speaker(feature_vector) if feature_vector is not None else None
            if speaker_id is None:
                enriched["speaker"] = int(segment.get("speaker", 1))
            else:
                enriched["speaker"] = speaker_id
            assigned_segments.append(enriched)

        if len({segment["speaker"] for segment in assigned_segments}) == 1 and len(assigned_segments) >= 2:
            return _turn_taking_speakers(segments)

        return assigned_segments

    def _match_speaker(self, feature_vector):
        if feature_vector is None:
            return None

        if not self._profiles:
            self._profiles.append({"speaker_id": 1, "centroid": feature_vector.copy(), "count": 1})
            return 1

        distances = []
        for profile in self._profiles:
            centroid = profile["centroid"]
            distance = float(np.linalg.norm(feature_vector - centroid))
            distances.append((distance, profile))

        best_distance, best_profile = min(distances, key=lambda item: item[0])
        if best_distance <= self.distance_threshold:
            count = best_profile["count"]
            best_profile["centroid"] = (best_profile["centroid"] * count + feature_vector) / float(count + 1)
            best_profile["count"] = count + 1
            return int(best_profile["speaker_id"])

        if len(self._profiles) < self.max_speakers:
            new_speaker_id = len(self._profiles) + 1
            self._profiles.append(
                {
                    "speaker_id": new_speaker_id,
                    "centroid": feature_vector.copy(),
                    "count": 1,
                }
            )
            return int(new_speaker_id)

        count = best_profile["count"]
        best_profile["centroid"] = (best_profile["centroid"] * count + feature_vector) / float(count + 1)
        best_profile["count"] = count + 1
        return int(best_profile["speaker_id"])


def _format_speaker_transcript(segments):
    lines = []
    for segment in segments or []:
        text = str(segment.get("text", "") or "").strip()
        if not text:
            continue
        speaker = int(segment.get("speaker", 1))
        lines.append(f"Person {speaker}: {text}")
    return "\n".join(lines).strip()


class LiveChunkProcessor:
    def __init__(self, *, language="auto", model_size=None):
        self.language = language
        self.agent = SpeechToTextAgent(model_size=model_size) if model_size else SpeechToTextAgent()
        self.context = deque(maxlen=2)
        self.collected_english = []
        self.speaker_tracker = OnlineSpeakerTracker()

    async def process_chunk(self, chunk):
        import asyncio

        return await asyncio.to_thread(self._process_chunk_sync, chunk)

    def _process_chunk_sync(self, chunk):
        language = None if not self.language or self.language == "auto" else self.language
        result = self.agent.model.transcribe(chunk.audio_float32, language=language, fp16=False)
        raw_segments = result.get("segments", []) or []
        if not raw_segments:
            return None

        speaker_segments = self.speaker_tracker.assign(chunk.audio_float32, raw_segments)
        speaker_chunk_text = _format_speaker_transcript(speaker_segments)
        if not speaker_chunk_text:
            return None

        corrected = self._correct_with_context(speaker_chunk_text)

        self.context.append(corrected)
        self.collected_english.append(corrected)

        return LiveChunkResult(
            english=corrected,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            start_seconds=chunk.start_seconds,
            end_seconds=chunk.end_seconds,
        )

    def _correct_with_context(self, raw_text):
        previous_context = "\n".join(self.context).strip()
        prompt = f"""
You are correcting one live meeting transcript chunk.

Correct grammar and improve clarity while preserving meaning.
Maintain continuity with previous context.
Do not summarize.
Do not add new facts.
Keep the wording natural and concise.
Return only the corrected English text for the current chunk.
Preserve speaker labels exactly as Person 1, Person 2, etc. when they are present.

Previous context from the last 2 chunks:
{previous_context or "No prior context available."}

Current raw transcript chunk:
{raw_text}
""".strip()

        try:
            corrected = generate_text(
                prompt,
                system_instruction="You are a real-time transcript correction assistant.",
                temperature=0.1,
                max_output_tokens=512,
            )
            return corrected.strip() or raw_text
        except Exception as exc:
            print(f"Live correction issue: {exc}")
            return raw_text

    async def generate_summary(self):
        import asyncio

        return await asyncio.to_thread(self._generate_summary_sync)

    def _generate_summary_sync(self):
        transcript = "\n".join(self.collected_english).strip()
        if not transcript:
            return ""

        prompt = f"""
Summarize the following meeting transcript into key points, decisions, and action items.

Return a clean bullet-point summary with these sections:
- Key Points
- Decisions
- Action Items

Transcript:
{transcript}
""".strip()

        try:
            return generate_text(
                prompt,
                system_instruction="You are a meeting summarization assistant.",
                temperature=0.2,
                max_output_tokens=1024,
            ).strip()
        except Exception as exc:
            print(f"Live summary issue: {exc}")
            return ""
