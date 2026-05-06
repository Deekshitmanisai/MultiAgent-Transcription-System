from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from audio_extraction import extract_audio
from pipeline_controller import run_feedback_correction_loop
from services.code_switch_service import normalize_code_switch
from services.subtitle_service import build_srt_from_segments
from services.translation_service import build_multilingual_transcripts
from speaker_diarization import diarize_segments, format_diarized_transcript
from speech_to_text import transcribe_audio_detailed
from validation_agent import ValidationResult


@dataclass
class StageResult:
    ok: bool
    data: Any = None
    error: str = ""
    errors: List[str] = field(default_factory=list)
    validation: ValidationResult = field(default_factory=ValidationResult)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    ok: bool
    raw_transcript: str = ""
    normalized_transcript: str = ""
    speaker_transcript: str = ""
    corrected_transcript: str = ""
    detected_language: str = "unknown"
    original_language: str = "unknown"
    is_mixed: bool = False
    normalization_mode: str = "keep-mixed"
    transcripts: Dict[str, str] = field(default_factory=dict)
    subtitles: Dict[str, str] = field(default_factory=dict)
    timestamped_transcript: str = ""
    segments: List[Dict[str, Any]] = field(default_factory=list)
    validation: ValidationResult = field(default_factory=ValidationResult)
    errors: List[str] = field(default_factory=list)
    audio_path: str = ""
    audio_quality: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def run_transcription_pipeline(
    video_path,
    speaker_count,
    transcription_language="auto",
    domain_mode="meeting",
    normalization_mode="hindi",
) -> PipelineResult:
    if not video_path:
        return PipelineResult(ok=False, errors=["Please upload a video file."])

    audio_stage = _extract_audio_stage(video_path)
    if not audio_stage.ok:
        return PipelineResult(ok=False, errors=[audio_stage.error])

    transcription_stage = _transcription_stage(audio_stage.data, transcription_language=transcription_language)
    if not transcription_stage.ok:
        return PipelineResult(
            ok=False,
            audio_path=audio_stage.data or "",
            errors=[transcription_stage.error],
        )

    transcription_data = transcription_stage.data or {}
    raw_transcript = transcription_data.get("text", "")
    segments = transcription_data.get("segments", [])
    detected_language = transcription_data.get("language", "unknown")
    is_mixed = bool(transcription_data.get("is_mixed"))
    diarization_stage = _diarization_stage(audio_stage.data, segments, speaker_count)
    normalized_transcript = _normalization_stage(
        raw_transcript,
        source_language=detected_language,
        is_mixed=is_mixed,
        normalization_mode=normalization_mode,
    )
    correction_loop = _correction_stage(
        normalized_transcript,
        domain_mode=domain_mode,
        source_language=detected_language,
        is_mixed=is_mixed,
        normalization_mode=normalization_mode,
        original_text=raw_transcript,
    )

    validation = correction_loop.validation
    errors = []
    if not diarization_stage.ok and diarization_stage.error:
        errors.append(diarization_stage.error)
    if correction_loop.errors:
        errors.extend(correction_loop.errors)

    audio_quality = _estimate_audio_quality(audio_stage.data)
    diarization_data = diarization_stage.data or {}
    multilingual_transcripts = _translation_stage(
        correction_loop.data or "",
        source_language=detected_language,
        validation=validation,
        errors=errors,
    )
    subtitles = _subtitle_stage(multilingual_transcripts, diarization_data.get("segments", []))
    return PipelineResult(
        ok=bool(raw_transcript and correction_loop.data),
        raw_transcript=raw_transcript,
        normalized_transcript=correction_loop.meta.get("normalized_text", "") or normalized_transcript,
        speaker_transcript=diarization_data.get("speaker_transcript", ""),
        corrected_transcript=correction_loop.data or "",
        detected_language=detected_language,
        original_language=detected_language,
        is_mixed=is_mixed,
        normalization_mode=normalization_mode,
        transcripts=multilingual_transcripts,
        subtitles=subtitles,
        timestamped_transcript=diarization_data.get("timestamped_transcript", ""),
        segments=diarization_data.get("segments", []),
        validation=validation,
        errors=errors,
        audio_path=audio_stage.data or "",
        audio_quality=audio_quality,
        meta={
            "detectedLanguage": detected_language,
            "originalLanguage": detected_language,
            "isMixed": is_mixed,
            "normalizationMode": normalization_mode,
            "transcriptionLanguage": transcription_language or "auto",
            "domainMode": domain_mode or "meeting",
            "speakerCount": 0 if str(speaker_count).strip().lower() == "auto" else int(speaker_count),
            "feedbackAttempts": correction_loop.meta.get("attempts", 0),
            "feedbackHistory": correction_loop.meta.get("feedback_history", []),
            "transcripts": multilingual_transcripts,
        },
    )


def _extract_audio_stage(video_path) -> StageResult:
    audio_path = extract_audio(video_path)
    if not audio_path:
        return StageResult(ok=False, error="Audio extraction failed.")
    return StageResult(ok=True, data=audio_path)


def _transcription_stage(audio_path, transcription_language="auto") -> StageResult:
    try:
        transcription = transcribe_audio_detailed(audio_path, language=transcription_language)
    except Exception as exc:
        return StageResult(ok=False, error=f"Transcription failed: {exc}")

    raw_transcript = (transcription.get("text") or "").strip()
    if not raw_transcript.strip():
        return StageResult(ok=False, error="Transcription produced no text.")
    return StageResult(ok=True, data=transcription)


def _diarization_stage(audio_path, segments, speaker_count) -> StageResult:
    try:
        normalized_speaker_count = speaker_count
        if str(speaker_count).strip().lower() != "auto":
            normalized_speaker_count = int(speaker_count)
        speaker_segments = diarize_segments(audio_path, segments, n_speakers=normalized_speaker_count)
        speaker_transcript = format_diarized_transcript(speaker_segments)
        timestamped_transcript = _format_timestamped_transcript(speaker_segments)
        serializable_segments = [_serialize_segment(segment) for segment in speaker_segments]
        return StageResult(
            ok=True,
            data={
                "speaker_transcript": speaker_transcript,
                "timestamped_transcript": timestamped_transcript,
                "segments": serializable_segments,
            },
        )
    except Exception as exc:
        return StageResult(ok=False, data={}, error=f"Speaker diarization issue: {exc}")


def _correction_stage(
    transcript,
    domain_mode="meeting",
    source_language="en",
    is_mixed=False,
    normalization_mode="keep-mixed",
    original_text="",
) -> StageResult:
    loop_result = run_feedback_correction_loop(
        transcript,
        domain_mode=domain_mode,
        source_language=source_language,
        is_mixed=is_mixed,
        normalization_mode=normalization_mode,
        original_text=original_text,
    )
    corrected = loop_result.corrected_text

    if not corrected.strip():
        return StageResult(
            ok=False,
            data=transcript,
            error="Transcript cleanup returned empty output; using the uncorrected transcript.",
            meta={
                "attempts": loop_result.attempts,
                "feedback_history": loop_result.feedback_history,
            },
        )
    return StageResult(
        ok=True,
        data=corrected,
        error="; ".join(loop_result.errors),
        meta={
            "attempts": loop_result.attempts,
            "feedback_history": loop_result.feedback_history,
            "normalized_text": loop_result.normalized_text,
        },
        validation=loop_result.validation,
        errors=loop_result.errors,
    )


def _normalization_stage(transcript, source_language="en", is_mixed=False, normalization_mode="hindi"):
    text = (transcript or "").strip()
    if not text:
        return ""
    if source_language == "hi" and is_mixed:
        return normalize_code_switch(text, mode=normalization_mode or "hindi")
    return text


def _translation_stage(corrected_transcript, source_language, validation, errors):
    source = (source_language or "unknown").strip().lower()
    base_text = (corrected_transcript or "").strip()
    transcripts = {source: base_text} if source and source != "unknown" and base_text else {}

    if not base_text:
        return {
            "hindi": "",
            "english": "",
            "tamil": "",
            "telugu": "",
        }

    if source == "hi":
        try:
            translated = build_multilingual_transcripts(base_text, original_language=source, targets=("en", "ta", "te"))
        except Exception as exc:
            errors.append(f"Translation issue: {exc}")
            translated = {"hi": base_text, "en": "", "ta": "", "te": ""}
    else:
        translated = transcripts
        translated.setdefault("en", base_text if source == "en" else "")
        translated.setdefault("hi", base_text if source == "hi" else "")
        translated.setdefault("ta", base_text if source == "ta" else "")
        translated.setdefault("te", base_text if source == "te" else "")

    # Translation must happen after correction and validation. We only fan out the outputs here.
    return {
        "hindi": translated.get("hi", base_text if source == "hi" else ""),
        "english": translated.get("en", base_text if source == "en" else ""),
        "tamil": translated.get("ta", base_text if source == "ta" else ""),
        "telugu": translated.get("te", base_text if source == "te" else ""),
    }


def _subtitle_stage(transcripts, segments):
    hindi_text = (transcripts or {}).get("hindi", "")
    english_text = (transcripts or {}).get("english", "")
    return {
        "hi": build_srt_from_segments(hindi_text, segments),
        "en": build_srt_from_segments(english_text, segments),
    }


def _format_timestamp(seconds):
    total = max(0, int(round(float(seconds or 0.0))))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _format_timestamped_transcript(segments):
    lines = []
    for segment in segments or []:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        speaker = int(segment.get("speaker", 1))
        start = _format_timestamp(segment.get("start", 0.0))
        end = _format_timestamp(segment.get("end", segment.get("start", 0.0)))
        lines.append(f"[{start} - {end}] Person {speaker}: {text}")
    return "\n".join(lines).strip()


def _serialize_segment(segment):
    return {
        "start": float(segment.get("start", 0.0)),
        "end": float(segment.get("end", segment.get("start", 0.0))),
        "text": (segment.get("text") or "").strip(),
        "speaker": int(segment.get("speaker", 1)),
    }


def _estimate_audio_quality(audio_path):
    try:
        import librosa
        import numpy as np
    except Exception:
        return {
            "label": "unknown",
            "score": 0,
            "summary": "Audio quality analysis unavailable.",
        }

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        return {
            "label": "unknown",
            "score": 0,
            "summary": "Audio quality analysis failed.",
        }

    if y.size == 0:
        return {
            "label": "poor",
            "score": 0,
            "summary": "Audio appears empty.",
        }

    rms = librosa.feature.rms(y=y).flatten()
    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    rms_std = float(np.std(rms)) if rms.size else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    voiced_ratio = float(np.mean(np.abs(y) > 0.01)) if y.size else 0.0

    score = 100.0
    score -= max(0.0, 0.02 - rms_mean) * 2200.0
    score -= max(0.0, 0.20 - peak) * 180.0
    score -= max(0.0, 0.12 - voiced_ratio) * 320.0
    score -= min(rms_std, 0.1) * 120.0
    score = int(max(0, min(100, round(score))))

    if score >= 75:
        label = "good"
        summary = "Audio quality looks strong for transcription."
    elif score >= 45:
        label = "moderate"
        summary = "Audio quality is usable, but noise or level changes may affect accuracy."
    else:
        label = "poor"
        summary = "Audio quality may hurt transcript accuracy."

    return {
        "label": label,
        "score": score,
        "summary": summary,
        "rmsMean": round(rms_mean, 5),
        "voicedRatio": round(voiced_ratio, 4),
    }
