from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from audio_extraction import extract_audio
from pipeline_controller import run_feedback_correction_loop
from speaker_diarization import diarize_segments, format_diarized_transcript
from speech_to_text import transcribe_audio_with_segments
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
    speaker_transcript: str = ""
    corrected_transcript: str = ""
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

    raw_transcript, segments = transcription_stage.data
    diarization_stage = _diarization_stage(audio_stage.data, segments, speaker_count)
    transcript_for_correction = diarization_stage.data.get("speaker_transcript") if diarization_stage.ok else raw_transcript
    correction_loop = _correction_stage(transcript_for_correction or raw_transcript, domain_mode=domain_mode)

    validation = correction_loop.validation
    errors = []
    if not diarization_stage.ok and diarization_stage.error:
        errors.append(diarization_stage.error)
    if correction_loop.errors:
        errors.extend(correction_loop.errors)

    audio_quality = _estimate_audio_quality(audio_stage.data)
    diarization_data = diarization_stage.data or {}
    return PipelineResult(
        ok=bool(raw_transcript and correction_loop.data),
        raw_transcript=raw_transcript,
        speaker_transcript=diarization_data.get("speaker_transcript", ""),
        corrected_transcript=correction_loop.data or "",
        timestamped_transcript=diarization_data.get("timestamped_transcript", ""),
        segments=diarization_data.get("segments", []),
        validation=validation,
        errors=errors,
        audio_path=audio_stage.data or "",
        audio_quality=audio_quality,
        meta={
            "transcriptionLanguage": transcription_language or "auto",
            "domainMode": domain_mode or "meeting",
            "speakerCount": 0 if str(speaker_count).strip().lower() == "auto" else int(speaker_count),
            "feedbackAttempts": correction_loop.meta.get("attempts", 0),
            "feedbackHistory": correction_loop.meta.get("feedback_history", []),
        },
    )


def _extract_audio_stage(video_path) -> StageResult:
    audio_path = extract_audio(video_path)
    if not audio_path:
        return StageResult(ok=False, error="Audio extraction failed.")
    return StageResult(ok=True, data=audio_path)


def _transcription_stage(audio_path, transcription_language="auto") -> StageResult:
    try:
        raw_transcript, segments = transcribe_audio_with_segments(audio_path, language=transcription_language)
    except Exception as exc:
        return StageResult(ok=False, error=f"Transcription failed: {exc}")

    if not raw_transcript.strip():
        return StageResult(ok=False, error="Transcription produced no text.")
    return StageResult(ok=True, data=(raw_transcript, segments))


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


def _correction_stage(transcript, domain_mode="meeting") -> StageResult:
    loop_result = run_feedback_correction_loop(transcript, domain_mode=domain_mode)
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
        },
        validation=loop_result.validation,
        errors=loop_result.errors,
    )


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
