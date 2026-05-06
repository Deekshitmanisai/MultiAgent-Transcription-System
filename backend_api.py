import mimetypes
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import (
    DEFAULT_DOMAIN_MODE,
    DEFAULT_TRANSCRIPTION_LANGUAGE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_VALIDATION_MODEL,
    TEMP_DIR,
    TTS_OUTPUT_DIR,
)
from mom_agent import generate_minutes_of_meeting
from pipeline import run_transcription_pipeline
from pipeline_controller import run_feedback_correction_loop
from services.translation_service import build_multilingual_transcripts, translate_transcript
from simplification_agent import simplify_transcript
from translation_agent import LANGUAGE_SPECS
from tts_agent import text_to_speech_file
from validation_agent import validate_transcript_detailed
from websocket import register_live_routes


TEMP_DIR.mkdir(exist_ok=True)
TTS_OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Adaptive Hybrid AI Transcription System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptRequest(BaseModel):
    corrected_transcript: str = ""
    raw_transcript: str = ""
    text: str = ""
    source_language: str = "en"
    source_lang: str = "en"
    target_language: str = "hi"
    target_lang: str = "hi"
    normalization_mode: str = "hindi"
    is_mixed: bool = False
    summary_style: str = "concise"
    domain_mode: str = DEFAULT_DOMAIN_MODE


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


class RefineRequest(TranscriptRequest):
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "hi"
    target_lang: str = "en"


def _public_audio_url(audio_path: str) -> str:
    path = Path(audio_path)
    return f"/api/media/audio/{path.name}"


def _serialize_validation(result):
    return {
        "isValid": result.is_valid,
        "verdict": result.verdict,
        "verdictDisplay": str(result.verdict or "").upper(),
        "confidenceScore": result.confidence_score,
        "scores": result.scores,
        "total": result.total,
        "summary": result.summary,
        "issues": result.issues,
        "strengths": result.strengths,
        "suggestedActions": result.suggested_actions,
        "suggestions": result.suggested_actions,
        "improvementCategories": result.improvement_categories,
        "editorFeedback": result.editor_feedback,
        "validator": result.validator,
        "metricScores": result.metric_scores,
        "criticalIssues": result.critical_issues,
    }


def _resolve_transcript_text(payload: TranscriptRequest) -> str:
    return (payload.corrected_transcript or payload.text or "").strip()


def _resolve_source_language(payload: TranscriptRequest) -> str:
    return (payload.source_language or payload.source_lang or "en").strip().lower()


def _database_record(result):
    return {
        "original_language": result.original_language,
        "is_mixed": result.is_mixed,
        "normalization_mode": result.normalization_mode,
        "raw_transcript": result.raw_transcript,
        "normalized_transcript": result.normalized_transcript,
        "corrected_transcript": result.corrected_transcript,
        "transcripts": result.transcripts,
    }


@app.get("/api/health")
def health():
    try:
        import sounddevice  # noqa: F401

        live_mode_available = True
    except Exception:
        live_mode_available = False

    return {
        "ok": True,
        "service": "adaptive-hybrid-ai-transcription-system",
        "geminiConfigured": bool(GEMINI_API_KEY),
        "geminiModel": GEMINI_MODEL,
        "openRouterConfigured": bool(OPENROUTER_API_KEY),
        "openRouterValidationModel": OPENROUTER_VALIDATION_MODEL,
        "defaultLanguage": DEFAULT_TRANSCRIPTION_LANGUAGE,
        "defaultDomainMode": DEFAULT_DOMAIN_MODE,
        "liveModeAvailable": live_mode_available,
    }


async def _process_upload(
    file: UploadFile = File(...),
    speaker_count: str = Form("auto"),
    transcription_language: str = Form(DEFAULT_TRANSCRIPTION_LANGUAGE),
    domain_mode: str = Form(DEFAULT_DOMAIN_MODE),
    normalization_mode: str = Form("hindi"),
):
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    upload_path = TEMP_DIR / f"upload_{uuid.uuid4().hex}{suffix}"

    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = run_transcription_pipeline(
            str(upload_path),
            speaker_count,
            transcription_language=transcription_language,
            domain_mode=domain_mode,
            normalization_mode=normalization_mode,
        )
        return {
            "ok": result.ok,
            "language": result.detected_language,
            "original_language": result.original_language,
            "is_mixed": result.is_mixed,
            "normalization_mode": result.normalization_mode,
            "raw_transcript": result.raw_transcript,
            "normalized_transcript": result.normalized_transcript,
            "corrected_transcript": result.corrected_transcript,
            "isMixed": result.is_mixed,
            "rawTranscript": result.raw_transcript,
            "normalizedTranscript": result.normalized_transcript,
            "speakerTranscript": result.speaker_transcript,
            "correctedTranscript": result.corrected_transcript,
            "detectedLanguage": result.detected_language,
            "originalLanguage": result.original_language,
            "transcripts": result.transcripts,
            "subtitles": result.subtitles,
            "timestampedTranscript": result.timestamped_transcript,
            "segments": result.segments,
            "validation": _serialize_validation(result.validation),
            "errors": result.errors,
            "audioQuality": result.audio_quality,
            "databaseRecord": _database_record(result),
            "meta": {
                "speakerCount": speaker_count,
                "sourceFilename": file.filename,
                "transcriptionLanguage": transcription_language,
                "domainMode": domain_mode,
                "detectedLanguage": result.detected_language,
                "originalLanguage": result.original_language,
                "isMixed": result.is_mixed,
                "normalizationMode": result.normalization_mode,
            },
        }
    finally:
        await file.close()


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    speaker_count: str = Form("auto"),
    transcription_language: str = Form(DEFAULT_TRANSCRIPTION_LANGUAGE),
    domain_mode: str = Form(DEFAULT_DOMAIN_MODE),
    normalization_mode: str = Form("hindi"),
):
    return await _process_upload(file, speaker_count, transcription_language, domain_mode, normalization_mode)


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    speaker_count: str = Form("auto"),
    transcription_language: str = Form(DEFAULT_TRANSCRIPTION_LANGUAGE),
    domain_mode: str = Form(DEFAULT_DOMAIN_MODE),
    normalization_mode: str = Form("hindi"),
):
    return await _process_upload(file, speaker_count, transcription_language, domain_mode, normalization_mode)


@app.post("/api/translate")
def translate(payload: TranslationRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured on the backend. Set GEMINI_API_KEY and restart the backend.",
        )

    transcript = (payload.text or "").strip()
    source_language = (payload.source_lang or "hi").strip().lower()
    target_language = (payload.target_lang or "en").strip().lower()
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided for translation.")
    if target_language not in LANGUAGE_SPECS:
        raise HTTPException(status_code=400, detail="Unsupported translation language.")

    translated_text = translate_transcript(
        transcript,
        source_lang=source_language,
        target_lang=target_language,
    )
    if not translated_text:
        raise HTTPException(
            status_code=502,
            detail="Translation failed. Check Gemini configuration and backend logs.",
        )

    return {
        "ok": True,
        "translatedText": translated_text,
        "sourceLanguage": source_language,
        "targetLanguage": target_language,
        "message": "Translation ready.",
    }


@app.post("/api/validate")
def validate(payload: TranscriptRequest):
    transcript = _resolve_transcript_text(payload)
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided for validation.")

    validation = validate_transcript_detailed(
        transcript,
        domain_mode=payload.domain_mode,
        source_language=_resolve_source_language(payload),
        normalization_mode=payload.normalization_mode,
        is_mixed=payload.is_mixed,
        original_text=payload.raw_transcript,
    )
    return {
        "ok": validation.is_valid,
        "validation": _serialize_validation(validation),
    }


@app.post("/api/refine")
def refine(payload: RefineRequest):
    transcript = _resolve_transcript_text(payload)
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided for refinement.")
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured on the backend. Set GEMINI_API_KEY and restart the backend.",
        )

    source_language = _resolve_source_language(payload)
    baseline_validation = validate_transcript_detailed(
        transcript,
        domain_mode=payload.domain_mode,
        source_language=source_language,
        normalization_mode=payload.normalization_mode,
        is_mixed=payload.is_mixed,
        original_text=payload.raw_transcript or transcript,
    )
    loop_result = run_feedback_correction_loop(
        transcript,
        domain_mode=payload.domain_mode,
        max_retries=1,
        source_language=source_language,
        is_mixed=payload.is_mixed,
        normalization_mode=payload.normalization_mode,
        original_text=payload.raw_transcript or transcript,
    )
    refined_text = (loop_result.corrected_text or "").strip() or transcript
    validation = loop_result.validation or validate_transcript_detailed(
        refined_text,
        domain_mode=payload.domain_mode,
        source_language=source_language,
        normalization_mode=payload.normalization_mode,
        is_mixed=payload.is_mixed,
        original_text=payload.raw_transcript or transcript,
    )

    if validation.confidence_score < baseline_validation.confidence_score:
        return {
            "ok": baseline_validation.is_valid,
            "normalizedTranscript": loop_result.normalized_text or transcript,
            "correctedTranscript": transcript,
            "validation": _serialize_validation(baseline_validation),
            "message": "Refinement did not improve the transcript, so the previous version was kept.",
        }

    return {
        "ok": validation.is_valid,
        "normalizedTranscript": loop_result.normalized_text or transcript,
        "correctedTranscript": refined_text,
        "validation": _serialize_validation(validation),
        "message": "Transcript refined with validation feedback.",
    }


@app.post("/api/minutes")
def minutes(payload: TranscriptRequest):
    transcript = _resolve_transcript_text(payload)
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided for meeting notes.")

    source_language = _resolve_source_language(payload)
    result = generate_minutes_of_meeting(
        transcript,
        summary_style=payload.summary_style,
        domain_mode=payload.domain_mode,
        source_language=source_language,
    )
    try:
        multilingual_transcripts = build_multilingual_transcripts(transcript, original_language=source_language)
    except Exception:
        multilingual_transcripts = {"en": transcript if source_language == "en" else ""}
    english_source = multilingual_transcripts.get("en", "")
    english_minutes = (
        generate_minutes_of_meeting(
            english_source,
            summary_style=payload.summary_style,
            domain_mode=payload.domain_mode,
            source_language="en",
        ).to_dict()
        if english_source
        else {}
    )

    payload_data = result.to_dict()
    source_language_minutes = dict(payload_data)
    payload_data.update(
        {
            "ok": True,
            "correctedTranscript": transcript,
            "minutesByLanguage": {
                source_language: source_language_minutes,
                "en": english_minutes,
            },
            "englishMinutes": english_minutes,
        }
    )
    return payload_data


@app.post("/api/simplify")
def simplify(payload: TranscriptRequest):
    transcript = _resolve_transcript_text(payload)
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided for simplification.")
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured on the backend. Set GEMINI_API_KEY and restart the backend.",
        )

    simplified_text = simplify_transcript(transcript, domain_mode=payload.domain_mode)
    if not simplified_text.strip():
        raise HTTPException(status_code=502, detail="Simplified explanation generation failed.")

    return {
        "ok": True,
        "simplifiedText": simplified_text,
        "message": "Simplified explanation ready.",
    }


@app.post("/api/tts")
def text_to_speech(payload: TTSRequest):
    text = (payload.text or "").strip()
    lang = (payload.lang or "en").strip() or "en"

    if not text:
        raise HTTPException(status_code=400, detail="No text provided for text-to-speech.")

    audio_path = text_to_speech_file(text, lang=lang)
    if not audio_path:
        raise HTTPException(status_code=502, detail="Text-to-speech generation failed.")

    return {
        "ok": True,
        "audioUrl": _public_audio_url(audio_path),
        "language": lang,
    }


@app.get("/api/media/audio/{filename}")
def get_audio(filename: str):
    file_path = TTS_OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    media_type, _ = mimetypes.guess_type(file_path.name)
    return FileResponse(file_path, media_type=media_type or "audio/mpeg", filename=file_path.name)


register_live_routes(app)
