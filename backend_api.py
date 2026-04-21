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
from simplification_agent import simplify_transcript
from translation_agent import LANGUAGE_SPECS, translate_text
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
    corrected_transcript: str
    target_language: str = "hi"
    summary_style: str = "concise"
    domain_mode: str = DEFAULT_DOMAIN_MODE


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


class RefineRequest(TranscriptRequest):
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


def _public_audio_url(audio_path: str) -> str:
    path = Path(audio_path)
    return f"/api/media/audio/{path.name}"


def _serialize_validation(result):
    return {
        "isValid": result.is_valid,
        "verdict": result.verdict,
        "confidenceScore": result.confidence_score,
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


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    speaker_count: str = Form("auto"),
    transcription_language: str = Form(DEFAULT_TRANSCRIPTION_LANGUAGE),
    domain_mode: str = Form(DEFAULT_DOMAIN_MODE),
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
        )
        return {
            "ok": result.ok,
            "rawTranscript": result.raw_transcript,
            "speakerTranscript": result.speaker_transcript,
            "correctedTranscript": result.corrected_transcript,
            "timestampedTranscript": result.timestamped_transcript,
            "segments": result.segments,
            "validation": _serialize_validation(result.validation),
            "errors": result.errors,
            "audioQuality": result.audio_quality,
            "meta": {
                "speakerCount": speaker_count,
                "sourceFilename": file.filename,
                "transcriptionLanguage": transcription_language,
                "domainMode": domain_mode,
            },
        }
    finally:
        await file.close()


@app.post("/api/translate")
def translate(payload: TranscriptRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured on the backend. Set GEMINI_API_KEY and restart the backend.",
        )

    transcript = (payload.corrected_transcript or "").strip()
    target_language = (payload.target_language or "hi").strip().lower()
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided for translation.")
    if target_language not in LANGUAGE_SPECS:
        raise HTTPException(status_code=400, detail="Unsupported translation language.")

    translated_text = translate_text(transcript, target_language=target_language)
    if not translated_text:
        raise HTTPException(
            status_code=502,
            detail="Translation failed. Check Gemini configuration and backend logs.",
        )

    return {
        "ok": True,
        "translatedText": translated_text,
        "targetLanguage": target_language,
        "message": "Translation ready.",
    }


@app.post("/api/validate")
def validate(payload: TranscriptRequest):
    transcript = (payload.corrected_transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided for validation.")

    validation = validate_transcript_detailed(transcript, domain_mode=payload.domain_mode)
    return {
        "ok": validation.is_valid,
        "validation": _serialize_validation(validation),
    }


@app.post("/api/refine")
def refine(payload: RefineRequest):
    transcript = (payload.corrected_transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided for refinement.")
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured on the backend. Set GEMINI_API_KEY and restart the backend.",
        )

    baseline_validation = validate_transcript_detailed(transcript, domain_mode=payload.domain_mode)
    loop_result = run_feedback_correction_loop(transcript, domain_mode=payload.domain_mode, max_retries=1)
    refined_text = (loop_result.corrected_text or "").strip() or transcript
    validation = loop_result.validation or validate_transcript_detailed(refined_text, domain_mode=payload.domain_mode)

    if validation.confidence_score < baseline_validation.confidence_score:
        return {
            "ok": baseline_validation.is_valid,
            "correctedTranscript": transcript,
            "validation": _serialize_validation(baseline_validation),
            "message": "Refinement did not improve the transcript, so the previous version was kept.",
        }

    return {
        "ok": validation.is_valid,
        "correctedTranscript": refined_text,
        "validation": _serialize_validation(validation),
        "message": "Transcript refined with validation feedback.",
    }


@app.post("/api/minutes")
def minutes(payload: TranscriptRequest):
    transcript = (payload.corrected_transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided for meeting notes.")

    mom, key_points, decisions, action_items = generate_minutes_of_meeting(
        transcript,
        summary_style=payload.summary_style,
        domain_mode=payload.domain_mode,
    )
    return {
        "ok": True,
        "minutes": mom,
        "keyPoints": key_points,
        "decisions": decisions,
        "actionItems": action_items,
    }


@app.post("/api/simplify")
def simplify(payload: TranscriptRequest):
    transcript = (payload.corrected_transcript or "").strip()
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
