from typing import Dict

from translation_agent import LANGUAGE_SPECS, translate_text


DEFAULT_TRANSLATION_TARGETS = ("en", "ta", "te")


def normalize_language_code(lang_code: str, fallback: str = "en") -> str:
    value = str(lang_code or fallback).strip().lower()
    return value if value in LANGUAGE_SPECS else fallback


def translate_transcript(text: str, source_lang: str, target_lang: str) -> str:
    source = normalize_language_code(source_lang, fallback="auto")
    target = normalize_language_code(target_lang)
    if not (text or "").strip():
        return ""
    if source == target:
        return (text or "").strip()
    return translate_text(text, source_language=source, target_language=target)


def build_multilingual_transcripts(
    corrected_text: str,
    original_language: str,
    targets=DEFAULT_TRANSLATION_TARGETS,
) -> Dict[str, str]:
    base_text = (corrected_text or "").strip()
    source = normalize_language_code(original_language, fallback="en")
    transcripts = {source: base_text} if base_text else {}

    for target in targets:
        normalized_target = normalize_language_code(target)
        if normalized_target in transcripts:
            continue
        transcripts[normalized_target] = translate_transcript(
            base_text,
            source_lang=source,
            target_lang=normalized_target,
        )

    for required_lang in ("hi", "en", "ta", "te"):
        transcripts.setdefault(required_lang, "")

    return transcripts
