import re

from gemini_service import generate_text


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")


def detect_code_switch(text: str) -> dict:
    content = str(text or "").strip()
    devanagari_count = len(DEVANAGARI_RE.findall(content))
    latin_tokens = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", content)
    latin_count = len(latin_tokens)
    is_mixed = devanagari_count > 0 and latin_count > 0
    return {
        "is_mixed": is_mixed,
        "devanagari_count": devanagari_count,
        "latin_token_count": latin_count,
        "latin_tokens": latin_tokens[:50],
    }


def normalize_code_switch(text: str, mode: str = "hindi") -> str:
    source_text = str(text or "").strip()
    normalization_mode = (mode or "hindi").strip().lower()
    if not source_text:
        return ""
    if normalization_mode not in {"hindi", "english", "keep-mixed"}:
        normalization_mode = "hindi"

    if normalization_mode == "keep-mixed":
        return source_text

    target_language = "Hindi" if normalization_mode == "hindi" else "English"
    script_instruction = (
        "Return the output primarily in Hindi written in Devanagari script."
        if normalization_mode == "hindi"
        else "Return the output in natural English."
    )

    prompt = f"""
You normalize Hindi-English code-switched transcripts.

Task:
- Detect mixed Hindi-English wording.
- Normalize the transcript into consistent {target_language}.
- Preserve meaning, tone, intent, and speaker flow.
- Avoid literal translation.
- Keep natural proper nouns, product names, and domain terms when translating them would sound unnatural.
- Preserve speaker labels like Person 1, Person 2 exactly.
- Return only the normalized transcript.

Rules:
- Do not summarize.
- Do not remove content.
- Do not add explanation.
- {script_instruction}

Transcript:
{source_text}
""".strip()

    try:
        normalized = generate_text(
            prompt,
            system_instruction="You are an expert code-switch normalization editor for spoken transcripts.",
            temperature=0.1,
            max_output_tokens=4096,
        )
        cleaned = str(normalized or "").strip()
        return cleaned or source_text
    except Exception as exc:
        print(f"Code-switch normalization issue: {exc}")
        return source_text
