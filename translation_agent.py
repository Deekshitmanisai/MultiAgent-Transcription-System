import re

from gemini_service import generate_text


LANGUAGE_SPECS = {
    "en": {"name": "English", "script": "Latin"},
    "hi": {"name": "Hindi", "script": "Devanagari"},
    "ta": {"name": "Tamil", "script": "Tamil"},
    "te": {"name": "Telugu", "script": "Telugu"},
    "kn": {"name": "Kannada", "script": "Kannada"},
}


def _extract_script_text(text, lang_code):
    t = text or ""
    spec = LANGUAGE_SPECS.get(lang_code, LANGUAGE_SPECS["hi"])
    ranges = {
        "Devanagari": [(0x0900, 0x097F)],
        "Tamil": [(0x0B80, 0x0BFF)],
        "Telugu": [(0x0C00, 0x0C7F)],
        "Kannada": [(0x0C80, 0x0CFF)],
        "Latin": [(0x0041, 0x005A), (0x0061, 0x007A)],
    }
    allowed_ranges = ranges.get(spec["script"], [])
    kept = []
    for ch in t:
        code = ord(ch)
        if any(start <= code <= end for start, end in allowed_ranges):
            kept.append(ch)
        elif kept and (ch.isdigit() or ch in (" ", "\n", "\t", ".", ",", "!", "?", "-", ":", ";", "(", ")", "'", "\"")):
            kept.append(ch)
    cleaned = "".join(kept)
    cleaned = "\n".join([line.strip() for line in cleaned.splitlines() if line.strip()])
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\s+[.,:;!?-]+$", "", cleaned)
    return cleaned.strip()


class TranslationAgent:
    def translate(self, text, source_language="auto", target_language="hi"):
        source_text = (text or "").strip()
        if not source_text:
            return ""

        spec = LANGUAGE_SPECS.get(target_language, LANGUAGE_SPECS["hi"])
        source_spec = LANGUAGE_SPECS.get(source_language, {"name": "the source language"})
        prompt = f"""
Translate the following text from {source_spec["name"]} into {spec["name"]}.

Rules:
- Output only {spec["script"]} script.
- Do not add explanations, headers, or notes.
- Preserve speaker labels such as Person 1, Person 2, etc.
- Keep mixed-language proper nouns naturally when translation would sound unnatural.
- Keep the meaning faithful and natural.

Text:
{source_text}
""".strip()

        try:
            response_text = generate_text(
                prompt,
                system_instruction=f"You are a precise {spec['name']} translation assistant.",
                temperature=0.2,
                max_output_tokens=3072,
            )
        except Exception as e:
            print(f"Gemini translation issue ({spec['name']}): {e}")
            return ""

        return _extract_script_text(response_text, target_language)


def translate_text(text, source_language="auto", target_language="hi"):
    agent = TranslationAgent()
    return agent.translate(text, source_language=source_language, target_language=target_language)


def translate_to_hindi(text):
    return translate_text(text, target_language="hi")
