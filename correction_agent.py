import re
from dataclasses import dataclass

from gemini_service import generate_text


def _basic_cleanup(text):
    cleaned_lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _post_process(text, fallback):
    cleaned = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip().lstrip("> *-").strip()
        if not line:
            cleaned.append("")
            continue
        if line.lower().startswith(("here is", "corrected transcript", "transcript:")):
            continue
        cleaned.append(line)

    final_text = "\n".join(cleaned).strip()
    return final_text if final_text else fallback


def _speaker_label_count(text):
    return len(re.findall(r"(?im)^\s*Person\s+\d+\s*:", text or ""))


def _looks_incomplete(candidate, source):
    candidate = (candidate or "").strip()
    source = (source or "").strip()
    if not candidate:
        return True

    source_labels = _speaker_label_count(source)
    candidate_labels = _speaker_label_count(candidate)
    if source_labels >= 4 and candidate_labels < max(1, int(source_labels * 0.75)):
        return True

    if len(source) >= 400 and len(candidate) < int(len(source) * 0.65):
        return True

    if len(candidate) >= 80 and candidate[-1] not in ".?!\"')":
        return True

    return False


def _split_transcript_chunks(transcript, max_lines=4, max_chars=900):
    lines = [line.strip() for line in (transcript or "").splitlines() if line.strip()]
    if not lines:
        return []

    chunks = []
    current = []
    current_chars = 0

    for line in lines:
        line_len = len(line)
        should_flush = current and (
            len(current) >= max_lines or current_chars + line_len > max_chars
        )
        if should_flush:
            chunks.append("\n".join(current))
            current = []
            current_chars = 0

        current.append(line)
        current_chars += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


@dataclass
class CorrectionResult:
    text: str
    used_gemini: bool = False
    fallback_reason: str = ""


class CorrectionAgent:
    """
    Fast transcript cleanup using Gemini.
    """

    def correct_text(self, transcript, domain_mode="meeting", feedback=None, source_language="en"):
        return self.correct_text_with_result(
            transcript,
            domain_mode=domain_mode,
            feedback=feedback,
            source_language=source_language,
        ).text

    def correct_text_with_result(self, transcript, domain_mode="meeting", feedback=None, source_language="en"):
        cleaned_input = _basic_cleanup(transcript)
        if not cleaned_input:
            return CorrectionResult(text="")

        domain_guidance = {
            "meeting": "Keep the tone professional and natural for workplace meeting notes.",
            "lecture": "Preserve explanatory flow and technical teaching language.",
            "interview": "Preserve question-and-answer structure clearly.",
            "discussion": "Preserve conversational back-and-forth naturally.",
        }.get((domain_mode or "meeting").strip().lower(), "Preserve the original context and tone.")

        def format_feedback():
            if not feedback:
                return ""

            issues = [str(item).strip() for item in feedback.get("issues", []) if str(item).strip()]
            suggestions = [
                str(item).strip()
                for item in feedback.get("suggestions", feedback.get("suggested_actions", []))
                if str(item).strip()
            ]
            improvement_categories = [
                str(item).strip()
                for item in feedback.get("improvement_categories", [])
                if str(item).strip()
            ]
            editor_feedback = str(feedback.get("editor_feedback", "") or "").strip()
            summary = str(feedback.get("summary", "") or "").strip()
            verdict = str(feedback.get("verdict", "") or "").strip()
            score = feedback.get("score")
            metric_scores = feedback.get("metric_scores", {}) or {}

            if (
                not issues
                and not suggestions
                and not improvement_categories
                and not editor_feedback
                and not summary
                and not verdict
                and score in (None, "")
                and not metric_scores
            ):
                return ""

            lines = [
                "VALIDATION FEEDBACK TO ADDRESS:",
                "Apply this feedback while preserving the original meaning and all speaker turns.",
            ]
            if summary:
                lines.append(f"Validation summary: {summary}")
            if verdict:
                lines.append(f"Validation verdict: {verdict}")
            if score not in (None, ""):
                lines.append(f"Validation confidence score: {score}/100")
            if improvement_categories:
                lines.append("Main improvement categories:")
                lines.extend([f"- {category}" for category in improvement_categories[:8]])
            if issues:
                lines.append("Issues:")
                lines.extend([f"- {issue}" for issue in issues[:8]])
            if suggestions:
                lines.append("Suggestions:")
                lines.extend([f"- {suggestion}" for suggestion in suggestions[:8]])
            if metric_scores:
                lines.append("Metric scores to improve:")
                for key, value in list(metric_scores.items())[:8]:
                    label = str(key).replace("_", " ").strip()
                    lines.append(f"- {label}: {value}/100")
            if editor_feedback:
                lines.append("English editor feedback:")
                lines.append(editor_feedback)
            return "\n".join(lines)

        feedback_guidance = format_feedback()

        source_language = (source_language or "en").strip().lower()

        def build_prompt(text, chunk_mode=False):
            chunk_rule = (
                "This is one chunk from a longer transcript. Correct every line in this chunk and return every speaker turn."
                if chunk_mode
                else "Return the full corrected transcript from beginning to end."
            )
            language_rules = """
LANGUAGE RULES:
- The input transcript is in Hindi and may contain natural English code-mixing.
- Improve grammar, punctuation, and readability in Hindi.
- Preserve the original meaning and conversational tone.
- Do not translate the transcript into English or any other language.
- Keep the final output in Hindi, while preserving natural proper nouns or mixed-language terms when needed.
""".strip() if source_language == "hi" else f"""
LANGUAGE RULES:
- Preserve the transcript in its original language: {source_language}.
- Do not translate the transcript into another language.
- Only improve grammar, punctuation, readability, and formatting.
""".strip()
            refinement_mode = bool(feedback_guidance.strip())
            refinement_rules = """
REFINEMENT MODE:
- Make surgical edits only.
- Prefer the smallest possible change that improves grammar, readability, or formatting.
- Do not compress or summarize any sentence.
- Do not remove a speaker turn unless it is completely empty.
- Do not merge multiple speaker turns into one.
- Preserve sequence, meaning, and coverage exactly.
- If a sentence is awkward but understandable, improve it lightly instead of rewriting it.
- Preserve uncertainty rather than guessing missing words.
""".strip() if refinement_mode else ""
            return f"""
You are an expert transcription editor.

Your task is to clean and correct a raw transcript while preserving the original meaning, tone, and speaker intent.

TRANSCRIPTION STYLE: Clean Verbatim

INSTRUCTIONS:

1. Fix grammar, spelling, and punctuation errors.
2. Improve sentence clarity, but do not rewrite or paraphrase heavily.
3. Remove filler words only when they do not add meaning:
   um, uh, like, you know, etc.
4. Remove unnecessary repetitions:
   Example: "we we we should go" -> "we should go"
   But keep intentional emphasis:
   Example: "no no this time we should go" -> keep as is
5. Preserve the natural conversational tone. Do not make it too formal.
6. Do not change the meaning, intent, or context of any sentence.
7. Do not summarize or remove content.
8. Keep all speaker labels consistent:
   Person 1, Person 2, etc.
9. Do not introduce new speakers if not clearly present.
10. Break long sentences into readable lines if needed.
11. If a word or phrase is unclear, preserve the closest transcript text instead of guessing.
- {domain_guidance}

IMPORTANT RULES:
- Maintain all original information.
- Do not add new information.
- Do not over-polish into formal written language.
- Keep it sounding like real spoken conversation.
- Return only the corrected transcript.
- {chunk_rule}
{language_rules}
{feedback_guidance}
{refinement_rules}

OUTPUT FORMAT:

Person 1: ...
Person 2: ...

Transcript:
{text}
""".strip()

        def run_correction(text, chunk_mode=False):
            prompt = build_prompt(text, chunk_mode=chunk_mode)
            corrected_text = generate_text(
                prompt,
                system_instruction="You are an expert clean-verbatim transcription editor.",
                temperature=0.05 if feedback_guidance else 0.1,
                max_output_tokens=4096,
            )
            return _post_process(corrected_text, text)

        try:
            corrected = run_correction(cleaned_input)
            if not _looks_incomplete(corrected, cleaned_input):
                return CorrectionResult(text=corrected, used_gemini=True)

            print("Gemini correction looked incomplete; retrying in smaller chunks.")
            corrected_chunks = []
            for chunk in _split_transcript_chunks(cleaned_input):
                try:
                    chunk_result = run_correction(chunk, chunk_mode=True)
                    if _looks_incomplete(chunk_result, chunk):
                        print("Gemini chunk correction looked incomplete; keeping original chunk.")
                        chunk_result = chunk
                    corrected_chunks.append(chunk_result)
                except Exception as chunk_exc:
                    print(f"Gemini chunk correction issue: {chunk_exc}")
                    corrected_chunks.append(chunk)

            chunked_result = "\n".join(part.strip() for part in corrected_chunks if part.strip()).strip()
            if not _looks_incomplete(chunked_result, cleaned_input):
                return CorrectionResult(text=chunked_result, used_gemini=True)

            print("Gemini correction remained incomplete; using untruncated cleaned transcript.")
            return CorrectionResult(
                text=cleaned_input,
                used_gemini=False,
                fallback_reason="Gemini returned incomplete correction output; using cleaned transcript.",
            )
        except Exception as e:
            print(f"Gemini correction issue: {e}")
            return CorrectionResult(
                text=cleaned_input,
                used_gemini=False,
                fallback_reason=f"Gemini correction failed: {e}",
            )


def correct_text(transcript, domain_mode="meeting", feedback=None, source_language="en"):
    agent = CorrectionAgent()
    return agent.correct_text(
        transcript,
        domain_mode=domain_mode,
        feedback=feedback,
        source_language=source_language,
    )


def correct_text_with_result(transcript, domain_mode="meeting", feedback=None, source_language="en"):
    agent = CorrectionAgent()
    return agent.correct_text_with_result(
        transcript,
        domain_mode=domain_mode,
        feedback=feedback,
        source_language=source_language,
    )


if __name__ == "__main__":
    raw_text = "Person 1: helo my nme is chaitanya"
    print(correct_text(raw_text))
