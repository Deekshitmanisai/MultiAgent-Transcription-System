import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dataclasses import dataclass, field
from typing import Dict, List

from config import OPENROUTER_API_KEY, OPENROUTER_VALIDATION_MODEL
from decision_engine import decide_verdict, is_valid_verdict
from scoring_engine import (
    calculate_confidence_score,
    missing_metric_issues,
    normalize_metric_scores,
    to_ten_point_scores,
    total_ten_point_score,
)


ERROR_MARKERS = (
    "gemini connection failed",
    "missing gemini_api_key",
    "audio extraction failed",
    "tts error",
    "validation issues",
)


@dataclass
class ValidationResult:
    is_valid: bool = False
    verdict: str = "unavailable"
    confidence_score: int = 0
    summary: str = ""
    issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    improvement_categories: List[str] = field(default_factory=list)
    editor_feedback: str = ""
    validator: str = "openrouter"
    metric_scores: Dict[str, int] = field(default_factory=dict)
    scores: Dict[str, int] = field(default_factory=dict)
    total: int = 0
    critical_issues: List[str] = field(default_factory=list)


def _openrouter_generate_text(prompt, *, system_instruction, temperature=0.1, max_output_tokens=1024):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Set it in .env before running validation.")

    user_prompt = f"{system_instruction}\n\n{prompt}"
    body = {
        "model": OPENROUTER_VALIDATION_MODEL,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }

    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1:5173",
            "X-Title": "TaaS Transcript Validator",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter connection failed: {exc.reason}") from exc

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter returned no validation choices.")

    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def _non_empty_line_count(text):
    return len([line for line in (text or "").splitlines() if line.strip()])


def _has_speaker_formatting_issues(text):
    speaker_lines = [
        line.strip()
        for line in (text or "").splitlines()
        if line.strip().lower().startswith("person ")
    ]
    if not speaker_lines:
        return False

    speaker_line_re = re.compile(r"^Person\s+\d+\s*:\s+\S+", flags=re.IGNORECASE)
    return any(not speaker_line_re.match(line) for line in speaker_lines)


def _basic_guardrail_issues(text):
    issues = []
    lowered = (text or "").lower()

    if any(marker in lowered for marker in ERROR_MARKERS):
        issues.append("Transcript still contains an upstream error message instead of natural conversation.")
    if len((text or "").strip()) < 20:
        issues.append("Transcript is too short to validate confidently.")
    if _non_empty_line_count(text) == 0:
        issues.append("Transcript has no non-empty lines.")
    if _has_speaker_formatting_issues(text):
        issues.append("Speaker labels are malformed.")

    alpha_chars = sum(ch.isalpha() for ch in (text or ""))
    if alpha_chars < 10:
        issues.append("Transcript does not contain enough alphabetic content.")

    return issues


def _extract_json_object(text):
    payload = (text or "").strip()
    if not payload:
        return {}

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", payload, flags=re.DOTALL)
    if fenced_match:
        payload = fenced_match.group(1).strip()
    else:
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1 and end > start:
            payload = payload[start : end + 1]

    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _clean_string_list(values):
    cleaned = []
    if not isinstance(values, list):
        return cleaned

    for value in values:
        text = str(value or "").strip()
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_validation_payload(payload):
    if not isinstance(payload, dict):
        return {}

    raw_metric_scores = payload.get("metric_scores", payload.get("metrics", {}))
    metric_scores = normalize_metric_scores(raw_metric_scores)
    confidence_score = calculate_confidence_score(metric_scores)
    issues = _clean_string_list(payload.get("issues", []))
    issues.extend(missing_metric_issues(raw_metric_scores))
    critical_issues = _clean_string_list(payload.get("critical_issues", []))
    verdict = decide_verdict(confidence_score, issues=issues, critical_issues=critical_issues)

    return {
        "is_valid": is_valid_verdict(verdict),
        "verdict": verdict,
        "confidence_score": confidence_score,
        "summary": str(payload.get("summary", "")).strip(),
        "issues": issues,
        "strengths": _clean_string_list(payload.get("strengths", [])),
        "suggested_actions": _clean_string_list(
            payload.get("suggested_actions", payload.get("suggestions", []))
        ),
        "improvement_categories": _clean_string_list(
            payload.get("improvement_categories", payload.get("improvement_types", []))
        ),
        "editor_feedback": str(
            payload.get("editor_feedback", payload.get("english_feedback", ""))
        ).strip(),
        "metric_scores": metric_scores,
        "scores": to_ten_point_scores(metric_scores),
        "total": total_ten_point_score(metric_scores),
        "critical_issues": critical_issues,
    }


class FinalValidationAgent:
    def validate(
        self,
        corrected_text,
        domain_mode="meeting",
        source_language="en",
        normalization_mode="keep-mixed",
        is_mixed=False,
        stage_label="corrected",
        original_text="",
    ):
        text = (corrected_text or "").strip()
        if not text:
            return ValidationResult(
                is_valid=False,
                verdict="fail",
                confidence_score=0,
                summary="Transcript is empty, so OpenRouter validation cannot verify it.",
                issues=["Transcript is empty."],
                suggested_actions=["Generate or paste a transcript before running validation."],
                improvement_categories=["missing_content"],
                editor_feedback=(
                    "The transcript is empty. Regenerate the transcript before attempting "
                    "grammar or readability cleanup."
                ),
                metric_scores={
                    "grammar_correctness": 0,
                    "clarity_readability": 0,
                    "sentence_structure": 0,
                    "completeness": 0,
                    "noise_reduction": 0,
                    "code_switch_consistency": 0,
                },
                scores={
                    "grammar": 0,
                    "clarity": 0,
                    "readability": 0,
                    "completeness": 0,
                    "noise": 0,
                    "code_switch_consistency": 0,
                },
                total=0,
                critical_issues=["Transcript is empty."],
            )

        guardrail_issues = _basic_guardrail_issues(text)
        if guardrail_issues:
            return ValidationResult(
                is_valid=False,
                verdict="fail",
                confidence_score=0,
                summary="Transcript failed basic validation checks before OpenRouter review.",
                issues=guardrail_issues,
                suggested_actions=["Fix the transcript content, then run validation again."],
                improvement_categories=["formatting", "content_integrity"],
                editor_feedback=(
                    "Repair the transcript content first. Remove upstream error text, ensure "
                    "speaker labels are valid, and provide enough natural-language content to review."
                ),
                metric_scores={
                    "grammar_correctness": 0,
                    "clarity_readability": 0,
                    "sentence_structure": 0,
                    "completeness": 0,
                    "noise_reduction": 0,
                    "code_switch_consistency": 0,
                },
                scores={
                    "grammar": 0,
                    "clarity": 0,
                    "readability": 0,
                    "completeness": 0,
                    "noise": 0,
                    "code_switch_consistency": 0,
                },
                total=0,
                critical_issues=guardrail_issues,
            )

        domain_guidance = {
            "meeting": "Judge it as a professional meeting transcript.",
            "lecture": "Judge it as a lecture or presentation transcript.",
            "interview": "Judge it as an interview transcript with speaker turns.",
            "discussion": "Judge it as a conversational discussion transcript.",
        }.get((domain_mode or "meeting").strip().lower(), "Judge it as a general spoken transcript.")
        language_guidance = (
            "The transcript is in Hindi and may include natural English code-mixing. Evaluate quality in Hindi, "
            "but write editor_feedback in English."
            if (source_language or "en").strip().lower() == "hi"
            else f"The transcript is in {(source_language or 'its original language').strip()}. "
            "Evaluate quality in that language, but write editor_feedback in English."
        )
        normalization_guidance = f"""
This transcript stage is: {stage_label}.
Original raw transcript may have mixed Hindi-English content: {bool(is_mixed)}.
Requested normalization mode: {normalization_mode}.
If the transcript should be normalized, evaluate whether the chosen language is consistent and whether meaning from the original transcript is preserved.
""".strip()

        prompt = f"""
Review the final corrected transcript for quality.

Evaluate only the transcript text. Do not validate the audio.
Use these criteria and scores from 0 to 100:
1. grammar_correctness, weight 30: tense correctness, subject-verb agreement, punctuation.
2. clarity_readability, weight 25: easy to understand, no ambiguity, logical phrasing.
3. sentence_structure, weight 20: proper sentence boundaries, no run-on sentences, good formatting.
4. completeness, weight 15: meaning preserved, no missing or distorted information.
5. noise_reduction, weight 10: filler words and unnecessary repetitions are removed where appropriate.
6. code_switch_consistency, weight 20: mixed Hindi-English content is normalized consistently when required, or preserved coherently when keep-mixed is intended.

{domain_guidance}
{language_guidance}
{normalization_guidance}

Return JSON only with this exact shape:
{{
  "metric_scores": {{
    "grammar_correctness": 0,
    "clarity_readability": 0,
    "sentence_structure": 0,
    "completeness": 0,
    "noise_reduction": 0,
    "code_switch_consistency": 0
  }},
  "summary": "short summary",
  "issues": ["issue"],
  "strengths": ["strength"],
  "suggestions": ["action"],
  "improvement_categories": ["grammar", "clarity", "sentence_structure", "completeness", "noise_reduction", "code_switch_consistency", "speaker_formatting", "contextual_accuracy", "normalization"],
  "editor_feedback": "English instructions for the transcript editor describing what to improve and what to preserve.",
  "critical_issues": ["critical issue"]
}}

Rules:
- return only JSON
- all metric scores must be integers from 0 to 100
- keep summary under 30 words
- write editor_feedback in plain English for another AI editor
- editor_feedback must explain what is wrong, what should be improved, and what must be preserved
- improvement_categories should name the main error types seen in the transcript
- use code_switch_consistency or normalization when mixed-language normalization is incomplete or inconsistent
- use contextual_accuracy when wording changes, unclear meaning, or context drift are present
- use speaker_formatting when speaker labels or turn structure need repair
- put meaning changes, missing content, hallucinated content, or malformed speaker labels in critical_issues
- use an empty list when there are no items
- do not include confidence_score or verdict; those are computed deterministically by the app
- return JSON only

Original raw transcript for meaning comparison:
{original_text or text}

Transcript:
{text}
""".strip()

        try:
            response_text = _openrouter_generate_text(
                prompt,
                system_instruction=(
                    "You are an independent strict transcript quality validator. "
                    "Respond with JSON only and do not assume the transcript is correct."
                ),
                temperature=0.1,
                max_output_tokens=1024,
            )
            payload = _normalize_validation_payload(_extract_json_object(response_text))
        except Exception as exc:
            metric_scores = {
                "grammar_correctness": 0,
                "clarity_readability": 0,
                "sentence_structure": 0,
                "completeness": 0,
                "noise_reduction": 0,
                "code_switch_consistency": 0,
            }
            return ValidationResult(
                is_valid=False,
                verdict="unavailable",
                confidence_score=0,
                summary="OpenRouter validation could not be completed.",
                issues=[f"OpenRouter validation failed: {exc}"],
                suggested_actions=["Check OpenRouter API configuration and try validation again."],
                improvement_categories=["validation_unavailable"],
                editor_feedback=(
                    "Validation feedback was unavailable because OpenRouter could not complete the review. "
                    "Keep meaning and speaker turns intact if you retry correction."
                ),
                metric_scores=metric_scores,
                scores=to_ten_point_scores(metric_scores),
                total=total_ten_point_score(metric_scores),
            )

        if not payload:
            metric_scores = {
                "grammar_correctness": 0,
                "clarity_readability": 0,
                "sentence_structure": 0,
                "completeness": 0,
                "noise_reduction": 0,
                "code_switch_consistency": 0,
            }
            return ValidationResult(
                is_valid=False,
                verdict="review",
                confidence_score=0,
                summary="OpenRouter returned an unreadable validation response.",
                issues=["Could not parse OpenRouter validation output."],
                suggested_actions=["Run validation again and inspect backend logs if the problem continues."],
                improvement_categories=["validation_unavailable"],
                editor_feedback=(
                    "The validator response could not be parsed. If you retry correction, preserve "
                    "meaning and speaker turns and focus on grammar, clarity, and formatting."
                ),
                metric_scores=metric_scores,
                scores=to_ten_point_scores(metric_scores),
                total=total_ten_point_score(metric_scores),
            )

        if not payload["summary"]:
            payload["summary"] = "OpenRouter completed transcript validation."

        return ValidationResult(**payload)


def validate_transcript_detailed(
    corrected_text,
    domain_mode="meeting",
    source_language="en",
    normalization_mode="keep-mixed",
    is_mixed=False,
    stage_label="corrected",
    original_text="",
):
    agent = FinalValidationAgent()
    return agent.validate(
        corrected_text,
        domain_mode=domain_mode,
        source_language=source_language,
        normalization_mode=normalization_mode,
        is_mixed=is_mixed,
        stage_label=stage_label,
        original_text=original_text,
    )


def validate_transcript(
    corrected_text,
    domain_mode="meeting",
    source_language="en",
    normalization_mode="keep-mixed",
    is_mixed=False,
    stage_label="corrected",
    original_text="",
):
    return validate_transcript_detailed(
        corrected_text,
        domain_mode=domain_mode,
        source_language=source_language,
        normalization_mode=normalization_mode,
        is_mixed=is_mixed,
        stage_label=stage_label,
        original_text=original_text,
    ).is_valid
