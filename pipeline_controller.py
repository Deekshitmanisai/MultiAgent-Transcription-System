from dataclasses import dataclass, field
from typing import List

from correction_agent import correct_text_with_result
from services.code_switch_service import normalize_code_switch
from validation_agent import ValidationResult, validate_transcript_detailed


MAX_FEEDBACK_RETRIES = 2


@dataclass
class FeedbackLoopResult:
    normalized_text: str = ""
    corrected_text: str = ""
    validation: ValidationResult = field(default_factory=ValidationResult)
    attempts: int = 0
    feedback_history: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _feedback_from_validation(validation):
    return {
        "issues": list(validation.issues or []) + list(validation.critical_issues or []),
        "suggestions": list(validation.suggested_actions or []),
        "improvement_categories": list(validation.improvement_categories or []),
        "editor_feedback": validation.editor_feedback,
        "summary": validation.summary,
        "score": validation.confidence_score,
        "verdict": validation.verdict,
        "metric_scores": dict(validation.metric_scores or {}),
    }


def run_feedback_correction_loop(
    transcript,
    domain_mode="meeting",
    max_retries=MAX_FEEDBACK_RETRIES,
    source_language="en",
    is_mixed=False,
    normalization_mode="keep-mixed",
    original_text="",
):
    source_text = (transcript or "").strip()
    if not source_text:
        validation = validate_transcript_detailed(
            "",
            domain_mode=domain_mode,
            source_language=source_language,
            normalization_mode=normalization_mode,
            is_mixed=is_mixed,
            original_text=original_text,
        )
        return FeedbackLoopResult(
            normalized_text="",
            corrected_text="",
            validation=validation,
            attempts=0,
            errors=["No transcript provided for correction."],
        )

    feedback = None
    best_text = source_text
    best_normalized = source_text
    best_validation = ValidationResult(verdict="unavailable", confidence_score=0)
    feedback_history = []
    errors = []

    total_attempts = max(1, int(max_retries or 0) + 1)
    for attempt in range(1, total_attempts + 1):
        working_text = source_text
        route_to_normalization = bool(feedback and any(
            token in " ".join(str(item or "").lower() for item in (feedback.get("improvement_categories", []) + feedback.get("issues", [])))
            for token in ("code_switch", "normalization", "mixed")
        ))
        if is_mixed and source_language == "hi" and normalization_mode != "keep-mixed":
            try:
                working_text = normalize_code_switch(source_text, mode=normalization_mode)
                if route_to_normalization and feedback:
                    working_text = normalize_code_switch(working_text, mode=normalization_mode)
            except Exception as exc:
                errors.append(f"Normalization attempt {attempt} failed: {exc}")
                working_text = source_text

        try:
            correction_result = correct_text_with_result(
                working_text,
                domain_mode=domain_mode,
                feedback=feedback,
                source_language=source_language,
            )
            corrected = correction_result.text
            if correction_result.fallback_reason:
                errors.append(f"Correction attempt {attempt}: {correction_result.fallback_reason}")
        except Exception as exc:
            errors.append(f"Correction attempt {attempt} failed: {exc}")
            corrected = best_text or source_text

        try:
            validation = validate_transcript_detailed(
                corrected,
                domain_mode=domain_mode,
                source_language=source_language,
                normalization_mode=normalization_mode,
                is_mixed=is_mixed,
                stage_label="corrected",
                original_text=original_text or source_text,
            )
        except Exception as exc:
            errors.append(f"Validation attempt {attempt} failed: {exc}")
            validation = ValidationResult(
                is_valid=False,
                verdict="unavailable",
                confidence_score=0,
                summary="Validation failed during feedback loop.",
                issues=[str(exc)],
            )

        feedback_history.append(
            {
                "attempt": attempt,
                "verdict": validation.verdict,
                "confidence_score": validation.confidence_score,
                "issues": validation.issues,
                "suggestions": validation.suggested_actions,
                "improvement_categories": validation.improvement_categories,
                "editor_feedback": validation.editor_feedback,
                "metric_scores": validation.metric_scores,
                "scores": validation.scores,
                "total": validation.total,
                "normalized_text": working_text,
            }
        )

        if validation.confidence_score >= best_validation.confidence_score:
            best_text = corrected
            best_normalized = working_text
            best_validation = validation

        if validation.verdict == "pass":
            return FeedbackLoopResult(
                normalized_text=working_text,
                corrected_text=corrected,
                validation=validation,
                attempts=attempt,
                feedback_history=feedback_history,
                errors=errors,
            )

        feedback = _feedback_from_validation(validation)

    return FeedbackLoopResult(
        normalized_text=best_normalized,
        corrected_text=best_text,
        validation=best_validation,
        attempts=total_attempts,
        feedback_history=feedback_history,
        errors=errors,
    )
