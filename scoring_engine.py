CRITERIA_WEIGHTS = {
    "grammar_correctness": 30,
    "clarity_readability": 25,
    "sentence_structure": 20,
    "completeness": 15,
    "noise_reduction": 10,
    "code_switch_consistency": 20,
}


CRITERIA_LABELS = {
    "grammar_correctness": "Grammar Correctness",
    "clarity_readability": "Clarity & Readability",
    "sentence_structure": "Sentence Structure",
    "completeness": "Completeness",
    "noise_reduction": "Noise Reduction",
    "code_switch_consistency": "Code-Switch Consistency",
}


_ALIASES = {
    "grammar": "grammar_correctness",
    "grammar_correctness": "grammar_correctness",
    "grammarCorrectness": "grammar_correctness",
    "clarity": "clarity_readability",
    "clarity_readability": "clarity_readability",
    "clarityReadability": "clarity_readability",
    "readability": "clarity_readability",
    "sentence_structure": "sentence_structure",
    "sentenceStructure": "sentence_structure",
    "structure": "sentence_structure",
    "completeness": "completeness",
    "meaning_preservation": "completeness",
    "meaningPreservation": "completeness",
    "noise_reduction": "noise_reduction",
    "noiseReduction": "noise_reduction",
    "filler_removal": "noise_reduction",
    "fillerRemoval": "noise_reduction",
    "code_switch_consistency": "code_switch_consistency",
    "codeSwitchConsistency": "code_switch_consistency",
    "code_switch": "code_switch_consistency",
    "normalization_consistency": "code_switch_consistency",
}


def clamp_score(value, default=0):
    try:
        score = int(round(float(value)))
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = default
    return max(0, min(100, score))


def normalize_metric_scores(raw_scores):
    raw_scores = raw_scores if isinstance(raw_scores, dict) else {}
    normalized = {key: 0 for key in CRITERIA_WEIGHTS}

    for raw_key, value in raw_scores.items():
        key = _ALIASES.get(str(raw_key).strip())
        if key:
            normalized[key] = clamp_score(value)

    return normalized


def calculate_confidence_score(metric_scores):
    normalized = normalize_metric_scores(metric_scores)
    weighted_total = 0.0
    total_weight = float(sum(CRITERIA_WEIGHTS.values()) or 1.0)

    for key, weight in CRITERIA_WEIGHTS.items():
        weighted_total += normalized[key] * (weight / total_weight)

    return clamp_score(weighted_total)


def to_ten_point_scores(metric_scores):
    normalized = normalize_metric_scores(metric_scores)
    score_map = {
        "grammar": round(normalized["grammar_correctness"] / 10),
        "clarity": round(normalized["clarity_readability"] / 10),
        "readability": round(normalized["sentence_structure"] / 10),
        "completeness": round(normalized["completeness"] / 10),
        "noise": round(normalized["noise_reduction"] / 10),
        "code_switch_consistency": round(normalized["code_switch_consistency"] / 10),
    }
    return {key: max(0, min(10, int(value))) for key, value in score_map.items()}


def total_ten_point_score(metric_scores):
    return sum(to_ten_point_scores(metric_scores).values())


def missing_metric_issues(metric_scores):
    provided = set((metric_scores or {}).keys()) if isinstance(metric_scores, dict) else set()
    missing = []

    for key, label in CRITERIA_LABELS.items():
        aliases_for_key = {alias for alias, target in _ALIASES.items() if target == key}
        if key not in provided and not aliases_for_key.intersection(provided):
            missing.append(f"Validation model did not provide a {label} score.")

    return missing
    return missin
