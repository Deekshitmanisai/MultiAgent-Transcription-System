import re
from typing import Dict, List


def _strip_speaker_label(text: str) -> str:
    return re.sub(r"^\s*Person\s+\d+\s*:\s*", "", str(text or "").strip(), flags=re.IGNORECASE)


def _subtitle_lines_from_transcript(transcript: str) -> List[str]:
    raw_lines = [_strip_speaker_label(line) for line in str(transcript or "").splitlines()]
    lines = [line.strip() for line in raw_lines if line.strip()]
    if lines:
        return lines

    sentence_chunks = re.split(r"(?<=[.!?।])\s+", str(transcript or "").strip())
    return [chunk.strip() for chunk in sentence_chunks if chunk.strip()]


def _partition_lines(lines: List[str], slot_count: int) -> List[str]:
    if slot_count <= 0:
        return []
    if not lines:
        return [""] * slot_count
    if len(lines) >= slot_count:
        return lines[:slot_count]

    output = []
    remaining_lines = list(lines)
    for index in range(slot_count):
        remaining_slots = slot_count - index
        take = max(1, round(len(remaining_lines) / remaining_slots))
        output.append(" ".join(remaining_lines[:take]).strip())
        remaining_lines = remaining_lines[take:]
    return output


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds or 0.0) * 1000)))
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt_from_segments(transcript: str, segments: List[Dict[str, float]]) -> str:
    clean_segments = [segment for segment in (segments or []) if str(segment.get("text", "")).strip()]
    if not clean_segments:
        return ""

    subtitle_lines = _partition_lines(_subtitle_lines_from_transcript(transcript), len(clean_segments))
    blocks = []
    for index, segment in enumerate(clean_segments, start=1):
        text = subtitle_lines[index - 1].strip() if index - 1 < len(subtitle_lines) else ""
        if not text:
            text = str(segment.get("text", "")).strip()
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_format_srt_timestamp(segment.get('start', 0.0))} --> {_format_srt_timestamp(segment.get('end', segment.get('start', 0.0)))}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks).strip()
