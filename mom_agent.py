import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from gemini_service import generate_text


def _strip_html(text):
    t = text or ""
    t = re.sub(r"</?p[^>]*>", "\n", t, flags=re.IGNORECASE)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.IGNORECASE)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _strip_markdown(text):
    t = text or ""
    t = t.replace("**", "")
    t = t.replace("__", "")
    t = t.replace("`", "")
    t = re.sub(r"^\s*\*\s+", "- ", t, flags=re.MULTILINE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _normalize_notes_text(text):
    t = _strip_html((text or "").strip())
    t = t.replace("__", "**")
    t = t.replace("`", "")
    t = re.sub(r"^\s*\*\s+", "- ", t, flags=re.MULTILINE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _strip_speaker_prefixes(text):
    lines = []
    for ln in (text or "").splitlines():
        stripped = re.sub(r"^(Person\s*\d+\s*:\s*)", "", ln.strip(), flags=re.IGNORECASE)
        if stripped:
            lines.append(stripped)
    return "\n".join(lines).strip()


def _extract_speakers(text):
    """Extract unique speakers from transcript."""
    speaker_pattern = re.compile(r"^(Person\s*\d+)\s*:", re.IGNORECASE | re.MULTILINE)
    speakers = set()
    for match in speaker_pattern.finditer(text or ""):
        speakers.add(match.group(1).strip())
    return sorted(list(speakers))


def _meaningful(text):
    t = _strip_markdown(_strip_html((text or "").strip()))
    t = t.replace("-", "").replace("*", "").strip()
    return len(t) >= 3


def _fallback_summary(transcript):
    t = (transcript or "").strip()
    if not t:
        return ""
    one_line = " ".join([line.strip() for line in t.splitlines() if line.strip()])
    return one_line[:300]


def _extract_json_payload(text):
    raw = (text or "").strip()
    if not raw:
        return {}

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else raw

    try:
        return json.loads(candidate)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return {}

    return {}


def _normalize_string_list(values):
    if not isinstance(values, list):
        return []
    output = []
    seen = set()
    for value in values:
        item = str(value or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _normalize_action_items(items):
    if not isinstance(items, list):
        return []

    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        task = str(item.get("task", "") or "").strip()
        if not task:
            continue
        owner = str(item.get("owner", "") or "").strip() or "unknown"
        deadline = str(item.get("deadline", "") or "").strip() or "none"
        priority = str(item.get("priority", "") or "").strip().title() or "Medium"
        normalized.append(
            {
                "task": task,
                "owner": owner,
                "deadline": deadline,
                "priority": priority,
                "action": task,
                "due_date": deadline,
            }
        )
    return normalized


def _normalize_highlights(items):
    if not isinstance(items, list):
        return []

    normalized = []
    seen = set()
    allowed_types = {"action", "deadline", "decision"}
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        kind = str(item.get("type", "") or "").strip().lower()
        if not text or kind not in allowed_types:
            continue
        key = (text.lower(), kind)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"text": text, "type": kind})
    return normalized


def _parse_action_items(text):
    """Parse action items with owner, action, due date, and priority."""
    action_items = []
    lines = (text or "").splitlines()
    
    priority_keywords = {
        "high": "High",
        "urgent": "High", 
        "critical": "High",
        "important": "High",
        "asap": "High",
        "medium": "Medium",
        "normal": "Medium",
        "low": "Low",
        "whenever": "Low",
        "optional": "Low",
    }
    
    due_date_pattern = re.compile(
        r"(?:due|by|deadline|before|on)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]?\d{0,4}|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}|"
        r"next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
        r"tomorrow|today|end\s+of\s+\w+)",
        re.IGNORECASE
    )
    
    owner_pattern = re.compile(r"^[-•*]?\s*(\[?([A-Z][a-z]+)\]?)\s*:", re.MULTILINE)
    
    for line in lines:
        line = line.strip().lstrip("-•*").strip()
        if not line or line.lower() in ["none", "- none", "no action items"]:
            continue
            
        # Extract owner
        owner = "Unassigned"
        owner_match = owner_pattern.match(line)
        if owner_match:
            owner = owner_match.group(1).strip()
            line = line[len(owner_match.group(0)):].strip()
        
        # Extract due date
        due_date = "TBD"
        due_match = due_date_pattern.search(line)
        if due_match:
            due_date = due_match.group(0).strip()
            line = due_date_pattern.sub("", line).strip()
        
        # Extract priority
        priority = "Medium"
        line_lower = line.lower()
        for keyword, prio in priority_keywords.items():
            if keyword in line_lower:
                priority = prio
                break
        
        # Clean up the action
        action = line.lstrip("-: ").strip()
        if action:
            action_items.append({
                "owner": owner,
                "action": action,
                "due_date": due_date,
                "priority": priority
            })
    
    return action_items


def _parse_decisions(text):
    """Parse decisions with participants."""
    decisions = []
    lines = (text or "").splitlines()
    
    for line in lines:
        line = line.strip().lstrip("-•*").strip()
        if not line or line.lower() in ["none", "- none", "no decisions"]:
            continue
        
        # Check for consensus indicators
        approved = any(word in line.lower() for word in ["approved", "agreed", "confirmed", "decided", "passed"])
        participants = []
        
        # Extract names in brackets or after [All], [Team], etc.
        bracket_match = re.search(r"\[([^\]]+)\]", line)
        if bracket_match:
            participants.append(bracket_match.group(1).strip())
        
        decisions.append({
            "decision": line,
            "approved": approved,
            "participants": participants
        })
    
    return decisions


def _parse_qa(text):
    """Extract Q&A pairs from transcript."""
    qa_pairs = []
    
    # Pattern for questions
    question_pattern = re.compile(
        r"(?:^|\n)\s*(?:Q:|Question:|Q\s*\d*:?)\s*(.+?)(?:\n|$)",
        re.IGNORECASE | re.MULTILINE
    )
    
    # Pattern for answers
    answer_pattern = re.compile(
        r"(?:^|\n)\s*(?:A:|Answer:|A\s*\d*:?)\s*(.+?)(?:\n|$)",
        re.IGNORECASE | re.MULTILINE
    )
    
    questions = question_pattern.findall(text or "")
    answers = answer_pattern.findall(text or "")
    
    for i in range(min(len(questions), len(answers))):
        qa_pairs.append({
            "question": questions[i].strip(),
            "answer": answers[i].strip()
        })
    
    return qa_pairs


def _extract_dates_mentioned(text):
    """Extract dates mentioned in the transcript."""
    dates = []
    
    date_patterns = [
        re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"),
        re.compile(r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4})\b", re.IGNORECASE),
        re.compile(r"\b(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", re.IGNORECASE),
        re.compile(r"\b(tomorrow|today|end\s+of\s+(?:week|month|quarter|year))\b", re.IGNORECASE),
    ]
    
    for pattern in date_patterns:
        matches = pattern.findall(text or "")
        dates.extend(matches)
    
    return list(set(dates))[:10]


def _extract_resources(text):
    """Extract resources mentioned (links, documents, tools)."""
    resources = []
    
    # URL pattern
    url_pattern = re.compile(r"https?://[^\s<]+")
    urls = url_pattern.findall(text or "")
    for url in urls:
        resources.append({"type": "link", "name": url})
    
    # Document patterns
    doc_patterns = [
        (r"\b([A-Z][a-zA-Z]+\s+(?:doc|document|file|report|plan|budget|proposal|presentation))\b", "document"),
        (r"\b(Google\s+Doc|Notion|Notion\s+page|Slack|Teams|Jira|Confluence)\b", "tool"),
        (r"\b([A-Z][a-zA-Z]+\s+Dashboard)\b", "tool"),
    ]
    
    for pattern, res_type in doc_patterns:
        matches = re.findall(pattern, text or "")
        for match in matches:
            resources.append({"type": res_type, "name": match})
    
    # Remove duplicates
    seen = set()
    unique_resources = []
    for r in resources:
        key = (r["type"], r["name"].lower())
        if key not in seen:
            seen.add(key)
            unique_resources.append(r)
    
    return unique_resources[:10]


def _analyze_sentiment(text):
    """Analyze overall sentiment/mood of the meeting."""
    positive_words = [
        "great", "excellent", "amazing", "wonderful", "fantastic", "good", "progress",
        "success", "achieved", "approved", "positive", "excited", "happy", "thank",
        "thanks", "appreciate", "glad", "pleased", "satisfied", "confident"
    ]
    
    negative_words = [
        "problem", "issue", "concern", "delay", "failed", "failure", "risk",
        "challenge", "difficult", "hard", "stuck", "blocked", "worry", "worried",
        "unfortunately", "sorry", "regret", "disappointed", "frustrated"
    ]
    
    neutral_words = [
        "discussed", "reviewed", "talked", "meeting", "agenda", "update",
        "presentation", "overview", "summary", "report"
    ]
    
    text_lower = (text or "").lower()
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    neu_count = sum(1 for word in neutral_words if word in text_lower)
    
    total = pos_count + neg_count + neu_count
    if total == 0:
        return "neutral", 50
    
    pos_pct = (pos_count / total) * 100
    neg_pct = (neg_count / total) * 100
    
    if pos_pct > neg_pct + 20:
        return "positive", int(pos_pct)
    elif neg_pct > pos_pct + 20:
        return "negative", int(100 - neg_pct)
    else:
        return "neutral", 50


def _split_enhanced_sections(text):
    """Split the enhanced response into sections."""
    t = _normalize_notes_text(text)
    
    heading_re = re.compile(
        r"^\s*(Meeting\s+(?:Topic|Title)|Agenda(?:\s+Items)?|Discussion\s+Topics|"
        r"Key\s+Takeaways|Minutes\s+of\s+Meeting|Decisions|Q\s*&\s*A|"
        r"Action\s+Items|Follow[-\s]?up(?:s)?|Next\s+Meeting|Resources|"
        r"Sentiment|Attendees)\s*:?\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    matches = list(heading_re.finditer(t))
    if not matches:
        return _split_sections(text)

    sections = {
        "meeting_topic": "",
        "agenda": "",
        "discussion_topics": "",
        "key_takeaways": "",
        "minutes": "",
        "decisions": "",
        "qa": "",
        "action_items": "",
        "follow_ups": "",
        "next_meeting": "",
        "resources": "",
        "sentiment": "",
        "attendees": "",
    }
    
    key_map = {
        "meeting topic": "meeting_topic",
        "agenda": "agenda",
        "agenda items": "agenda",
        "discussion topics": "discussion_topics",
        "key takeaways": "key_takeaways",
        "minutes of meeting": "minutes",
        "decisions": "decisions",
        "q&a": "qa",
        "q and a": "qa",
        "action items": "action_items",
        "follow-ups": "follow_ups",
        "follow ups": "follow_ups",
        "next meeting": "next_meeting",
        "resources": "resources",
        "sentiment": "sentiment",
        "attendees": "attendees",
    }

    for index, match in enumerate(matches):
        key = key_map.get((match.group(1) or "").strip().lower())
        if not key:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(t)
        sections[key] = t[start:end].strip()

    return sections


def _split_sections(text):
    """Legacy section splitter for backward compatibility."""
    t = _normalize_notes_text(text)
    heading_re = re.compile(
        r"^\s*(Minutes of Meeting|Key Points|Decisions|Action Items)\s*:?\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    matches = list(heading_re.finditer(t))
    if not matches:
        return t, "", "", ""

    sections = {"minutes": "", "key_points": "", "decisions": "", "action_items": ""}
    key_map = {
        "minutes of meeting": "minutes",
        "key points": "key_points",
        "decisions": "decisions",
        "action items": "action_items",
    }

    for index, match in enumerate(matches):
        key = key_map.get((match.group(1) or "").strip().lower())
        if not key:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(t)
        sections[key] = t[start:end].strip()

    minutes_text = "Minutes of Meeting:\n" + (sections["minutes"] or "").strip()
    return (
        _normalize_notes_text(minutes_text).strip(),
        _normalize_notes_text(sections["key_points"]).strip(),
        _normalize_notes_text(sections["decisions"]).strip(),
        _normalize_notes_text(sections["action_items"]).strip(),
    )


@dataclass
class EnhancedMinutesResult:
    meeting_summary: str = ""
    key_discussions: List[str] = field(default_factory=list)
    decisions: str = ""
    action_items: str = ""
    parsed_action_items: List[Dict[str, str]] = field(default_factory=list)
    deadlines: List[str] = field(default_factory=list)
    important_highlights: List[Dict[str, str]] = field(default_factory=list)
    parsed_decisions: List[Dict[str, str]] = field(default_factory=list)
    language: str = "en"
    
    def to_dict(self):
        legacy_minutes = self.meeting_summary or "Not generated"
        legacy_key_points = "\n".join(f"- {item}" for item in self.key_discussions) if self.key_discussions else "- None"
        legacy_action_items = (
            "\n".join(
                f"- {item['owner']}: {item['task']} (Due: {item['deadline']})"
                for item in self.parsed_action_items
            )
            if self.parsed_action_items
            else "- None"
        )
        legacy_decisions = self.decisions or "- None"
        return {
            "meetingSummary": self.meeting_summary,
            "keyDiscussions": self.key_discussions,
            "decisions": self.parsed_decisions,
            "actionItems": self.parsed_action_items,
            "deadlines": self.deadlines,
            "importantHighlights": self.important_highlights,
            "minutes": legacy_minutes,
            "keyPoints": legacy_key_points,
            "decisionsText": legacy_decisions,
            "actionItemsText": legacy_action_items,
            "parsedActionItems": self.parsed_action_items,
            "parsedDecisions": self.parsed_decisions,
            "language": self.language,
        }


class MinutesOfMeetingAgent:
    def generate_minutes(self, corrected_transcript, summary_style="concise", domain_mode="meeting", source_language="en"):
        labeled_transcript = _strip_markdown(_strip_html((corrected_transcript or "").strip()))
        transcript = _strip_markdown(_strip_speaker_prefixes(_strip_html((corrected_transcript or "").strip())))
        if not labeled_transcript.strip():
            return EnhancedMinutesResult()

        style_guidance = {
            "concise": "Keep the notes crisp and compact.",
            "detailed": "Include fuller context for each important point, while staying structured.",
            "actions_only": "Prioritize actionable next steps and concrete follow-ups over narrative detail.",
            "executive": "Use a polished executive-summary tone with high-signal takeaways only.",
        }.get((summary_style or "concise").strip().lower(), "Keep the notes concise and useful.")
        
        domain_guidance = {
            "meeting": "Treat the conversation as a professional meeting.",
            "lecture": "Treat the conversation as a lecture or presentation with key teaching points.",
            "interview": "Treat the conversation as an interview and preserve the most important answers.",
            "discussion": "Treat the conversation as a collaborative discussion.",
        }.get((domain_mode or "meeting").strip().lower(), "Treat the conversation as a spoken transcript.")
        language_guidance = (
            "The transcript is in Hindi and may include some English words. Extract decisions and actions from Hindi accurately."
            if (source_language or "en").strip().lower() == "hi"
            else f"The transcript is in {(source_language or 'its original language').strip()}."
        )

        prompt = f"""
You are an advanced AI Meeting Intelligence Assistant.

Your task is to convert a raw or corrected meeting transcript into a highly structured, professional Minutes of Meeting (MoM).

This is NOT a summarization task.
This is an information extraction and structuring task.

You must extract only meaningful, actionable, and decision-relevant content.

OBJECTIVE:
Transform the transcript into structured MoM that includes:
- Clear decisions
- Actionable tasks
- Deadlines and timelines
- Key discussion topics
- Important highlight phrases for UI

STRICT RULES:
1. DO NOT include filler conversation.
2. DO NOT hallucinate or assume missing information.
3. ONLY extract what is explicitly present.
4. KEEP output concise and structured.
5. DO NOT return explanations. ONLY JSON.
6. If something is missing, use "unknown" for owner and "none" for deadline.
7. {style_guidance}
8. {domain_guidance}
9. {language_guidance}

WHAT TO EXTRACT:
- meeting_summary: 2-3 lines maximum
- key_discussions: concise meaningful discussion topics
- decisions: finalized decisions only
- action_items: objects with task, owner, deadline
- deadlines: all time-related mentions
- important_highlights: short high-importance phrases only

HIGHLIGHTING RULES:
- Keep phrases short, 2-6 words
- Avoid duplication
- Only include high-importance phrases

Return ONLY valid JSON in this format:
{{
  "meeting_summary": "string",
  "key_discussions": ["string"],
  "decisions": ["string"],
  "action_items": [
    {{
      "task": "string",
      "owner": "string",
      "deadline": "string"
    }}
  ],
  "deadlines": ["string"],
  "important_highlights": [
    {{
      "text": "string",
      "type": "action | deadline | decision"
    }}
  ]
}}

INPUT TRANSCRIPT:
{labeled_transcript}
""".strip()

        try:
            content = generate_text(
                prompt,
                system_instruction="You extract structured meeting intelligence and return valid JSON only.",
                temperature=0.1,
                max_output_tokens=4096,
            )
        except Exception as e:
            print(f"Gemini minutes issue: {e}")
            content = ""

        result = EnhancedMinutesResult(language=(source_language or "en").strip().lower() or "en")

        payload = _extract_json_payload(content)
        if payload:
            result.meeting_summary = str(payload.get("meeting_summary", "") or "").strip()
            result.key_discussions = _normalize_string_list(payload.get("key_discussions"))

            decision_list = _normalize_string_list(payload.get("decisions"))
            result.decisions = "\n".join(f"- {item}" for item in decision_list)
            result.parsed_decisions = [
                {"decision": item, "approved": True, "participants": []}
                for item in decision_list
            ]

            result.parsed_action_items = _normalize_action_items(payload.get("action_items"))
            result.action_items = "\n".join(
                f"- {item['owner']}: {item['task']} (Due: {item['deadline']})"
                for item in result.parsed_action_items
            )

            result.deadlines = _normalize_string_list(payload.get("deadlines"))
            result.important_highlights = _normalize_highlights(payload.get("important_highlights"))

        if not _meaningful(result.meeting_summary):
            result.meeting_summary = _fallback_summary(transcript)

        if not result.key_discussions:
            result.key_discussions = _normalize_string_list(_extract_dates_mentioned(transcript)[:3])

        if not result.deadlines:
            result.deadlines = _extract_dates_mentioned(transcript)

        if not result.parsed_action_items:
            fallback_items = _parse_action_items(transcript)
            result.parsed_action_items = [
                {
                    "task": item.get("action", ""),
                    "owner": item.get("owner", "unknown") or "unknown",
                    "deadline": item.get("due_date", "none") or "none",
                }
                for item in fallback_items
                if item.get("action")
            ]
            result.action_items = "\n".join(
                f"- {item['owner']}: {item['task']} (Due: {item['deadline']})"
                for item in result.parsed_action_items
            )

        if not result.parsed_decisions:
            fallback_decisions = _parse_decisions(transcript)
            result.parsed_decisions = fallback_decisions
            result.decisions = "\n".join(f"- {item['decision']}" for item in fallback_decisions)

        if not result.important_highlights:
            for item in result.parsed_action_items[:3]:
                phrase = str(item.get("task", "")).strip()
                if phrase:
                    result.important_highlights.append({"text": " ".join(phrase.split()[:6]), "type": "action"})
            for item in result.deadlines[:2]:
                result.important_highlights.append({"text": str(item), "type": "deadline"})
            for item in result.parsed_decisions[:2]:
                phrase = str(item.get("decision", "")).strip()
                if phrase:
                    result.important_highlights.append({"text": " ".join(phrase.split()[:6]), "type": "decision"})

        if not result.decisions.strip():
            result.decisions = "- None"
        if not result.action_items.strip():
            result.action_items = "- None"

        return result

    def generate_minutes_legacy(self, corrected_transcript, summary_style="concise", domain_mode="meeting", source_language="en"):
        """Legacy method for backward compatibility."""
        result = self.generate_minutes(corrected_transcript, summary_style, domain_mode, source_language=source_language)
        return result.minutes, result.key_points, result.decisions, result.action_items


def generate_minutes_of_meeting(corrected_transcript, summary_style="concise", domain_mode="meeting", source_language="en"):
    agent = MinutesOfMeetingAgent()
    return agent.generate_minutes(
        corrected_transcript,
        summary_style=summary_style,
        domain_mode=domain_mode,
        source_language=source_language,
    )


def generate_minutes_of_meeting_legacy(corrected_transcript, summary_style="concise", domain_mode="meeting", source_language="en"):
    """Legacy function for backward compatibility."""
    agent = MinutesOfMeetingAgent()
    return agent.generate_minutes_legacy(
        corrected_transcript,
        summary_style=summary_style,
        domain_mode=domain_mode,
        source_language=source_language,
    )
