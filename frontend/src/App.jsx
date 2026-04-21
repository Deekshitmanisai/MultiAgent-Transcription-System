import { useEffect, useMemo, useState } from "react";
import LiveMode from "./LiveMode";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const LANGUAGE_OPTIONS = [
  { code: "auto", label: "Auto Detect" },
  { code: "en", label: "English" },
  { code: "hi", label: "Hindi" },
  { code: "te", label: "Telugu" },
  { code: "ta", label: "Tamil" },
  { code: "kn", label: "Kannada" },
];
const TRANSLATION_OPTIONS = [
  { code: "hi", label: "Hindi" },
  { code: "te", label: "Telugu" },
  { code: "kn", label: "Kannada" },
  { code: "ta", label: "Tamil" },
];
const DOMAIN_OPTIONS = [
  { code: "meeting", label: "Meeting" },
  { code: "lecture", label: "Lecture" },
  { code: "interview", label: "Interview" },
  { code: "discussion", label: "Discussion" },
];
const SUMMARY_STYLES = [
  { code: "concise", label: "Concise" },
  { code: "detailed", label: "Detailed" },
  { code: "actions_only", label: "Actions Only" },
  { code: "executive", label: "Executive" },
];
const tabs = [
  { id: "transcript", label: "Transcript Workspace" },
  { id: "compare", label: "Comparison" },
  { id: "simplified", label: "Simplified View" },
  { id: "voice", label: "Voice + Translation" },
  { id: "notes", label: "Meeting Notes" },
  { id: "live", label: "🔴 Live Mode" },
];

const initialResult = {
  rawTranscript: "",
  speakerTranscript: "",
  correctedTranscript: "",
  timestampedTranscript: "",
  segments: [],
  validation: {
    isValid: false,
    verdict: "unavailable",
    confidenceScore: 0,
    summary: "",
    issues: [],
    strengths: [],
    suggestedActions: [],
    validator: "openrouter",
    metricScores: {},
    criticalIssues: [],
  },
  audioQuality: { label: "unknown", score: 0, summary: "" },
  meta: {},
  errors: [],
};

function App() {
  const [geminiConfigured, setGeminiConfigured] = useState(false);
  const [openRouterConfigured, setOpenRouterConfigured] = useState(false);
  const [liveModeAvailable, setLiveModeAvailable] = useState(false);
  const [speakerCount, setSpeakerCount] = useState("auto");
  const [selectedTab, setSelectedTab] = useState("transcript");
  const [videoFile, setVideoFile] = useState(null);
  const [status, setStatus] = useState("Waiting for input.");
  const [selectedLanguage, setSelectedLanguage] = useState("hi");
  const [transcriptionLanguage, setTranscriptionLanguage] = useState("auto");
  const [domainMode, setDomainMode] = useState("meeting");
  const [summaryStyle, setSummaryStyle] = useState("concise");
  const [searchQuery, setSearchQuery] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isGeneratingEnglish, setIsGeneratingEnglish] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [isGeneratingHindi, setIsGeneratingHindi] = useState(false);
  const [isValidatingFinal, setIsValidatingFinal] = useState(false);
  const [isRefiningWithFeedback, setIsRefiningWithFeedback] = useState(false);
  const [isSimplifying, setIsSimplifying] = useState(false);
  const [result, setResult] = useState(initialResult);
  const [speakerAliases, setSpeakerAliases] = useState({});
  const [minutes, setMinutes] = useState("");
  const [keyPoints, setKeyPoints] = useState("");
  const [decisions, setDecisions] = useState("");
  const [actionItems, setActionItems] = useState("");
  const [simplifiedText, setSimplifiedText] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [translationStatus, setTranslationStatus] = useState("Translation and voice status will appear here.");
  const [englishAudioUrl, setEnglishAudioUrl] = useState("");
  const [targetAudioUrl, setTargetAudioUrl] = useState("");

  const hasTranscript = useMemo(
    () => Boolean(result.correctedTranscript.trim()),
    [result.correctedTranscript]
  );

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((response) => response.json())
      .then((payload) => {
        setGeminiConfigured(Boolean(payload.geminiConfigured));
        setOpenRouterConfigured(Boolean(payload.openRouterConfigured));
        setLiveModeAvailable(Boolean(payload.liveModeAvailable));
        if (payload.defaultLanguage) setTranscriptionLanguage(payload.defaultLanguage);
        if (payload.defaultDomainMode) setDomainMode(payload.defaultDomainMode);
      })
      .catch(() => {
        setGeminiConfigured(false);
        setOpenRouterConfigured(false);
        setLiveModeAvailable(false);
      });
  }, []);

  useEffect(() => {
    if (!result.segments?.length) {
      setSpeakerAliases({});
      return;
    }
    setSpeakerAliases((current) => {
      const next = { ...current };
      const speakerIds = [...new Set(result.segments.map((segment) => Number(segment.speaker || 1)))].sort(
        (a, b) => a - b
      );
      speakerIds.forEach((id) => {
        if (!next[id]) next[id] = `Person ${id}`;
      });
      return next;
    });
  }, [result.segments]);

  async function checkResponse(response) {
    if (response.ok) return response.json();
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Request failed.");
  }

  async function handleProcessSubmit(event) {
    event.preventDefault();
    if (!videoFile) {
      setStatus("Please choose a meeting recording before running the pipeline.");
      return;
    }

    const formData = new FormData();
    formData.append("file", videoFile);
    formData.append("speaker_count", String(speakerCount));
    formData.append("transcription_language", transcriptionLanguage);
    formData.append("domain_mode", domainMode);

    setIsProcessing(true);
    setStatus("Running full transcription pipeline...");
    setMinutes("");
    setKeyPoints("");
    setDecisions("");
    setActionItems("");
    setSimplifiedText("");
    setTranslatedText("");
    setEnglishAudioUrl("");
    setTargetAudioUrl("");
    setTranslationStatus("Translation and voice status will appear here.");

    try {
      const response = await fetch(`${API_BASE}/api/process`, { method: "POST", body: formData });
      const payload = await checkResponse(response);
      setResult({
        rawTranscript: payload.rawTranscript || "",
        speakerTranscript: payload.speakerTranscript || "",
        correctedTranscript: payload.correctedTranscript || "",
        timestampedTranscript: payload.timestampedTranscript || "",
        segments: payload.segments || [],
        validation: normalizeValidation(payload.validation),
        audioQuality: payload.audioQuality || initialResult.audioQuality,
        meta: payload.meta || {},
        errors: payload.errors || [],
      });

      if (payload.ok && payload.validation?.isValid) {
        setStatus("OpenRouter validated the final transcript. It is ready for follow-up actions.");
      } else if (payload.ok && payload.validation?.verdict === "review") {
        setStatus("OpenRouter found final-stage quality concerns. Review the transcript before delivery.");
      } else if (payload.ok) {
        setStatus("Transcript generated with review notes. Check validation issues before delivery.");
      } else {
        setStatus("Pipeline finished with errors. Review the status and try again.");
      }
      setSelectedTab("transcript");
    } catch (error) {
      setResult(initialResult);
      setStatus(error.message || "Pipeline request failed.");
    } finally {
      setIsProcessing(false);
    }
  }

  async function handleValidateFinalTranscript() {
    if (!hasTranscript) return;

    setIsValidatingFinal(true);
    setStatus("Running OpenRouter final validation...");
    try {
      const response = await fetch(`${API_BASE}/api/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          corrected_transcript: result.correctedTranscript,
          domain_mode: domainMode,
        }),
      });
      const payload = await checkResponse(response);
      setResult((current) => ({ ...current, validation: normalizeValidation(payload.validation) }));

      if (payload.validation?.isValid) {
        setStatus("OpenRouter says the final transcript looks correct.");
      } else if (payload.validation?.verdict === "review") {
        setStatus("OpenRouter says the final transcript needs review before delivery.");
      } else {
        setStatus("OpenRouter says the final transcript is not ready yet.");
      }
    } catch (error) {
      setStatus(error.message || "Final validation failed.");
    } finally {
      setIsValidatingFinal(false);
    }
  }

  async function handleRefineWithFeedback() {
    if (!hasTranscript) return;

    const issues = [
      ...(result.validation?.issues || []),
      ...(result.validation?.criticalIssues || []),
    ];
    const suggestions = result.validation?.suggestedActions || [];

    if (!issues.length && !suggestions.length) {
      setStatus("No validation feedback is available for refinement.");
      return;
    }

    setIsRefiningWithFeedback(true);
    setStatus("Refining transcript with OpenRouter feedback...");
    try {
      const response = await fetch(`${API_BASE}/api/refine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          corrected_transcript: result.correctedTranscript,
          domain_mode: domainMode,
          issues,
          suggestions,
        }),
      });
      const payload = await checkResponse(response);
      setResult((current) => ({
        ...current,
        correctedTranscript: payload.correctedTranscript || current.correctedTranscript,
        validation: normalizeValidation(payload.validation),
      }));
      setSelectedTab("transcript");

      if (payload.validation?.isValid) {
        setStatus("Feedback refinement passed OpenRouter validation.");
      } else {
        setStatus("Feedback refinement completed. Review the remaining OpenRouter notes.");
      }
    } catch (error) {
      setStatus(error.message || "Feedback refinement failed.");
    } finally {
      setIsRefiningWithFeedback(false);
    }
  }

  async function handleGenerateMinutes() {
    if (!hasTranscript) return;

    setStatus("Generating meeting notes...");
    try {
      const response = await fetch(`${API_BASE}/api/minutes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          corrected_transcript: result.correctedTranscript,
          summary_style: summaryStyle,
          domain_mode: domainMode,
        }),
      });
      const payload = await checkResponse(response);
      setMinutes(payload.minutes || "");
      setKeyPoints(payload.keyPoints || "");
      setDecisions(payload.decisions || "");
      setActionItems(payload.actionItems || "");
      setSelectedTab("notes");
      setStatus("Meeting notes are ready.");
    } catch (error) {
      setStatus(error.message || "Meeting note generation failed.");
    }
  }

  async function handleSimplifyTranscript() {
    if (!hasTranscript) return;

    setIsSimplifying(true);
    setStatus("Generating simplified explanation...");
    try {
      const response = await fetch(`${API_BASE}/api/simplify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          corrected_transcript: result.correctedTranscript,
          domain_mode: domainMode,
        }),
      });
      const payload = await checkResponse(response);
      setSimplifiedText(payload.simplifiedText || "");
      setSelectedTab("simplified");
      setStatus(payload.message || "Simplified explanation ready.");
    } catch (error) {
      setStatus(error.message || "Simplified explanation generation failed.");
    } finally {
      setIsSimplifying(false);
    }
  }

  async function handleTranslate() {
    if (!hasTranscript) return;

    setIsTranslating(true);
    setTranslationStatus(`Translating transcript into ${getLanguageLabel(selectedLanguage, TRANSLATION_OPTIONS)}...`);
    try {
      const response = await fetch(`${API_BASE}/api/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          corrected_transcript: result.correctedTranscript,
          target_language: selectedLanguage,
        }),
      });
      const payload = await checkResponse(response);
      setTranslatedText(payload.translatedText || "");
      setTranslationStatus(payload.message || "Translation ready.");
      setSelectedTab("voice");
    } catch (error) {
      setTranslationStatus(error.message || "Translation failed.");
    } finally {
      setIsTranslating(false);
    }
  }

  async function handleGenerateEnglishVoice() {
    if (!hasTranscript) return;

    setIsGeneratingEnglish(true);
    setStatus("Generating English voice output...");
    try {
      const response = await fetch(`${API_BASE}/api/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: result.correctedTranscript, lang: "en" }),
      });
      const payload = await checkResponse(response);
      setEnglishAudioUrl(`${API_BASE}${payload.audioUrl}`);
      setSelectedTab("voice");
      setStatus("English audio is ready.");
    } catch (error) {
      setStatus(error.message || "English voice generation failed.");
    } finally {
      setIsGeneratingEnglish(false);
    }
  }

  async function handleGenerateTargetVoice() {
    if (!translatedText.trim()) {
      setTranslationStatus("Generate translation first.");
      return;
    }

    setIsGeneratingHindi(true);
    const label = getLanguageLabel(selectedLanguage, TRANSLATION_OPTIONS);
    setTranslationStatus(`Generating ${label} voice output...`);
    try {
      const response = await fetch(`${API_BASE}/api/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: translatedText, lang: selectedLanguage }),
      });
      const payload = await checkResponse(response);
      setTargetAudioUrl(`${API_BASE}${payload.audioUrl}`);
      setTranslationStatus(`${label} audio is ready.`);
    } catch (error) {
      setTranslationStatus(error.message || "Voice generation failed.");
    } finally {
      setIsGeneratingHindi(false);
    }
  }

  function updateSpeakerAlias(speakerId, value) {
    setSpeakerAliases((current) => ({
      ...current,
      [speakerId]: value || `Person ${speakerId}`,
    }));
  }

  function downloadTranscriptTxt() {
    const content = [
      "TaaS Transcript Export",
      "",
      "Raw Transcript:",
      result.rawTranscript || "N/A",
      "",
      "Speaker Transcript:",
      renamedSpeakerTranscript || "N/A",
      "",
      "Final Transcript:",
      result.correctedTranscript || "N/A",
      "",
      "Timestamped Transcript:",
      renamedTimestampedTranscript || "N/A",
    ].join("\n");
    downloadFile("taas-transcript.txt", content, "text/plain;charset=utf-8");
  }

  function downloadNotesMarkdown() {
    const content = [
      "# TaaS Meeting Notes",
      "",
      "## Minutes of Meeting",
      minutes || "Not generated",
      "",
      "## Key Points",
      keyPoints || "Not generated",
      "",
      "## Decisions",
      decisions || "Not generated",
      "",
      "## Action Items",
      actionItems || "Not generated",
    ].join("\n");
    downloadFile("taas-meeting-notes.md", content, "text/markdown;charset=utf-8");
  }

  function downloadReportJson() {
    const payload = {
      transcript: {
        raw: result.rawTranscript,
        speaker: renamedSpeakerTranscript,
        corrected: result.correctedTranscript,
        timestamped: renamedTimestampedTranscript,
      },
      validation: result.validation,
      audioQuality: result.audioQuality,
      notes: { minutes, keyPoints, decisions, actionItems },
      meta: { ...result.meta, speakerAliases, summaryStyle, domainMode },
    };
    downloadFile("taas-report.json", JSON.stringify(payload, null, 2), "application/json;charset=utf-8");
  }

  function downloadEvaluationHtml() {
    const html = `
      <html>
        <head><meta charset="utf-8"><title>TaaS Report</title></head>
        <body style="font-family: Arial, sans-serif; padding: 24px;">
          <h1>TaaS Evaluation Report</h1>
          <h2>Validation</h2>
          <p><strong>Verdict:</strong> ${escapeHtml(validationVerdict)}</p>
          <p><strong>Confidence:</strong> ${validationConfidence}/100</p>
          <p>${escapeHtml(validationSummary)}</p>
          <h2>Audio Quality</h2>
          <p><strong>Label:</strong> ${escapeHtml(audioQualityLabel)}</p>
          <p><strong>Score:</strong> ${audioQualityScore}/100</p>
          <p>${escapeHtml(result.audioQuality?.summary || "")}</p>
          <h2>Final Transcript</h2>
          <pre style="white-space: pre-wrap;">${escapeHtml(result.correctedTranscript)}</pre>
          <h2>Meeting Notes</h2>
          <pre style="white-space: pre-wrap;">${escapeHtml(minutes || "Not generated")}</pre>
        </body>
      </html>
    `;
    downloadFile("taas-report.html", html, "text/html;charset=utf-8");
  }

  const validationIssues = result.validation?.issues || [];
  const validationCriticalIssues = result.validation?.criticalIssues || [];
  const validationStrengths = result.validation?.strengths || [];
  const suggestedActions = result.validation?.suggestedActions || [];
  const validationVerdict = result.validation?.verdict || "unavailable";
  const validationSummary = result.validation?.summary || "OpenRouter validation has not run yet.";
  const validationConfidence = result.validation?.confidenceScore || 0;
  const validationProvider = result.validation?.validator || "openrouter";
  const validationProviderLabel = validationProvider === "openrouter" ? "OpenRouter" : validationProvider;
  const validationMetricRows = [
    ["Grammar Correctness", "grammar_correctness"],
    ["Clarity & Readability", "clarity_readability"],
    ["Sentence Structure", "sentence_structure"],
    ["Completeness", "completeness"],
    ["Noise Reduction", "noise_reduction"],
  ].map(([label, key]) => ({
    label,
    value: Number(result.validation?.metricScores?.[key] || 0),
  }));
  const validationHeadline =
    validationVerdict === "pass"
      ? "OpenRouter approved the transcript."
      : validationVerdict === "review"
        ? "OpenRouter recommends reviewing the transcript."
        : validationVerdict === "fail"
          ? "OpenRouter does not consider the transcript final yet."
          : "OpenRouter validation is unavailable.";
  const qualityClass =
    validationVerdict === "pass"
      ? "quality-good"
      : validationVerdict === "review"
        ? "quality-medium"
        : "quality-poor";
  const audioQualityLabel = result.audioQuality?.label || "unknown";
  const audioQualityScore = result.audioQuality?.score || 0;
  const uniqueSpeakers = useMemo(
    () => [...new Set((result.segments || []).map((segment) => Number(segment.speaker || 1)))].sort((a, b) => a - b),
    [result.segments]
  );
  const renamedSpeakerTranscript = useMemo(
    () => applySpeakerAliasesToText(result.speakerTranscript, speakerAliases),
    [result.speakerTranscript, speakerAliases]
  );
  const renamedTimestampedTranscript = useMemo(
    () => formatTimestampedSegments(result.segments, speakerAliases),
    [result.segments, speakerAliases]
  );
  const filteredSegments = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return result.segments || [];
    return (result.segments || []).filter((segment) => {
      const label = speakerAliases[segment.speaker] || `Person ${segment.speaker}`;
      return `${label} ${segment.text}`.toLowerCase().includes(query);
    });
  }, [result.segments, searchQuery, speakerAliases]);
  const comparisonStats = useMemo(
    () => [
      { label: "Raw Words", value: countWords(result.rawTranscript) },
      { label: "Speaker View Words", value: countWords(result.speakerTranscript) },
      { label: "Final Words", value: countWords(result.correctedTranscript) },
      { label: "Search Matches", value: filteredSegments.length },
    ],
    [result.rawTranscript, result.speakerTranscript, result.correctedTranscript, filteredSegments.length]
  );
  const importantMoments = useMemo(() => {
    const combined = [keyPoints, decisions, actionItems].join("\n");
    return combined
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line && line !== "- None")
      .slice(0, 8);
  }, [keyPoints, decisions, actionItems]);

  return (
    <div className="shell">
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">Transcription as a Service</span>
          <h1>TaaS</h1>
          <p>
            A full workflow for transcription, comparison, correction, validation, search, notes, translation,
            and export from one polished workspace.
          </p>
          <div className="badge-row">
            <span className="badge">Whisper + Gemini + OpenRouter</span>
            <span className="badge">Search + Timestamps</span>
            <span className="badge">Download Ready</span>
          </div>
        </div>
        <div className="hero-grid">
          <MetricCard label="Pipeline" value="Transcribe, validate, refine, summarize, translate, and voice" />
          <MetricCard label="Modes" value="Meeting, lecture, interview, and discussion workflows" />
          <MetricCard
            label="Accuracy"
            value="Compare transcript quality with validation and without validation before final delivery"
          />
        </div>
      </section>

      <main className="workspace">
        <aside className="panel control-panel">
          <div className="section-heading">Control Panel</div>
          <p className="section-copy">
            Upload your recording, choose transcription options, then review, edit, validate, and export the
            final result.
          </p>

          <form onSubmit={handleProcessSubmit} className="stack">
            <label className="field">
              <span>Meeting video or audio</span>
              <input
                type="file"
                accept="video/*,audio/*"
                onChange={(event) => setVideoFile(event.target.files?.[0] || null)}
              />
            </label>

            <label className="field">
              <span>Expected speaker count</span>
              <select value={speakerCount} onChange={(event) => setSpeakerCount(event.target.value)}>
                <option value="auto">Auto Detect</option>
                <option value="1">1 speaker</option>
                <option value="2">2 speakers</option>
                <option value="3">3 speakers</option>
                <option value="4">4 speakers</option>
                <option value="5">5 speakers</option>
                <option value="6">6 speakers</option>
              </select>
            </label>

            <label className="field">
              <span>Transcription language</span>
              <select value={transcriptionLanguage} onChange={(event) => setTranscriptionLanguage(event.target.value)}>
                {LANGUAGE_OPTIONS.map((option) => (
                  <option key={option.code} value={option.code}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Domain mode</span>
              <select value={domainMode} onChange={(event) => setDomainMode(event.target.value)}>
                {DOMAIN_OPTIONS.map((option) => (
                  <option key={option.code} value={option.code}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Meeting note style</span>
              <select value={summaryStyle} onChange={(event) => setSummaryStyle(event.target.value)}>
                {SUMMARY_STYLES.map((option) => (
                  <option key={option.code} value={option.code}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <button className="primary-button" type="submit" disabled={isProcessing}>
              {isProcessing ? "Running pipeline..." : "Run Full Pipeline"}
            </button>
          </form>

          <div className="status-card">
            <strong>Pipeline status</strong>
            <p>{status}</p>
          </div>

          <div className="status-card">
            <strong>AI providers</strong>
            <p>Correction, translation, and notes: Gemini</p>
            <p>Independent final validation: OpenRouter</p>
          </div>

          <div className="quality-card">
            <div className="quality-header">
              <strong>Final Validation</strong>
              <span className={`quality-badge ${qualityClass}`}>{validationVerdict}</span>
            </div>
            <p>{validationHeadline}</p>
            <p>{validationSummary}</p>
            <div className="progress-track">
              <div className={`progress-fill ${qualityClass}`} style={{ width: `${validationConfidence}%` }} />
            </div>
            <small>Confidence {validationConfidence}/100 | Validator: {validationProviderLabel}</small>
            <div className="metric-score-list">
              {validationMetricRows.map((metric) => (
                <div className="metric-score-row" key={metric.label}>
                  <span>{metric.label}</span>
                  <strong>{metric.value}/100</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="quality-card">
            <div className="quality-header">
              <strong>Audio Quality</strong>
              <span
                className={`quality-badge ${
                  audioQualityScore >= 75 ? "quality-good" : audioQualityScore >= 45 ? "quality-medium" : "quality-poor"
                }`}
              >
                {audioQualityLabel}
              </span>
            </div>
            <p>{result.audioQuality?.summary || "Run the pipeline to inspect audio quality."}</p>
            <div className="progress-track">
              <div
                className={`progress-fill ${
                  audioQualityScore >= 75 ? "quality-good" : audioQualityScore >= 45 ? "quality-medium" : "quality-poor"
                }`}
                style={{ width: `${audioQualityScore}%` }}
              />
            </div>
            <small>Score {audioQualityScore}/100</small>
          </div>

          <div className="hint-card">
            The corrected transcript is editable in the workspace. You can refine it manually, then run OpenRouter
            validation again before exporting.
            {!geminiConfigured
              ? " Gemini is not configured on the backend right now, so notes, correction follow-ups, and translation are limited."
              : ""}
            {!openRouterConfigured
              ? " OpenRouter is not configured on the backend right now, so independent validation is limited."
              : ""}
          </div>

          <div className="action-stack">
            <button
              disabled={!hasTranscript || isValidatingFinal || !openRouterConfigured}
              onClick={handleValidateFinalTranscript}
            >
              {isValidatingFinal ? "Validating Final Transcript..." : "Validate Final Transcript"}
            </button>
            <button
              disabled={
                !hasTranscript ||
                isRefiningWithFeedback ||
                !geminiConfigured ||
                !openRouterConfigured ||
                (validationIssues.length === 0 && validationCriticalIssues.length === 0 && suggestedActions.length === 0)
              }
              onClick={handleRefineWithFeedback}
            >
              {isRefiningWithFeedback ? "Refining With Feedback..." : "Refine With Feedback"}
            </button>
            <button disabled={!hasTranscript || isSimplifying || !geminiConfigured} onClick={handleSimplifyTranscript}>
              {isSimplifying ? "Simplifying..." : "Simplified Explanation"}
            </button>
            <button disabled={!hasTranscript || isGeneratingEnglish} onClick={handleGenerateEnglishVoice}>
              {isGeneratingEnglish ? "Generating English Voice..." : "Create English Voice"}
            </button>
            <button disabled={!hasTranscript || isTranslating || !geminiConfigured} onClick={handleTranslate}>
              {isTranslating ? "Translating..." : `Translate to ${getLanguageLabel(selectedLanguage, TRANSLATION_OPTIONS)}`}
            </button>
            <button disabled={!translatedText.trim() || isGeneratingHindi} onClick={handleGenerateTargetVoice}>
              {isGeneratingHindi ? "Generating Voice..." : `Generate ${getLanguageLabel(selectedLanguage, TRANSLATION_OPTIONS)} Voice`}
            </button>
            <button disabled={!hasTranscript || !geminiConfigured} onClick={handleGenerateMinutes}>
              Generate Meeting Notes
            </button>
          </div>

          <div className="export-grid">
            <button className="secondary-button" disabled={!hasTranscript} onClick={downloadTranscriptTxt}>
              Download Transcript TXT
            </button>
            <button className="secondary-button" disabled={!minutes && !keyPoints} onClick={downloadNotesMarkdown}>
              Download Notes MD
            </button>
            <button className="secondary-button" disabled={!hasTranscript} onClick={downloadReportJson}>
              Export Report JSON
            </button>
            <button className="secondary-button" disabled={!hasTranscript} onClick={downloadEvaluationHtml}>
              Export Report HTML
            </button>
          </div>
        </aside>

        <section className="panel main-panel">
          <div className="section-heading">Workspace</div>
          <p className="section-copy">
            Review the transcript, compare stages, rename speakers, search timestamps, generate notes, run live
            transcription, and export the final result from one place.
          </p>

          <div className="tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className={selectedTab === tab.id ? "tab active" : "tab"}
                onClick={() => setSelectedTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {selectedTab === "transcript" && (
            <div className="tab-panel">
              <div className="transcript-grid">
                <TextPanel label="Raw Whisper Transcript" value={result.rawTranscript} />
                <TextPanel label="Speaker-Labeled Transcript" value={renamedSpeakerTranscript} />
              </div>

              <TextPanel
                label="Final Corrected Transcript"
                value={result.correctedTranscript}
                tall
                readOnly={false}
                onChange={(value) =>
                  setResult((current) => ({
                    ...current,
                    correctedTranscript: value,
                  }))
                }
              />

              <div className="review-grid review-grid-three">
                <div className="info-panel">
                  <strong>Search Transcript</strong>
                  <input
                    className="search-input"
                    type="text"
                    value={searchQuery}
                    onChange={(event) => setSearchQuery(event.target.value)}
                    placeholder="Search within timestamped transcript..."
                  />
                  <p>{filteredSegments.length} segment(s) match the current search.</p>
                </div>

                <div className="info-panel">
                  <strong>Speaker Rename</strong>
                  {uniqueSpeakers.length > 0 ? (
                    <div className="speaker-rename-list">
                      {uniqueSpeakers.map((speakerId) => (
                        <label key={speakerId} className="speaker-rename-item">
                          <span>{`Person ${speakerId}`}</span>
                          <input
                            type="text"
                            value={speakerAliases[speakerId] || `Person ${speakerId}`}
                            onChange={(event) => updateSpeakerAlias(speakerId, event.target.value)}
                          />
                        </label>
                      ))}
                    </div>
                  ) : (
                    <p>Run transcription first to rename speakers.</p>
                  )}
                </div>

                <div className="info-panel">
                  <strong>Pipeline Meta</strong>
                  <p>Language: {getLanguageLabel(transcriptionLanguage, LANGUAGE_OPTIONS)}</p>
                  <p>Domain: {getDomainLabel(domainMode)}</p>
                  <p>Summary style: {getSummaryStyleLabel(summaryStyle)}</p>
                </div>
              </div>

              <SegmentTimeline segments={filteredSegments} speakerAliases={speakerAliases} searchQuery={searchQuery} />

              <div className="review-grid review-grid-three">
                <div className="info-panel">
                  <strong>Validation Issues</strong>
                  {validationIssues.length > 0 ? (
                    <ul>
                      {validationIssues.map((issue) => (
                        <li key={issue}>{issue}</li>
                      ))}
                    </ul>
                  ) : (
                    <p>No validation issues reported.</p>
                  )}
                </div>

                <div className="info-panel">
                  <strong>Strengths</strong>
                  {validationStrengths.length > 0 ? (
                    <ul>
                      {validationStrengths.map((strength) => (
                        <li key={strength}>{strength}</li>
                      ))}
                    </ul>
                  ) : (
                    <p>No strengths highlighted by OpenRouter.</p>
                  )}
                </div>

                <div className="info-panel">
                  <strong>Next Steps</strong>
                  {suggestedActions.length > 0 ? (
                    <ul>
                      {suggestedActions.map((action) => (
                        <li key={action}>{action}</li>
                      ))}
                    </ul>
                  ) : result.errors.length > 0 ? (
                    <ul>
                      {result.errors.map((error) => (
                        <li key={error}>{error}</li>
                      ))}
                    </ul>
                  ) : (
                    <p>No immediate follow-up actions suggested.</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {selectedTab === "compare" && (
            <div className="tab-panel">
              <div className="comparison-metrics">
                {comparisonStats.map((metric) => (
                  <MetricCard key={metric.label} label={metric.label} value={String(metric.value)} />
                ))}
              </div>
              <div className="notes-grid notes-grid-three">
                <TextPanel label="Raw Transcript" value={result.rawTranscript} tall />
                <TextPanel label="Speaker Transcript" value={renamedSpeakerTranscript} tall />
                <TextPanel label="Final Transcript" value={result.correctedTranscript} tall />
              </div>
              <TextPanel label="Timestamped Transcript" value={renamedTimestampedTranscript} tall />
            </div>
          )}

          {selectedTab === "simplified" && (
            <div className="tab-panel">
              <div className="status-card">
                <strong>Simplified Explanation Mode</strong>
                <p>
                  This mode rewrites the final transcript in shorter sentences and simpler language for easier
                  understanding.
                </p>
              </div>
              <div className="notes-grid">
                <TextPanel label="Final Transcript" value={result.correctedTranscript} tall />
                <TextPanel label="Simplified Explanation" value={simplifiedText} tall />
              </div>
            </div>
          )}

          {selectedTab === "voice" && (
            <div className="tab-panel">
              <div className="status-card">
                <strong>Translation + voice status</strong>
                <p>{translationStatus}</p>
              </div>
              <label className="field">
                <span>Translation language</span>
                <select value={selectedLanguage} onChange={(event) => setSelectedLanguage(event.target.value)}>
                  {TRANSLATION_OPTIONS.map((option) => (
                    <option key={option.code} value={option.code}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <div className="inline-actions">
                <button
                  className="secondary-button"
                  disabled={!hasTranscript || isTranslating || !geminiConfigured}
                  onClick={handleTranslate}
                >
                  {isTranslating ? "Translating..." : `Translate to ${getLanguageLabel(selectedLanguage, TRANSLATION_OPTIONS)}`}
                </button>
                <button
                  className="secondary-button"
                  disabled={!translatedText.trim() || isGeneratingHindi}
                  onClick={handleGenerateTargetVoice}
                >
                  {isGeneratingHindi ? "Generating..." : `Generate ${getLanguageLabel(selectedLanguage, TRANSLATION_OPTIONS)} Voice`}
                </button>
              </div>
              <div className="audio-grid">
                <AudioCard label="English Audio" src={englishAudioUrl} />
                <AudioCard label={`${getLanguageLabel(selectedLanguage, TRANSLATION_OPTIONS)} Audio`} src={targetAudioUrl} />
              </div>
              <TextPanel label={`${getLanguageLabel(selectedLanguage, TRANSLATION_OPTIONS)} Translation`} value={translatedText} tall />
            </div>
          )}

          {selectedTab === "notes" && (
            <div className="tab-panel">
              <div className="notes-grid notes-grid-three">
                <RichTextPanel label="Minutes of Meeting" value={minutes} tall />
                <RichTextPanel label="Key Points" value={keyPoints} tall />
                <RichTextPanel label="Decisions" value={decisions} tall />
              </div>
              <div className="notes-grid">
                <RichTextPanel label="Action Items" value={actionItems} tall />
                <article className="text-panel rich-text-panel tall">
                  <div className="panel-label">Important Moments</div>
                  <div className="rich-text-content">
                    {importantMoments.length > 0 ? (
                      importantMoments.map((line, index) => <RichLine key={`moment-${index}`} line={line} />)
                    ) : (
                      <p className="rich-placeholder">Generate notes to surface important moments.</p>
                    )}
                  </div>
                </article>
              </div>
            </div>
          )}

          {selectedTab === "live" && <LiveMode apiBase={API_BASE} isAvailable={liveModeAvailable} />}
        </section>
      </main>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <p>{value}</p>
    </article>
  );
}

function TextPanel({ label, value, tall = false, readOnly = true, onChange }) {
  return (
    <article className={tall ? "text-panel tall" : "text-panel"}>
      <div className="panel-label">{label}</div>
      <textarea
        readOnly={readOnly}
        value={value || ""}
        onChange={readOnly ? undefined : (event) => onChange?.(event.target.value)}
        placeholder={`${label} will appear here.`}
      />
    </article>
  );
}

function RichTextPanel({ label, value, tall = false }) {
  const lines = (value || "").split("\n");
  return (
    <article className={tall ? "text-panel rich-text-panel tall" : "text-panel rich-text-panel"}>
      <div className="panel-label">{label}</div>
      <div className="rich-text-content">
        {lines.some((line) => line.trim()) ? (
          lines.map((line, index) => <RichLine key={`${label}-${index}`} line={line} />)
        ) : (
          <p className="rich-placeholder">{label} will appear here.</p>
        )}
      </div>
    </article>
  );
}

function RichLine({ line }) {
  const trimmed = line.trim();
  if (!trimmed) return <div className="rich-line spacer" />;

  const isBullet = trimmed.startsWith("- ");
  const content = isBullet ? trimmed.slice(2) : trimmed;
  const parts = content.split(/(\*\*.*?\*\*)/g).filter(Boolean);

  return (
    <p className={isBullet ? "rich-line bullet" : "rich-line"}>
      {isBullet ? <span className="bullet-mark">•</span> : null}
      <span>
        {parts.map((part, index) => {
          const isHighlighted = part.startsWith("**") && part.endsWith("**") && part.length > 4;
          const text = isHighlighted ? part.slice(2, -2) : part;
          return isHighlighted ? (
            <strong className="highlight-mark" key={index}>
              {text}
            </strong>
          ) : (
            <span key={index}>{text}</span>
          );
        })}
      </span>
    </p>
  );
}

function AudioCard({ label, src }) {
  return (
    <article className="audio-card">
      <div className="panel-label">{label}</div>
      {src ? <audio controls src={src} /> : <p>Audio output will appear here.</p>}
    </article>
  );
}

function normalizeValidation(validation) {
  return {
    ...initialResult.validation,
    ...(validation || {}),
    validator: "openrouter",
  };
}

function SegmentTimeline({ segments, speakerAliases, searchQuery }) {
  return (
    <article className="timeline-card">
      <div className="panel-label">Timestamped Transcript</div>
      {segments?.length ? (
        <div className="timeline-list">
          {segments.map((segment, index) => (
            <div className="timeline-item" key={`${segment.start}-${segment.end}-${index}`}>
              <div className="timeline-meta">
                <span className="timeline-time">
                  {formatClock(segment.start)} - {formatClock(segment.end)}
                </span>
                <span className="timeline-speaker">{speakerAliases[segment.speaker] || `Person ${segment.speaker}`}</span>
              </div>
              <p className="timeline-text">
                <HighlightText text={segment.text || ""} query={searchQuery} />
              </p>
            </div>
          ))}
        </div>
      ) : (
        <p className="rich-placeholder">Timestamped transcript will appear here.</p>
      )}
    </article>
  );
}

function HighlightText({ text, query }) {
  const cleanQuery = (query || "").trim();
  if (!cleanQuery) return text;

  const pattern = new RegExp(`(${escapeRegExp(cleanQuery)})`, "ig");
  const parts = String(text || "").split(pattern);
  return parts.map((part, index) =>
    part.toLowerCase() === cleanQuery.toLowerCase() ? <mark key={index}>{part}</mark> : <span key={index}>{part}</span>
  );
}

function applySpeakerAliasesToText(text, aliases) {
  let output = text || "";
  Object.entries(aliases || {}).forEach(([speakerId, label]) => {
    output = output.replaceAll(`Person ${speakerId}:`, `${label}:`);
  });
  return output;
}

function formatTimestampedSegments(segments, aliases) {
  return (segments || [])
    .map((segment) => {
      const speaker = aliases[segment.speaker] || `Person ${segment.speaker}`;
      return `[${formatClock(segment.start)} - ${formatClock(segment.end)}] ${speaker}: ${segment.text || ""}`;
    })
    .join("\n");
}

function formatClock(value) {
  const total = Math.max(0, Math.round(Number(value || 0)));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const seconds = String(total % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function countWords(text) {
  return String(text || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
}

function downloadFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function getLanguageLabel(code, options) {
  return options.find((option) => option.code === code)?.label || code;
}

function getDomainLabel(code) {
  return DOMAIN_OPTIONS.find((option) => option.code === code)?.label || code;
}

function getSummaryStyleLabel(code) {
  return SUMMARY_STYLES.find((option) => option.code === code)?.label || code;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export default App;
