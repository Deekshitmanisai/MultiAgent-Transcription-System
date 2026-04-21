import { useEffect, useMemo, useRef, useState } from "react";


const LIVE_SOURCE_OPTIONS = [
  { code: "both", label: "Meeting + My Microphone" },
  { code: "system", label: "System Audio (Google Meet)" },
  { code: "microphone", label: "Microphone Only" },
];


function toWebSocketUrl(apiBase, source) {
  if (!apiBase) return "ws://127.0.0.1:8000/live";
  const suffix = `/live?source=${encodeURIComponent(source || "system")}`;
  if (apiBase.startsWith("https://")) return apiBase.replace("https://", "wss://") + suffix;
  if (apiBase.startsWith("http://")) return apiBase.replace("http://", "ws://") + suffix;
  return `${apiBase}${suffix}`;
}


function formatChunkLine(chunk) {
  const timeLabel = chunk.timestamp ? `[${chunk.timestamp}] ` : "";
  return `${timeLabel}${chunk.english || ""}`.trim();
}


export default function LiveMode({ apiBase, isAvailable }) {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("Ready to start live transcription.");
  const [source, setSource] = useState("both");
  const [englishChunks, setEnglishChunks] = useState([]);
  const [summary, setSummary] = useState("");
  const socketRef = useRef(null);

  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
    };
  }, []);

  const englishTranscript = useMemo(() => englishChunks.join("\n"), [englishChunks]);

  function handleStart() {
    if (isRunning || !isAvailable) return;

    setEnglishChunks([]);
    setSummary("");
    setStatus("Connecting to live transcription...");

    const socket = new WebSocket(toWebSocketUrl(apiBase, source));
    socketRef.current = socket;

    socket.onopen = () => {
      setIsRunning(true);
      setStatus("Live Transcription Running...");
    };

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "status") {
        setStatus(payload.message || "Live status updated.");
        return;
      }
      if (payload.type === "chunk") {
        if (payload.english) {
          setEnglishChunks((current) => [...current, formatChunkLine(payload)]);
        }
        return;
      }
      if (payload.type === "summary") {
        setSummary(payload.summary || "");
        return;
      }
      if (payload.type === "error") {
        setStatus(payload.message || "Live transcription failed.");
        setIsRunning(false);
      }
    };

    socket.onerror = () => {
      setStatus("Live transcription connection failed.");
      setIsRunning(false);
    };

    socket.onclose = () => {
      setIsRunning(false);
      socketRef.current = null;
      setStatus((current) =>
        current === "Live Transcription Running..." ? "Live transcription stopped." : current
      );
    };
  }

  function handleStop() {
    if (!socketRef.current) return;
    setStatus("Stopping live transcription and generating summary...");
    socketRef.current.send(JSON.stringify({ action: "stop" }));
  }

  return (
    <div className="tab-panel">
      <div className="status-card">
        <strong>Live Mode</strong>
        <p>{status}</p>
      </div>

      {!isAvailable ? (
        <div className="hint-card">
          Live audio capture is unavailable on the backend. Install `sounddevice`, allow audio device access,
          and restart the backend.
        </div>
      ) : null}

      <label className="field">
        <span>Live audio source</span>
        <select value={source} onChange={(event) => setSource(event.target.value)} disabled={isRunning}>
          {LIVE_SOURCE_OPTIONS.map((option) => (
            <option key={option.code} value={option.code}>
              {option.label}
            </option>
          ))}
        </select>
      </label>

      <div className="inline-actions">
        <button className="secondary-button" disabled={isRunning || !isAvailable} onClick={handleStart}>
          Start Live Mode
        </button>
        <button className="secondary-button" disabled={!isRunning} onClick={handleStop}>
          Stop Live Mode
        </button>
      </div>

      <article className="text-panel tall">
        <div className="panel-label">Live English Transcript</div>
        <textarea readOnly value={englishTranscript} placeholder="Live English transcript will append here." />
      </article>

      <article className="text-panel rich-text-panel tall">
        <div className="panel-label">Meeting Summary</div>
        <div className="rich-text-content">
          {summary.trim() ? (
            summary.split("\n").map((line, index) => (
              <p className="rich-line" key={`live-summary-${index}`}>
                <span>{line}</span>
              </p>
            ))
          ) : (
            <p className="rich-placeholder">Stop live mode to generate the meeting summary.</p>
          )}
        </div>
      </article>
    </div>
  );
}
