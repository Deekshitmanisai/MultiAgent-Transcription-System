import numpy as np


def _segment_duration(segment):
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    return max(0.0, end - start)


def _turn_taking_speakers(segments):
    if not segments:
        return []

    def is_question(txt):
        t = (txt or "").strip()
        if not t:
            return False
        if t.endswith("?"):
            return True
        starters = (
            "what ",
            "why ",
            "how ",
            "when ",
            "where ",
            "who ",
            "can ",
            "do ",
            "did ",
            "are ",
            "is ",
            "will ",
            "would ",
            "could ",
            "should ",
        )
        return t.lower().startswith(starters)

    def looks_like_answer(txt):
        t = (txt or "").strip().lower()
        if not t:
            return False
        answer_starters = (
            "yes",
            "no",
            "yeah",
            "yep",
            "nope",
            "i ",
            "i'm ",
            "my ",
            "because",
            "it is ",
            "today is ",
        )
        return t.startswith(answer_starters)

    speakers = np.ones(len(segments), dtype=int)
    last_end = None
    cur = 1
    prev_was_question = False

    for i, segment in enumerate(segments):
        start = float(segment.get("start", 0.0))
        gap = (start - last_end) if last_end is not None else 0.0
        txt = (segment.get("text") or "").strip()

        if gap >= 0.8:
            cur = 2 if cur == 1 else 1
        else:
            if prev_was_question and looks_like_answer(txt):
                cur = 2 if cur == 1 else 1
            elif is_question(txt) and i > 0 and looks_like_answer((segments[i - 1].get("text") or "")):
                cur = 2 if cur == 1 else 1

        speakers[i] = cur
        prev_was_question = is_question(txt)
        last_end = float(segment.get("end", start))

    return [{**segment, "speaker": int(speakers[i])} for i, segment in enumerate(segments)]


def _extract_segment_features(y, sr, segment, librosa):
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    duration = max(0.0, end - start)
    if duration <= 0:
        return None

    min_window = 1.2
    center = 0.5 * (start + end)
    half_window = 0.5 * max(min_window, duration)
    sample_a = int(max(0, (center - half_window) * sr))
    sample_b = int(min(len(y), (center + half_window) * sr))
    if sample_b - sample_a < int(0.35 * sr):
        return None

    seg = y[sample_a:sample_b]
    if not np.any(np.abs(seg) > 1e-5):
        return None

    hop_length = 160
    fmin = 50
    fmax = 400

    try:
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        spectral_centroid = librosa.feature.spectral_centroid(y=seg, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(seg)
        rms = librosa.feature.rms(y=seg, hop_length=hop_length)

        try:
            spectral_contrast = librosa.feature.spectral_contrast(y=seg, sr=sr)
        except Exception:
            spectral_contrast = np.zeros((7, max(1, mfcc.shape[1])), dtype=np.float32)

        try:
            chroma = librosa.feature.chroma_stft(y=seg, sr=sr)
        except Exception:
            chroma = np.zeros((12, max(1, mfcc.shape[1])), dtype=np.float32)

        f0, voiced_flag, _ = librosa.pyin(seg, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
        f0 = np.array([]) if f0 is None else f0
        voiced_flag = np.array([]) if voiced_flag is None else voiced_flag
        f0_valid = f0[np.isfinite(f0)] if f0.size else np.array([])
        voiced_ratio = float(np.mean(voiced_flag.astype(np.float32))) if voiced_flag.size else 0.0

        pause_before = float(segment.get("_pause_before", 0.0))
        text = (segment.get("text") or "").strip()
        text_len = len(text)

        feature_vector = np.concatenate(
            [
                mfcc.mean(axis=1),
                mfcc.std(axis=1),
                delta.mean(axis=1),
                delta2.mean(axis=1),
                spectral_centroid.mean(axis=1),
                spectral_bandwidth.mean(axis=1),
                spectral_rolloff.mean(axis=1),
                spectral_contrast.mean(axis=1),
                chroma.mean(axis=1),
                np.array(
                    [
                        float(np.mean(rms)) if rms.size else 0.0,
                        float(np.std(rms)) if rms.size else 0.0,
                        float(np.mean(zero_crossing)) if zero_crossing.size else 0.0,
                        float(np.std(zero_crossing)) if zero_crossing.size else 0.0,
                        float(np.mean(f0_valid)) if f0_valid.size else 0.0,
                        float(np.std(f0_valid)) if f0_valid.size else 0.0,
                        voiced_ratio,
                        duration,
                        pause_before,
                        min(400.0, float(text_len)) / 400.0,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        return feature_vector.astype(np.float32)
    except Exception:
        return None


def _normalize_features(X, StandardScaler):
    if X.shape[0] < 2:
        return X
    return StandardScaler().fit_transform(X)


def _cluster_features(X, n_speakers, AgglomerativeClustering):
    if X.shape[0] < 2:
        return np.zeros(X.shape[0], dtype=int)

    cluster_count = max(1, min(int(n_speakers), X.shape[0]))
    if cluster_count == 1:
        return np.zeros(X.shape[0], dtype=int)

    model = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward")
    return model.fit_predict(X)


def _estimate_speaker_count(X, AgglomerativeClustering, silhouette_score, max_speakers=6):
    sample_count = int(X.shape[0])
    if sample_count < 2:
        return 1
    if sample_count == 2:
        return 2

    upper_bound = max(2, min(int(max_speakers), sample_count - 1))
    best_count = 2
    best_score = float("-inf")

    for cluster_count in range(2, upper_bound + 1):
        try:
            labels = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward").fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(X, labels))
            if score > best_score:
                best_score = score
                best_count = cluster_count
        except Exception:
            continue

    return best_count


def _assign_missing_labels(speakers, valid_idx, segments):
    if not valid_idx:
        return speakers

    for i in range(len(speakers)):
        if speakers[i] != 1 or i in valid_idx:
            continue

        left = None
        right = None

        for j in range(i - 1, -1, -1):
            if speakers[j] > 0:
                left = speakers[j]
                break
        for j in range(i + 1, len(speakers)):
            if speakers[j] > 0:
                right = speakers[j]
                break

        text = (segments[i].get("text") or "").strip()
        if left is not None and right is not None and left == right:
            speakers[i] = left
        elif left is not None:
            speakers[i] = left
        elif right is not None:
            speakers[i] = right
        elif text:
            speakers[i] = 1

    return speakers


def _smooth_speaker_sequence(speakers, segments):
    if len(speakers) < 3:
        return speakers

    smoothed = speakers.copy()
    durations = [_segment_duration(segment) for segment in segments]

    for i in range(1, len(smoothed) - 1):
        left = smoothed[i - 1]
        cur = smoothed[i]
        right = smoothed[i + 1]
        cur_duration = durations[i]

        if left == right and cur != left and cur_duration <= 1.6:
            smoothed[i] = left
            continue

        if cur != left and cur != right and cur_duration <= 0.75:
            left_duration = durations[i - 1]
            right_duration = durations[i + 1]
            smoothed[i] = left if left_duration >= right_duration else right

    run_start = 0
    while run_start < len(smoothed):
        run_end = run_start + 1
        while run_end < len(smoothed) and smoothed[run_end] == smoothed[run_start]:
            run_end += 1

        run_duration = sum(durations[run_start:run_end])
        run_len = run_end - run_start
        if run_duration <= 0.9 and run_len <= 2:
            left_label = smoothed[run_start - 1] if run_start > 0 else None
            right_label = smoothed[run_end] if run_end < len(smoothed) else None
            if left_label is not None and right_label == left_label:
                for i in range(run_start, run_end):
                    smoothed[i] = left_label
            elif left_label is not None and right_label is None:
                for i in range(run_start, run_end):
                    smoothed[i] = left_label
            elif right_label is not None and left_label is None:
                for i in range(run_start, run_end):
                    smoothed[i] = right_label

        run_start = run_end

    return smoothed


def diarize_segments(audio_path, segments, n_speakers=2):
    if not segments:
        return []

    enriched_segments = []
    last_end = None
    for segment in segments:
        start = float(segment.get("start", 0.0))
        pause_before = max(0.0, start - last_end) if last_end is not None else 0.0
        enriched = dict(segment)
        enriched["_pause_before"] = pause_before
        enriched_segments.append(enriched)
        last_end = float(segment.get("end", start))

    try:
        import librosa
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        print(f"Diarization dependency issue: {exc}")
        return _turn_taking_speakers(segments)

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as exc:
        print(f"Audio load error: {exc}")
        return [{**segment, "speaker": 1} for segment in segments]

    features = []
    valid_idx = []
    for i, segment in enumerate(enriched_segments):
        feature_vector = _extract_segment_features(y, sr, segment, librosa)
        if feature_vector is None:
            continue
        features.append(feature_vector)
        valid_idx.append(i)

    if len(features) < 2:
        return _turn_taking_speakers(segments)

    X = np.stack(features, axis=0)
    X = _normalize_features(X, StandardScaler)

    target_speakers = n_speakers
    if target_speakers in (None, "", 0, "0", "auto"):
        target_speakers = _estimate_speaker_count(
            X,
            AgglomerativeClustering=AgglomerativeClustering,
            silhouette_score=silhouette_score,
            max_speakers=6,
        )

    try:
        labels = _cluster_features(
            X,
            n_speakers=target_speakers,
            AgglomerativeClustering=AgglomerativeClustering,
        )
    except Exception as exc:
        print(f"Diarization clustering error: {exc}")
        return _turn_taking_speakers(segments)

    speakers = np.ones(len(segments), dtype=int)
    for idx, label in zip(valid_idx, labels):
        speakers[idx] = int(label) + 1

    speakers = _assign_missing_labels(speakers, set(valid_idx), segments)
    speakers = _smooth_speaker_sequence(speakers, segments)

    unique = set(int(speakers[i]) for i in valid_idx) if valid_idx else {1}
    if len(unique) < 2 and len(segments) >= 2:
        return _turn_taking_speakers(segments)

    return [{**segment, "speaker": int(speakers[i])} for i, segment in enumerate(segments)]


def format_diarized_transcript(segments_with_speakers, speaker_prefix="Person"):
    segs = segments_with_speakers or []
    if not segs:
        return ""

    speakers = [int(segment.get("speaker", 1)) for segment in segs]
    unique = set(speakers)

    if len(unique) <= 1:
        parts = []
        for segment in segs:
            txt = (segment.get("text") or "").strip()
            if txt:
                parts.append(txt)
        return " ".join(parts).strip()

    merged = []
    cur_spk = None
    cur_txt = []
    for segment in segs:
        spk = int(segment.get("speaker", 1))
        txt = (segment.get("text") or "").strip()
        if not txt:
            continue
        if cur_spk is None:
            cur_spk = spk
            cur_txt = [txt]
            continue
        if spk == cur_spk:
            cur_txt.append(txt)
        else:
            merged.append((cur_spk, " ".join(cur_txt).strip()))
            cur_spk = spk
            cur_txt = [txt]

    if cur_spk is not None and cur_txt:
        merged.append((cur_spk, " ".join(cur_txt).strip()))

    lines = [f"{speaker_prefix} {spk}: {txt}" for spk, txt in merged if txt]
    return "\n".join(lines).strip()
