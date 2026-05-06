import numpy as np


def _segment_duration(segment):
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    return max(0.0, end - start)


def _heuristic_turn_taking_speakers(segments):
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


def _cosine_distance(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 1.0
    similarity = float(np.dot(a, b) / denom)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def _build_segment_profile(y, sr, segment, librosa):
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    duration = max(0.0, end - start)
    if duration <= 0:
        return None

    min_window = 1.6
    center = 0.5 * (start + end)
    half_window = 0.5 * max(min_window, duration)
    sample_a = int(max(0, (center - half_window) * sr))
    sample_b = int(min(y.shape[-1], (center + half_window) * sr))
    if sample_b - sample_a < int(0.45 * sr):
        return None

    if getattr(y, "ndim", 1) > 1:
        seg_channels = y[:, sample_a:sample_b]
        seg = np.mean(seg_channels, axis=0)
        channel_rms = np.sqrt(np.mean(np.square(seg_channels), axis=1) + 1e-8)
        total_channel_rms = float(np.sum(channel_rms))
        if total_channel_rms > 1e-8 and len(channel_rms) >= 2:
            channel_balance = float((channel_rms[0] - channel_rms[1]) / total_channel_rms)
        else:
            channel_balance = 0.0
    else:
        seg = y[sample_a:sample_b]
        channel_balance = 0.0

    if not np.any(np.abs(seg) > 1e-5):
        return None

    hop_length = 160
    fmin = 50
    fmax = 400

    try:
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        spectral_centroid = librosa.feature.spectral_centroid(y=seg, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(seg)
        rms = librosa.feature.rms(y=seg, hop_length=hop_length)

        f0, voiced_flag, _ = librosa.pyin(seg, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
        f0 = np.array([]) if f0 is None else f0
        voiced_flag = np.array([]) if voiced_flag is None else voiced_flag
        f0_valid = f0[np.isfinite(f0)] if f0.size else np.array([])
        voiced_ratio = float(np.mean(voiced_flag.astype(np.float32))) if voiced_flag.size else 0.0
        pitch_mean = float(np.mean(f0_valid)) if f0_valid.size else 0.0
        pitch_std = float(np.std(f0_valid)) if f0_valid.size else 0.0
        rms_mean = float(np.mean(rms)) if rms.size else 0.0
        rms_std = float(np.std(rms)) if rms.size else 0.0
        zcr_mean = float(np.mean(zero_crossing)) if zero_crossing.size else 0.0
        zcr_std = float(np.std(zero_crossing)) if zero_crossing.size else 0.0

        pause_before = float(segment.get("_pause_before", 0.0))
        text = (segment.get("text") or "").strip()
        text_len = len(text)

        feature_vector = np.concatenate(
            [
                mfcc.mean(axis=1),
                mfcc.std(axis=1),
                delta.mean(axis=1),
                spectral_centroid.mean(axis=1),
                spectral_bandwidth.mean(axis=1),
                np.array(
                    [
                        rms_mean,
                        rms_std,
                        zcr_mean,
                        zcr_std,
                        pitch_mean,
                        pitch_std,
                        voiced_ratio,
                        channel_balance,
                        duration,
                        pause_before,
                        min(400.0, float(text_len)) / 400.0,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        return {
            "vector": feature_vector.astype(np.float32),
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "voiced_ratio": voiced_ratio,
            "rms_mean": rms_mean,
            "channel_balance": channel_balance,
            "duration": duration,
            "pause_before": pause_before,
        }
    except Exception:
        return None


def _extract_segment_features(y, sr, segment, librosa):
    profile = _build_segment_profile(y, sr, segment, librosa)
    return None if profile is None else profile["vector"]


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


def _coerce_speaker_count(n_speakers, default=2, max_speakers=6):
    try:
        normalized = int(n_speakers)
    except (TypeError, ValueError):
        normalized = int(default)
    return max(1, min(int(max_speakers), normalized))


def _count_unique_speakers(speakers):
    return len({int(speaker) for speaker in speakers if int(speaker) > 0})


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


def _profile_distance(feature_vector, speaker_profile, profile_meta=None):
    distance = _cosine_distance(feature_vector, speaker_profile["centroid"])

    if profile_meta:
        pitch_mean = float(profile_meta.get("pitch_mean", 0.0) or 0.0)
        speaker_pitch = float(speaker_profile.get("pitch_mean", 0.0) or 0.0)
        if pitch_mean > 0.0 and speaker_pitch > 0.0:
            octave_gap = abs(np.log2(pitch_mean / speaker_pitch))
            distance += min(0.35, octave_gap * 0.25)

        voiced_ratio = float(profile_meta.get("voiced_ratio", 0.0) or 0.0)
        speaker_voiced = float(speaker_profile.get("voiced_ratio", 0.0) or 0.0)
        distance += min(0.15, abs(voiced_ratio - speaker_voiced) * 0.2)

        channel_balance = float(profile_meta.get("channel_balance", 0.0) or 0.0)
        speaker_balance = float(speaker_profile.get("channel_balance", 0.0) or 0.0)
        distance += min(0.2, abs(channel_balance - speaker_balance) * 0.35)

    return float(distance)


def _assign_profiles_online(profiles, max_speakers=2, distance_threshold=0.32):
    if not profiles:
        return []

    assignments = []
    speaker_profiles = []
    previous_speaker = None

    for profile in profiles:
        if profile is None:
            assignments.append(None)
            continue

        feature_vector = profile["vector"]
        if not speaker_profiles:
            speaker_profiles.append(
                {
                    "speaker_id": 1,
                    "centroid": feature_vector.copy(),
                    "pitch_mean": float(profile.get("pitch_mean", 0.0) or 0.0),
                    "voiced_ratio": float(profile.get("voiced_ratio", 0.0) or 0.0),
                    "channel_balance": float(profile.get("channel_balance", 0.0) or 0.0),
                    "count": 1,
                }
            )
            assignments.append(1)
            previous_speaker = 1
            continue

        ranked = sorted(
            (
                _profile_distance(feature_vector, speaker_profile, profile),
                speaker_profile,
            )
            for speaker_profile in speaker_profiles
        )
        best_distance, best_profile = ranked[0]
        previous_profile = None
        if previous_speaker is not None:
            previous_profile = next(
                (speaker for speaker in speaker_profiles if speaker["speaker_id"] == previous_speaker),
                None,
            )

        # Prefer continuity when the previous speaker is still a close match.
        if previous_profile is not None:
            previous_distance = _profile_distance(feature_vector, previous_profile, profile)
            if previous_distance <= best_distance + 0.08:
                best_distance = previous_distance
                best_profile = previous_profile

        chosen_speaker = None
        if best_distance <= distance_threshold:
            chosen_speaker = int(best_profile["speaker_id"])
            count = best_profile["count"]
            best_profile["centroid"] = (best_profile["centroid"] * count + feature_vector) / float(count + 1)
            best_profile["pitch_mean"] = (
                (best_profile["pitch_mean"] * count) + float(profile.get("pitch_mean", 0.0) or 0.0)
            ) / float(count + 1)
            best_profile["voiced_ratio"] = (
                (best_profile["voiced_ratio"] * count) + float(profile.get("voiced_ratio", 0.0) or 0.0)
            ) / float(count + 1)
            best_profile["channel_balance"] = (
                (best_profile["channel_balance"] * count) + float(profile.get("channel_balance", 0.0) or 0.0)
            ) / float(count + 1)
            best_profile["count"] = count + 1
        elif len(speaker_profiles) < max(1, int(max_speakers or 1)):
            chosen_speaker = len(speaker_profiles) + 1
            speaker_profiles.append(
                {
                    "speaker_id": chosen_speaker,
                    "centroid": feature_vector.copy(),
                    "pitch_mean": float(profile.get("pitch_mean", 0.0) or 0.0),
                    "voiced_ratio": float(profile.get("voiced_ratio", 0.0) or 0.0),
                    "channel_balance": float(profile.get("channel_balance", 0.0) or 0.0),
                    "count": 1,
                }
            )
        else:
            chosen_speaker = int(best_profile["speaker_id"])

        assignments.append(chosen_speaker)
        previous_speaker = chosen_speaker

    return assignments


def _assign_missing_labels(speakers, valid_idx, segments):
    if not valid_idx:
        return speakers

    for i in range(len(speakers)):
        if speakers[i] > 0 or i in valid_idx:
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


def _reindex_speakers_by_first_appearance(segments):
    if not segments:
        return []

    speaker_map = {}
    next_speaker_id = 1
    normalized_segments = []

    for segment in segments:
        original_speaker = int(segment.get("speaker", 1) or 1)
        if original_speaker not in speaker_map:
            speaker_map[original_speaker] = next_speaker_id
            next_speaker_id += 1
        normalized_segments.append({**segment, "speaker": int(speaker_map[original_speaker])})

    return normalized_segments


def _fallback_diarization(segments, profiles, n_speakers):
    valid_profiles = [profile for profile in profiles if profile is not None]
    if len(valid_profiles) >= 2:
        fallback_speaker_limit = _coerce_speaker_count(
            n_speakers,
            default=min(2, len(valid_profiles)) or 1,
            max_speakers=max(2, len(valid_profiles)),
        )
        assignments = _assign_profiles_online(valid_profiles, max_speakers=fallback_speaker_limit)
        speakers = np.zeros(len(segments), dtype=int)
        valid_idx = [i for i, profile in enumerate(profiles) if profile is not None]
        for idx, speaker_id in zip(valid_idx, assignments):
            speakers[idx] = int(speaker_id or 0)
        speakers = _assign_missing_labels(speakers, set(valid_idx), segments)
        smoothed_speakers = _smooth_speaker_sequence(speakers, segments)
        if _count_unique_speakers(smoothed_speakers) < min(fallback_speaker_limit, len(valid_profiles)):
            final_speakers = speakers
        else:
            final_speakers = smoothed_speakers
        if _count_unique_speakers(final_speakers) >= 2:
            normalized = [{**segment, "speaker": int(max(1, final_speakers[i]))} for i, segment in enumerate(segments)]
            return _reindex_speakers_by_first_appearance(normalized)

    return _reindex_speakers_by_first_appearance(_heuristic_turn_taking_speakers(segments))


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
        return _heuristic_turn_taking_speakers(segments)

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=False)
    except Exception as exc:
        print(f"Audio load error: {exc}")
        return [{**segment, "speaker": 1} for segment in segments]

    profiles = []
    features = []
    valid_idx = []
    for i, segment in enumerate(enriched_segments):
        profile = _build_segment_profile(y, sr, segment, librosa)
        profiles.append(profile)
        if profile is None:
            continue
        features.append(profile["vector"])
        valid_idx.append(i)

    if len(features) < 2:
        return _fallback_diarization(segments, profiles, n_speakers)

    X = np.stack(features, axis=0)
    X = _normalize_features(X, StandardScaler)

    auto_detect_requested = n_speakers in (None, "", 0, "0", "auto")
    target_speakers = n_speakers
    if auto_detect_requested:
        target_speakers = _estimate_speaker_count(
            X,
            AgglomerativeClustering=AgglomerativeClustering,
            silhouette_score=silhouette_score,
            max_speakers=6,
        )
    else:
        target_speakers = _coerce_speaker_count(target_speakers, default=2, max_speakers=len(features))

    try:
        labels = _cluster_features(
            X,
            n_speakers=target_speakers,
            AgglomerativeClustering=AgglomerativeClustering,
        )
    except Exception as exc:
        print(f"Diarization clustering error: {exc}")
        return _fallback_diarization(segments, profiles, target_speakers)

    clustering_confidence = 0.0
    if len(set(labels)) >= 2 and len(labels) >= 3:
        try:
            clustering_confidence = float(silhouette_score(X, labels))
        except Exception:
            clustering_confidence = 0.0
    if auto_detect_requested and (len(set(labels)) < 2 or clustering_confidence < 0.02):
        return _fallback_diarization(segments, profiles, target_speakers)

    speakers = np.zeros(len(segments), dtype=int)
    for idx, label in zip(valid_idx, labels):
        speakers[idx] = int(label) + 1

    raw_speakers = _assign_missing_labels(speakers, set(valid_idx), segments)
    smoothed_speakers = _smooth_speaker_sequence(raw_speakers, segments)

    if auto_detect_requested:
        speakers = smoothed_speakers
    else:
        requested_unique_count = min(int(target_speakers), len(valid_idx))
        if _count_unique_speakers(smoothed_speakers) < requested_unique_count:
            speakers = raw_speakers
        else:
            speakers = smoothed_speakers

    unique = set(int(speakers[i]) for i in valid_idx) if valid_idx else {1}
    if auto_detect_requested and len(unique) < 2 and len(segments) >= 2:
        return _fallback_diarization(segments, profiles, target_speakers)

    normalized = [{**segment, "speaker": int(speakers[i])} for i, segment in enumerate(segments)]
    return _reindex_speakers_by_first_appearance(normalized)


def _turn_taking_speakers(segments):
    return _heuristic_turn_taking_speakers(segments)


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
