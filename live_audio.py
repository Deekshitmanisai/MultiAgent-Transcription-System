import asyncio
import io
import inspect
import threading
import time
import wave
from dataclasses import dataclass

import numpy as np


SOURCE_MICROPHONE = "microphone"
SOURCE_SYSTEM = "system"
SOURCE_BOTH = "both"


@dataclass
class LiveAudioChunk:
    index: int
    start_seconds: float
    end_seconds: float
    sample_rate: int
    audio_float32: np.ndarray
    wav_bytes: bytes


class LiveAudioCapture:
    def __init__(
        self,
        loop,
        queue,
        *,
        source=SOURCE_SYSTEM,
        sample_rate=16000,
        channels=1,
        chunk_seconds=3,
    ):
        self.loop = loop
        self.queue = queue
        self.source = str(source or SOURCE_SYSTEM).strip().lower()
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.chunk_seconds = int(chunk_seconds)
        self._streams = []
        self._devices = []
        self._chunk_index = 0
        self._running = False
        self._buffers = {"microphone": [], "system": []}
        self._lock = threading.Lock()
        self._chunk_thread = None

    def start(self):
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError(
                "Live mode requires the sounddevice package. Install dependencies and allow audio device access."
            ) from exc

        stream_specs = self._resolve_streams(sd)
        blocksize = max(1024, self.sample_rate // 2)
        self._running = True

        self._streams = []
        self._devices = []
        for buffer_key, device, extra_settings in stream_specs:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=blocksize,
                device=device,
                extra_settings=extra_settings,
                callback=self._audio_callback_factory(buffer_key),
            )
            stream.start()
            self._streams.append(stream)
            self._devices.append(device)

        self._chunk_thread = threading.Thread(target=self._chunk_loop, daemon=True)
        self._chunk_thread.start()

    def stop(self):
        self._running = False
        for stream in self._streams:
            try:
                stream.stop()
            finally:
                stream.close()
        self._streams = []
        self._devices = []

    def _resolve_streams(self, sd):
        if self.source == SOURCE_MICROPHONE:
            microphone_device = sd.default.device[0]
            if microphone_device is None or microphone_device < 0:
                raise RuntimeError("No default microphone input device found.")
            return [("microphone", microphone_device, None)]

        if self.source == SOURCE_SYSTEM:
            try:
                return [("system", *self._resolve_system_capture_device(sd))]
            except Exception as exc:
                raise RuntimeError(
                    "System audio capture is unavailable. On Windows, a WASAPI loopback-compatible output device is required."
                ) from exc

        if self.source == SOURCE_BOTH:
            microphone_device = sd.default.device[0]
            if microphone_device is None or microphone_device < 0:
                raise RuntimeError("No default microphone input device found.")
            try:
                system_device, system_settings = self._resolve_system_capture_device(sd)
            except Exception as exc:
                raise RuntimeError(
                    "System audio capture is unavailable. On Windows, a WASAPI loopback-compatible output device is required."
                ) from exc
            return [
                ("microphone", microphone_device, None),
                ("system", system_device, system_settings),
            ]

        if self.source not in {SOURCE_MICROPHONE, SOURCE_SYSTEM, SOURCE_BOTH}:
            raise RuntimeError(f"Unsupported live source: {self.source}")
        return []

    def _resolve_system_capture_device(self, sd):
        try:
            supports_loopback_flag = "loopback" in inspect.signature(sd.WasapiSettings).parameters
        except Exception:
            supports_loopback_flag = False
        if supports_loopback_flag:
            output_device = self._resolve_wasapi_output_device(sd)
            return output_device, sd.WasapiSettings(loopback=True)

        stereo_mix_device = self._resolve_stereo_mix_input_device(sd)
        if stereo_mix_device is not None:
            return stereo_mix_device, None

        output_device = self._resolve_wasapi_output_device(sd)
        return output_device, sd.WasapiSettings()

    def _resolve_wasapi_output_device(self, sd):
        default_output = sd.default.device[1]
        if default_output is None or default_output < 0:
            raise RuntimeError("No default output device found for system audio capture.")

        try:
            hostapis = sd.query_hostapis()
            default_hostapi_name = hostapis[sd.query_devices(default_output)["hostapi"]]["name"].lower()
            if "wasapi" in default_hostapi_name:
                return default_output
        except Exception:
            pass

        for index, device in enumerate(sd.query_devices()):
            if int(device.get("max_output_channels", 0) or 0) <= 0:
                continue
            hostapi = hostapis[device["hostapi"]]["name"].lower()
            if "wasapi" in hostapi:
                return index

        raise RuntimeError("No WASAPI loopback-compatible output device found.")

    def _resolve_stereo_mix_input_device(self, sd):
        for index, device in enumerate(sd.query_devices()):
            if int(device.get("max_input_channels", 0) or 0) <= 0:
                continue
            name = str(device.get("name", "") or "").lower()
            if "stereo mix" in name or "loopback" in name:
                return index
        return None

    def _audio_callback_factory(self, buffer_key):
        def _audio_callback(indata, frames, callback_time, status):
            if status:
                print(f"Live audio status ({buffer_key}): {status}")

            mono_audio = np.asarray(indata[:, 0], dtype=np.float32).copy()
            with self._lock:
                self._buffers[buffer_key].append(mono_audio)

        return _audio_callback

    def _chunk_loop(self):
        while self._running:
            time.sleep(self.chunk_seconds)
            chunk_audio = self._drain_buffer()
            if chunk_audio.size == 0:
                continue
            chunk = self._build_chunk(chunk_audio)
            self.loop.call_soon_threadsafe(self._publish_chunk, chunk)

        final_audio = self._drain_buffer()
        if final_audio.size:
            chunk = self._build_chunk(final_audio)
            self.loop.call_soon_threadsafe(self._publish_chunk, chunk)

    def _drain_buffer(self):
        with self._lock:
            microphone_pieces = self._buffers["microphone"]
            system_pieces = self._buffers["system"]
            if not microphone_pieces and not system_pieces:
                return np.zeros((0,), dtype=np.float32)
            self._buffers = {"microphone": [], "system": []}

        mic_audio = self._merge_pieces(microphone_pieces)
        system_audio = self._merge_pieces(system_pieces)

        if mic_audio.size and system_audio.size:
            target_size = max(mic_audio.size, system_audio.size)
            mixed = np.zeros((target_size,), dtype=np.float32)
            mixed[: mic_audio.size] += mic_audio
            mixed[: system_audio.size] += system_audio
            return np.clip(mixed * 0.5, -1.0, 1.0)
        if mic_audio.size:
            return mic_audio
        return system_audio

    def _merge_pieces(self, pieces):
        if not pieces:
            return np.zeros((0,), dtype=np.float32)
        audio = np.concatenate(pieces, axis=0)
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio.astype(np.float32, copy=False)

    def _build_chunk(self, audio_float32):
        chunk_start = self._chunk_index * self.chunk_seconds
        chunk_end = chunk_start + (len(audio_float32) / float(self.sample_rate))
        chunk = LiveAudioChunk(
            index=self._chunk_index,
            start_seconds=round(chunk_start, 3),
            end_seconds=round(chunk_end, 3),
            sample_rate=self.sample_rate,
            audio_float32=audio_float32,
            wav_bytes=self._to_wav_bytes(audio_float32),
        )
        self._chunk_index += 1
        return chunk

    def _publish_chunk(self, chunk):
        try:
            self.queue.put_nowait(chunk)
        except asyncio.QueueFull:
            print("Live queue full; dropping chunk to keep latency bounded.")

    def _to_wav_bytes(self, audio_float32):
        if audio_float32.size == 0:
            return b""

        pcm_audio = np.clip(audio_float32, -1.0, 1.0)
        pcm_audio = (pcm_audio * 32767.0).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_audio.tobytes())
        return buffer.getvalue()
