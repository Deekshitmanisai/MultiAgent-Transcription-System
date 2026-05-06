import asyncio

from fastapi import WebSocket, WebSocketDisconnect

from live_audio import LiveAudioCapture, SOURCE_BOTH, SOURCE_MICROPHONE, SOURCE_SYSTEM
from live_pipeline import LiveChunkProcessor


class LiveSession:
    def __init__(self, websocket: WebSocket, *, source=SOURCE_SYSTEM):
        self.websocket = websocket
        self.queue = asyncio.Queue(maxsize=5)
        self.processor = LiveChunkProcessor(model_size="tiny", enable_chunk_correction=False)
        self.capture = LiveAudioCapture(asyncio.get_running_loop(), self.queue, source=source)
        self.stop_requested = asyncio.Event()

    async def run(self):
        await self.websocket.accept()
        try:
            self.capture.start()
        except Exception as exc:
            source_label = {
                SOURCE_SYSTEM: "system audio",
                SOURCE_MICROPHONE: "microphone",
                SOURCE_BOTH: "microphone + system audio",
            }.get(self.capture.source, "audio")
            await self.websocket.send_json(
                {
                    "type": "error",
                    "message": f"Unable to start live {source_label} capture: {exc}",
                }
            )
            await self.websocket.close()
            return

        await self.websocket.send_json({"type": "status", "message": "Live transcription started."})

        processing_task = asyncio.create_task(self._processing_loop())
        receiver_task = asyncio.create_task(self._receiver_loop())

        done, pending = await asyncio.wait(
            [processing_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        self.capture.stop()
        self.stop_requested.set()

        if not processing_task.done():
            await processing_task

        summary = await self.processor.generate_summary()
        await self.websocket.send_json(
            {
                "type": "summary",
                "summary": summary,
                "englishTranscript": "\n".join(self.processor.collected_english).strip(),
            }
        )
        await self.websocket.send_json({"type": "status", "message": "Live transcription stopped."})
        await self.websocket.close()

    async def _receiver_loop(self):
        try:
            while True:
                payload = await self.websocket.receive_json()
                action = str(payload.get("action", "") or "").strip().lower()
                if action == "stop":
                    self.stop_requested.set()
                    self.capture.stop()
                    return
        except WebSocketDisconnect:
            self.stop_requested.set()
            self.capture.stop()
        except Exception as exc:
            await self.websocket.send_json({"type": "error", "message": str(exc)})
            self.stop_requested.set()
            self.capture.stop()

    async def _processing_loop(self):
        while True:
            if self.stop_requested.is_set() and self.queue.empty():
                return

            try:
                chunk = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            result = await self.processor.process_chunk(chunk)
            if result is None:
                continue

            await self.websocket.send_json(
                {
                    "type": "chunk",
                    "english": result.english,
                    "timestamp": result.timestamp,
                    "startSeconds": result.start_seconds,
                    "endSeconds": result.end_seconds,
                }
            )


def register_live_routes(app):
    @app.websocket("/live")
    async def live_endpoint(websocket: WebSocket):
        source = str(websocket.query_params.get("source", SOURCE_SYSTEM) or SOURCE_SYSTEM).strip().lower()
        if source not in {SOURCE_MICROPHONE, SOURCE_SYSTEM, SOURCE_BOTH}:
            source = SOURCE_SYSTEM
        session = LiveSession(websocket, source=source)
        await session.run()
