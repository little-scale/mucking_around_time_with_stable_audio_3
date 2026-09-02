from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from drift_engine import MultitrackEngine


API_PREFIX = "/api/drift"


class PromptTag(BaseModel):
    text: str = Field(min_length=1, max_length=120)
    weight: Literal["subtle", "normal", "strong"] = "normal"


class TrackSettings(BaseModel):
    model: Literal["sm-music", "sm-sfx", "medium"] | None = None
    prompt_tags: list[PromptTag] | None = None
    negative_prompt: str | None = None
    role_prompt: str | None = Field(default=None, max_length=500)
    beats: int | None = Field(default=None, ge=1, le=512)
    mutation: float | None = Field(default=None, ge=0.01, le=1.0)
    cfg: float | None = Field(default=None, ge=-20, le=20)
    steps: int | None = Field(default=None, ge=1, le=64)
    mutation_interval_loops: Literal[1, 2, 4, 8] | None = None
    cadence_seconds: float | None = Field(default=None, ge=0, le=86400)
    seed_mode: Literal["random", "increment", "fixed"] | None = None
    seed: int | None = None

    def changes(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class TrackMixSettings(BaseModel):
    volume: float | None = Field(default=None, ge=0, le=1.5)
    muted: bool | None = None
    solo: bool | None = None
    highpass_hz: float | None = Field(default=None, ge=20, le=20000)
    lowpass_hz: float | None = Field(default=None, ge=20, le=20000)
    reverb_send: float | None = Field(default=None, ge=0, le=1)
    delay_send: float | None = Field(default=None, ge=0, le=1)


class SidechainSettings(BaseModel):
    enabled: bool | None = None
    duck_melodic: bool | None = None
    duck_texture: bool | None = None
    threshold_db: float | None = Field(default=None, ge=-80, le=0)
    ratio: float | None = Field(default=None, ge=1, le=20)
    attack_ms: float | None = Field(default=None, ge=0.1, le=1000)
    release_ms: float | None = Field(default=None, ge=1, le=5000)
    makeup_db: float | None = Field(default=None, ge=-24, le=24)
    depth: float | None = Field(default=None, ge=0, le=1)


class ReverbSettings(BaseModel):
    enabled: bool | None = None
    return_level: float | None = Field(default=None, ge=0, le=1)
    decay_seconds: float | None = Field(default=None, ge=0.1, le=20)
    predelay_ms: float | None = Field(default=None, ge=0, le=500)
    damping_hz: float | None = Field(default=None, ge=200, le=20000)


class DelaySettings(BaseModel):
    enabled: bool | None = None
    return_level: float | None = Field(default=None, ge=0, le=1)
    division: Literal["1/1", "1/2", "1/4D", "1/4", "1/8D", "1/8", "1/8T", "1/16"] | None = None
    feedback: float | None = Field(default=None, ge=0, le=0.92)
    lowpass_hz: float | None = Field(default=None, ge=200, le=20000)
    stereo_width: float | None = Field(default=None, ge=0, le=1)


class SessionSettings(BaseModel):
    bpm: float | None = Field(default=None, ge=30, le=300)
    master_volume: float | None = Field(default=None, ge=0, le=1.5)
    mixer: dict[str, TrackMixSettings] | None = None
    sidechain: SidechainSettings | None = None
    reverb: ReverbSettings | None = None
    delay: DelaySettings | None = None

    def changes(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class FreezeRequest(BaseModel):
    frozen: bool = True


def create_app(
    loop: MultitrackEngine,
    *,
    api_prefix: str = API_PREFIX,
    title: str = "Drift Multitrack",
    description: str = "Synchronized Stable Audio 3 feedback lineages with a non-blocking render queue.",
    interface_url: str = "/drift",
) -> FastAPI:
    app = FastAPI(
        title=title,
        version="0.2.0",
        description=description,
    )
    app.state.loop_engine = loop

    @app.get("/api")
    async def api_root() -> dict[str, Any]:
        return {
            "name": title,
            "phase": "multitrack-feedback",
            "tracks": list(loop.track_ids),
            "interface_url": interface_url,
            "docs_url": f"{api_prefix}/docs",
            "status_url": f"{api_prefix}/status",
        }

    async def decorated_status() -> dict[str, Any]:
        value = await loop.status()
        for track_id, track in value["tracks"].items():
            track["current_audio_url"] = f"{api_prefix}/tracks/{track_id}/audio/current.wav" if track["current_available"] else None
            if track["last_completed"]:
                generation = track["last_completed"]["generation"]
                if loop.get_version(track_id, generation) is None:
                    track["last_completed"]["audio_url"] = track["current_audio_url"]
                    track["last_completed"]["archived"] = False
                else:
                    track["last_completed"]["audio_url"] = f"{api_prefix}/tracks/{track_id}/audio/versions/{generation}.wav"
                    track["last_completed"]["archived"] = True
        return value

    @app.get("/status")
    async def status() -> dict[str, Any]:
        return await decorated_status()

    @app.post("/loop/start", status_code=202)
    async def start(request: Request) -> dict[str, Any]:
        loop.note_all_clients(request.client.host if request.client else "unknown")
        try:
            started = await loop.start()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"accepted": True, "started": started, "status": await decorated_status()}

    @app.post("/loop/stop", status_code=202)
    async def stop() -> dict[str, Any]:
        stopping = await loop.stop()
        return {
            "accepted": True,
            "was_running": stopping,
            "note": "The active render may finish; no further jobs will be scheduled.",
            "status": await decorated_status(),
        }

    @app.put("/session")
    async def update_session(
        request: SessionSettings,
        latch: bool = Query(default=False),
    ) -> dict[str, Any]:
        if not request.changes():
            raise HTTPException(status_code=422, detail="provide at least one session setting")
        try:
            config = await loop.update_session(latch_for_next=latch, **request.changes())
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"updated": True, "latched": latch, "session": asdict(config)}

    @app.put("/tracks/{track_id}")
    async def update_track(
        track_id: str,
        settings: TrackSettings,
        request: Request,
        latch: bool = Query(default=False),
    ) -> dict[str, Any]:
        if track_id not in loop.track_ids:
            raise HTTPException(status_code=404, detail="Track not found")
        loop.note_client(track_id, request.client.host if request.client else "unknown")
        changes = settings.changes()
        legacy_cadence = changes.pop("cadence_seconds", None)
        if legacy_cadence is not None and "mutation_interval_loops" not in changes:
            # Pre-1.6 browser compatibility. Seconds-based cadence is retired;
            # its safe musical equivalent is one mutation per loop.
            changes["mutation_interval_loops"] = 1
        if not changes:
            raise HTTPException(status_code=422, detail="provide at least one track setting")
        try:
            config = await loop.update_track(track_id, latch_for_next=latch, **changes)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "updated": True,
            "latched": latch,
            "track_id": track_id,
            "config": {
                **asdict(config),
                "constructed_prompt": config.prompt(track_id, loop.session.bpm),
                "loop_seconds": config.beats * 60.0 / loop.session.bpm,
                "mutation_interval_seconds": loop._mutation_period_seconds(config, loop.session.bpm),
            },
        }

    @app.put("/tracks/{track_id}/freeze")
    async def freeze_track(track_id: str, request: FreezeRequest) -> dict[str, Any]:
        if track_id not in loop.track_ids:
            raise HTTPException(status_code=404, detail="Track not found")
        config = await loop.set_frozen(track_id, request.frozen)
        return {"updated": True, "track_id": track_id, "frozen": config.frozen}

    @app.put("/loop/freeze")
    async def freeze_all(request: FreezeRequest) -> dict[str, Any]:
        await loop.freeze_all(request.frozen)
        return {"updated": True, "frozen": request.frozen, "status": await decorated_status()}

    @app.put("/tracks/{track_id}/reference", status_code=201)
    async def import_reference(
        track_id: str,
        request: Request,
        mode: Literal["auto", "introduce", "replace"] = Query(default="replace"),
        blend: float = Query(default=0.25, ge=0.01, le=0.75),
        generations: int = Query(default=4, ge=1, le=8),
    ) -> dict[str, Any]:
        if track_id not in loop.track_ids:
            raise HTTPException(status_code=404, detail="Track not found")
        loop.note_client(track_id, request.client.host if request.client else "unknown")
        declared_size = int(request.headers.get("content-length", "0") or 0)
        if declared_size > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Reference file is limited to 100 MB")
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=422, detail="Reference file is empty")
        if len(payload) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Reference file is limited to 100 MB")
        temporary = loop._track(track_id).work_dir / f"upload-{uuid.uuid4().hex}.wav"
        temporary.write_bytes(payload)
        try:
            encoded_name = request.headers.get("x-file-name")
            record = await loop.import_reference(
                track_id,
                temporary,
                unquote(encoded_name) if encoded_name else None,
                mode=mode,
                blend=blend,
                generations=generations,
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            temporary.unlink(missing_ok=True)
        if record.get("queued"):
            return {"introduced": True, "track_id": track_id, "introduction": record}
        value = dict(record)
        value["audio_url"] = f"{api_prefix}/tracks/{track_id}/audio/versions/{record['generation']}.wav"
        return {"imported": True, "track_id": track_id, "version": value}

    @app.get("/tracks/{track_id}/versions")
    async def versions(
        track_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        if track_id not in loop.track_ids:
            raise HTTPException(status_code=404, detail="Track not found")
        items = [dict(item) for item in loop.list_versions(track_id, offset=offset, limit=limit)]
        for item in items:
            item["audio_url"] = f"{api_prefix}/tracks/{track_id}/audio/versions/{item['generation']}.wav"
        return {"track_id": track_id, "offset": offset, "limit": limit, "versions": items}

    @app.get("/tracks/{track_id}/versions/{generation}")
    async def version(track_id: str, generation: int) -> dict[str, Any]:
        if track_id not in loop.track_ids:
            raise HTTPException(status_code=404, detail="Track not found")
        item = loop.get_version(track_id, generation)
        if item is None:
            raise HTTPException(status_code=404, detail="Generation not found")
        value = dict(item)
        value["audio_url"] = f"{api_prefix}/tracks/{track_id}/audio/versions/{generation}.wav"
        return value

    @app.delete("/tracks/{track_id}/versions")
    async def clear_track_versions(track_id: str) -> dict[str, Any]:
        if track_id not in loop.track_ids:
            raise HTTPException(status_code=404, detail="Track not found")
        result = await loop.clear_versions(track_id)
        return {"cleared": True, **result}

    @app.delete("/versions")
    async def clear_all_versions() -> dict[str, Any]:
        results = await loop.clear_all_versions()
        return {
            "cleared": True,
            "bytes_freed": sum(item["bytes_freed"] for item in results),
            "files_removed": sum(item["files_removed"] for item in results),
            "tracks": results,
        }

    @app.delete("/tracks/{track_id}/output")
    async def clear_track_output(track_id: str) -> dict[str, Any]:
        if track_id not in loop.track_ids:
            raise HTTPException(status_code=404, detail="Track not found")
        result = await loop.clear_output(track_id)
        return {"cleared": True, **result}

    @app.delete("/outputs")
    async def clear_all_outputs() -> dict[str, Any]:
        results = await loop.clear_all_outputs()
        return {
            "cleared": True,
            "bytes_freed": sum(item["bytes_freed"] for item in results),
            "files_removed": sum(item["files_removed"] for item in results),
            "tracks": results,
        }

    def current_response(track_id: str) -> FileResponse:
        if track_id not in loop.track_ids:
            raise HTTPException(status_code=404, detail="Track not found")
        path = loop.current_path_for(track_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="No current loop exists for this track")
        return FileResponse(
            path,
            media_type="audio/wav",
            filename=f"{track_id}-current.wav",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/tracks/{track_id}/audio/current.wav", response_class=FileResponse)
    async def current_audio(track_id: str) -> FileResponse:
        return current_response(track_id)

    @app.get("/tracks/{track_id}/audio/versions/{generation}.wav", response_class=FileResponse)
    async def version_audio(track_id: str, generation: int) -> FileResponse:
        if track_id not in loop.track_ids:
            raise HTTPException(status_code=404, detail="Track not found")
        path = loop.version_path(track_id, generation)
        if path is None:
            raise HTTPException(status_code=404, detail="Generation not found")
        return FileResponse(
            path,
            media_type="audio/wav",
            filename=path.name,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    # Compatibility aliases retain the original melodic-track contract.
    @app.get("/audio/current.wav", response_class=FileResponse)
    async def legacy_current_audio() -> FileResponse:
        return current_response("melodic")

    @app.get("/audio/latest.wav", response_class=FileResponse)
    async def legacy_latest_audio() -> FileResponse:
        return current_response("melodic")

    @app.get("/versions")
    async def legacy_versions(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
        items = [dict(item) for item in loop.list_versions("melodic", limit=limit)]
        for item in items:
            item["audio_url"] = f"{api_prefix}/tracks/melodic/audio/versions/{item['generation']}.wav"
        return {"offset": 0, "limit": limit, "versions": items}

    return app
