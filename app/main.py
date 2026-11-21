"""FastAPI 入口：网页 + REST API。"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .replica import ReplicaStore
from .sample import build_sample_manifest
from .store import CorruptStoreError, NotFoundError, Store, StoreError

STATIC_DIR = Path(__file__).parent / "static"


def create_app(data_dir: str | None = None,
               replica_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="虚拟机备份维护台")
    store = Store(data_dir or os.environ.get("DATA_DIR", "./data"))
    replica = ReplicaStore(
        replica_dir or os.environ.get("REPLICA_DIR", "./replica")
    )

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(StoreError)
    async def _store_error(_: Request, exc: StoreError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(CorruptStoreError)
    async def _corrupt(_: Request, exc: CorruptStoreError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=500)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    async def get_state() -> dict:
        return store.state()

    @app.get("/api/sample")
    async def get_sample() -> dict:
        return build_sample_manifest()

    @app.post("/api/import")
    async def import_manifest(request: Request) -> dict:
        try:
            payload = await request.json()
        except Exception:
            raise StoreError("请求体不是合法 JSON")
        return store.import_manifest(payload)

    @app.post("/api/remove/{layer_id}")
    async def remove_layer(layer_id: str) -> dict:
        return store.remove(layer_id)

    @app.get("/api/download/{layer_id}")
    async def download_layer(layer_id: str) -> Response:
        data = store.resolve_image(layer_id)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{layer_id}.raw"',
                "Content-Length": str(len(data)),
            },
        )

    # ------------------------------------------------------------------ #
    # 分层副本同步（备用卷）
    # ------------------------------------------------------------------ #
    @app.get("/api/replica/state")
    async def replica_state() -> dict:
        return replica.state()

    @app.post("/api/replica/sync/{layer_id}")
    async def replica_sync(layer_id: str) -> dict:
        # 先在源卷锁内取一致链快照，再在副本锁内提交；
        # 锁顺序固定为“源 -> 副本”，避免死锁。复制全程不修改源卷。
        snapshot = store.snapshot_chain(layer_id)
        return replica.sync(snapshot)

    @app.get("/api/replica/download/{sync_id}/{layer_id}")
    async def replica_download(sync_id: str, layer_id: str) -> Response:
        data = replica.resolve_image(sync_id, layer_id)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="replica-{layer_id}.raw"'
                ),
                "Content-Length": str(len(data)),
            },
        )

    return app


# uvicorn 入口（本地默认 ./data 与 ./replica，容器内由环境变量指定 /data、/replica）
app = create_app()
