"""admin_panel：管理面板后端（挂载在 NoneBot 的 FastAPI 驱动上）。

安全：面板只监听 127.0.0.1，公网访问由 nginx 反代并**必须**套 BasicAuth
（见 deploy/nginx.conf.example）。应用层不再重复做鉴权。

接口：
  GET  /            -> 面板页面
  GET  /api/config  -> 当前配置
  POST /api/config  -> 覆盖写入配置（热生效）
  GET  /api/stats   -> 运行统计
  POST /api/reload  -> 让 RAG 缓存失效（删数据后调用）
"""

from __future__ import annotations

from pathlib import Path

from nonebot import get_driver, logger

from qqbot_core import config, db

driver = get_driver()

_PANEL_HTML = Path(__file__).resolve().parent / "index.html"


def _json(payload, status: int = 200):
    from fastapi.responses import JSONResponse

    return JSONResponse(payload, status_code=status)


def _register(app) -> None:
    from fastapi import Request
    from fastapi.responses import HTMLResponse

    @app.get("/", response_class=HTMLResponse)
    async def _index() -> HTMLResponse:
        try:
            return HTMLResponse(_PANEL_HTML.read_text(encoding="utf-8"))
        except OSError as exc:
            return HTMLResponse(f"面板页面缺失：{exc}", status_code=500)

    @app.get("/api/config")
    async def _get_config():
        return _json(config.load(force=True))

    @app.post("/api/config")
    async def _post_config(request: Request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _json({"ok": False, "error": "请求体不是合法 JSON"}, status=400)
        if not isinstance(body, dict):
            return _json({"ok": False, "error": "配置必须是对象"}, status=400)
        saved = config.save(body)
        logger.info("admin_panel: 配置已更新")
        return _json({"ok": True, "config": saved})

    @app.get("/api/stats")
    async def _stats():
        return _json({"ok": True, "stats": db.stats()})

    @app.post("/api/reload")
    async def _reload():
        try:
            from plugins.rag import invalidate

            invalidate()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"admin_panel: 失效缓存失败：{exc}")
        return _json({"ok": True})


@driver.on_startup
async def _mount() -> None:
    app = getattr(driver, "server_app", None)
    if app is None:
        logger.warning("admin_panel: 当前驱动未提供 ASGI server_app，面板未挂载")
        return
    _register(app)
    logger.info("admin_panel: 面板已挂载（http://127.0.0.1:8081/）")
