"""admin_panel：管理面板后端（挂载在 NoneBot 的 FastAPI 驱动上）。

安全：面板只监听 127.0.0.1，公网访问由 nginx 反代并**必须**套 BasicAuth
（见 deploy/nginx.conf.example）。应用层不再重复做鉴权。

接口：
  GET  /            -> 面板页面
  GET  /api/config  -> 当前配置
  POST /api/config  -> 覆盖写入配置（热生效）
  GET  /api/stats   -> 运行统计
  POST /api/reload  -> 让 RAG 缓存失效（删数据后调用）
  GET  /api/context/groups -> 已加入的群 + 每群统计与开关状态
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any

# 必须在模块顶层导入：FastAPI 用 get_type_hints() 在「模块全局作用域」解析注解，
# 若在函数内部才 import，注解里的 Request/HTMLResponse 会解析失败，
# 导致 request 被当成查询参数 → POST 一律 422。
from fastapi import Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from nonebot import get_bots, get_driver, logger

from qqbot_core import config, db

driver = get_driver()

_PANEL_HTML = Path(__file__).resolve().parent / "index.html"


def _json(payload, status: int = 200):
    return JSONResponse(payload, status_code=status)


def _schedule_restart(delay: float = 1.0) -> None:
    """延迟向自身发 SIGTERM，由 systemd（unit 里 Restart=always）拉起。

    这样无需 sudo，也不与 NoNewPrivileges=true 冲突。
    """
    loop = asyncio.get_event_loop()
    loop.call_later(delay, lambda: os.kill(os.getpid(), signal.SIGTERM))


# ---------------------------------------------------------------------------
# NapCat / QQ 登录管理
#   容器把运行缓存挂到了 <repo>/napcat/data/cache，因此宿主可直接读到二维码。
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_QR_PATH = _REPO_ROOT / "napcat" / "data" / "cache" / "qrcode.png"
_QQ_CONTAINER = "qqbot-napcat"


async def _docker(*args: str, timeout: float = 60.0) -> tuple[int, str]:
    """执行 docker 子进程；需要服务用户（ubuntu）在 docker 组内。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except (FileNotFoundError, PermissionError) as exc:
        return 127, f"无法执行 docker：{exc}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "docker 命令超时"
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace")


def _clean_napcat_log(text: str) -> list[str]:
    """滤掉二维码字符画与 MMKV 噪声，只留可读行。"""
    keep: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("[I] "):
            continue
        if any(ch in line for ch in "█▄▀▐▌") or len(line) > 200:
            continue
        keep.append(line)
    return keep[-10:]


async def _joined_groups() -> list[dict[str, str]]:
    """向 OneBot 要机器人已加入的群；未连接或超时就返回空，面板退回只显示有数据的群。"""
    try:
        bots = list(get_bots().values())
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"admin_panel: 取 bot 实例失败：{exc}")
        return []

    for bot in bots:
        try:
            raw = await asyncio.wait_for(bot.get_group_list(), timeout=5.0)
        except Exception as exc:  # noqa: BLE001 —— 拉群列表失败不该让面板报错
            logger.warning(f"admin_panel: 获取群列表失败：{type(exc).__name__}: {exc}")
            continue
        return [
            {
                "group_id": str(item.get("group_id", "")),
                "group_name": str(item.get("group_name") or ""),
            }
            for item in raw
            if item.get("group_id") is not None
        ]
    return []


def _register(app) -> None:

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
        """软重载：丢弃配置与检索缓存，不重启进程。"""
        config.load(force=True)
        try:
            from plugins.rag import invalidate

            invalidate()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"admin_panel: 失效缓存失败：{exc}")
        logger.info("admin_panel: 配置已重载")
        return _json({"ok": True, "config": config.load(force=True)})

    @app.post("/api/restart")
    async def _restart():
        """硬重启：先回响应，约 1 秒后优雅退出，systemd 自动拉起。"""
        _schedule_restart()
        logger.info("admin_panel: 已请求重启服务")
        return _json({"ok": True, "message": "服务将在约 1 秒后重启"})

    # ---- 风格蒸馏（接话对）----
    @app.get("/api/style/status")
    async def _style_status():
        try:
            from plugins.style_distill import status
        except Exception as exc:  # noqa: BLE001
            return _json({"ok": False, "error": str(exc)}, status=500)
        return _json({"ok": True, "status": status()})

    @app.post("/api/style/rebuild")
    async def _style_rebuild(request: Request):
        """触发后台重建；立即返回，进度用 /api/style/status 查。"""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        group_id = str((body or {}).get("group_id") or "") or None
        try:
            from plugins.style_distill import start_rebuild
        except Exception as exc:  # noqa: BLE001
            return _json({"ok": False, "error": str(exc)}, status=500)
        started = start_rebuild(group_id)
        if not started:
            return _json({"ok": False, "error": "已有重建任务在运行"}, status=409)
        logger.info(f"admin_panel: 已触发风格蒸馏重建 group={group_id or '全部'}")
        return _json({"ok": True, "message": "重建已在后台开始"})

    @app.get("/api/style/pairs")
    async def _style_pairs(group_id: str, limit: int = 50):
        return _json(
            {
                "ok": True,
                "pairs": db.list_style_pairs(group_id, limit),
                "total": db.style_pair_stats().get(group_id, 0),
            }
        )

    @app.post("/api/style/delete")
    async def _style_delete(request: Request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        raw = (body or {}).get("ids") if isinstance(body, dict) else None
        ids = (
            [int(i) for i in raw if str(i).lstrip("-").isdigit()]
            if isinstance(raw, list)
            else []
        )
        removed = db.delete_style_pairs(ids)
        try:
            from plugins.style_distill import invalidate

            invalidate()
        except Exception:  # noqa: BLE001
            pass
        logger.info(f"admin_panel: 删除接话对 {removed} 条")
        return _json({"ok": True, "deleted": removed})

    # ---- 上下文 / 数据管理 ----
    @app.get("/api/context/groups")
    async def _ctx_groups():
        """已加入的全部群（含尚无消息的）+ 每群消息量、上下文条数与功能开关。"""
        joined = await _joined_groups()
        rows: dict[str, dict[str, Any]] = {g["group_id"]: g for g in db.list_groups()}
        for info in joined:
            row = rows.setdefault(
                info["group_id"],
                {
                    "group_id": info["group_id"],
                    "messages": 0,
                    "bot_messages": 0,
                    "last_ts": None,
                    "sessions": 0,
                },
            )
            row["group_name"] = info["group_name"]
            row["joined"] = True
        for row in rows.values():
            row.setdefault("group_name", "")
            row.setdefault("joined", False)
            row["enabled"] = config.group_enabled(row["group_id"])
        ordered = sorted(
            rows.values(), key=lambda r: (not r["joined"], -(r["last_ts"] or 0.0))
        )
        return _json({"ok": True, "groups": ordered, "joined_ok": bool(joined)})

    @app.get("/api/context/messages")
    async def _ctx_messages(group_id: str, limit: int = 50, offset: int = 0):
        return _json(
            {"ok": True, "messages": db.recent_messages(group_id, limit, offset)}
        )

    @app.post("/api/context/clear")
    async def _ctx_clear(request: Request):
        """scope: messages | sessions | both；可按群、按用户或按消息 id 清理。"""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if not isinstance(body, dict):
            body = {}

        scope = str(body.get("scope", "both"))
        group_id = str(body["group_id"]) if body.get("group_id") else None
        user_id = str(body["user_id"]) if body.get("user_id") else None
        raw_ids = body.get("ids")
        ids = (
            [int(i) for i in raw_ids if str(i).isdigit()]
            if isinstance(raw_ids, list)
            else None
        )

        deleted = cleared = 0
        if scope in ("messages", "both"):
            deleted = db.delete_messages(group_id=group_id, user_id=user_id, ids=ids)
            # 复读学习表与接话对都是从这些消息派生出来的，一起清掉免得留下脏数据
            if not ids and not user_id:
                db.clear_repeat_learn(group_id)
                db.clear_style_pairs(group_id)
        if scope in ("sessions", "both"):
            cleared = db.clear_sessions(group_id=group_id, user_id=user_id)
        try:
            from plugins.rag import invalidate

            invalidate(group_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            from plugins.style_distill import invalidate as invalidate_pairs

            invalidate_pairs(group_id)
        except Exception:  # noqa: BLE001
            pass

        logger.info(
            f"admin_panel: 清理 scope={scope} group={group_id} user={user_id} "
            f"ids={ids} -> 消息 {deleted} / 上下文 {cleared}"
        )
        return _json(
            {"ok": True, "deleted_messages": deleted, "cleared_sessions": cleared}
        )
    # ---- QQ / NapCat 登录状态与管理 ----
    @app.get("/api/napcat/status")
    async def _napcat_status():
        try:
            bots = [str(key) for key in get_bots().keys()]
        except Exception:  # noqa: BLE001
            bots = []

        qr_info = None
        if _QR_PATH.exists():
            st = _QR_PATH.stat()
            qr_info = {"mtime": st.st_mtime, "size": st.st_size}

        rc_state, state = await _docker(
            "inspect", "-f", "{{.State.Status}}", _QQ_CONTAINER, timeout=20
        )
        rc_log, logs = await _docker(
            "logs", "--tail", "60", _QQ_CONTAINER, timeout=20
        )
        return _json(
            {
                "ok": True,
                "online": bool(bots),
                "bots": bots,
                "container": state.strip() if rc_state == 0 else "unknown",
                "qr": qr_info,
                "log": _clean_napcat_log(logs) if rc_log == 0 else [],
            }
        )

    @app.get("/api/napcat/qr")
    async def _napcat_qr():
        """把容器里的登录二维码原样吐给浏览器（no-store，Always 取最新的）。"""
        if not _QR_PATH.exists():
            return _json({"ok": False, "error": "暂无二维码文件"}, status=404)
        try:
            data = _QR_PATH.read_bytes()
        except OSError as exc:
            return _json({"ok": False, "error": str(exc)}, status=500)
        return Response(
            content=data,
            media_type="image/png",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.post("/api/napcat/relogin")
    async def _napcat_relogin():
        """重启 NapCat 容器，逼它重新生成登录二维码。"""
        rc, out = await _docker("restart", _QQ_CONTAINER, timeout=120)
        logger.info(f"admin_panel: 重启 NapCat 容器 rc={rc}")
        if rc != 0:
            return _json({"ok": False, "error": out.strip()[:500]}, status=500)
        return _json(
            {"ok": True, "message": "NapCat 已重启，二维码生成中（约 10~20 秒）"}
        )


@driver.on_startup
async def _mount() -> None:
    app = getattr(driver, "server_app", None)
    if app is None:
        logger.warning("admin_panel: 当前驱动未提供 ASGI server_app，面板未挂载")
        return
    _register(app)
    logger.info("admin_panel: 面板已挂载（http://127.0.0.1:8081/）")
