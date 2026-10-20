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
  GET  /api/context/private -> 有私聊记录/上下文的 QQ 号
  GET  /api/context/private/messages -> 某人的私聊消息
  POST /api/context/private/clear   -> 重置某人的私聊上下文 / 删私聊记录
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import signal
from pathlib import Path
from typing import Any

# 必须在模块顶层导入：FastAPI 用 get_type_hints() 在「模块全局作用域」解析注解，
# 若在函数内部才 import，注解里的 Request/HTMLResponse 会解析失败，
# 导致 request 被当成查询参数 → POST 一律 422。
from fastapi import Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from nonebot import get_bots, get_driver, logger

from qqbot_core import config, db, stickers

driver = get_driver()

_PANEL_HTML = Path(__file__).resolve().parent / "index.html"


def _json(payload, status: int = 200):
    return JSONResponse(payload, status_code=status)


# Windows 下没有守护进程兜底，进程一旦退出就再也没人把它拉起来
_IS_WINDOWS = os.name == "nt"


def _schedule_restart(delay: float = 1.0) -> bool:
    """延迟向自身发 SIGTERM，由 systemd（unit 里 Restart=always）拉起。

    这样无需 sudo，也不与 NoNewPrivileges=true 冲突。

    Windows 上没有 systemd 兜底，自杀之后就真的起不来了，
    所以这里什么也不做并返回 False，由调用方提示用户手动重启。
    """
    if _IS_WINDOWS:
        return False
    loop = asyncio.get_event_loop()
    loop.call_later(delay, lambda: os.kill(os.getpid(), signal.SIGTERM))
    return True


# ---------------------------------------------------------------------------
# NapCat / QQ 登录管理
#
# 默认按仓库里的 Docker 编排取路径：容器把 NapCat 的 config/ cache/ logs/
# 挂到了 <仓库>/napcat/data/ 下面，所以宿主能直接读到二维码与 WebUI token。
#
# Windows 上直接跑原生 NapCat（NapCat.Shell）时不需要 Docker，而 NapCat 自身
# 用的也是 config/ cache/ logs/ 这三个子目录 —— 把 config.json 里的
# `napcat.mode` 改成 native、`napcat.dir` 指向 NapCat 安装目录即可，
# 机器人本体（OneBot 连接）本来就不依赖 Docker。
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_NAPCAT_DIR = _REPO_ROOT / "napcat" / "data"


def _napcat_cfg() -> dict:
    return config.get("napcat", {}) or {}


def _napcat_mode() -> str:
    """`docker`（默认，走仓库里的容器编排）或 `native`（直跑 NapCat）。"""
    return str(_napcat_cfg().get("mode") or "docker").strip().lower()


def _napcat_dir() -> Path:
    """NapCat 数据目录（下面固定有 config/ cache/ logs/ 三个子目录）。

    留空则用仓库内的 docker bind mount 路径，因此**默认行为与以前完全一致**。
    """
    raw = str(_napcat_cfg().get("dir") or "").strip()
    return Path(raw).expanduser() if raw else _DEFAULT_NAPCAT_DIR


def _qr_path() -> Path:
    return _napcat_dir() / "cache" / "qrcode.png"


def _webui_conf_path() -> Path:
    return _napcat_dir() / "config" / "webui.json"


def _webui_base() -> str:
    return str(_napcat_cfg().get("webui_base") or "http://127.0.0.1:6099").rstrip("/")


def _qq_container() -> str:
    return str(_napcat_cfg().get("container") or "qqbot-napcat")


def _webui_token() -> str:
    """从 NapCat 持久化的 webui.json 读 WebUI token（无需 docker）。"""
    try:
        data = json.loads(_webui_conf_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)
        elif isinstance(node, str):
            yield node

    if not isinstance(data, dict):
        return ""
    for key in ("token", "Token", "webUiToken", "webuiToken"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    # 兜底：找第一个 12 位十六进制串
    for text in walk(data):
        if re.fullmatch(r"[0-9a-f]{12}", text):
            return text
    return ""


async def _http_json(url: str, payload: dict, timeout: float, headers: dict | None = None):
    """POST JSON 并解析响应；任何异常都转成带 code=-1 的字典，不往上抛。"""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload or {}, headers=headers or {})
            return resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"code": -1, "message": f"{type(exc).__name__}: {exc}"}


async def _napcat_webui(path: str, payload: dict | None = None, timeout: float = 30.0):
    """调用 NapCat WebUI API：先 sha256(token+".napcat") 登录，再带 Bearer 请求。

    算法与请求头均取自 WebUI 前端打包代码（loginWithToken）。
    """
    token = _webui_token()
    if not token:
        return {"code": -1, "message": "读取 NapCat WebUI token 失败（webui.json）"}
    digest = hashlib.sha256((token + ".napcat").encode()).hexdigest()
    login = await _http_json(
        f"{_webui_base()}/api/auth/login", {"hash": digest, "totpCode": ""}, timeout
    )
    cred = (login.get("data") or {}).get("Credential") if isinstance(login, dict) else None
    if not cred:
        return {"code": -1, "message": f"登录 NapCat WebUI 失败：{login}"}
    return await _http_json(
        f"{_webui_base()}/api{path}",
        payload or {},
        timeout,
        headers={"Authorization": f"Bearer {cred}"},
    )


async def _docker(*args: str, timeout: float = 60.0) -> tuple[int, str]:
    """执行 docker 子进程；需要服务用户（ubuntu）在 docker 组内。

    `napcat.mode=native`（不走 Docker 跑原生 NapCat）时直接返回错误，
    免得面板每轮状态轮询都白起一个进程。
    """
    if _napcat_mode() != "docker":
        return 127, "native 模式（未使用 Docker），该操作不可用"
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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{32,}$")


def _clean_napcat_log(text: str) -> list[str]:
    """滤掉颜色码、二维码字符画、哈希行与 MMKV 噪声，只留可读行。"""
    keep: list[str] = []
    for raw in text.splitlines():
        line = _ANSI_RE.sub("", raw).strip()
        if not line or line.startswith("[I] ") or _HEX_RE.match(line):
            continue
        if any(ch in line for ch in "█▄▀▐▌") or len(line) > 200:
            continue
        keep.append(line)
    return keep[-10:]


def _native_logs() -> list[str]:
    """原生模式没有 docker logs，直接读 NapCat 自己的日志目录（取最新的那个文件）。"""
    log_dir = _napcat_dir() / "logs"
    try:
        files = sorted(
            (p for p in log_dir.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    for path in files[:1]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # 只取尾部：日志文件可能很大，没必要整个读进来
        return _clean_napcat_log("\n".join(text.splitlines()[-400:]))
    return []


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
        # 面板每次部署都会变。不给 Cache-Control 时浏览器会走启发式缓存，
        # 可能拿着旧页面让人误以为改动没生效，这里直接禁止缓存。
        try:
            return HTMLResponse(
                _PANEL_HTML.read_text(encoding="utf-8"),
                headers={"Cache-Control": "no-store"},
            )
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

    # ---- 表情包管理 ----
    @app.get("/api/stickers")
    async def _stickers_list(
        page: int = 1,
        page_size: int = 24,
        status: str = "all",
        source: str = "",
        q: str = "",
    ):
        data = db.list_stickers(
            page=page, page_size=page_size, status=status, source=source, q=q
        )
        return _json(
            {"ok": True, **data, "stats": db.sticker_stats()}
        )

    @app.get("/api/stickers/stats")
    async def _stickers_stats():
        return _json({"ok": True, "stats": db.sticker_stats()})

    @app.post("/api/stickers/upload")
    async def _stickers_upload(request: Request):
        """用 JSON base64 上传，避免引入 python-multipart 依赖。"""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _json({"ok": False, "error": "请求体不是合法 JSON"}, status=400)
        files = body.get("files") if isinstance(body, dict) else None
        if files is None and isinstance(body, dict) and body.get("data"):
            files = [body]
        if not isinstance(files, list) or not files:
            return _json({"ok": False, "error": "没有收到文件"}, status=400)

        saved: list[dict[str, Any]] = []
        errors: list[str] = []
        for item in files[:30]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            raw_b64 = str(item.get("data") or "")
            try:
                raw = base64.b64decode(raw_b64, validate=True)
                row = stickers.save_upload(raw, name)
                if row is not None:
                    saved.append(row)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name or '未命名'}：{exc}")
        logger.info(f"admin_panel: 上传表情 {len(saved)} 张，失败 {len(errors)} 张")
        return _json({"ok": bool(saved), "saved": saved, "errors": errors})

    @app.post("/api/stickers/batch")
    async def _stickers_batch(request: Request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _json({"ok": False, "error": "请求体不是合法 JSON"}, status=400)
        if not isinstance(body, dict):
            return _json({"ok": False, "error": "参数必须是对象"}, status=400)
        action = str(body.get("action") or "")
        ids = [int(x) for x in (body.get("ids") or []) if str(x).isdigit()]
        if not ids:
            return _json({"ok": False, "error": "没有选择表情"}, status=400)

        changed = 0
        for sticker_id in ids:
            row = db.get_sticker(sticker_id)
            if row is None:
                continue
            if action == "delete":
                stickers.remove_file(row)
                db.delete_sticker(sticker_id)
            elif action == "approve":
                db.update_sticker(sticker_id, {"approved": True, "enabled": True})
            elif action == "enable":
                db.update_sticker(sticker_id, {"enabled": True})
            elif action == "disable":
                db.update_sticker(sticker_id, {"enabled": False})
            else:
                return _json({"ok": False, "error": f"未知操作：{action}"}, status=400)
            changed += 1
        return _json({"ok": True, "changed": changed})

    @app.get("/api/stickers/{sticker_id}/file")
    async def _sticker_file(sticker_id: int):
        row = db.get_sticker(sticker_id)
        path = stickers.resolve_path(row) if row is not None else None
        if row is None or path is None:
            return _json({"ok": False, "error": "文件不存在"}, status=404)
        return FileResponse(
            path,
            media_type=str(row.get("mime") or "application/octet-stream"),
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.post("/api/stickers/{sticker_id}/update")
    async def _sticker_update(sticker_id: int, request: Request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _json({"ok": False, "error": "请求体不是合法 JSON"}, status=400)
        if not isinstance(body, dict):
            return _json({"ok": False, "error": "参数必须是对象"}, status=400)
        row = db.update_sticker(sticker_id, body)
        if row is None:
            return _json({"ok": False, "error": "表情不存在"}, status=404)
        return _json({"ok": True, "sticker": row})

    @app.post("/api/stickers/{sticker_id}/delete")
    async def _sticker_delete(sticker_id: int):
        row = db.get_sticker(sticker_id)
        if row is None:
            return _json({"ok": False, "error": "表情不存在"}, status=404)
        stickers.remove_file(row)
        db.delete_sticker(sticker_id)
        return _json({"ok": True})

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
        """硬重启：先回响应，约 1 秒后优雅退出，systemd 自动拉起。

        Windows 上没有 systemd 兜底，进程退出了就没人拉起，这里干脆不自杀，
        直接告诉调用者需要手动重启（否则面板会把人坑得以为重启成功了）。
        """
        if not _schedule_restart():
            return _json(
                {
                    "ok": False,
                    "error": "当前平台没有守护进程兜底（Windows），"
                    "请手动停止并重新运行 bot.py",
                },
                status=501,
            )
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
    async def _style_pairs(group_id: str, limit: int = 50, offset: int = 0):
        """分页浏览某群的接话对；同时返回 total 供前端算页数。"""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        return _json(
            {
                "ok": True,
                "pairs": db.list_style_pairs(group_id, limit, offset),
                "total": db.style_pair_stats().get(group_id, 0),
                "limit": limit,
                "offset": offset,
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

    # ---- 工具注册表（@ 对话的 function calling）----
    @app.get("/api/tools")
    async def _tools_list(text: str = ""):
        """列出已注册工具；给了 text 就额外返回「这句话会带上哪些工具」。"""
        try:
            from qqbot_core import tools as registry

            registry.register_builtins()
            try:
                from plugins import ai_chat

                ai_chat.register_tools()
            except Exception:  # noqa: BLE001 —— 插件侧工具缺失不影响列出内置工具
                pass
            rows = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "visibility": tool.visibility,
                    "hints": list(tool.hints),
                    "read_only": tool.read_only,
                    "source": tool.source,
                }
                for tool in registry.all_tools()
            ]
            return _json(
                {
                    "ok": True,
                    "enabled": registry.enabled(),
                    "tools": rows,
                    "selected": [t.name for t in registry.select(text)] if text else [],
                }
            )
        except Exception as exc:  # noqa: BLE001
            return _json(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tools": []}, status=500
            )

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

        # 先记下受影响的范围：复读学习表与接话对都是从消息派生出来的
        if group_id:
            affected: list[str] | None = [group_id]
        elif user_id:
            affected = db.groups_of_user(user_id)
        else:
            affected = None  # None = 全部

        deleted = cleared = 0
        if scope in ("messages", "both"):
            deleted = db.delete_messages(group_id=group_id, user_id=user_id, ids=ids)
            # 按单条 id 删除时不碰整群的派生数据；其余情况同步清理，免得留下脏数据
            if not ids:
                if affected is None:
                    db.clear_repeat_learn(None)
                    db.clear_style_pairs(None)
                else:
                    for gid in affected:
                        db.clear_repeat_learn(gid)
                        db.clear_style_pairs(gid)
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

    # ---- 私聊 / 上下文管理 ----
    @app.get("/api/context/private")
    async def _ctx_private():
        """有私聊记录或私聊上下文的 QQ 号（两者都要看，见 db.list_private_users）。"""
        return _json({"ok": True, "users": db.list_private_users()})

    @app.get("/api/context/private/messages")
    async def _ctx_private_messages(user_id: str, limit: int = 60, offset: int = 0):
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        return _json(
            {
                "ok": True,
                "messages": db.recent_private_messages(user_id, limit, offset),
                "limit": limit,
                "offset": offset,
            }
        )

    @app.post("/api/context/private/clear")
    async def _ctx_private_clear(request: Request):
        """scope: sessions（重置上下文）| messages（删私聊记录）| both。

        只动 `group_id IS NULL` 的私聊消息和 `user:<uid>` 上下文，
        **不会碰这个人在各群里的发言** —— 那要走 /api/context/clear，
        两者刻意分开，避免“只想重置私聊上下文，结果把群语料也删了”。
        """
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if not isinstance(body, dict):
            body = {}

        user_id = str(body.get("user_id") or "").strip()
        if not user_id:
            return _json({"ok": False, "error": "缺少 user_id"}, status=400)
        scope = str(body.get("scope") or "sessions")
        if scope not in ("sessions", "messages", "both"):
            return _json({"ok": False, "error": f"未知 scope：{scope}"}, status=400)

        deleted = cleared = 0
        if scope in ("messages", "both"):
            deleted = db.delete_messages(user_id=user_id, private_only=True)
        if scope in ("sessions", "both"):
            cleared = db.clear_sessions(user_id=user_id)

        logger.info(
            f"admin_panel: 私聊清理 user={user_id} scope={scope} "
            f"-> 消息 {deleted} / 上下文 {cleared}"
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
        qr_file = _qr_path()
        if qr_file.exists():
            st = qr_file.stat()
            qr_info = {"mtime": st.st_mtime, "size": st.st_size}

        if _napcat_mode() != "docker":
            # 原生模式（如 Windows 直跑 NapCat）：没有容器，改读 NapCat 自己的日志目录
            return _json(
                {
                    "ok": True,
                    "online": bool(bots),
                    "bots": bots,
                    "mode": "native",
                    "container": "原生模式（未使用 Docker）",
                    "qr": qr_info,
                    "log": _native_logs(),
                }
            )

        rc_state, state = await _docker(
            "inspect", "-f", "{{.State.Status}}", _qq_container(), timeout=20
        )
        rc_log, logs = await _docker(
            "logs", "--tail", "60", _qq_container(), timeout=20
        )
        return _json(
            {
                "ok": True,
                "online": bool(bots),
                "bots": bots,
                "mode": "docker",
                "container": state.strip() if rc_state == 0 else "unknown",
                "qr": qr_info,
                "log": _clean_napcat_log(logs) if rc_log == 0 else [],
            }
        )

    @app.get("/api/napcat/qr")
    async def _napcat_qr():
        """把容器里的登录二维码原样吐给浏览器（no-store，Always 取最新的）。"""
        if not _qr_path().exists():
            return _json({"ok": False, "error": "暂无二维码文件"}, status=404)
        try:
            data = _qr_path().read_bytes()
        except OSError as exc:
            return _json({"ok": False, "error": str(exc)}, status=500)
        return Response(
            content=data,
            media_type="image/png",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.post("/api/napcat/refresh-qr")
    async def _napcat_refresh_qr():
        """向 NapCat WebUI API 索取登录二维码（会写入 cache/qrcode.png）。

        比重启容器温和：不丢登录态、不断开连接，失败时二维码仍可扫码。
        """
        res = await _napcat_webui("/QQLogin/GetQQLoginQrcode")
        if not isinstance(res, dict) or res.get("code") != 0:
            logger.warning(f"admin_panel: 获取登录二维码失败：{res}")
            return _json({"ok": False, "error": str(res)[:300]}, status=502)
        data = res.get("data") or {}
        url = data.get("qrcode") if isinstance(data, dict) else ""
        logger.info("admin_panel: 已获取登录二维码")
        return _json({"ok": True, "qrcode": url or "", "message": "二维码已生成"})

    @app.post("/api/napcat/restart")
    async def _napcat_restart():
        """重启 NapCat 容器（兜底手段，会中断连接）。原生模式不支持。"""
        if _napcat_mode() != "docker":
            return _json(
                {
                    "ok": False,
                    "error": "原生模式（未使用 Docker）无法从这里重启，请手动重启 NapCat 进程",
                },
                status=501,
            )
        rc, out = await _docker("restart", _qq_container(), timeout=120)
        logger.info(f"admin_panel: 重启 NapCat 容器 rc={rc}")
        if rc != 0:
            return _json({"ok": False, "error": out.strip()[:500]}, status=500)
        return _json({"ok": True, "message": "NapCat 已重启"})


@driver.on_startup
async def _mount() -> None:
    app = getattr(driver, "server_app", None)
    if app is None:
        logger.warning("admin_panel: 当前驱动未提供 ASGI server_app，面板未挂载")
        return
    _register(app)
    logger.info("admin_panel: 面板已挂载（http://127.0.0.1:8081/）")
