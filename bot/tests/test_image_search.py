from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from plugins.image_search import parse, providers, render
from plugins.image_search.models import Candidate, SearchResult
from qqbot_core import secrets
from qqbot_core.vision_context import current_image_sources


class ParseTests(unittest.TestCase):
    def test_x_urls(self):
        cases = {
            "https://x.com/mamamia1008/status/123": ("X (Twitter)", "mamamia1008"),
            "https://twitter.com/foo_bar": ("X (Twitter)", "foo_bar"),
            "https://mobile.twitter.com/foo/status/1": ("X (Twitter)", "foo"),
            "https://x.com/i/status/1": ("X (Twitter)", ""),
        }
        for url, expected in cases.items():
            self.assertEqual(parse.platform_for_url(url), expected)

    def test_lens_redirect(self):
        url = "https://lens.google.com/url?url=https%3A%2F%2Fx.com%2Fbar%2Fstatus%2F1"
        self.assertEqual(parse.normalize_url(url), "https://x.com/bar/status/1")
        self.assertEqual(parse.platform_for_url(url), ("X (Twitter)", "bar"))

    def test_other_platforms(self):
        self.assertEqual(
            parse.platform_for_url("https://www.weibo.com/u/123456"), ("微博", "123456")
        )
        self.assertEqual(
            parse.platform_for_url("https://space.bilibili.com/789"), ("Bilibili", "789")
        )
        self.assertEqual(
            parse.platform_for_url("https://www.instagram.com/foo/"), ("Instagram", "foo")
        )

    def test_extract_name(self):
        self.assertEqual(
            parse.extract_name("咪娅酱 (@mamamia1008) / X", "mamamia1008"), "咪娅酱"
        )
        self.assertEqual(
            parse.extract_name("咪娅酱 on X: hello", "mamamia1008"), "咪娅酱"
        )
        self.assertEqual(parse.extract_name("咪娅酱 - X", ""), "咪娅酱")
        self.assertEqual(parse.extract_name("", "mamamia1008"), "mamamia1008")


class ProviderParseTests(unittest.TestCase):
    def test_serpapi(self):
        payload = {
            "exact_matches": [
                {
                    "title": "咪娅酱 (@mamamia1008) / X",
                    "link": "https://x.com/mamamia1008/status/1",
                    "source": "X",
                }
            ],
            "visual_matches": [
                {
                    "title": "Other on X",
                    "link": "https://twitter.com/other/status/2",
                    "source": "X",
                }
            ],
        }
        items = providers.candidates_from_serpapi(payload, 3)
        self.assertGreaterEqual(len(items), 2)
        self.assertEqual(items[0].platform, "X (Twitter)")
        self.assertEqual(items[0].handle, "mamamia1008")
        self.assertEqual(items[0].name, "咪娅酱")
        self.assertEqual(items[0].match, "精确匹配")

    def test_saucenao(self):
        payload = {
            "results": [
                {
                    "header": {"similarity": "96.2", "index_name": "Twitter"},
                    "data": {
                        "twitter_user_handle": "mamamia1008",
                        "author": "咪娅酱",
                        "ext_urls": ["https://x.com/mamamia1008/status/1"],
                    },
                }
            ]
        }
        items = providers.candidates_from_saucenao(payload, 3)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].handle, "mamamia1008")
        self.assertEqual(items[0].name, "咪娅酱")
        self.assertIn("96", items[0].match)

    def test_rank_dedupes_same_handle(self):
        items = [
            Candidate("X (Twitter)", "A", "same", confidence=0.8, position=1),
            Candidate("X (Twitter)", "B", "same", confidence=0.9, position=2),
        ]
        ranked = providers._rank(items, 3)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].name, "B")


class RenderTests(unittest.TestCase):
    def test_card(self):
        card = render.render_card(
            [
                Candidate(
                    platform="X (Twitter)",
                    name="咪娅酱🐰",
                    handle="mamamia1008",
                    url="https://x.com/mamamia1008/status/1",
                    match="视觉匹配",
                )
            ],
            title="识图结果",
        )
        self.assertIn("🎯 识图结果", card)
        self.assertIn("🌐 平台：X (Twitter)", card)
        self.assertIn("🔍 匹配：视觉匹配", card)
        self.assertIn("👤 名字：咪娅酱🐰", card)
        self.assertIn("🆔 账号：@mamamia1008", card)
        self.assertIn("⚠️ 识图存在误差", card)

    def test_empty(self):
        self.assertEqual(render.render_card([]), "")


class PanelTests(unittest.TestCase):
    def test_secret_state_uses_backend_key_name(self):
        path = Path(__file__).resolve().parents[1] / "plugins" / "admin_panel" / "index.html"
        html = path.read_text(encoding="utf-8")
        self.assertIn("state.SAUCENAO_API_KEY", html)
        self.assertNotIn("state.SAUCENAO_KEY", html)
        self.assertIn("btnSaveSecrets", html)
        self.assertIn("/api/secrets", html)


class SecretsTests(unittest.TestCase):
    def setUp(self):
        self._old = {key: os.environ.get(key) for key in secrets.ALLOWED_KEYS}
        for key in secrets.ALLOWED_KEYS:
            os.environ.pop(key, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.env_path = Path(self.tmp.name) / ".env"
        self.env_path.write_text("# keep\nOTHER=1\nSERPAPI_KEY=old-key\n", encoding="utf-8")
        self.patch = patch.object(secrets, "ENV_PATH", self.env_path)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_set_and_mask(self):
        secrets.set_many({"SERPAPI_KEY": "sk-test-1234", "SAUCENAO_API_KEY": "sn-5678"})
        text = self.env_path.read_text(encoding="utf-8")
        self.assertIn("SERPAPI_KEY=sk-test-1234", text)
        self.assertIn("SAUCENAO_API_KEY=sn-5678", text)
        self.assertIn("# keep", text)
        self.assertIn("OTHER=1", text)
        state = secrets.status()
        self.assertTrue(state["SERPAPI_KEY"]["configured"])
        self.assertNotIn("sk-test-1234", json.dumps(state))
        self.assertIn("1234", state["SERPAPI_KEY"]["preview"])

    def test_clear(self):
        secrets.set_many({"SERPAPI_KEY": "", "SAUCENAO_API_KEY": ""})
        text = self.env_path.read_text(encoding="utf-8")
        self.assertNotIn("SERPAPI_KEY=", text)
        self.assertNotIn("SAUCENAO_API_KEY=", text)
        self.assertFalse(secrets.status()["SERPAPI_KEY"]["configured"])


def _serpapi_payload() -> dict:
    return {
        "visual_matches": [
            {
                "title": "咪娅酱 (@mamamia1008) / X",
                "link": "https://x.com/mamamia1008/status/1",
                "source": "X",
            }
        ]
    }


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_serpapi_via_mock_transport(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "serpapi.com")
            return httpx.Response(200, json=_serpapi_payload())

        cfg = {
            "enabled": True,
            "providers": ["serpapi"],
            "max_results": 3,
            "timeout_sec": 5,
            "max_image_mb": 8,
            "safe_search": True,
        }
        with patch.object(providers, "_cfg", return_value=cfg), patch.object(
            secrets, "get", side_effect=lambda key: "k" if key == "SERPAPI_KEY" else ""
        ):
            result = await providers.search_image(
                ["https://example.com/a.jpg"], transport=httpx.MockTransport(handler)
            )
        self.assertEqual(result.provider, "serpapi")
        self.assertEqual(result.best.handle, "mamamia1008")

    async def test_saucenao_fallback_downloads_image(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "img.example":
                return httpx.Response(
                    200,
                    content=b"\xff\xd8\xff\xe0fakejpeg",
                    headers={"content-type": "image/jpeg"},
                )
            self.assertEqual(request.url.host, "saucenao.com")
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "header": {"similarity": "95.0", "index_name": "Twitter"},
                            "data": {
                                "twitter_user_handle": "mamamia1008",
                                "author": "咪娅酱",
                                "ext_urls": ["https://x.com/mamamia1008/status/1"],
                            },
                        }
                    ]
                },
            )

        cfg = {
            "enabled": True,
            "providers": ["serpapi", "saucenao"],
            "max_results": 3,
            "timeout_sec": 5,
            "max_image_mb": 8,
            "safe_search": True,
        }
        with patch.object(providers, "_cfg", return_value=cfg), patch.object(
            secrets, "get", side_effect=lambda key: "k" if key == "SAUCENAO_API_KEY" else ""
        ):
            result = await providers.search_image(
                ["https://img.example/a.jpg"], transport=httpx.MockTransport(handler)
            )
        self.assertEqual(result.provider, "saucenao")
        self.assertEqual(result.best.handle, "mamamia1008")

    async def test_no_keys(self):
        cfg = {"enabled": True, "providers": ["serpapi", "saucenao"]}
        with patch.object(providers, "_cfg", return_value=cfg), patch.object(
            secrets, "get", return_value=""
        ):
            result = await providers.search_image(
                ["https://example.com/a.jpg"], transport=httpx.MockTransport(lambda r: None)
            )
        self.assertFalse(result.candidates)
        self.assertTrue(all("未配置 Key" in item for item in result.errors))


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_returns_card(self):
        from plugins.image_search import _tool_handler

        result = SearchResult(
            candidates=[
                Candidate(
                    platform="X (Twitter)",
                    name="咪娅酱",
                    handle="mamamia1008",
                    url="https://x.com/mamamia1008/status/1",
                    match="视觉匹配",
                )
            ],
            provider="serpapi",
        )
        token = current_image_sources.set(("https://example.com/a.jpg",))
        try:
            with patch("plugins.image_search.search_image", return_value=result):
                raw = await _tool_handler({})
        finally:
            current_image_sources.reset(token)
        payload = json.loads(raw)
        self.assertTrue(payload["ok"])
        self.assertIn("🎯 识图结果", payload["card"])
        self.assertEqual(payload["direct_reply"], payload["card"])
        self.assertLessEqual(len(raw), 1600)

    async def test_tool_without_image(self):
        from plugins.image_search import _tool_handler

        token = current_image_sources.set(())
        try:
            raw = await _tool_handler({})
        finally:
            current_image_sources.reset(token)
        payload = json.loads(raw)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "no_image")

    async def test_tool_without_keys(self):
        from plugins.image_search import _tool_handler

        result = SearchResult(
            candidates=[],
            errors=["serpapi: 未配置 Key", "saucenao: 未配置 Key"],
        )
        token = current_image_sources.set(("https://example.com/a.jpg",))
        try:
            with patch("plugins.image_search.search_image", return_value=result):
                raw = await _tool_handler({})
        finally:
            current_image_sources.reset(token)
        payload = json.loads(raw)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "not_configured")
        self.assertIn("管理面板", payload["hint"])


if __name__ == "__main__":
    unittest.main()
