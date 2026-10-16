# QQ 聊天机器人（NoneBot2 + NapCat + DeepSeek）

在单台服务器（`ubuntu@starst.site`）上自建一个 QQ 群聊 AI 机器人：

- **NapCatQQ**（Docker）托管 QQ 小号，对外提供 OneBot 11 协议；
- **NoneBot2**（systemd + venv）连接 OneBot，接入 **DeepSeek `deepseek-flash`**；
- **触发**：群内默认「@ 机器人」；可切换为 **semantic** —— 任意群消息先过一道低成本接话判定；
- **读图**：消息里的图片下载后内联给模型（`deepseek-flash` 支持图片输入）；
- **风格注入**：全量落库群消息 → embedding 语义检索 → 作为 few-shot 学习群体语气；
- **@ 触发 + 三层限流**，避免刷屏与烧额度；
- 自带**管理面板**（`https://qq.starst.site`，nginx + BasicAuth），可在线调参。

```
QQ 群消息( @机器人 / 语义判定 ) ──▶ NapCatQQ(Docker, :3001 WS) ──▶ NoneBot2(systemd, :8081)
                                                     │  ├─ SQLite 消息库 + 向量
                                                     │  └─ DeepSeek deepseek-flash
管理面板 ◀── nginx qq.starst.site ◀── 127.0.0.1:8081
```

## 目录结构

```
qqbot/
├── bot/                      # NoneBot2 项目
│   ├── bot.py                # 入口
│   ├── pyproject.toml
│   ├── requirements.txt      # 供部署钩子增量安装
│   ├── .env.example          # 环境变量模板（复制为 .env 填密钥）
│   ├── qqbot_core/           # 共享核心：config / db / embeddings / media
│   └── plugins/
│       ├── msg_store/        # 全量落库群消息（图片折算为占位符）
│       ├── rag/              # embedding 语义检索 + 后台向量化
│       ├── ai_chat/          # 触发（@ / 语义判定）+ 限流 + 读图 + 调用模型
│       └── admin_panel/      # 管理面板后端 + 页面
├── napcat/docker-compose.yml # NapCat 容器编排
├── deploy/                   # systemd / nginx / sudoers / 钩子 / 安装脚本
└── README.md
```

## 运行期数据（均在仓库外或已 gitignore）

| 路径 | 内容 |
| --- | --- |
| `data/config.json` | 可热改配置（面板读写） |
| `data/qqbot.db` | 消息、向量、会话（SQLite WAL） |
| `data/hf/` | embedding 模型缓存（HF_HOME） |
| `napcat/data/` | QQ 登录态与 NapCat 配置（**切勿删除**） |
| `bot/.env` | 密钥（DeepSeek Key、OneBot token） |

---

## 触发方式与读图

两项都在面板在线调整，热生效（`data/config.json`）。

**触发（`trigger.mode`）**

- `at`（默认）：群内必须 @机器人，或命中 `trigger.prefixes`；`trigger.require_at_in_group=false`
  可放宽为"每条群消息都回"（仅靠限流兜底，活跃群慎用）。
- `semantic`：群内任意消息都先过一道**接话判定**——一次极小成本的模型调用，只回答 `1/0`。
  判定通过才真正回复；被 @或命中前缀的消息跳过判定直接回。成本随群活跃度上升，
  因此有 `trigger.semantic.max_per_minute`（每群每分钟判定次数上限，超出后只响应直接点名）；
  `trigger.aliases` 填机器人昵称/别名，判定时可识别点名。私聊不受 `mode` 影响。

**读图（`vision`）**

- 消息里的图片会下载后以 base64 内联给模型（图床直链常有防盗链/过期，不能直接交给模型）；
- `max_images` / `max_mb` / `timeout_sec` 限制张数、单张大小与下载超时，读取失败的单张跳过；
- 模型必须支持图片输入，所以 `model` 要用 `deepseek-flash`（`deepseek-v4-pro` 是纯文本）；
- 服务器没装 `Pillow`：图片只做大小/类型校验，不做缩放或转码。

---

## 部署步骤

### 0. 前置检查

```bash
free -h && df -h && nproc
ss -lntp | grep -E ':(3001|6099|8081)' || echo "端口空闲"
docker --version || echo "需要安装 Docker"
```

- NapCat（NTQQ 无头进程）约占 **1.5–2 GB 内存**，加上 embedding 模型（`bge-m3` 约 2 GB）。
  **可用内存 < 4 GB 时**：先把面板里的 embedding 模型换成 `BAAI/bge-small-zh-v1.5`（约 100 MB），
  或先加 2 GB swap。
- 确认 `qq.starst.site` 已解析到本机（你已完成）。

### 1. 上传代码并建裸仓库

```bash
# 服务器上：
git init --bare /home/ubuntu/qqbot.git
mkdir -p /home/ubuntu/qqbot

# 本地（d:\gits\qqbot）：
git init && git add -A && git commit -m "init"
git remote add origin ubuntu@starst.site:/home/ubuntu/qqbot.git
git push -u origin master
```

### 2. 建虚拟环境

```bash
cd /home/ubuntu/qqbot
python3 -m venv venv
venv/bin/pip install -U pip
venv/bin/pip install -r bot/requirements.txt
```

### 3. 配置密钥

```bash
cp bot/.env.example bot/.env
vi bot/.env      # 填 DEEPSEEK_API_KEY、ONEBOT_V11_ACCESS_TOKEN
```

### 4. 启动 NapCat 并登录 QQ

```bash
cd /home/ubuntu/qqbot/napcat
docker compose up -d
docker compose logs -f      # 看是否起来

# WebUI 只监听 127.0.0.1，用 SSH 隧道访问：
# 本地：ssh -L 6099:127.0.0.1:6099 ubuntu@starst.site
# 浏览器打开 http://127.0.0.1:6099 → 扫码登录 QQ 小号
```

登录后，在 NapCat 的 **网络配置** 里新增：

- 类型：**WebSocket 服务器**（正向 WS）
- 主机：`0.0.0.0`（或留空；容器内监听全部地址，才能经端口映射出去）
- 端口：`3001`
- Access Token：与 `bot/.env` 的 `ONEBOT_V11_ACCESS_TOKEN` 完全一致

### 5. 安装并启动机器人服务

```bash
sudo bash deploy/install.sh --dry-run     # 先预览
sudo bash deploy/install.sh \
     --app-dir /home/ubuntu/qqbot \
     --bare-repo /home/ubuntu/qqbot.git \
     --domain qq.starst.site

sudo visudo -c
sudo systemctl daemon-reload
sudo systemctl enable --now qqbot.service
journalctl -u qqbot -f      # 应看到 OneBot 连接成功
```

### 6. 配置管理面板（nginx + TLS）

```bash
sudo cp deploy/nginx.conf.example /etc/nginx/sites-available/qqbot
sudo ln -sf /etc/nginx/sites-available/qqbot /etc/nginx/sites-enabled/qqbot
sudo htpasswd -c /etc/nginx/.htpasswd-qqbot admin   # 首次创建口令
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d qq.starst.site
```

之后浏览器打开 `https://qq.starst.site`（需 BasicAuth），即可在线调整模型、限流、开关等。

### 7. 验证

```bash
ss -lntp | grep 3001                    # NapCat WS 在听
systemctl status qqbot --no-pager
curl -su admin https://qq.starst.site/  # 面板可达
```

在群里 @ 机器人 提问 → 应收到回复；连续 @ 第二次会被限流静默丢弃；
面板「运行统计」的消息/向量数会增长。

---

## 日常运维

| 操作 | 命令 |
| --- | --- |
| 查看机器人日志 | `journalctl -u qqbot -f` |
| 查看 NapCat 日志 | `cd napcat && docker compose logs -f` |
| 升级 NapCat | `docker compose pull && docker compose up -d` |
| 改代码部署 | 本地 `git push origin master`（钩子自动重启） |
| 改配置 | 面板在线改，或编辑 `data/config.json`（自动热载） |
| 备份 | `data/qqbot.db`、`data/config.json`、`napcat/data/`、`bot/.env` |

## 隐私与合规

- 群消息**全量落库**用于风格学习，请**在群内公开说明**并取得群友默许；
  图片本身不落库，只保存 `[图片]` 一类占位符；
- 数据仅存本机；但**对话内容与读图时的图片会发送给模型服务商**（DeepSeek API），
  发图前请确认群里没有敏感信息；
- 删除某群数据时：清 `messages`/`embeddings` 对应行后调用面板「保存并失效缓存」；
- 自动化行为受 QQ 平台条款约束，建议使用小号并控制发言频率。

## 关键取舍

- **语义检索默认 `BAAI/bge-m3`**：中文效果最好但约 2 GB 内存；无 GPU 的服务器若吃紧，
  在面板把 `embedding 模型` 改为 `BAAI/bge-small-zh-v1.5` 后重启服务即可。
  embedding 加载失败会**自动降级**为 FTS5 关键词检索，机器人不会因此不可用。
- **思考模式默认关闭**：群聊场景求快；需要推理时在面板打开。
- **semantic 触发有额外调用成本**：每条群消息都要判一次"该不该接话"，活跃群建议配合
  `trigger.semantic.max_per_minute` 与限流一起用；不确定就先留在 `at` 模式。
- **读图会占带宽与上下文**：图片走 base64 内联，单张上限默认 5 MB、单条最多 3 张；
  群里图多时模型输入会明显变长，可调小 `vision.max_mb` / `vision.max_images` 或直接关掉读图。
- OneBot 端口仅 `127.0.0.1`；面板公网访问必须经 nginx BasicAuth。
