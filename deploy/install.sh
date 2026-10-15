#!/usr/bin/env bash
# ============================================================================
# 渲染并安装 QQ 机器人部署文件（systemd + sudoers + git 钩子）。
#
# 预览（无需 root，产物在 deploy/out/）：
#   bash deploy/install.sh --dry-run
#
# 正式安装：
#   sudo bash deploy/install.sh \
#        --app-dir /home/ubuntu/qqbot \
#        --bare-repo /home/ubuntu/qqbot.git \
#        --domain qq.starst.site
# ============================================================================
set -euo pipefail

APP_DIR="/home/ubuntu/qqbot"
VENV_DIR=""
DATA_DIR=""
APP_USER="ubuntu"
SERVICE_NAME="qqbot"
DOMAIN="qq.starst.site"
BRANCH="master"
BARE_REPO=""
DRY_RUN=0

usage() {
    cat <<'EOF'
用法： sudo bash deploy/install.sh [选项]
  --app-dir DIR      应用运行目录            默认 /home/ubuntu/qqbot
  --venv-dir DIR     Python 虚拟环境目录      默认 <app-dir>/venv
  --data-dir DIR     可写数据目录             默认 <app-dir>/data
  --user USER        运行服务的系统用户        默认 ubuntu
  --service NAME     systemd 服务名           默认 qqbot
  --domain DOMAIN    对外域名                 默认 qq.starst.site
  --bare-repo DIR    裸仓库路径；提供则安装钩子
  --branch NAME      触发部署的分支            默认 master
  --dry-run          只渲染到 deploy/out/，不改系统
  -h, --help         帮助
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --app-dir)   APP_DIR="$2"; shift 2 ;;
        --venv-dir)  VENV_DIR="$2"; shift 2 ;;
        --data-dir)  DATA_DIR="$2"; shift 2 ;;
        --user)      APP_USER="$2"; shift 2 ;;
        --service)   SERVICE_NAME="$2"; shift 2 ;;
        --domain)    DOMAIN="$2"; shift 2 ;;
        --bare-repo) BARE_REPO="$2"; shift 2 ;;
        --branch)    BRANCH="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "未知参数：$1" >&2; usage >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -d "$APP_DIR" ] && APP_DIR="$(cd "$APP_DIR" && pwd)"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"
DATA_DIR="${DATA_DIR:-$APP_DIR/data}"

render() {
    sed -e "s|__APP_DIR__|${APP_DIR}|g" \
        -e "s|__VENV_DIR__|${VENV_DIR}|g" \
        -e "s|__DATA_DIR__|${DATA_DIR}|g" \
        -e "s|__APP_USER__|${APP_USER}|g" \
        -e "s|__SERVICE_NAME__|${SERVICE_NAME}|g" \
        -e "s|__DOMAIN__|${DOMAIN}|g" \
        -e "s|__BRANCH__|${BRANCH}|g" \
        "$1" > "$2"
}

echo "渲染参数："
cat <<EOF
  服务名   : $SERVICE_NAME
  运行用户 : $APP_USER
  应用目录 : $APP_DIR
  虚拟环境 : $VENV_DIR
  数据目录 : $DATA_DIR
  域名     : $DOMAIN
  分支     : $BRANCH
  裸仓库   : ${BARE_REPO:-（未提供，跳过钩子）}
EOF

OUT_DIR="$SCRIPT_DIR/out"
mkdir -p "$OUT_DIR"
render "$SCRIPT_DIR/qqbot.service.template" "$OUT_DIR/$SERVICE_NAME.service"
render "$SCRIPT_DIR/qqbot-sudoers.template" "$OUT_DIR/$SERVICE_NAME-sudoers"
render "$SCRIPT_DIR/post-receive.template" "$OUT_DIR/post-receive"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "✓ 已渲染到 $OUT_DIR（未改动系统）"
    exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "错误：安装 systemd/sudoers 需要 root，请用 sudo 运行（或加 --dry-run 预览）。" >&2
    exit 1
fi

mkdir -p "$DATA_DIR"
install -m 0644 "$OUT_DIR/$SERVICE_NAME.service" "/etc/systemd/system/$SERVICE_NAME.service"
install -m 0440 "$OUT_DIR/$SERVICE_NAME-sudoers" "/etc/sudoers.d/$SERVICE_NAME"
install -m 0644 "$SCRIPT_DIR/nginx.conf.example" "$OUT_DIR/nginx.conf"
echo "✓ 已安装 systemd 单元与 sudoers"
echo "  请手动："
echo "    sudo visudo -c                        # 校验 sudoers"
echo "    sudo systemctl daemon-reload"
echo "    sudo systemctl enable --now $SERVICE_NAME.service"
echo "    sudo cp $OUT_DIR/nginx.conf /etc/nginx/sites-available/$SERVICE_NAME"

if [ -n "$BARE_REPO" ]; then
    if [ -d "$BARE_REPO/hooks" ]; then
        install -m 0755 "$OUT_DIR/post-receive" "$BARE_REPO/hooks/post-receive"
        echo "✓ 已安装 git 钩子到 $BARE_REPO/hooks/post-receive"
    else
        echo "⚠️  $BARE_REPO 不是裸仓库，跳过钩子安装" >&2
    fi
fi
