#!/usr/bin/env python3
"""small_agent: 4 tools (read/write/edit/bash) + OpenAI-compatible /v1 + web UI.

    cp .env.example .env   # fill OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL
    python3 agent.py       # http://127.0.0.1:8090
    python3 agent.py --selftest
"""
import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import unquote
from urllib.request import Request, urlopen

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORKSPACE = os.getcwd()  # ponytail: no sandbox beyond "must be an existing dir"; bash can reach anything anyway
MAX_ROUNDS = 20
OUT_CAP = 20000    # chars of tool output fed back to the model
HIST_CAP = 60000   # chars of history kept before dropping the oldest full turns
RETRY_CODES = (429, 500, 502, 503, 504)

SYSTEM = (
    "You are a minimal coding agent. You have four tools: read, write, edit, bash. "
    "read before edit. Use edit for surgical changes, write only for new/whole files. "
    "Work in small verified steps and state what you changed and why. Reply in the user's language."
)
# 技能属于 agent 本身，跟工作区无关，固定放在脚本目录下
# ponytail: 这两个环境变量只为测试隔离（把 config/技能指到临时目录），正常启动不设
SKILLS_DIR = os.environ.get("SMALL_AGENT_SKILLS") or os.path.join(APP_DIR, "skills")

_TLS = threading.local()  # per-request workspace: ThreadingHTTPServer serves chats in parallel

# 正在跑的对话：rid -> {"stop": bool, "resp": 上游响应}。前端点「停止」时按 rid 掐掉，
# 只掐自己那条，多标签页互不影响。
_RUNS = {}
_RUNS_LOCK = threading.Lock()


class _Stopped(Exception):
    """客户端点了停止。"""


def _stop_requested():
    with _RUNS_LOCK:
        r = _RUNS.get(getattr(_TLS, "rid", None))
        return bool(r and r["stop"])


def _kill_resp(resp):
    """掐上游连接。只调 close() 不行：读线程正占着文件的锁，close 会一直等；
    shutdown 才让阻塞中的 recv 立刻返回。"""
    sock = None
    try:
        sock = resp.fp.raw._sock      # urllib -> http.client.HTTPResponse -> BufferedReader -> SocketIO
    except AttributeError:
        pass
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    try:
        resp.close()
    except Exception:
        pass


def _run_put(key, val):
    """把「可被掐的东西」（上游响应 / 正在跑的子进程）挂到当前请求上，给 /api/stop 用。"""
    with _RUNS_LOCK:
        r = _RUNS.get(getattr(_TLS, "rid", None))
        if r is not None:
            r[key] = val


def stop_run(rid):
    with _RUNS_LOCK:
        r = _RUNS.get(rid)
        if not r:
            return False
        r["stop"] = True
        resp, proc = r.get("resp"), r.get("proc")
    if resp is not None:
        _kill_resp(resp)   # 立刻断开上游：不再烧 token，卡在 read 里的线程马上醒来
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)   # 正在跑的工具也一起掐掉，别等它自己结束
        except (ProcessLookupError, PermissionError):
            pass
    return True


def workspace():
    return getattr(_TLS, "workspace", None) or CFG.get("workspace") or DEFAULT_WORKSPACE


TOOLS = [
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a text file with line numbers.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-indexed start line"},
            "limit": {"type": "integer", "description": "max lines, default 2000"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write",
        "description": "Create or fully overwrite a file (parents are created).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit",
        "description": "Replace an exact string in a file. Fails unless old_string is unique (unless replace_all).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "exact text, include surrounding context"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in the workspace and return stdout+stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "seconds, default 120"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "skill",
        "description": "Load the full instructions of a skill by name. Skills are reusable procedures; "
                       "call this before starting a task one of them covers.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}},
            "required": ["name"]}}},
]


def _frontmatter(raw):
    """Split '---\\nkey: value\\n---\\nbody' -> ({key: value}, body). No YAML dep."""
    if raw.startswith("---"):
        head, sep, body = raw[3:].partition("\n---")
        if sep:
            meta = {}
            for line in head.splitlines():
                if ":" in line and not line.lstrip().startswith("#"):
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip("\"'")
            return meta, body.lstrip("\n")
    return {}, raw


def load_skills():
    """{name: (description, body)} scanned fresh each call, so the agent can write a skill and use it."""
    out = {}
    if not os.path.isdir(SKILLS_DIR):
        return out
    for entry in sorted(os.listdir(SKILLS_DIR)):
        p = os.path.join(SKILLS_DIR, entry, "SKILL.md")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            meta, body = _frontmatter(f.read())
        desc = meta.get("description") or next((l.strip() for l in body.splitlines() if l.strip()), "")
        out[meta.get("name") or entry] = (desc, body.strip())
    return out


def system_prompt():
    skills = load_skills()
    if not skills:
        return SYSTEM
    return SYSTEM + "\n\n## Skills\nReusable procedures — call skill(name) to load one before doing that kind of task.\n" + \
        "\n".join("- %s: %s" % (n, d) for n, (d, _) in skills.items())


# ---------------------------------------------------------------- tools
def _abs(path):
    path = os.path.expanduser(path or "")
    return path if os.path.isabs(path) else os.path.join(workspace(), path)


def _cap(s):
    return s if len(s) <= OUT_CAP else s[:OUT_CAP] + "\n... [truncated %d chars]" % (len(s) - OUT_CAP)


def tool_read(path, offset=1, limit=2000):
    p = _abs(path)
    offset, limit = max(1, int(offset)), min(int(limit), 2000)
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    part = lines[offset - 1: offset - 1 + limit]
    if not part:
        return "(empty: file has %d lines)" % len(lines)
    out = "".join("%6d|%s" % (offset + i, l if l.endswith("\n") else l + "\n") for i, l in enumerate(part))
    tail = offset - 1 + len(part)
    if tail < len(lines):
        out += "... [%d more lines, continue with offset=%d]" % (len(lines) - tail, tail + 1)
    return _cap(out)


def tool_write(path, content):
    p = _abs(path)
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return "wrote %s (%d chars, %d lines)" % (path, len(content), content.count("\n") + 1)


def tool_edit(path, old_string, new_string, replace_all=False):
    if not old_string:
        raise ValueError("old_string is empty")
    p = _abs(path)
    with open(p, "r", encoding="utf-8") as f:
        src = f.read()
    n = src.count(old_string)
    if n == 0:
        raise ValueError("old_string not found in %s" % path)
    if n > 1 and not replace_all:
        raise ValueError("old_string appears %d times in %s; add more context or set replace_all" % (n, path))
    with open(p, "w", encoding="utf-8") as f:
        f.write(src.replace(old_string, new_string) if replace_all else src.replace(old_string, new_string, 1))
    return "edited %s (%d replacement%s)" % (path, n if replace_all else 1, "s" if replace_all else "")


def tool_bash(command, timeout=120):
    """Stream stdout+stderr, keep only OUT_CAP bytes: `cat bigfile` / `yes` can't eat RAM.
    ponytail: 整组杀进程（start_new_session + killpg），别留 `npm run dev` 的孙进程当孤儿。"""
    t = min(int(timeout), 600)
    p = subprocess.Popen(command, shell=True, cwd=workspace(), stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    _run_put("proc", p)   # 前端点停止时 killpg 掉它，不然得等 timeout
    out, total, timed_out = bytearray(), 0, False
    end = time.monotonic() + t
    with p.stdout:
        while True:
            left = end - time.monotonic()
            if left <= 0:
                timed_out = True
                break
            if not select.select([p.stdout], [], [], left)[0]:
                continue  # select 超时 -> 下一圈 left<=0 走 timeout 分支
            chunk = p.stdout.read1(65536)
            if not chunk:
                break
            total += len(chunk)
            if len(out) < OUT_CAP:
                out += chunk
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    code = p.wait()
    _run_put("proc", None)
    s = out.decode("utf-8", "replace")
    if total > len(out):
        s += "\n... [truncated %d bytes]" % (total - len(out))
    head = "timeout %ds, killed\n" % t if timed_out else ""
    return _cap("exit=%d\n%s%s" % (code, head, s.strip() or "(no output)"))


def tool_skill(name):
    skills = load_skills()
    hit = skills.get(name) or skills.get(os.path.basename(name.rstrip("/")))
    if not hit:
        return "ERROR: no skill %r. Available: %s" % (name, ", ".join(skills) or "(none)")
    return "# skill: %s\n\n%s" % (name, hit[1])


DISPATCH = {"read": tool_read, "write": tool_write, "edit": tool_edit, "bash": tool_bash, "skill": tool_skill}


def call_tool(name, args):
    fn = DISPATCH.get(name)
    if not fn:
        return "ERROR: unknown tool %r" % name
    try:
        return fn(**args)
    except Exception as e:  # tool errors go back to the model, not up the stack
        return "ERROR: %s: %s" % (type(e).__name__, e)


# ---------------------------------------------------------------- llm
CFG_PATH = os.environ.get("SMALL_AGENT_CONFIG") or os.path.join(APP_DIR, "config.json")  # web 配置，优先级高于 .env
CFG_FIELDS = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "MAX_ROUNDS")
CFG = {}
if os.path.exists(CFG_PATH):
    try:
        with open(CFG_PATH, encoding="utf-8") as f:
            CFG = json.load(f)
    except ValueError:
        CFG = {}


def env(key, default=""):
    # CFG 里有这个键（哪怕是空串）= 用户在网页上明确设过，不再回落到 .env
    if key in CFG and CFG[key] is not None:
        return (str(CFG[key]) or default).strip()
    return (os.environ.get(key, default) or "").strip()


def _write_cfg():
    # 先写临时文件再替换：写到一半崩了也不会把 config.json 弄成半截 JSON（那等于丢 key）
    tmp = CFG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=1)
    os.chmod(tmp, 0o600)
    os.replace(tmp, CFG_PATH)


def max_rounds():
    """config.json 的 MAX_ROUNDS 优先，否则用默认；非法值回落到默认。"""
    try:
        return min(max(int(env("MAX_ROUNDS", "") or MAX_ROUNDS), 1), 200)
    except ValueError:
        return MAX_ROUNDS


def public_config():
    key = env("OPENAI_API_KEY")
    return {"base_url": env("OPENAI_BASE_URL"), "model": env("OPENAI_MODEL"),
            "key_set": bool(key),
            "key_hint": (key[:6] + "…" + key[-4:]) if len(key) > 12 else ("已配置" if key else ""),
            "max_rounds": max_rounds(),
            "workspace": CFG.get("workspace") or DEFAULT_WORKSPACE,
            "recent": CFG.get("recent") or []}


def save_config(patch):
    for k, v in patch.items():
        if k not in CFG_FIELDS:
            continue
        v = str(v if v is not None else "").strip()
        if k == "OPENAI_API_KEY" and not v:
            continue  # 空 = 保持原 key，不覆盖
        if k == "MAX_ROUNDS":
            if not v.isdigit() or not 1 <= int(v) <= 200:
                raise ValueError("循环次数要填 1-200 的整数")
        CFG[k] = v
    _write_cfg()
    return public_config()


def _resolve_dir(path):
    p = os.path.abspath(os.path.expanduser((path or "").strip() or DEFAULT_WORKSPACE))
    if not os.path.isdir(p):
        raise ValueError("目录不存在：%s" % p)
    return p


def set_workspace(path):
    p = _resolve_dir(path)
    CFG["workspace"] = p
    CFG["recent"] = ([p] + [r for r in CFG.get("recent") or [] if r != p])[:8]
    _write_cfg()
    return public_config()


def browse(path):
    """列子目录，给前端点选。ponytail: 隐藏点目录；要选 .config 之类就手输路径。"""
    p = _resolve_dir(path)
    try:
        dirs = sorted(d for d in os.listdir(p)
                      if not d.startswith(".") and os.path.isdir(os.path.join(p, d)))
    except OSError as e:
        raise ValueError("读不了 %s：%s" % (p, e))
    parent = os.path.dirname(p)
    return {"path": p, "parent": None if parent == p else parent, "dirs": dirs}


SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def save_skill(name, description, body):
    name = (name or "").strip()
    if not SKILL_NAME.match(name):
        raise ValueError("技能名只能用小写字母/数字/_/-，长度 1-64")
    d = os.path.join(SKILLS_DIR, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: %s\ndescription: %s\n---\n\n%s\n"
                % (name, (description or "").strip().replace("\n", " "), (body or "").strip()))
    return name


def delete_skill(name):
    p = os.path.join(SKILLS_DIR, os.path.basename(name or ""), "SKILL.md")
    if not os.path.isfile(p):
        raise ValueError("没有技能 %r" % name)
    os.remove(p)
    try:
        os.rmdir(os.path.dirname(p))
    except OSError:
        pass  # 目录里还有别的文件，留着
    return name


def skills_payload():
    return [{"name": n, "description": d, "body": b} for n, (d, b) in load_skills().items()]


_STREAM_OPTS = True   # 要 usage 就得带 stream_options；个别网关不认，400 之后自动关掉再试


def _completion_req(messages):
    body = {"model": env("OPENAI_MODEL", "gpt-4o-mini"), "messages": messages,
            "tools": TOOLS, "stream": True}
    if _STREAM_OPTS:
        body["stream_options"] = {"include_usage": True}
    base = env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    return Request(base + "/chat/completions", data=json.dumps(body).encode(),
                   headers={"Content-Type": "application/json",
                            "Authorization": "Bearer " + env("OPENAI_API_KEY")})


def chat_stream(messages, emit):
    """One streamed completion -> (text, tool_calls). Text deltas are emitted live."""
    global _STREAM_OPTS
    if _stop_requested():
        raise _Stopped()
    # 只在还没往外吐任何字之前重试；中途断流就直接报错，不重放
    resp, attempt = None, 0
    while resp is None:
        try:
            resp = urlopen(_completion_req(messages), timeout=600)
        except HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")
            except OSError:
                detail = ""
            if _STREAM_OPTS and e.code == 400 and "stream_options" in detail:
                _STREAM_OPTS = False   # 这个网关不支持，关掉换 token 统计
                continue
            if e.code not in RETRY_CODES or attempt >= 2:
                raise
            attempt += 1
            time.sleep(2 ** (attempt - 1))
    with _RUNS_LOCK:   # 登记给 /api/stop 掐
        r = _RUNS.get(getattr(_TLS, "rid", None))
        if r is not None:
            if r["stop"]:
                resp.close()
                raise _Stopped()
            r["resp"] = resp
    try:
        return _read_stream(resp, emit)
    except (_Stopped, ValueError, OSError):
        # 停止会从别的线程 close 掉 response，读循环会抛 ValueError/OSError；
        # 只有真的是「被停止」时才当成正常收尾，否则照常报错
        if _stop_requested():
            raise _Stopped()
        raise
    finally:
        with _RUNS_LOCK:
            r = _RUNS.get(getattr(_TLS, "rid", None))
            if r is not None:
                r["resp"] = None


def _read_stream(resp, emit):
    text, calls = "", {}
    with resp as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            # 有些网关把业务错误当 SSE 帧推进来：{"error": {...}}，别静默吞成空白回复
            if isinstance(chunk, dict) and chunk.get("error"):
                err = chunk["error"]
                raise RuntimeError("上游返回错误：%s" % (err.get("message") if isinstance(err, dict) else err))
            if isinstance(chunk, dict) and chunk.get("usage"):   # 上游要不要给是它的事，有就透传
                u = chunk["usage"]
                emit({"type": "usage", "usage": {"prompt": u.get("prompt_tokens") or 0,
                                                 "completion": u.get("completion_tokens") or 0,
                                                 "total": u.get("total_tokens") or 0}})
            try:
                delta = chunk["choices"][0].get("delta") or {}
            except (KeyError, IndexError, TypeError, AttributeError):
                continue
            if delta.get("content"):
                text += delta["content"]
                emit({"type": "text", "delta": delta["content"]})
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index")
                if idx is None:  # 没给 index 的服务：按 id 认领已有槽位，认不出就新开一个
                    idx = next((i for i, s in calls.items() if tc.get("id") and s["id"] == tc["id"]), len(calls))
                slot = calls.setdefault(idx, {"id": "", "type": "function",
                                              "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
    ordered = [calls[i] for i in sorted(calls)]
    for i, c in enumerate(ordered):
        if not c["id"]:  # 没有 id 的工具消息下一轮会被上游 400
            c["id"] = "call_%d" % i
    return text, ordered


def _trim(msgs):
    """超预算就从最早的整段开始丢：user 起、到下一个 user 之前，别拆散 tool_calls / tool 配对。"""
    # ponytail: 每轮全量求和，会话规模下无所谓；真长了就用前缀和
    while len(msgs) > 2 and sum(len(m.get("content") or "") for m in msgs) > HIST_CAP:
        j = next((k for k in range(2, len(msgs)) if msgs[k]["role"] == "user"), len(msgs))
        del msgs[1:j]
    return msgs


def run_agent(history, emit):
    msgs = [{"role": "system", "content": system_prompt()}] + [
        {"role": m["role"], "content": m["content"]} for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")]
    rounds = max_rounds()
    for _ in range(rounds):
        if _stop_requested():
            raise _Stopped()
        text, calls = chat_stream(_trim(msgs), emit)
        msgs.append({"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})})
        if not calls:
            return
        for c in calls:
            if _stop_requested():   # 停止后不再执行后面的工具，别把没敲定的改动做下去
                raise _Stopped()
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args, result = {}, "ERROR: arguments were not valid JSON"
            else:
                result = None
            emit({"type": "tool", "id": c["id"], "name": name, "args": args})
            result = result if result is not None else call_tool(name, args)
            if _stop_requested():   # 工具是被掐死的，别再把「killed」当结果回给模型/前端
                raise _Stopped()
            emit({"type": "tool_result", "id": c["id"], "output": result[:2000]})
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    emit({"type": "text", "delta": "\n\n[已跑满 %d 轮，停在最后一步；需要继续就再说一句]" % rounds})


# ---------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % a))

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8", PAGE.encode())
        if path == "/api/config":
            return self._json(public_config())
        if path == "/api/skills":
            return self._json(skills_payload())
        if path == "/api/workspaces":
            q = self.path.partition("?")[2]
            want = dict(p.split("=", 1) for p in q.split("&") if "=" in p).get("path", "")
            try:
                return self._json(browse(unquote(want)))
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
        self._send(404, "text/plain", b"not found")

    def _json(self, obj, code=200):
        self._send(code, "application/json; charset=utf-8",
                   json.dumps(obj, ensure_ascii=False).encode())

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return self._send(400, "text/plain", b"bad json")
        if path == "/api/chat":
            return self.chat(body)
        if path == "/api/stop":
            return self._json({"ok": stop_run((body.get("rid") or "").strip())})
        try:
            if path == "/api/config":
                out = save_config(body)
            elif path == "/api/skills":
                out = {"ok": save_skill(body.get("name"), body.get("description"), body.get("body"))}
            elif path == "/api/skills/delete":
                out = {"ok": delete_skill(body.get("name"))}
            elif path == "/api/workspaces":
                out = set_workspace(body.get("path"))
            else:
                return self._send(404, "text/plain", b"not found")
        except (ValueError, OSError) as e:
            return self._json({"error": str(e)}, 400)
        self._json(out)

    def chat(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        rid = (body.get("rid") or "").strip()
        if rid:
            with _RUNS_LOCK:
                _RUNS.setdefault(rid, {"stop": False, "resp": None})
            _TLS.rid = rid

        def emit(ev):
            try:
                self.wfile.write(("data: " + json.dumps(ev, ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise SystemExit  # browser went away

        try:
            want = (body.get("workspace") or "").strip()
            if want:
                _TLS.workspace = _resolve_dir(want)  # 每个请求一个工作区，线程局部，不污染别的会话
            if not env("OPENAI_API_KEY"):
                emit({"type": "error", "message": "还没配置 API Key：点右上角「设置 → 模型」填一下"})
            else:
                emit({"type": "workspace", "path": workspace()})
                run_agent(body.get("messages") or [], emit)
        except _Stopped:
            pass          # 客户端点了停止：正常收尾，不算错误
        except SystemExit:
            raise
        except Exception as e:
            emit({"type": "error", "message": "%s: %s" % (type(e).__name__, e)})
        finally:
            _TLS.workspace = None
            _TLS.rid = None
            if rid:
                with _RUNS_LOCK:
                    _RUNS.pop(rid, None)
        try:
            emit({"type": "done"})
        except SystemExit:
            pass          # 停止后前端已经断开，没人接这条了
        self.close_connection = True


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>small_agent</title>
<style>
:root{--fg:#f4f6ff;--dim:rgba(244,246,255,.62);--line:rgba(255,255,255,.16)}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;font:15px/1.65 -apple-system,"PingFang SC",system-ui,sans-serif;color:var(--fg);
background:radial-gradient(120% 80% at 15% 0%,#3b2d6e 0%,#1b1b3a 45%,#0d0f1f 100%);background-attachment:fixed}
.wrap{max-width:880px;margin:0 auto;padding:28px 20px 150px}
h1{font-size:16px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin:0 0 22px}
h1 b{color:#fff;letter-spacing:.06em}
.msg{display:flex;margin:0 0 16px;animation:in .42s cubic-bezier(.2,.9,.25,1.2) both}
@keyframes in{from{opacity:0;transform:translateY(14px) scale(.985)}to{opacity:1;transform:none}}
.msg.user{justify-content:flex-end}
.bubble{max-width:86%;padding:12px 16px;border-radius:18px;white-space:pre-wrap;word-break:break-word;
background:rgba(255,255,255,.10);border:1px solid var(--line);backdrop-filter:blur(14px);
box-shadow:0 8px 30px rgba(0,0,0,.22)}
.user .bubble{background:linear-gradient(135deg,#6d5cff,#a855f7 55%,#22d3ee);border-color:transparent}
.bubble p{margin:.4em 0}.bubble pre{background:rgba(0,0,0,.38);padding:10px 12px;border-radius:10px;overflow:auto;font-size:13px}
.bubble code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.tool{margin:0 0 10px;border-left:2px solid rgba(120,190,255,.55);padding:6px 0 6px 12px;
font:12px/1.5 ui-monospace,Menlo,monospace;color:var(--dim);animation:in .3s both}
/* 一轮一个折叠组：默认展开，点标题收起 */
.round{margin:0 0 14px;border-left:2px solid rgba(255,255,255,.16);padding-left:12px}
.round>summary{cursor:pointer;font:12px/1.9 ui-monospace,Menlo,monospace;color:var(--dim);
list-style:none;user-select:none;outline:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.round>summary::-webkit-details-marker{display:none}
.round>summary::before{content:'▸ ';color:#8fd0ff}
.round[open]>summary::before{content:'▾ '}
.round>summary:hover{color:var(--fg)}
.round-body{padding:4px 0 0}
.round-body>.tool{border-left-color:rgba(120,190,255,.35)}
.tool b{color:#8fd0ff;font-weight:600}
.tool details{margin-top:4px}.tool summary{cursor:pointer;outline:none}
.tool pre{margin:6px 0 0;max-height:220px;overflow:auto;background:rgba(0,0,0,.3);padding:8px;border-radius:8px;white-space:pre-wrap}
.err{color:#ffb4b4}
form{position:fixed;left:0;right:0;bottom:0;padding:16px 20px 22px;
background:linear-gradient(180deg,rgba(13,15,31,0),rgba(13,15,31,.92) 42%);backdrop-filter:blur(8px)}
.row{max-width:880px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
textarea{flex:1;resize:none;min-height:50px;max-height:180px;padding:14px 16px;border-radius:16px;
background:rgba(255,255,255,.10);border:1px solid var(--line);color:var(--fg);font:inherit;outline:none}
textarea:focus{border-color:rgba(255,255,255,.42)}
button{height:50px;padding:0 22px;border:none;border-radius:999px;cursor:pointer;font:600 15px/1 inherit;
background:#fff;color:#1b1b3a;transition:transform .16s,opacity .16s}
button:hover{transform:translateY(-1px)}button:disabled{opacity:.45;cursor:default;transform:none}
.hint{max-width:880px;margin:8px auto 0;font-size:12px;color:var(--dim);text-align:right}
header{max-width:880px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;gap:12px}
/* 停止按钮：回复中替换「发送」，红色语义但不刺眼 */
.stop{background:rgba(255,110,110,.9);color:#2a0d0d;animation:in .2s both}
.gear{height:36px;padding:0 16px;font-size:13px;background:rgba(255,255,255,.14);color:var(--fg);
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.head-actions{display:flex;gap:8px;align-items:center;min-width:0}
#wsBtn{max-width:240px}
.panel{position:fixed;top:0;right:0;bottom:0;width:min(540px,100%);padding:22px;overflow:auto;z-index:9;
background:rgba(18,18,38,.94);backdrop-filter:blur(24px);border-left:1px solid var(--line);
box-shadow:-20px 0 60px rgba(0,0,0,.35);animation:slide .34s cubic-bezier(.2,.9,.25,1.12) both}
@keyframes slide{from{opacity:0;transform:translateX(30px)}to{opacity:1;transform:none}}
.tabs{display:flex;gap:8px;margin-bottom:18px;align-items:center}
.tabs button{height:38px;padding:0 18px;font-size:13px;background:rgba(255,255,255,.10);color:var(--dim)}
.tabs button.on{background:#fff;color:#1b1b3a}
.tabs .x{margin-left:auto;width:38px;padding:0;font-size:15px;background:rgba(255,255,255,.10);color:var(--fg)}
label{display:block;margin:0 0 14px;font-size:12px;letter-spacing:.04em;color:var(--dim)}
label input,label textarea{display:block;width:100%;margin-top:6px;padding:11px 13px;border-radius:12px;
border:1px solid var(--line);background:rgba(0,0,0,.3);color:var(--fg);font:inherit;font-size:14px;
outline:none;resize:vertical;min-height:auto}
label textarea{min-height:52px}
#skBody{min-height:260px;font-family:ui-monospace,Menlo,monospace;font-size:13px;line-height:1.6}
label input:focus,label textarea:focus{border-color:rgba(255,255,255,.45)}
.actions{display:flex;gap:8px;align-items:center;margin-top:6px}
.actions button{height:38px;padding:0 18px;font-size:13px}
.ghost{background:rgba(255,255,255,.12);color:var(--fg)}
.danger{background:rgba(255,120,120,.18);color:#ffc9c9}
.note{font-size:12px;color:var(--dim)}
#skillList{margin-bottom:14px}
.skitem{display:flex;align-items:center;gap:10px;padding:11px 13px;margin-bottom:8px;border-radius:12px;
border:1px solid var(--line);background:rgba(255,255,255,.06);cursor:pointer;transition:background .15s}
.skitem:hover{background:rgba(255,255,255,.12)}
.skitem b{font-size:14px;display:block}
.skitem small{font-size:12px;color:var(--dim);display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.skitem span{margin-left:auto;font-size:12px;color:var(--dim)}
.hidden{display:none!important}
#cmds{max-width:880px;margin:0 auto 8px;border-radius:14px;overflow:hidden;border:1px solid var(--line);
background:rgba(18,18,38,.96);backdrop-filter:blur(20px);box-shadow:0 -8px 30px rgba(0,0,0,.3);
animation:in .16s both}
#cmds div{padding:9px 14px;font-size:13px;cursor:pointer;display:flex;gap:12px;align-items:baseline}
#cmds div.on{background:rgba(255,255,255,.14)}
#cmds b{font-family:ui-monospace,Menlo,monospace;color:#8fd0ff;min-width:76px}
#cmds span{color:var(--dim);font-size:12px}
</style></head><body>
<div class="wrap"><header><h1><b>small_agent</b> &nbsp;read · write · edit · bash</h1>
<div class="head-actions"><button class="gear hidden" id="foldBtn" title="展开/收起所有轮次">全部收起</button>
<button class="gear" id="wsBtn" title="切换工作区">📁 <span id="wsChip">…</span></button>
<button class="gear" id="cfgBtn">⚙ 设置</button></div></header><div id="log"></div></div>
<form id="f"><div id="cmds" class="hidden"></div><div class="row"><textarea id="i" rows="1" placeholder="想让我做什么？（Enter 发送 / Shift+Enter 换行，/ 看命令）"></textarea>
<button id="stop" class="stop hidden" title="停止这次回复">■ 停止</button>
<button id="b">发送</button></div><div class="hint" id="hint">workspace: 服务启动目录</div></form>
<div class="panel hidden" id="panel">
  <div class="tabs">
    <button class="on" data-tab="model">模型</button><button data-tab="skills">技能</button>
    <button data-tab="ws">工作区</button>
    <button class="x" id="closePanel">✕</button>
  </div>
  <section id="tab-model">
    <label>BASE_URL（OpenAI 兼容 /v1）<input id="cfgBase" placeholder="https://api.openai.com/v1"></label>
    <label>API KEY<input id="cfgKey" autocomplete="off" spellcheck="false"></label>
    <label>MODEL<input id="cfgModel" placeholder="gpt-4o-mini"></label>
    <label>循环次数（一轮 = 一次模型调用；用满就停，1-200）<input id="cfgRounds" type="number" min="1" max="200" placeholder="20"></label>
    <div class="actions"><button id="saveCfg">保存</button><span class="note" id="cfgMsg"></span></div>
  </section>
  <section id="tab-skills" class="hidden">
    <div id="skillList"></div>
    <div class="actions" style="margin-bottom:16px"><button class="ghost" id="newSkill">+ 新建技能</button>
      <span class="note">保存即生效，无需重启</span></div>
    <div id="skillEdit" class="hidden">
      <label>NAME（小写字母/数字/_/-）<input id="skName" spellcheck="false"></label>
      <label>DESCRIPTION（模型据此判断何时加载）<input id="skDesc"></label>
      <label>正文（Markdown 步骤）<textarea id="skBody"></textarea></label>
      <div class="actions"><button id="saveSkill">保存</button>
        <button class="danger" id="delSkill">删除</button><span class="note" id="skMsg"></span></div>
    </div>
  </section>
  <section id="tab-ws" class="hidden">
    <label>工作区路径（read/write/edit/bash 的相对根）<input id="wsPath" spellcheck="false"></label>
    <div class="actions" style="margin-bottom:14px"><button id="wsUse">切到这里</button>
      <button class="ghost" id="wsUp">↑ 上级</button><span class="note" id="wsMsg"></span></div>
    <div id="wsRecent"></div>
    <div id="wsDirs"></div>
  </section>
</div>
<script>
/* marked v12.0.2 — MIT, 内联以免 jsdelivr 不通时 Markdown 静默退化成纯文本 */
/**
 * marked v12.0.2 - a markdown parser
 * Copyright (c) 2011-2024, Christopher Jeffrey. (MIT Licensed)
 * https://github.com/markedjs/marked
 */
!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?t(exports):"function"==typeof define&&define.amd?define(["exports"],t):t((e="undefined"!=typeof globalThis?globalThis:e||self).marked={})}(this,(function(e){"use strict";function t(){return{async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null}}function n(t){e.defaults=t}e.defaults={async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null};const s=/[&<>"']/,r=new RegExp(s.source,"g"),i=/[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/,l=new RegExp(i.source,"g"),o={"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"},a=e=>o[e];function c(e,t){if(t){if(s.test(e))return e.replace(r,a)}else if(i.test(e))return e.replace(l,a);return e}const h=/&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/gi;function p(e){return e.replace(h,((e,t)=>"colon"===(t=t.toLowerCase())?":":"#"===t.charAt(0)?"x"===t.charAt(1)?String.fromCharCode(parseInt(t.substring(2),16)):String.fromCharCode(+t.substring(1)):""))}const u=/(^|[^\[])\^/g;function k(e,t){let n="string"==typeof e?e:e.source;t=t||"";const s={replace:(e,t)=>{let r="string"==typeof t?t:t.source;return r=r.replace(u,"$1"),n=n.replace(e,r),s},getRegex:()=>new RegExp(n,t)};return s}function g(e){try{e=encodeURI(e).replace(/%25/g,"%")}catch(e){return null}return e}const f={exec:()=>null};function d(e,t){const n=e.replace(/\|/g,((e,t,n)=>{let s=!1,r=t;for(;--r>=0&&"\\"===n[r];)s=!s;return s?"|":" |"})).split(/ \|/);let s=0;if(n[0].trim()||n.shift(),n.length>0&&!n[n.length-1].trim()&&n.pop(),t)if(n.length>t)n.splice(t);else for(;n.length<t;)n.push("");for(;s<n.length;s++)n[s]=n[s].trim().replace(/\\\|/g,"|");return n}function x(e,t,n){const s=e.length;if(0===s)return"";let r=0;for(;r<s;){const i=e.charAt(s-r-1);if(i!==t||n){if(i===t||!n)break;r++}else r++}return e.slice(0,s-r)}function b(e,t,n,s){const r=t.href,i=t.title?c(t.title):null,l=e[1].replace(/\\([\[\]])/g,"$1");if("!"!==e[0].charAt(0)){s.state.inLink=!0;const e={type:"link",raw:n,href:r,title:i,text:l,tokens:s.inlineTokens(l)};return s.state.inLink=!1,e}return{type:"image",raw:n,href:r,title:i,text:c(l)}}class w{options;rules;lexer;constructor(t){this.options=t||e.defaults}space(e){const t=this.rules.block.newline.exec(e);if(t&&t[0].length>0)return{type:"space",raw:t[0]}}code(e){const t=this.rules.block.code.exec(e);if(t){const e=t[0].replace(/^ {1,4}/gm,"");return{type:"code",raw:t[0],codeBlockStyle:"indented",text:this.options.pedantic?e:x(e,"\n")}}}fences(e){const t=this.rules.block.fences.exec(e);if(t){const e=t[0],n=function(e,t){const n=e.match(/^(\s+)(?:```)/);if(null===n)return t;const s=n[1];return t.split("\n").map((e=>{const t=e.match(/^\s+/);if(null===t)return e;const[n]=t;return n.length>=s.length?e.slice(s.length):e})).join("\n")}(e,t[3]||"");return{type:"code",raw:e,lang:t[2]?t[2].trim().replace(this.rules.inline.anyPunctuation,"$1"):t[2],text:n}}}heading(e){const t=this.rules.block.heading.exec(e);if(t){let e=t[2].trim();if(/#$/.test(e)){const t=x(e,"#");this.options.pedantic?e=t.trim():t&&!/ $/.test(t)||(e=t.trim())}return{type:"heading",raw:t[0],depth:t[1].length,text:e,tokens:this.lexer.inline(e)}}}hr(e){const t=this.rules.block.hr.exec(e);if(t)return{type:"hr",raw:t[0]}}blockquote(e){const t=this.rules.block.blockquote.exec(e);if(t){let e=t[0].replace(/\n {0,3}((?:=+|-+) *)(?=\n|$)/g,"\n    $1");e=x(e.replace(/^ *>[ \t]?/gm,""),"\n");const n=this.lexer.state.top;this.lexer.state.top=!0;const s=this.lexer.blockTokens(e);return this.lexer.state.top=n,{type:"blockquote",raw:t[0],tokens:s,text:e}}}list(e){let t=this.rules.block.list.exec(e);if(t){let n=t[1].trim();const s=n.length>1,r={type:"list",raw:"",ordered:s,start:s?+n.slice(0,-1):"",loose:!1,items:[]};n=s?`\\d{1,9}\\${n.slice(-1)}`:`\\${n}`,this.options.pedantic&&(n=s?n:"[*+-]");const i=new RegExp(`^( {0,3}${n})((?:[\t ][^\\n]*)?(?:\\n|$))`);let l="",o="",a=!1;for(;e;){let n=!1;if(!(t=i.exec(e)))break;if(this.rules.block.hr.test(e))break;l=t[0],e=e.substring(l.length);let s=t[2].split("\n",1)[0].replace(/^\t+/,(e=>" ".repeat(3*e.length))),c=e.split("\n",1)[0],h=0;this.options.pedantic?(h=2,o=s.trimStart()):(h=t[2].search(/[^ ]/),h=h>4?1:h,o=s.slice(h),h+=t[1].length);let p=!1;if(!s&&/^ *$/.test(c)&&(l+=c+"\n",e=e.substring(c.length+1),n=!0),!n){const t=new RegExp(`^ {0,${Math.min(3,h-1)}}(?:[*+-]|\\d{1,9}[.)])((?:[ \t][^\\n]*)?(?:\\n|$))`),n=new RegExp(`^ {0,${Math.min(3,h-1)}}((?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$)`),r=new RegExp(`^ {0,${Math.min(3,h-1)}}(?:\`\`\`|~~~)`),i=new RegExp(`^ {0,${Math.min(3,h-1)}}#`);for(;e;){const a=e.split("\n",1)[0];if(c=a,this.options.pedantic&&(c=c.replace(/^ {1,4}(?=( {4})*[^ ])/g,"  ")),r.test(c))break;if(i.test(c))break;if(t.test(c))break;if(n.test(e))break;if(c.search(/[^ ]/)>=h||!c.trim())o+="\n"+c.slice(h);else{if(p)break;if(s.search(/[^ ]/)>=4)break;if(r.test(s))break;if(i.test(s))break;if(n.test(s))break;o+="\n"+c}p||c.trim()||(p=!0),l+=a+"\n",e=e.substring(a.length+1),s=c.slice(h)}}r.loose||(a?r.loose=!0:/\n *\n *$/.test(l)&&(a=!0));let u,k=null;this.options.gfm&&(k=/^\[[ xX]\] /.exec(o),k&&(u="[ ] "!==k[0],o=o.replace(/^\[[ xX]\] +/,""))),r.items.push({type:"list_item",raw:l,task:!!k,checked:u,loose:!1,text:o,tokens:[]}),r.raw+=l}r.items[r.items.length-1].raw=l.trimEnd(),r.items[r.items.length-1].text=o.trimEnd(),r.raw=r.raw.trimEnd();for(let e=0;e<r.items.length;e++)if(this.lexer.state.top=!1,r.items[e].tokens=this.lexer.blockTokens(r.items[e].text,[]),!r.loose){const t=r.items[e].tokens.filter((e=>"space"===e.type)),n=t.length>0&&t.some((e=>/\n.*\n/.test(e.raw)));r.loose=n}if(r.loose)for(let e=0;e<r.items.length;e++)r.items[e].loose=!0;return r}}html(e){const t=this.rules.block.html.exec(e);if(t){return{type:"html",block:!0,raw:t[0],pre:"pre"===t[1]||"script"===t[1]||"style"===t[1],text:t[0]}}}def(e){const t=this.rules.block.def.exec(e);if(t){const e=t[1].toLowerCase().replace(/\s+/g," "),n=t[2]?t[2].replace(/^<(.*)>$/,"$1").replace(this.rules.inline.anyPunctuation,"$1"):"",s=t[3]?t[3].substring(1,t[3].length-1).replace(this.rules.inline.anyPunctuation,"$1"):t[3];return{type:"def",tag:e,raw:t[0],href:n,title:s}}}table(e){const t=this.rules.block.table.exec(e);if(!t)return;if(!/[:|]/.test(t[2]))return;const n=d(t[1]),s=t[2].replace(/^\||\| *$/g,"").split("|"),r=t[3]&&t[3].trim()?t[3].replace(/\n[ \t]*$/,"").split("\n"):[],i={type:"table",raw:t[0],header:[],align:[],rows:[]};if(n.length===s.length){for(const e of s)/^ *-+: *$/.test(e)?i.align.push("right"):/^ *:-+: *$/.test(e)?i.align.push("center"):/^ *:-+ *$/.test(e)?i.align.push("left"):i.align.push(null);for(const e of n)i.header.push({text:e,tokens:this.lexer.inline(e)});for(const e of r)i.rows.push(d(e,i.header.length).map((e=>({text:e,tokens:this.lexer.inline(e)}))));return i}}lheading(e){const t=this.rules.block.lheading.exec(e);if(t)return{type:"heading",raw:t[0],depth:"="===t[2].charAt(0)?1:2,text:t[1],tokens:this.lexer.inline(t[1])}}paragraph(e){const t=this.rules.block.paragraph.exec(e);if(t){const e="\n"===t[1].charAt(t[1].length-1)?t[1].slice(0,-1):t[1];return{type:"paragraph",raw:t[0],text:e,tokens:this.lexer.inline(e)}}}text(e){const t=this.rules.block.text.exec(e);if(t)return{type:"text",raw:t[0],text:t[0],tokens:this.lexer.inline(t[0])}}escape(e){const t=this.rules.inline.escape.exec(e);if(t)return{type:"escape",raw:t[0],text:c(t[1])}}tag(e){const t=this.rules.inline.tag.exec(e);if(t)return!this.lexer.state.inLink&&/^<a /i.test(t[0])?this.lexer.state.inLink=!0:this.lexer.state.inLink&&/^<\/a>/i.test(t[0])&&(this.lexer.state.inLink=!1),!this.lexer.state.inRawBlock&&/^<(pre|code|kbd|script)(\s|>)/i.test(t[0])?this.lexer.state.inRawBlock=!0:this.lexer.state.inRawBlock&&/^<\/(pre|code|kbd|script)(\s|>)/i.test(t[0])&&(this.lexer.state.inRawBlock=!1),{type:"html",raw:t[0],inLink:this.lexer.state.inLink,inRawBlock:this.lexer.state.inRawBlock,block:!1,text:t[0]}}link(e){const t=this.rules.inline.link.exec(e);if(t){const e=t[2].trim();if(!this.options.pedantic&&/^</.test(e)){if(!/>$/.test(e))return;const t=x(e.slice(0,-1),"\\");if((e.length-t.length)%2==0)return}else{const e=function(e,t){if(-1===e.indexOf(t[1]))return-1;let n=0;for(let s=0;s<e.length;s++)if("\\"===e[s])s++;else if(e[s]===t[0])n++;else if(e[s]===t[1]&&(n--,n<0))return s;return-1}(t[2],"()");if(e>-1){const n=(0===t[0].indexOf("!")?5:4)+t[1].length+e;t[2]=t[2].substring(0,e),t[0]=t[0].substring(0,n).trim(),t[3]=""}}let n=t[2],s="";if(this.options.pedantic){const e=/^([^'"]*[^\s])\s+(['"])(.*)\2/.exec(n);e&&(n=e[1],s=e[3])}else s=t[3]?t[3].slice(1,-1):"";return n=n.trim(),/^</.test(n)&&(n=this.options.pedantic&&!/>$/.test(e)?n.slice(1):n.slice(1,-1)),b(t,{href:n?n.replace(this.rules.inline.anyPunctuation,"$1"):n,title:s?s.replace(this.rules.inline.anyPunctuation,"$1"):s},t[0],this.lexer)}}reflink(e,t){let n;if((n=this.rules.inline.reflink.exec(e))||(n=this.rules.inline.nolink.exec(e))){const e=t[(n[2]||n[1]).replace(/\s+/g," ").toLowerCase()];if(!e){const e=n[0].charAt(0);return{type:"text",raw:e,text:e}}return b(n,e,n[0],this.lexer)}}emStrong(e,t,n=""){let s=this.rules.inline.emStrongLDelim.exec(e);if(!s)return;if(s[3]&&n.match(/[\p{L}\p{N}]/u))return;if(!(s[1]||s[2]||"")||!n||this.rules.inline.punctuation.exec(n)){const n=[...s[0]].length-1;let r,i,l=n,o=0;const a="*"===s[0][0]?this.rules.inline.emStrongRDelimAst:this.rules.inline.emStrongRDelimUnd;for(a.lastIndex=0,t=t.slice(-1*e.length+n);null!=(s=a.exec(t));){if(r=s[1]||s[2]||s[3]||s[4]||s[5]||s[6],!r)continue;if(i=[...r].length,s[3]||s[4]){l+=i;continue}if((s[5]||s[6])&&n%3&&!((n+i)%3)){o+=i;continue}if(l-=i,l>0)continue;i=Math.min(i,i+l+o);const t=[...s[0]][0].length,a=e.slice(0,n+s.index+t+i);if(Math.min(n,i)%2){const e=a.slice(1,-1);return{type:"em",raw:a,text:e,tokens:this.lexer.inlineTokens(e)}}const c=a.slice(2,-2);return{type:"strong",raw:a,text:c,tokens:this.lexer.inlineTokens(c)}}}}codespan(e){const t=this.rules.inline.code.exec(e);if(t){let e=t[2].replace(/\n/g," ");const n=/[^ ]/.test(e),s=/^ /.test(e)&&/ $/.test(e);return n&&s&&(e=e.substring(1,e.length-1)),e=c(e,!0),{type:"codespan",raw:t[0],text:e}}}br(e){const t=this.rules.inline.br.exec(e);if(t)return{type:"br",raw:t[0]}}del(e){const t=this.rules.inline.del.exec(e);if(t)return{type:"del",raw:t[0],text:t[2],tokens:this.lexer.inlineTokens(t[2])}}autolink(e){const t=this.rules.inline.autolink.exec(e);if(t){let e,n;return"@"===t[2]?(e=c(t[1]),n="mailto:"+e):(e=c(t[1]),n=e),{type:"link",raw:t[0],text:e,href:n,tokens:[{type:"text",raw:e,text:e}]}}}url(e){let t;if(t=this.rules.inline.url.exec(e)){let e,n;if("@"===t[2])e=c(t[0]),n="mailto:"+e;else{let s;do{s=t[0],t[0]=this.rules.inline._backpedal.exec(t[0])?.[0]??""}while(s!==t[0]);e=c(t[0]),n="www."===t[1]?"http://"+t[0]:t[0]}return{type:"link",raw:t[0],text:e,href:n,tokens:[{type:"text",raw:e,text:e}]}}}inlineText(e){const t=this.rules.inline.text.exec(e);if(t){let e;return e=this.lexer.state.inRawBlock?t[0]:c(t[0]),{type:"text",raw:t[0],text:e}}}}const m=/^ {0,3}((?:-[\t ]*){3,}|(?:_[ \t]*){3,}|(?:\*[ \t]*){3,})(?:\n+|$)/,y=/(?:[*+-]|\d{1,9}[.)])/,$=k(/^(?!bull |blockCode|fences|blockquote|heading|html)((?:.|\n(?!\s*?\n|bull |blockCode|fences|blockquote|heading|html))+?)\n {0,3}(=+|-+) *(?:\n+|$)/).replace(/bull/g,y).replace(/blockCode/g,/ {4}/).replace(/fences/g,/ {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g,/ {0,3}>/).replace(/heading/g,/ {0,3}#{1,6}/).replace(/html/g,/ {0,3}<[^\n>]+>\n/).getRegex(),z=/^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/,T=/(?!\s*\])(?:\\.|[^\[\]\\])+/,R=k(/^ {0,3}\[(label)\]: *(?:\n *)?([^<\s][^\s]*|<.*?>)(?:(?: +(?:\n *)?| *\n *)(title))? *(?:\n+|$)/).replace("label",T).replace("title",/(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/).getRegex(),_=k(/^( {0,3}bull)([ \t][^\n]+?)?(?:\n|$)/).replace(/bull/g,y).getRegex(),A="address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul",S=/<!--(?:-?>|[\s\S]*?(?:-->|$))/,I=k("^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n *)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n *)+\\n|$))","i").replace("comment",S).replace("tag",A).replace("attribute",/ +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/).getRegex(),E=k(z).replace("hr",m).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("|lheading","").replace("|table","").replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",A).getRegex(),q={blockquote:k(/^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/).replace("paragraph",E).getRegex(),code:/^( {4}[^\n]+(?:\n(?: *(?:\n|$))*)?)+/,def:R,fences:/^ {0,3}(`{3,}(?=[^`\n]*(?:\n|$))|~{3,})([^\n]*)(?:\n|$)(?:|([\s\S]*?)(?:\n|$))(?: {0,3}\1[~`]* *(?=\n|$)|$)/,heading:/^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/,hr:m,html:I,lheading:$,list:_,newline:/^(?: *(?:\n|$))+/,paragraph:E,table:f,text:/^[^\n]+/},Z=k("^ *([^\\n ].*)\\n {0,3}((?:\\| *)?:?-+:? *(?:\\| *:?-+:? *)*(?:\\| *)?)(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)").replace("hr",m).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("blockquote"," {0,3}>").replace("code"," {4}[^\\n]").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",A).getRegex(),L={...q,table:Z,paragraph:k(z).replace("hr",m).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("|lheading","").replace("table",Z).replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",A).getRegex()},P={...q,html:k("^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:\"[^\"]*\"|'[^']*'|\\s[^'\"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))").replace("comment",S).replace(/tag/g,"(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b").getRegex(),def:/^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/,heading:/^(#{1,6})(.*)(?:\n+|$)/,fences:f,lheading:/^(.+?)\n {0,3}(=+|-+) *(?:\n+|$)/,paragraph:k(z).replace("hr",m).replace("heading"," *#{1,6} *[^\n]").replace("lheading",$).replace("|table","").replace("blockquote"," {0,3}>").replace("|fences","").replace("|list","").replace("|html","").replace("|tag","").getRegex()},Q=/^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/,v=/^( {2,}|\\)\n(?!\s*$)/,B="\\p{P}\\p{S}",C=k(/^((?![*_])[\spunctuation])/,"u").replace(/punctuation/g,B).getRegex(),M=k(/^(?:\*+(?:((?!\*)[punct])|[^\s*]))|^_+(?:((?!_)[punct])|([^\s_]))/,"u").replace(/punct/g,B).getRegex(),O=k("^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)[punct](\\*+)(?=[\\s]|$)|[^punct\\s](\\*+)(?!\\*)(?=[punct\\s]|$)|(?!\\*)[punct\\s](\\*+)(?=[^punct\\s])|[\\s](\\*+)(?!\\*)(?=[punct])|(?!\\*)[punct](\\*+)(?!\\*)(?=[punct])|[^punct\\s](\\*+)(?=[^punct\\s])","gu").replace(/punct/g,B).getRegex(),D=k("^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)[punct](_+)(?=[\\s]|$)|[^punct\\s](_+)(?!_)(?=[punct\\s]|$)|(?!_)[punct\\s](_+)(?=[^punct\\s])|[\\s](_+)(?!_)(?=[punct])|(?!_)[punct](_+)(?!_)(?=[punct])","gu").replace(/punct/g,B).getRegex(),j=k(/\\([punct])/,"gu").replace(/punct/g,B).getRegex(),H=k(/^<(scheme:[^\s\x00-\x1f<>]*|email)>/).replace("scheme",/[a-zA-Z][a-zA-Z0-9+.-]{1,31}/).replace("email",/[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/).getRegex(),U=k(S).replace("(?:--\x3e|$)","--\x3e").getRegex(),X=k("^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>").replace("comment",U).replace("attribute",/\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/).getRegex(),F=/(?:\[(?:\\.|[^\[\]\\])*\]|\\.|`[^`]*`|[^\[\]\\`])*?/,N=k(/^!?\[(label)\]\(\s*(href)(?:\s+(title))?\s*\)/).replace("label",F).replace("href",/<(?:\\.|[^\n<>\\])+>|[^\s\x00-\x1f]*/).replace("title",/"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/).getRegex(),G=k(/^!?\[(label)\]\[(ref)\]/).replace("label",F).replace("ref",T).getRegex(),J=k(/^!?\[(ref)\](?:\[\])?/).replace("ref",T).getRegex(),K={_backpedal:f,anyPunctuation:j,autolink:H,blockSkip:/\[[^[\]]*?\]\([^\(\)]*?\)|`[^`]*?`|<[^<>]*?>/g,br:v,code:/^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/,del:f,emStrongLDelim:M,emStrongRDelimAst:O,emStrongRDelimUnd:D,escape:Q,link:N,nolink:J,punctuation:C,reflink:G,reflinkSearch:k("reflink|nolink(?!\\()","g").replace("reflink",G).replace("nolink",J).getRegex(),tag:X,text:/^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/,url:f},V={...K,link:k(/^!?\[(label)\]\((.*?)\)/).replace("label",F).getRegex(),reflink:k(/^!?\[(label)\]\s*\[([^\]]*)\]/).replace("label",F).getRegex()},W={...K,escape:k(Q).replace("])","~|])").getRegex(),url:k(/^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/,"i").replace("email",/[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/).getRegex(),_backpedal:/(?:[^?!.,:;*_'"~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_'"~)]+(?!$))+/,del:/^(~~?)(?=[^\s~])([\s\S]*?[^\s~])\1(?=[^~]|$)/,text:/^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|https?:\/\/|ftp:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/},Y={...W,br:k(v).replace("{2,}","*").getRegex(),text:k(W.text).replace("\\b_","\\b_| {2,}\\n").replace(/\{2,\}/g,"*").getRegex()},ee={normal:q,gfm:L,pedantic:P},te={normal:K,gfm:W,breaks:Y,pedantic:V};class ne{tokens;options;state;tokenizer;inlineQueue;constructor(t){this.tokens=[],this.tokens.links=Object.create(null),this.options=t||e.defaults,this.options.tokenizer=this.options.tokenizer||new w,this.tokenizer=this.options.tokenizer,this.tokenizer.options=this.options,this.tokenizer.lexer=this,this.inlineQueue=[],this.state={inLink:!1,inRawBlock:!1,top:!0};const n={block:ee.normal,inline:te.normal};this.options.pedantic?(n.block=ee.pedantic,n.inline=te.pedantic):this.options.gfm&&(n.block=ee.gfm,this.options.breaks?n.inline=te.breaks:n.inline=te.gfm),this.tokenizer.rules=n}static get rules(){return{block:ee,inline:te}}static lex(e,t){return new ne(t).lex(e)}static lexInline(e,t){return new ne(t).inlineTokens(e)}lex(e){e=e.replace(/\r\n|\r/g,"\n"),this.blockTokens(e,this.tokens);for(let e=0;e<this.inlineQueue.length;e++){const t=this.inlineQueue[e];this.inlineTokens(t.src,t.tokens)}return this.inlineQueue=[],this.tokens}blockTokens(e,t=[]){let n,s,r,i;for(e=this.options.pedantic?e.replace(/\t/g,"    ").replace(/^ +$/gm,""):e.replace(/^( *)(\t+)/gm,((e,t,n)=>t+"    ".repeat(n.length)));e;)if(!(this.options.extensions&&this.options.extensions.block&&this.options.extensions.block.some((s=>!!(n=s.call({lexer:this},e,t))&&(e=e.substring(n.raw.length),t.push(n),!0)))))if(n=this.tokenizer.space(e))e=e.substring(n.raw.length),1===n.raw.length&&t.length>0?t[t.length-1].raw+="\n":t.push(n);else if(n=this.tokenizer.code(e))e=e.substring(n.raw.length),s=t[t.length-1],!s||"paragraph"!==s.type&&"text"!==s.type?t.push(n):(s.raw+="\n"+n.raw,s.text+="\n"+n.text,this.inlineQueue[this.inlineQueue.length-1].src=s.text);else if(n=this.tokenizer.fences(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.heading(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.hr(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.blockquote(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.list(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.html(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.def(e))e=e.substring(n.raw.length),s=t[t.length-1],!s||"paragraph"!==s.type&&"text"!==s.type?this.tokens.links[n.tag]||(this.tokens.links[n.tag]={href:n.href,title:n.title}):(s.raw+="\n"+n.raw,s.text+="\n"+n.raw,this.inlineQueue[this.inlineQueue.length-1].src=s.text);else if(n=this.tokenizer.table(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.lheading(e))e=e.substring(n.raw.length),t.push(n);else{if(r=e,this.options.extensions&&this.options.extensions.startBlock){let t=1/0;const n=e.slice(1);let s;this.options.extensions.startBlock.forEach((e=>{s=e.call({lexer:this},n),"number"==typeof s&&s>=0&&(t=Math.min(t,s))})),t<1/0&&t>=0&&(r=e.substring(0,t+1))}if(this.state.top&&(n=this.tokenizer.paragraph(r)))s=t[t.length-1],i&&"paragraph"===s.type?(s.raw+="\n"+n.raw,s.text+="\n"+n.text,this.inlineQueue.pop(),this.inlineQueue[this.inlineQueue.length-1].src=s.text):t.push(n),i=r.length!==e.length,e=e.substring(n.raw.length);else if(n=this.tokenizer.text(e))e=e.substring(n.raw.length),s=t[t.length-1],s&&"text"===s.type?(s.raw+="\n"+n.raw,s.text+="\n"+n.text,this.inlineQueue.pop(),this.inlineQueue[this.inlineQueue.length-1].src=s.text):t.push(n);else if(e){const t="Infinite loop on byte: "+e.charCodeAt(0);if(this.options.silent){console.error(t);break}throw new Error(t)}}return this.state.top=!0,t}inline(e,t=[]){return this.inlineQueue.push({src:e,tokens:t}),t}inlineTokens(e,t=[]){let n,s,r,i,l,o,a=e;if(this.tokens.links){const e=Object.keys(this.tokens.links);if(e.length>0)for(;null!=(i=this.tokenizer.rules.inline.reflinkSearch.exec(a));)e.includes(i[0].slice(i[0].lastIndexOf("[")+1,-1))&&(a=a.slice(0,i.index)+"["+"a".repeat(i[0].length-2)+"]"+a.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex))}for(;null!=(i=this.tokenizer.rules.inline.blockSkip.exec(a));)a=a.slice(0,i.index)+"["+"a".repeat(i[0].length-2)+"]"+a.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);for(;null!=(i=this.tokenizer.rules.inline.anyPunctuation.exec(a));)a=a.slice(0,i.index)+"++"+a.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);for(;e;)if(l||(o=""),l=!1,!(this.options.extensions&&this.options.extensions.inline&&this.options.extensions.inline.some((s=>!!(n=s.call({lexer:this},e,t))&&(e=e.substring(n.raw.length),t.push(n),!0)))))if(n=this.tokenizer.escape(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.tag(e))e=e.substring(n.raw.length),s=t[t.length-1],s&&"text"===n.type&&"text"===s.type?(s.raw+=n.raw,s.text+=n.text):t.push(n);else if(n=this.tokenizer.link(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.reflink(e,this.tokens.links))e=e.substring(n.raw.length),s=t[t.length-1],s&&"text"===n.type&&"text"===s.type?(s.raw+=n.raw,s.text+=n.text):t.push(n);else if(n=this.tokenizer.emStrong(e,a,o))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.codespan(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.br(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.del(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.autolink(e))e=e.substring(n.raw.length),t.push(n);else if(this.state.inLink||!(n=this.tokenizer.url(e))){if(r=e,this.options.extensions&&this.options.extensions.startInline){let t=1/0;const n=e.slice(1);let s;this.options.extensions.startInline.forEach((e=>{s=e.call({lexer:this},n),"number"==typeof s&&s>=0&&(t=Math.min(t,s))})),t<1/0&&t>=0&&(r=e.substring(0,t+1))}if(n=this.tokenizer.inlineText(r))e=e.substring(n.raw.length),"_"!==n.raw.slice(-1)&&(o=n.raw.slice(-1)),l=!0,s=t[t.length-1],s&&"text"===s.type?(s.raw+=n.raw,s.text+=n.text):t.push(n);else if(e){const t="Infinite loop on byte: "+e.charCodeAt(0);if(this.options.silent){console.error(t);break}throw new Error(t)}}else e=e.substring(n.raw.length),t.push(n);return t}}class se{options;constructor(t){this.options=t||e.defaults}code(e,t,n){const s=(t||"").match(/^\S*/)?.[0];return e=e.replace(/\n$/,"")+"\n",s?'<pre><code class="language-'+c(s)+'">'+(n?e:c(e,!0))+"</code></pre>\n":"<pre><code>"+(n?e:c(e,!0))+"</code></pre>\n"}blockquote(e){return`<blockquote>\n${e}</blockquote>\n`}html(e,t){return e}heading(e,t,n){return`<h${t}>${e}</h${t}>\n`}hr(){return"<hr>\n"}list(e,t,n){const s=t?"ol":"ul";return"<"+s+(t&&1!==n?' start="'+n+'"':"")+">\n"+e+"</"+s+">\n"}listitem(e,t,n){return`<li>${e}</li>\n`}checkbox(e){return"<input "+(e?'checked="" ':"")+'disabled="" type="checkbox">'}paragraph(e){return`<p>${e}</p>\n`}table(e,t){return t&&(t=`<tbody>${t}</tbody>`),"<table>\n<thead>\n"+e+"</thead>\n"+t+"</table>\n"}tablerow(e){return`<tr>\n${e}</tr>\n`}tablecell(e,t){const n=t.header?"th":"td";return(t.align?`<${n} align="${t.align}">`:`<${n}>`)+e+`</${n}>\n`}strong(e){return`<strong>${e}</strong>`}em(e){return`<em>${e}</em>`}codespan(e){return`<code>${e}</code>`}br(){return"<br>"}del(e){return`<del>${e}</del>`}link(e,t,n){const s=g(e);if(null===s)return n;let r='<a href="'+(e=s)+'"';return t&&(r+=' title="'+t+'"'),r+=">"+n+"</a>",r}image(e,t,n){const s=g(e);if(null===s)return n;let r=`<img src="${e=s}" alt="${n}"`;return t&&(r+=` title="${t}"`),r+=">",r}text(e){return e}}class re{strong(e){return e}em(e){return e}codespan(e){return e}del(e){return e}html(e){return e}text(e){return e}link(e,t,n){return""+n}image(e,t,n){return""+n}br(){return""}}class ie{options;renderer;textRenderer;constructor(t){this.options=t||e.defaults,this.options.renderer=this.options.renderer||new se,this.renderer=this.options.renderer,this.renderer.options=this.options,this.textRenderer=new re}static parse(e,t){return new ie(t).parse(e)}static parseInline(e,t){return new ie(t).parseInline(e)}parse(e,t=!0){let n="";for(let s=0;s<e.length;s++){const r=e[s];if(this.options.extensions&&this.options.extensions.renderers&&this.options.extensions.renderers[r.type]){const e=r,t=this.options.extensions.renderers[e.type].call({parser:this},e);if(!1!==t||!["space","hr","heading","code","table","blockquote","list","html","paragraph","text"].includes(e.type)){n+=t||"";continue}}switch(r.type){case"space":continue;case"hr":n+=this.renderer.hr();continue;case"heading":{const e=r;n+=this.renderer.heading(this.parseInline(e.tokens),e.depth,p(this.parseInline(e.tokens,this.textRenderer)));continue}case"code":{const e=r;n+=this.renderer.code(e.text,e.lang,!!e.escaped);continue}case"table":{const e=r;let t="",s="";for(let t=0;t<e.header.length;t++)s+=this.renderer.tablecell(this.parseInline(e.header[t].tokens),{header:!0,align:e.align[t]});t+=this.renderer.tablerow(s);let i="";for(let t=0;t<e.rows.length;t++){const n=e.rows[t];s="";for(let t=0;t<n.length;t++)s+=this.renderer.tablecell(this.parseInline(n[t].tokens),{header:!1,align:e.align[t]});i+=this.renderer.tablerow(s)}n+=this.renderer.table(t,i);continue}case"blockquote":{const e=r,t=this.parse(e.tokens);n+=this.renderer.blockquote(t);continue}case"list":{const e=r,t=e.ordered,s=e.start,i=e.loose;let l="";for(let t=0;t<e.items.length;t++){const n=e.items[t],s=n.checked,r=n.task;let o="";if(n.task){const e=this.renderer.checkbox(!!s);i?n.tokens.length>0&&"paragraph"===n.tokens[0].type?(n.tokens[0].text=e+" "+n.tokens[0].text,n.tokens[0].tokens&&n.tokens[0].tokens.length>0&&"text"===n.tokens[0].tokens[0].type&&(n.tokens[0].tokens[0].text=e+" "+n.tokens[0].tokens[0].text)):n.tokens.unshift({type:"text",text:e+" "}):o+=e+" "}o+=this.parse(n.tokens,i),l+=this.renderer.listitem(o,r,!!s)}n+=this.renderer.list(l,t,s);continue}case"html":{const e=r;n+=this.renderer.html(e.text,e.block);continue}case"paragraph":{const e=r;n+=this.renderer.paragraph(this.parseInline(e.tokens));continue}case"text":{let i=r,l=i.tokens?this.parseInline(i.tokens):i.text;for(;s+1<e.length&&"text"===e[s+1].type;)i=e[++s],l+="\n"+(i.tokens?this.parseInline(i.tokens):i.text);n+=t?this.renderer.paragraph(l):l;continue}default:{const e='Token with "'+r.type+'" type was not found.';if(this.options.silent)return console.error(e),"";throw new Error(e)}}}return n}parseInline(e,t){t=t||this.renderer;let n="";for(let s=0;s<e.length;s++){const r=e[s];if(this.options.extensions&&this.options.extensions.renderers&&this.options.extensions.renderers[r.type]){const e=this.options.extensions.renderers[r.type].call({parser:this},r);if(!1!==e||!["escape","html","link","image","strong","em","codespan","br","del","text"].includes(r.type)){n+=e||"";continue}}switch(r.type){case"escape":{const e=r;n+=t.text(e.text);break}case"html":{const e=r;n+=t.html(e.text);break}case"link":{const e=r;n+=t.link(e.href,e.title,this.parseInline(e.tokens,t));break}case"image":{const e=r;n+=t.image(e.href,e.title,e.text);break}case"strong":{const e=r;n+=t.strong(this.parseInline(e.tokens,t));break}case"em":{const e=r;n+=t.em(this.parseInline(e.tokens,t));break}case"codespan":{const e=r;n+=t.codespan(e.text);break}case"br":n+=t.br();break;case"del":{const e=r;n+=t.del(this.parseInline(e.tokens,t));break}case"text":{const e=r;n+=t.text(e.text);break}default:{const e='Token with "'+r.type+'" type was not found.';if(this.options.silent)return console.error(e),"";throw new Error(e)}}}return n}}class le{options;constructor(t){this.options=t||e.defaults}static passThroughHooks=new Set(["preprocess","postprocess","processAllTokens"]);preprocess(e){return e}postprocess(e){return e}processAllTokens(e){return e}}class oe{defaults={async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null};options=this.setOptions;parse=this.#e(ne.lex,ie.parse);parseInline=this.#e(ne.lexInline,ie.parseInline);Parser=ie;Renderer=se;TextRenderer=re;Lexer=ne;Tokenizer=w;Hooks=le;constructor(...e){this.use(...e)}walkTokens(e,t){let n=[];for(const s of e)switch(n=n.concat(t.call(this,s)),s.type){case"table":{const e=s;for(const s of e.header)n=n.concat(this.walkTokens(s.tokens,t));for(const s of e.rows)for(const e of s)n=n.concat(this.walkTokens(e.tokens,t));break}case"list":{const e=s;n=n.concat(this.walkTokens(e.items,t));break}default:{const e=s;this.defaults.extensions?.childTokens?.[e.type]?this.defaults.extensions.childTokens[e.type].forEach((s=>{const r=e[s].flat(1/0);n=n.concat(this.walkTokens(r,t))})):e.tokens&&(n=n.concat(this.walkTokens(e.tokens,t)))}}return n}use(...e){const t=this.defaults.extensions||{renderers:{},childTokens:{}};return e.forEach((e=>{const n={...e};if(n.async=this.defaults.async||n.async||!1,e.extensions&&(e.extensions.forEach((e=>{if(!e.name)throw new Error("extension name required");if("renderer"in e){const n=t.renderers[e.name];t.renderers[e.name]=n?function(...t){let s=e.renderer.apply(this,t);return!1===s&&(s=n.apply(this,t)),s}:e.renderer}if("tokenizer"in e){if(!e.level||"block"!==e.level&&"inline"!==e.level)throw new Error("extension level must be 'block' or 'inline'");const n=t[e.level];n?n.unshift(e.tokenizer):t[e.level]=[e.tokenizer],e.start&&("block"===e.level?t.startBlock?t.startBlock.push(e.start):t.startBlock=[e.start]:"inline"===e.level&&(t.startInline?t.startInline.push(e.start):t.startInline=[e.start]))}"childTokens"in e&&e.childTokens&&(t.childTokens[e.name]=e.childTokens)})),n.extensions=t),e.renderer){const t=this.defaults.renderer||new se(this.defaults);for(const n in e.renderer){if(!(n in t))throw new Error(`renderer '${n}' does not exist`);if("options"===n)continue;const s=n,r=e.renderer[s],i=t[s];t[s]=(...e)=>{let n=r.apply(t,e);return!1===n&&(n=i.apply(t,e)),n||""}}n.renderer=t}if(e.tokenizer){const t=this.defaults.tokenizer||new w(this.defaults);for(const n in e.tokenizer){if(!(n in t))throw new Error(`tokenizer '${n}' does not exist`);if(["options","rules","lexer"].includes(n))continue;const s=n,r=e.tokenizer[s],i=t[s];t[s]=(...e)=>{let n=r.apply(t,e);return!1===n&&(n=i.apply(t,e)),n}}n.tokenizer=t}if(e.hooks){const t=this.defaults.hooks||new le;for(const n in e.hooks){if(!(n in t))throw new Error(`hook '${n}' does not exist`);if("options"===n)continue;const s=n,r=e.hooks[s],i=t[s];le.passThroughHooks.has(n)?t[s]=e=>{if(this.defaults.async)return Promise.resolve(r.call(t,e)).then((e=>i.call(t,e)));const n=r.call(t,e);return i.call(t,n)}:t[s]=(...e)=>{let n=r.apply(t,e);return!1===n&&(n=i.apply(t,e)),n}}n.hooks=t}if(e.walkTokens){const t=this.defaults.walkTokens,s=e.walkTokens;n.walkTokens=function(e){let n=[];return n.push(s.call(this,e)),t&&(n=n.concat(t.call(this,e))),n}}this.defaults={...this.defaults,...n}})),this}setOptions(e){return this.defaults={...this.defaults,...e},this}lexer(e,t){return ne.lex(e,t??this.defaults)}parser(e,t){return ie.parse(e,t??this.defaults)}#e(e,t){return(n,s)=>{const r={...s},i={...this.defaults,...r};!0===this.defaults.async&&!1===r.async&&(i.silent||console.warn("marked(): The async option was set to true by an extension. The async: false option sent to parse will be ignored."),i.async=!0);const l=this.#t(!!i.silent,!!i.async);if(null==n)return l(new Error("marked(): input parameter is undefined or null"));if("string"!=typeof n)return l(new Error("marked(): input parameter is of type "+Object.prototype.toString.call(n)+", string expected"));if(i.hooks&&(i.hooks.options=i),i.async)return Promise.resolve(i.hooks?i.hooks.preprocess(n):n).then((t=>e(t,i))).then((e=>i.hooks?i.hooks.processAllTokens(e):e)).then((e=>i.walkTokens?Promise.all(this.walkTokens(e,i.walkTokens)).then((()=>e)):e)).then((e=>t(e,i))).then((e=>i.hooks?i.hooks.postprocess(e):e)).catch(l);try{i.hooks&&(n=i.hooks.preprocess(n));let s=e(n,i);i.hooks&&(s=i.hooks.processAllTokens(s)),i.walkTokens&&this.walkTokens(s,i.walkTokens);let r=t(s,i);return i.hooks&&(r=i.hooks.postprocess(r)),r}catch(e){return l(e)}}}#t(e,t){return n=>{if(n.message+="\nPlease report this to https://github.com/markedjs/marked.",e){const e="<p>An error occurred:</p><pre>"+c(n.message+"",!0)+"</pre>";return t?Promise.resolve(e):e}if(t)return Promise.reject(n);throw n}}}const ae=new oe;function ce(e,t){return ae.parse(e,t)}ce.options=ce.setOptions=function(e){return ae.setOptions(e),ce.defaults=ae.defaults,n(ce.defaults),ce},ce.getDefaults=t,ce.defaults=e.defaults,ce.use=function(...e){return ae.use(...e),ce.defaults=ae.defaults,n(ce.defaults),ce},ce.walkTokens=function(e,t){return ae.walkTokens(e,t)},ce.parseInline=ae.parseInline,ce.Parser=ie,ce.parser=ie.parse,ce.Renderer=se,ce.TextRenderer=re,ce.Lexer=ne,ce.lexer=ne.lex,ce.Tokenizer=w,ce.Hooks=le,ce.parse=ce;const he=ce.options,pe=ce.setOptions,ue=ce.use,ke=ce.walkTokens,ge=ce.parseInline,fe=ce,de=ie.parse,xe=ne.lex;e.Hooks=le,e.Lexer=ne,e.Marked=oe,e.Parser=ie,e.Renderer=se,e.TextRenderer=re,e.Tokenizer=w,e.getDefaults=t,e.lexer=xe,e.marked=ce,e.options=he,e.parse=fe,e.parseInline=ge,e.parser=de,e.setOptions=pe,e.use=ue,e.walkTokens=ke}));
</script>
<script>
const log=document.getElementById('log'),input=document.getElementById('i'),btn=document.getElementById('b');
const stopBtn=document.getElementById('stop');
let history=[],busy=false;
const el=(t,c)=>{const e=document.createElement(t);if(c)e.className=c;return e};
const md=s=>{try{return window.marked?marked.parse(s):null}catch(e){return null}};
// 模型输出进 DOM 前必须洗一遍。页面和 read/write/edit/bash 同源，
// 一次 XSS = 直接把工具链（含 key）交出去：白名单标签 + 去掉 on* / javascript: / data:
const OK=new Set('B STRONG I EM U S DEL CODE PRE A P BR HR UL OL LI H1 H2 H3 H4 H5 H6 BLOCKQUOTE TABLE THEAD TBODY TR TH TD IMG SPAN DIV'.split(' '));
function clean(h){
  const doc=new DOMParser().parseFromString(h,'text/html');   // inert 文档：脚本不跑、图片不加载
  doc.body.querySelectorAll('*').forEach(n=>{   // 只走 body：html/head/body 本身不能 replaceWith
    if(!OK.has(n.tagName)){n.replaceWith(...n.childNodes);return}
    [...n.attributes].forEach(a=>{
      const k=a.name.toLowerCase();
      // 只留这 4 个属性：on*、style、id 之类一律去掉（style 能拿来 url() 外泄）
      if(!(k==='href'||k==='src'||k==='alt'||k==='title')
         ||((k==='href'||k==='src')&&/^\s*(javascript|data):/i.test(a.value)))n.removeAttribute(a.name);
    });
  });
  return doc.body.innerHTML;
}
const html=s=>{const h=md(s);return h?clean(h):''};
function bubble(role,parent){const m=el('div','msg '+role),b=el('div','bubble');m.appendChild(b);
  (parent||log).appendChild(m);m.scrollIntoView({block:'end',behavior:'smooth'});return b}
function toolCard(ev,parent){const d=el('div','tool'),n=el('b');
  n.textContent=ev.name;d.appendChild(n);
  d.appendChild(document.createTextNode(' '+JSON.stringify(ev.args)));
  const det=el('details'),sum=el('summary');sum.textContent='result';det.appendChild(sum);
  const pre=el('pre');pre.textContent='…';det.appendChild(pre);d.appendChild(det);
  (parent||log).appendChild(d);d.scrollIntoView({block:'end',behavior:'smooth'});
  return pre}
// 全部展开/收起：有轮次才显示，标签跟着当前状态走
const foldBtn=document.getElementById('foldBtn');
function paintFold(){
  const rs=log.querySelectorAll('.round');
  foldBtn.classList.toggle('hidden',!rs.length);
  foldBtn.textContent=[...rs].some(r=>r.open)?'全部收起':'全部展开';
}
foldBtn.onclick=()=>{
  const rs=[...log.querySelectorAll('.round')];if(!rs.length)return;
  const open=rs.some(r=>r.open);rs.forEach(r=>r.open=!open);paintFold();
};
// 停止：前端掐掉 fetch，同时按 rid 让服务端关掉上游（不然模型会继续生成到结束）
let ac=null,curRid='';
async function stopNow(){
  if(!busy)return;
  const rid=curRid;
  stopBtn.disabled=true;
  if(ac)ac.abort();                                   // 立刻断开本地的流
  try{await post('/api/stop',{rid});}catch(e){}       // 服务端掐上游；失败也无所谓，abort 已生效
}
async function send(text){
  if(busy||!text.trim())return;
  busy=true;btn.disabled=true;stopBtn.classList.remove('hidden');stopBtn.disabled=false;
  history.push({role:'user',content:text});
  bubble('user').textContent=text;
  // 一轮 = 一次模型调用，每轮一个 <details> 分组（默认展开）：
  // 带工具的轮留在分组里可折叠，纯文字的轮（最终回答）拆掉壳直接平铺。
  let out=null,acc='',raf=0,failed=false,turn='',rn=0,grp=null,gbody=null,gnames=[],t0=0,ntok=0;
  let stable=0,tailEl=null;   // stable = 已定稿前缀长度；只有尾巴每帧重绘
  const cards={};
  // 定稿边界：最后一个空行，且 ``` 计数为偶数（不在未闭合的围栏里）
  // ponytail: 每帧扫一遍 indexOf + 数围栏，60k 文本约 0.1ms；真嫌慢就增量记游标。
  // 已知取舍：空行分隔的 <ol> 会被切成两个列表（序号重新从 1 起），模型一般不在条目间留空行。
  const cutOf=s=>{
    let cut=s.lastIndexOf('\n\n');
    while(cut>=0){
      if(((s.slice(0,cut+2).match(/```/g)||[]).length)%2===0)return cut+2;
      cut=s.lastIndexOf('\n\n',cut-3);
    }
    return 0;
  };
  const paintNow=()=>{raf=0;if(failed||!out)return;
    if(!tailEl){tailEl=el('div');out.appendChild(tailEl)}
    const cut=cutOf(acc);
    if(cut>stable){   // 定稿的块只渲染一次，插在尾巴前面，已有 DOM 一概不动
      tailEl.insertAdjacentHTML('beforebegin',html(acc.slice(stable,cut)));
      stable=cut;
    }
    const rest=acc.slice(stable),h=html(rest);   // 未定稿的尾巴才每帧重解析
    tailEl.innerHTML=h;
    if(!h)tailEl.textContent=rest;
    out.scrollIntoView({block:'end'})};
  const paint=()=>{if(!raf)raf=requestAnimationFrame(paintNow)};  // 一帧最多重渲染一次，别每个 token 全量 parse
  const endText=()=>{if(!out)return;if(raf)cancelAnimationFrame(raf);raf=0;paintNow();out=null;stable=0;tailEl=null};
  const say=()=>{if(!out){out=bubble('assistant',gbody||log);acc='';stable=0;tailEl=null}return out};
  // 标题：第 N 轮 · 工具(bash ×2) · 1.4s；没工具的轮在跑的时候显示「思考中」
  const toolLabel=()=>{const c={};gnames.forEach(n=>c[n]=(c[n]||0)+1);
    return Object.keys(c).map(n=>c[n]>1?n+' ×'+c[n]:n).join(' + ')};
  const tok=n=>n>=1000?(n/1000).toFixed(1)+'k':String(n);
  const label=()=>'第 '+rn+' 轮 · '+(gnames.length?toolLabel():'思考中…')+
    ' · '+((performance.now()-t0)/1000).toFixed(1)+'s'+(ntok?' · '+tok(ntok)+' tok':'');
  const openRound=()=>{
    endText();
    rn++;gnames=[];t0=performance.now();ntok=0;
    grp=el('details','round');grp.open=true;
    const s=el('summary');s.textContent=label();grp.appendChild(s);
    gbody=el('div','round-body');grp.appendChild(gbody);
    log.appendChild(grp);grp.scrollIntoView({block:'end',behavior:'smooth'});
    // 新一轮展开，之前的自动收起（手动点开的那轮也会被收，别处没有状态可存）
    log.querySelectorAll('.round').forEach(d=>{if(d!==grp)d.open=false});
  };
  const endRound=()=>{
    endText();if(!grp)return;
    if(gnames.length){
      grp.querySelector('summary').textContent=label()+' · 点开看';
    }else{  // 纯文字轮：不要折叠壳，答案照常平铺
      const f=document.createDocumentFragment();
      while(gbody.firstChild)f.appendChild(gbody.firstChild);
      grp.replaceWith(f);
    }
    grp=null;gbody=null;
  };
  const tick=setInterval(()=>{if(grp)grp.querySelector('summary').textContent=label()},1000);
  let stopped=false;
  try{
    ac=new AbortController();curRid=(crypto.randomUUID?crypto.randomUUID():String(Date.now()+Math.random()));
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      signal:ac.signal,body:JSON.stringify({messages:history,workspace:ws,rid:curRid})});
    const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
    for(;;){
      const {value,done}=await rd.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      let i;while((i=buf.indexOf('\n\n'))>=0){
        const chunk=buf.slice(0,i);buf=buf.slice(i+2);
        if(!chunk.startsWith('data:'))continue;
        const ev=JSON.parse(chunk.slice(5));
        if(ev.type==='text'){if(gnames.length||!grp){endRound();openRound()}say();acc+=ev.delta;turn+=ev.delta;paint()}
        else if(ev.type==='workspace'){setWs(ev.path)}
        else if(ev.type==='tool'){
          if(!grp)openRound();endText();
          gnames.push(ev.name);
          grp.querySelector('summary').textContent=label();
          cards[ev.id]=toolCard(ev,gbody)}
        else if(ev.type==='tool_result'&&cards[ev.id]){cards[ev.id].textContent=ev.output;
          cards[ev.id].closest('.tool').scrollIntoView({block:'end'})}
        else if(ev.type==='usage'){ntok=ev.usage.total||0;if(grp)grp.querySelector('summary').textContent=label()}
        else if(ev.type==='error'){failed=true;const b=say();b.textContent=ev.message;b.classList.add('err')}
      }
    }
  }catch(e){
    if(e&&e.name==='AbortError'){stopped=true}   // 自己点的停止，不算失败
    else{failed=true;const b=say();b.textContent='请求失败：'+e;b.classList.add('err')}
  }
  clearInterval(tick);
  endRound();
  if(stopped)note('⏹ 已停止（这一轮的部分输出已保留在上下文里）');
  paintFold();
  if(turn)history.push({role:'assistant',content:turn});
  busy=false;btn.disabled=false;ac=null;curRid='';
  stopBtn.classList.add('hidden');
}
stopBtn.onclick=()=>stopNow();
// ---------------- 斜杠命令 ----------------
function note(t){const d=el('div','tool');d.style.borderLeftColor='rgba(255,255,255,.32)';
  d.style.whiteSpace='pre-wrap';d.textContent=t;log.appendChild(d);
  d.scrollIntoView({block:'end',behavior:'smooth'})}
const CMDS=[
  {c:'/new',d:'新开一个会话（清空上下文和消息）',run:()=>{history=[];log.textContent='';paintFold();note('已新开会话')}},
  {c:'/ws',d:'查看当前工作区；/ws <路径> 切换',run:async a=>{
     if(!a)return note('当前工作区：'+ws+'\n完整路径可直接输入，或用 📁 按钮逐级选');
     await useWs(a);note('工作区 → '+ws)}},
  {c:'/model',d:'打开模型设置',run:()=>openPanel('model')},
  {c:'/skills',d:'打开技能设置',run:()=>openPanel('skills')},
  {c:'/help',d:'显示所有命令',run:()=>note(CMDS.map(x=>x.c+'  '+x.d).join('\n'))},
];
const cmds=document.getElementById('cmds');let cmdPick=0,cmdList=[];
function paintCmds(){
  const v=input.value,first=v.split(' ')[0];
  const show=v.startsWith('/')&&v===first;   // 只在敲命令名时提示，开始敲参数就收起
  cmds.classList.toggle('hidden',!show);
  if(!show)return;
  cmdList=CMDS.filter(x=>x.c.startsWith(first));
  cmdPick=Math.min(cmdPick,Math.max(cmdList.length-1,0));
  cmds.textContent='';
  cmdList.forEach((x,i)=>{
    const d=el('div',i===cmdPick?'on':'');d.innerHTML='<b></b><span></span>';
    d.querySelector('b').textContent=x.c;d.querySelector('span').textContent=x.d;
    d.onmousedown=e=>{e.preventDefault();input.value=x.c+' ';runCmd(x.c+' ')};
    cmds.appendChild(d);
  });
  if(!cmdList.length)cmds.classList.add('hidden');
}
// 返回 true 表示这条输入被命令消费，不发请求
function runCmd(text){
  const [c,...rest]=text.trim().split(/\s+/),hit=CMDS.find(x=>x.c===c);
  if(!hit)return false;
  cmds.classList.add('hidden');input.value='';input.style.height='auto';
  if(busy&&c!=='/help'&&c!=='/model'&&c!=='/skills'){
    note('正在回复中，等这条结束再执行 '+c+'（或刷新页面中断）');return true}
  note(c);hit.run(rest.join(' '));return true;
}
document.getElementById('f').addEventListener('submit',e=>{e.preventDefault();
  const t=input.value;
  if(t.trim().startsWith('/')){runCmd(t);return}
  input.value='';input.style.height='auto';send(t)});
input.addEventListener('keydown',e=>{
  const open=!cmds.classList.contains('hidden');
  if(open&&(e.key==='ArrowDown'||e.key==='ArrowUp')){
    e.preventDefault();cmdPick=(cmdPick+(e.key==='ArrowDown'?1:-1)+cmdList.length)%cmdList.length;
    paintCmds();return}
  if(open&&e.key==='Tab'){
    e.preventDefault();input.value=cmdList[cmdPick].c+' ';paintCmds();return}
  if(open&&e.key==='Enter'&&!e.shiftKey){
    e.preventDefault();
    const c=input.value.trim();
    if(CMDS.some(x=>x.c===c)){runCmd(c);return}          // 完整命令才直接跑
    input.value=cmdList[cmdPick].c+' ';paintCmds();return  // 否则先补全，避免误触发
  }
  if(e.key==='Escape'){cmds.classList.add('hidden');return}
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();
    const t=input.value;
    if(t.trim().startsWith('/')){runCmd(t);return}
    document.getElementById('f').requestSubmit()}});
input.addEventListener('input',()=>{input.style.height='auto';
  input.style.height=Math.min(input.scrollHeight,180)+'px';paintCmds()});
input.focus();

// ---------------- 设置面板 ----------------
const $=id=>document.getElementById(id),panel=$('panel');
const j=async(url,opt)=>{const r=await fetch(url,opt);const d=await r.json().catch(()=>({}));
  if(!r.ok)throw new Error(d.error||('HTTP '+r.status));return d};
const post=(url,body)=>j(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
const findSkill=name=>skills.find(s=>s.name===name);

function openPanel(tab){
  panel.classList.remove('hidden');
  document.querySelectorAll('.tabs button[data-tab]').forEach(x=>x.classList.toggle('on',x.dataset.tab===tab));
  ['model','skills','ws'].forEach(t=>$('tab-'+t).classList.toggle('hidden',t!==tab));
  if(tab==='model')loadCfg();
  if(tab==='skills')loadSkills();
  if(tab==='ws')loadWs(ws);
}
$('cfgBtn').onclick=()=>openPanel('model');
$('wsBtn').onclick=()=>openPanel('ws');
$('closePanel').onclick=()=>panel.classList.add('hidden');
document.querySelectorAll('.tabs button[data-tab]').forEach(b=>b.onclick=()=>openPanel(b.dataset.tab));

// ---------------- 工作区 ----------------
let ws='';
function setWs(p){
  ws=p||'';$('wsChip').textContent=ws?ws.split('/').pop()||'/':'…';
  $('wsChip').title=ws;$('wsBtn').title='切换工作区：'+ws;
  $('hint').textContent=(ws||'未选工作区')+' · '+($('cfgModel').value||'未选模型');
}
async function loadWs(path){
  try{
    const d=await j('/api/workspaces'+(path?'?path='+encodeURIComponent(path):''));
    $('wsPath').value=d.path;
    const box=$('wsDirs');box.textContent='';
    if(!d.dirs.length)box.innerHTML='<div class="note">没有子目录</div>';
    d.dirs.forEach(n=>{
      const it=el('div','skitem');it.innerHTML='<div style="min-width:0"><b></b></div><span>进入 ›</span>';
      it.querySelector('b').textContent=n;
      it.onclick=()=>loadWs(d.path+'/'+n);
      box.appendChild(it);
    });
    $('wsUp').disabled=!d.parent;$('wsUp').dataset.parent=d.parent||'';
  }catch(e){$('wsMsg').textContent=e.message}
}
function paintRecent(list){
  const box=$('wsRecent');box.textContent='';
  if(!list.length)return;
  const h=el('div','note');h.textContent='最近使用';h.style.margin='0 0 8px';box.appendChild(h);
  list.forEach(p=>{
    const it=el('div','skitem');it.innerHTML='<div style="min-width:0"><b></b></div><span>使用</span>';
    it.querySelector('b').textContent=p;
    it.querySelector('b').style.fontSize='12px';
    it.onclick=()=>useWs(p);
    box.appendChild(it);
  });
  box.appendChild(el('div','note')).style.margin='0 0 12px';
}
async function useWs(path){
  try{
    const c=await post('/api/workspaces',{path});
    setWs(c.workspace);paintRecent(c.recent||[]);$('wsPath').value=c.workspace;
    $('wsMsg').textContent='✓ 已切换';loadWs(c.workspace);
  }catch(e){$('wsMsg').textContent='切换失败：'+e.message}
}
$('wsUse').onclick=()=>useWs($('wsPath').value);
$('wsUp').onclick=()=>{const p=$('wsUp').dataset.parent;if(p)loadWs(p)};

async function loadCfg(){
  try{
    const c=await j('/api/config');
    $('cfgBase').value=c.base_url||'https://api.openai.com/v1';
    $('cfgModel').value=c.model||'';
    $('cfgRounds').value=c.max_rounds||20;
    $('cfgKey').value='';
    $('cfgKey').placeholder=c.key_set?('已配置 '+c.key_hint+'，留空则不修改'):'sk-…';
    setWs(c.workspace);paintRecent(c.recent||[]);
    $('hint').textContent=ws+' · '+(c.model||'未选模型')+' · 最多 '+c.max_rounds+' 轮';
  }catch(e){$('cfgMsg').textContent='读取失败：'+e.message}
}
$('saveCfg').onclick=async()=>{
  const m=$('cfgMsg');m.textContent='保存中…';btn.disabled=true;
  try{
    const c=await post('/api/config',{OPENAI_BASE_URL:$('cfgBase').value,OPENAI_MODEL:$('cfgModel').value,
      OPENAI_API_KEY:$('cfgKey').value,MAX_ROUNDS:$('cfgRounds').value});
    $('cfgKey').value='';m.textContent='✓ 已生效 · '+(c.model||'?')+' · 最多 '+c.max_rounds+' 轮';
    $('hint').textContent=ws+' · '+(c.model||'未选模型')+' · 最多 '+c.max_rounds+' 轮';
    $('cfgKey').placeholder=c.key_set?('已配置 '+c.key_hint+'，留空则不修改'):'sk-…';
  }catch(e){m.textContent='失败：'+e.message}
  btn.disabled=busy;
};

let skills=[];
async function loadSkills(){
  try{skills=await j('/api/skills');paintSkills()}catch(e){$('skMsg').textContent='读取失败：'+e.message}
}
function paintSkills(){
  const box=$('skillList');box.textContent='';
  if(!skills.length){box.innerHTML='<div class="note" style="margin-bottom:12px">还没有技能，新建一个吧。</div>'}
  skills.forEach(s=>{
    const d=el('div','skitem');
    d.innerHTML='<div style="min-width:0"><b></b><small></small></div><span>编辑</span>';
    d.querySelector('b').textContent=s.name;
    d.querySelector('small').textContent=s.description||'（无描述）';
    d.onclick=()=>editSkill(s.name);
    box.appendChild(d);
  });
}
function editSkill(name){
  const s=name?findSkill(name):{name:'',description:'',body:''};
  $('skillEdit').classList.remove('hidden');
  $('skName').value=s.name;$('skDesc').value=s.description||'';$('skBody').value=s.body||'';
  $('skName').readOnly=!!name;$('delSkill').classList.toggle('hidden',!name);
  $('skMsg').textContent=name?'编辑 '+name:'新技能';
  $('skName').focus();
}
$('newSkill').onclick=()=>editSkill('');
$('saveSkill').onclick=async()=>{
  const m=$('skMsg'),name=$('skName').value.trim();m.textContent='保存中…';
  try{
    await post('/api/skills',{name,description:$('skDesc').value,body:$('skBody').value});
    await loadSkills();editSkill(name);m.textContent='✓ 已保存，下一次请求生效';
  }catch(e){m.textContent='失败：'+e.message}
};
$('delSkill').onclick=async()=>{
  const name=$('skName').value.trim();
  if(!name||!confirm('删除技能 '+name+'？'))return;
  try{
    await post('/api/skills/delete',{name});
    $('skillEdit').classList.add('hidden');await loadSkills();
    $('skMsg').textContent='已删除 '+name;
  }catch(e){$('skMsg').textContent='删除失败：'+e.message}
};
loadCfg();
</script></body></html>"""


# ---------------------------------------------------------------- main
def selftest():
    global SKILLS_DIR, CFG_PATH, CFG
    import tempfile
    d = tempfile.mkdtemp()
    os.chdir(d)
    CFG["workspace"] = d  # 工作区裁决 read/write/edit/bash 的相对根
    f = os.path.join(d, "a.txt")
    assert tool_write(f, "one\ntwo\nthree\n").startswith("wrote")
    assert "2|two" in tool_read(f, 2, 1).replace(" ", "")
    assert "edited" in tool_edit(f, "two", "TWO")
    assert open(f).read() == "one\nTWO\nthree\n"

    def raises(fn, needle):
        try:
            fn()
        except ValueError as e:
            return needle in str(e)
        return False

    assert raises(lambda: tool_edit(f, "nope", "x"), "not found")
    tool_write(f, "x\nx\n")
    assert raises(lambda: tool_edit(f, "x", "y"), "2 times")
    assert "2 replacements" in tool_edit(f, "x", "y", True) and open(f).read() == "y\ny\n"
    hi = tool_bash("echo hi")
    assert "exit=0" in hi and "hi" in hi, hi
    assert "exit=7" in tool_bash("exit 7")
    # 大输出只留 OUT_CAP，别整坨进内存
    big = tool_bash("head -c 300000 /dev/zero | tr '\\0' x")
    assert "truncated" in big and len(big) < OUT_CAP + 500, len(big)
    # 超时连子孙进程一起杀
    killed = tool_bash("sleep 30", timeout=1)
    assert "timeout 1s" in killed and "exit=-9" in killed, killed
    # 上下文预算：只丢完整的回合，system 和最后一轮留着
    hist = [{"role": "system", "content": "s"}] + [{"role": "user", "content": "u" * HIST_CAP}] * 3
    assert _trim(hist) is hist and len(hist) == 2, len(hist)

    SKILLS_DIR = os.path.join(d, "skills")
    os.makedirs(os.path.join(SKILLS_DIR, "fmt"))
    with open(os.path.join(SKILLS_DIR, "fmt", "SKILL.md"), "w") as f:
        f.write("---\nname: fmt\ndescription: 代码格式化\ntags: [a, b]\n---\n\n# steps\n1. 先读文件\n")
    assert load_skills()["fmt"] == ("代码格式化", "# steps\n1. 先读文件")
    assert "- fmt: 代码格式化" in system_prompt() and "- fmt:" not in SYSTEM
    assert tool_skill("fmt").startswith("# skill: fmt") and "先读文件" in tool_skill("fmt")
    assert "Available: fmt" in tool_skill("nope")
    # no frontmatter -> dir name + first body line
    os.makedirs(os.path.join(SKILLS_DIR, "bare"))
    with open(os.path.join(SKILLS_DIR, "bare", "SKILL.md"), "w") as f:
        f.write("\n部署到测试环境\n第二步……\n")
    assert load_skills()["bare"][0] == "部署到测试环境"

    # --- web 配置 & 技能 CRUD
    CFG_PATH = os.path.join(d, "config.json")
    CFG = {}
    # 自检不能受 shell 里导出的 OPENAI_* 影响（否则断言"应为空"必挂），先摘掉，结束后还原
    _ambient = {k: os.environ.pop(k) for k in
                ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "MAX_ROUNDS")
                if k in os.environ}
    p0 = public_config()
    assert p0["base_url"] == "" and p0["model"] == "" and not p0["key_set"], p0
    assert p0["workspace"] == DEFAULT_WORKSPACE and p0["recent"] == [], p0
    c = save_config({"OPENAI_BASE_URL": "https://x/v1", "OPENAI_MODEL": "m1",
                     "OPENAI_API_KEY": "sk-1234567890abcdef", "HOST": "ignored"})
    assert c["model"] == "m1" and c["key_set"] and c["key_hint"] == "sk-123…cdef", c
    assert json.load(open(CFG_PATH)) == {"OPENAI_BASE_URL": "https://x/v1",
                                         "OPENAI_MODEL": "m1", "OPENAI_API_KEY": "sk-1234567890abcdef"}
    save_config({"OPENAI_API_KEY": "", "OPENAI_MODEL": "m2"})  # 空 key 不覆盖
    assert env("OPENAI_API_KEY") == "sk-1234567890abcdef" and env("OPENAI_MODEL") == "m2"
    assert env("OPENAI_API_KEY") not in json.dumps(public_config())  # key 不回传前端

    # 循环次数：默认值 / 网页设置 / 越界拒绝
    assert p0["max_rounds"] == MAX_ROUNDS and max_rounds() == MAX_ROUNDS
    assert save_config({"MAX_ROUNDS": "5"})["max_rounds"] == 5 and max_rounds() == 5
    assert json.load(open(CFG_PATH))["MAX_ROUNDS"] == "5"
    assert raises(lambda: save_config({"MAX_ROUNDS": "0"}), "1-200")
    assert raises(lambda: save_config({"MAX_ROUNDS": "abc"}), "1-200")
    assert raises(lambda: save_config({"MAX_ROUNDS": "201"}), "1-200")
    assert max_rounds() == 5, max_rounds()
    save_config({"MAX_ROUNDS": "20"})
    os.environ.update(_ambient)  # 还原调用方的环境变量

    save_skill("code-review", "review 代码时用\n第二行", "1. git diff")
    assert load_skills()["code-review"] == ("review 代码时用 第二行", "1. git diff")
    assert [s["name"] for s in skills_payload()] == ["bare", "code-review", "fmt"]
    assert raises(lambda: save_skill("../evil", "x", "y"), "技能名")
    assert raises(lambda: delete_skill("nope"), "没有技能")
    delete_skill("code-review")
    assert "code-review" not in load_skills()

    # --- 工作区
    w1, w2 = os.path.join(d, "w1"), os.path.join(d, "w2")
    os.makedirs(os.path.join(w1, "src"))
    os.makedirs(os.path.join(w2, "deep", "deeper"))
    os.makedirs(os.path.join(w1, ".git"))  # 点目录不列出
    assert set_workspace(w1)["workspace"] == w1
    assert workspace() == w1, workspace()
    tool_write("src/x.py", "print(1)\n")  # 相对路径落在工作区
    assert open(os.path.join(w1, "src", "x.py")).read() == "print(1)\n"
    assert tool_bash("pwd").rstrip().endswith("/w1"), tool_bash("pwd")  # bash 也跟工作区
    assert "x.py" in tool_bash("ls src")
    b = browse(w1)
    assert b["dirs"] == ["src"] and b["parent"] == d, b
    assert browse(w2)["dirs"] == ["deep"]
    assert browse(os.path.join(w2, "deep"))["parent"] == w2
    assert set_workspace(w2)["workspace"] == w2
    assert CFG["recent"][:2] == [w2, w1], CFG["recent"]
    assert len(CFG["recent"]) == len(set(CFG["recent"]))  # 不重复
    set_workspace(w1)
    assert CFG["recent"][:2] == [w1, w2], CFG["recent"]
    assert raises(lambda: set_workspace(os.path.join(d, "nope")), "目录不存在")
    assert raises(lambda: browse("/etc/hosts"), "目录不存在")
    # 线程局部覆盖：并发请求各用各的工作区
    _TLS.workspace = w2
    assert workspace() == w2 and "w2" in tool_bash("pwd")
    _TLS.workspace = None
    assert workspace() == w1
    # 斜杠命令：菜单项必须有对应 handler（前端代码，这里只防漏写）
    for c in ("/new", "/ws", "/model", "/skills", "/help"):
        assert "c:'%s'" % c in PAGE, c
    assert "function runCmd(" in PAGE and "function paintCmds(" in PAGE
    # 停止：只掐自己那条 rid，别影响别人的
    _RUNS["r1"] = {"stop": False, "resp": None}
    _RUNS["r2"] = {"stop": False, "resp": None}
    _TLS.rid = "r1"
    assert not _stop_requested()
    assert stop_run("r1") and _stop_requested() and not _RUNS["r2"]["stop"]
    assert not stop_run("nope")
    _TLS.rid = "r2"
    assert not _stop_requested(), "掐 r1 不该影响 r2"
    _RUNS.pop("r1"), _RUNS.pop("r2"), setattr(_TLS, "rid", None)
    # 前端安全网：markdown 引擎内联（别退回 CDN）、清洗函数还在（DOM 行为见 test_frontend_probe.py）
    assert "function clean(" in PAGE and "DOMParser" in PAGE and "cdn.jsdelivr" not in PAGE
    # 每轮一个折叠组：工具卡排在分组里（不是插到最后一个气泡前面），纯文字轮拆壳平铺
    assert "function endRound(" not in PAGE and "const endRound=" in PAGE and "openRound" in PAGE
    assert "insertBefore(d" not in PAGE and "grp.replaceWith" in PAGE
    # 增量重绘：定稿块只 append 一次，每帧只重解析尾巴（60k 长回复 20.6ms/帧 -> 1.5ms/帧）
    assert "cutOf" in PAGE and "insertAdjacentHTML('beforebegin'" in PAGE and "tailEl" in PAGE
    # 停止按钮 + AbortController + rid 上报
    assert "id=\"stop\"" in PAGE and "function stopNow(" in PAGE and "AbortController" in PAGE
    assert "/api/stop" in PAGE and "rid:curRid" in PAGE
    # 新一轮自动收起旧的 + 组内耗时（1s 心跳刷新，结束时 clearInterval）
    assert "d.open=false" in PAGE and "performance.now()" in PAGE and "clearInterval(tick)" in PAGE
    # 全部展开/收起按钮 + 每轮 token 数（usage 事件透传）
    assert "paintFold" in PAGE and "id=\"foldBtn\"" in PAGE and "ev.type==='usage'" in PAGE
    assert _STREAM_OPTS is True and "include_usage" in _completion_req([{"role": "user", "content": "x"}]).data.decode()
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    for line in open(".env", encoding="utf-8") if os.path.exists(".env") else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    host, port = env("HOST", "127.0.0.1"), int(env("PORT", "8090"))
    if host not in ("127.0.0.1", "localhost", "::1") and not env("SMALL_AGENT_ALLOW_PUBLIC"):
        sys.exit("拒绝绑定 %s：bash 能执行任意命令、配置里还有 key。真要暴露就设 SMALL_AGENT_ALLOW_PUBLIC=1。" % host)
    print("small_agent -> http://%s:%d  (model=%s)" % (host, port, env("OPENAI_MODEL", "gpt-4o-mini")))
    ThreadingHTTPServer((host, port), Handler).serve_forever()
