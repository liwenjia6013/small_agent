#!/usr/bin/env python3
"""End-to-end smoke: mock OpenAI /v1 streaming server + real agent.py run.
Run: python3 test_smoke.py   (spawns agent.py on :8099, writes into a temp dir)
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MOCK_PORT, AGENT_PORT = 8098, 8099
seen = []
stop_wait = threading.Event()   # 让 mock 的慢速流一直挂着，直到测试掐断


def sse(events):
    return "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"


USAGE = {"usage": {"prompt_tokens": 30, "completion_tokens": 60, "total_tokens": 90}, "choices": []}


SCRIPT = [  # one entry per upstream turn; each is a list of streamed deltas
    [{"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                      "function": {"name": "skill", "arguments": '{"name":"demo"}'}}]}],
    # tool call split across chunks, like real providers
    [{"tool_calls": [{"index": 0, "id": "call_2", "type": "function",
                      "function": {"name": "write", "arguments": '{"path":"hello.txt",'}}]},
     {"tool_calls": [{"index": 0, "function": {"arguments": '"content":"hi 中文\\n"}'}}]}],
    [{"content": "done: "}, {"content": "wrote hello.txt"}],
]


class Mock(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append(body)
        last = body["messages"][-1].get("content")
        if last == "boom":  # 网关把错误当 SSE 帧推进来
            payload = sse([{"error": {"message": "quota exceeded"}}])
        elif last == "sleepy":  # 让模型要跑一个 30 秒的 bash，用来测「停止能不能掐掉正在跑的工具」
            payload = sse([{"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_sleep", "type": "function",
                 "function": {"name": "bash", "arguments": '{"command":"sleep 30","timeout":60}'}}]}}]}, USAGE])
        elif last == "slow":
            # 先吐一段就装死，模拟「模型还在生成」；没有 Content-Length，靠 close 结束
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            first = "".join("data: " + json.dumps(e) + "\n\n"
                            for e in [{"choices": [{"delta": {"content": "第一段。"}}]}])
            self.wfile.write(first.encode())
            self.wfile.flush()
            self.close_connection = True
            stop_wait.wait(5)      # 等着被 /api/stop 掐（测试里 set）或超时
            try:
                self.wfile.write(sse([{"choices": [{"delta": {"content": "尾巴"}}]}]).encode())
            except OSError:
                pass
            return
        else:
            turn = sum(1 for m in body["messages"] if m.get("role") == "tool")
            deltas = SCRIPT[min(turn, len(SCRIPT) - 1)]
            payload = sse([{"choices": [{"delta": d}]} for d in deltas] + [USAGE])
            assert "stream_options" in body, "没要 usage 就统计不了 token"
        b = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def main():
    assert free(MOCK_PORT) and free(AGENT_PORT), "ports busy"
    work = tempfile.mkdtemp()
    os.makedirs(os.path.join(work, "skills", "demo"))
    with open(os.path.join(work, "skills", "demo", "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: demo\ndescription: 演示技能\n---\n\nSTEP ONE: 先写下 hello.txt\n")
    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", MOCK_PORT), Mock).serve_forever,
                     daemon=True).start()
    env = dict(os.environ, OPENAI_BASE_URL="http://127.0.0.1:%d/v1" % MOCK_PORT,
               OPENAI_API_KEY="test", OPENAI_MODEL="mock-model", PORT=str(AGENT_PORT),
               SMALL_AGENT_CONFIG=os.path.join(work, "config.json"),
               SMALL_AGENT_SKILLS=os.path.join(work, "skills"))  # 别碰真实 config.json/技能
    agent = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "agent.py")],
                             cwd=work, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            if not free(AGENT_PORT):
                break
            time.sleep(0.1)
        req = Request("http://127.0.0.1:%d/api/chat" % AGENT_PORT,
                      data=json.dumps({"messages": [{"role": "user", "content": "写个文件"}]}).encode(),
                      headers={"Content-Type": "application/json"})
        raw = urlopen(req, timeout=20).read().decode()
        events = [json.loads(l[5:]) for l in raw.splitlines() if l.startswith("data:")]

        kinds = [e["type"] for e in events]
        assert kinds.count("tool") == 2 and kinds.count("tool_result") == 2, kinds
        # usage 透传：每轮一条，前端拿它显示 token 数
        assert kinds.count("usage") == 3, kinds
        assert next(e for e in events if e["type"] == "usage")["usage"]["total"] == 90
        skill_ev = next(e for e in events if e["type"] == "tool")
        assert skill_ev["name"] == "skill" and skill_ev["args"] == {"name": "demo"}, skill_ev
        write_ev = [e for e in events if e["type"] == "tool"][1]
        assert write_ev["args"] == {"path": "hello.txt", "content": "hi 中文\n"}, write_ev
        assert open(os.path.join(work, "hello.txt"), encoding="utf-8").read() == "hi 中文\n"
        assert "STEP ONE" in next(e for e in events if e["type"] == "tool_result")["output"]
        assert "wrote hello.txt" in "".join(e.get("delta", "") for e in events if e["type"] == "text")
        assert len(seen) == 3, len(seen)
        # skill index in system prompt, then skill body fed back to the model
        assert "- demo: 演示技能" in seen[0]["messages"][0]["content"], seen[0]["messages"][0]["content"]
        assert "STEP ONE" in seen[1]["messages"][-1]["content"], seen[1]["messages"][-1]
        assert seen[1]["messages"][-1]["role"] == "tool" and seen[2]["messages"][-1]["role"] == "tool"
        assert seen[0]["stream"] is True and seen[0]["tools"][0]["function"]["name"] == "read"
        assert any(t["function"]["name"] == "skill" for t in seen[0]["tools"])

        # --- web 配置 & 技能 API
        api = "http://127.0.0.1:%d" % AGENT_PORT

        def get(p):
            return json.load(urlopen(api + p, timeout=10))

        def post(p, obj):
            return json.load(urlopen(Request(api + p, data=json.dumps(obj).encode(),
                                             headers={"Content-Type": "application/json"}),
                                     timeout=10))

        cfg = get("/api/config")
        assert cfg["key_set"] and cfg["model"] == "mock-model" and "test" not in json.dumps(cfg), cfg
        post("/api/config", {"OPENAI_BASE_URL": "http://127.0.0.1:%d/v1" % MOCK_PORT,
                             "OPENAI_MODEL": "mock-2", "OPENAI_API_KEY": ""})
        assert get("/api/config")["model"] == "mock-2"
        assert json.load(open(os.path.join(work, "config.json")))["OPENAI_MODEL"] == "mock-2"

        assert [s["name"] for s in get("/api/skills")] == ["demo"]
        post("/api/skills", {"name": "deploy", "description": "部署时用", "body": "1. rsync"})
        assert [s["name"] for s in get("/api/skills")] == ["demo", "deploy"]
        post("/api/skills/delete", {"name": "demo"})
        assert [s["name"] for s in get("/api/skills")] == ["deploy"]
        try:
            post("/api/skills", {"name": "../evil", "description": "", "body": ""})
            bad = 0
        except HTTPError as e:
            bad = e.code
        assert bad == 400, bad
        assert not os.path.exists(os.path.join(work, "..", "evil")), "path traversal escaped skills/"

        # 新配置和新技能对下一次对话立即生效
        urlopen(Request(api + "/api/chat",
                        data=json.dumps({"messages": [{"role": "user", "content": "下次"}]}).encode(),
                        headers={"Content-Type": "application/json"}), timeout=20).read()
        last = seen[-1]
        assert last["model"] == "mock-2", last["model"]
        assert "- deploy: 部署时用" in last["messages"][0]["content"], last["messages"][0]["content"]

        # --- 工作区 API + 每次对话按请求切换
        ws1 = os.path.join(work, "ws1")
        ws2 = os.path.join(work, "ws2")
        for p in (ws1, ws2):
            os.makedirs(os.path.join(p, "proj"))
        os.makedirs(os.path.join(ws1, "proj", "inner"))
        c = post("/api/workspaces", {"path": ws1})
        assert c["workspace"] == ws1 and c["recent"][0] == ws1, c
        assert json.load(open(os.path.join(work, "config.json")))["workspace"] == ws1
        b = get("/api/workspaces?path=" + ws1)
        assert b["path"] == ws1 and b["dirs"] == ["proj"] and b["parent"] == work, b
        assert get("/api/workspaces?path=" + ws1 + "/proj")["dirs"] == ["inner"]
        try:
            get("/api/workspaces?path=" + ws1 + "/nope")
            bad = 0
        except HTTPError as e:
            bad = e.code
        assert bad == 400, bad

        # 请求里的 workspace 决定 read/write/edit/bash 的根（这里让 mock 再走一遍 write hello.txt）
        def chat(path=None, content="再写一个"):
            body = {"messages": [{"role": "user", "content": content}]}
            if path:
                body["workspace"] = path
            return urlopen(Request(api + "/api/chat", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"}), timeout=20).read()

        chat(ws2)
        assert open(os.path.join(ws2, "hello.txt"), encoding="utf-8").read() == "hi 中文\n"
        assert not os.path.exists(os.path.join(ws1, "hello.txt")), "workspace 没生效"
        # 不传 workspace 时回落到持久化的 ws1
        chat()
        assert open(os.path.join(ws1, "hello.txt"), encoding="utf-8").read() == "hi 中文\n"

        # 上游错误帧要冒成 error 事件，不能静默给个空白回复
        errs = [json.loads(l[5:]) for l in chat(content="boom").decode().splitlines() if l.startswith("data:")]
        assert any(e["type"] == "error" and "quota exceeded" in e["message"] for e in errs), errs

        # 循环次数可在网页配置：设 1 轮就只跑一次模型调用
        assert get("/api/config")["max_rounds"] == 20
        assert post("/api/config", {"MAX_ROUNDS": "1"})["max_rounds"] == 1
        evs = [json.loads(l[5:]) for l in chat(content="再来一次").decode().splitlines() if l.startswith("data:")]
        assert sum(1 for e in evs if e["type"] == "tool") == 1, evs
        assert "已跑满 1 轮" in "".join(e.get("delta", "") for e in evs), evs
        try:
            post("/api/config", {"MAX_ROUNDS": "0"})
            bad = 0
        except HTTPError as e:
            bad = e.code
        assert bad == 400, bad

        # --- 主动停止：慢速流 + /api/stop，客户端应很快拿到 close，而不是等模型生成完
        stop_wait.clear()
        got = {}

        def slow_chat():
            t0 = time.time()
            try:
                raw = urlopen(Request(api + "/api/chat",
                                      data=json.dumps({"messages": [{"role": "user", "content": "slow"}],
                                                       "rid": "rid-slow"}).encode(),
                                      headers={"Content-Type": "application/json"}), timeout=20).read()
                got["events"] = [json.loads(l[5:]) for l in raw.decode().splitlines() if l.startswith("data:")]
            except Exception as e:   # 被掐断也算正常
                got["error"] = repr(e)
            got["secs"] = time.time() - t0

        th = threading.Thread(target=slow_chat, daemon=True)
        th.start()
        time.sleep(0.6)                                  # 等第一条 delta 到达
        assert post("/api/stop", {"rid": "rid-slow"})["ok"] is True
        stop_wait.set()                                   # 放开 mock，别拖住整个测试
        th.join(5)
        assert th.is_alive() is False, "停止后请求还挂着"
        assert got["secs"] < 3, "停止没生效，等了 %.1fs" % got["secs"]
        assert any(e["type"] == "text" and "第一段" in e.get("delta", "") for e in got.get("events", [])), got
        assert not any(e.get("delta") == "尾巴" for e in got.get("events", [])), "停止后还在收内容"
        assert post("/api/stop", {"rid": "rid-slow"})["ok"] is False   # 已经结束的运行，掐不到

        # 停止也要掐掉正在跑的工具：30 秒的 sleep 应该立刻断，而不是等 timeout
        got2 = {}

        def sleepy_chat():
            t0 = time.time()
            try:
                raw = urlopen(Request(api + "/api/chat",
                                      data=json.dumps({"messages": [{"role": "user", "content": "sleepy"}],
                                                       "rid": "rid-sleep"}).encode(),
                                      headers={"Content-Type": "application/json"}), timeout=40).read()
                got2["events"] = [json.loads(l[5:]) for l in raw.decode().splitlines() if l.startswith("data:")]
            except Exception as e:
                got2["error"] = repr(e)
            got2["secs"] = time.time() - t0

        th2 = threading.Thread(target=sleepy_chat, daemon=True)
        th2.start()
        time.sleep(1.2)                                  # 等 bash 真的跑起来
        assert post("/api/stop", {"rid": "rid-sleep"})["ok"] is True
        th2.join(10)
        assert th2.is_alive() is False, "停止没掐断工具，请求还挂着"
        assert got2["secs"] < 5, "停止没掐掉子进程，等了 %.1fs" % got2["secs"]
        assert not any(e["type"] == "tool_result" for e in got2.get("events", [])), "停止后工具还跑完了"
        print("smoke ok:", kinds)
    finally:
        agent.terminate()


if __name__ == "__main__":
    main()
