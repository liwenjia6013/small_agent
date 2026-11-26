# small_agent

最小 AI agent：read / write / edit / bash 四个工具，OpenAI 兼容 `/v1` 接口，纯 Python 标准库 + 单 HTML 页面，零依赖。

```bash
cp .env.example .env   # 填 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL
python3 agent.py       # http://127.0.0.1:8090
```

- 工具作用于**服务启动目录**（相对路径），bash 也在该目录下执行。
- 前端把整段对话回传，服务端无状态；回复走 SSE 流式，工具调用/结果实时显示。
- 每轮（一次模型调用）在页面上是一个可折叠分组：标题显示「第 N 轮 · 用了哪些工具 ×次数 · 耗时 · token 数」，跑的时候每秒刷新；一轮结束或下一轮开始时自动收起，最后一轮纯文字回答不套壳、直接平铺。顶栏「全部展开/收起」一键操作所有轮次。
- token 数来自上游流末尾的 `usage`（请求里带 `stream_options: {"include_usage": true}`）。网关不认这个参数（返回 400）时会自动关掉重试，此时不显示 token 数，其余功能不受影响。
- 历史超过 6 万字符时，服务端从最早的完整回合开始丢（不会拆散 tool_calls / tool 配对）。
- 回复中可以点「■ 停止」随时中止：前端断开流，同时 POST `/api/stop` 带本次请求的 `rid`，服务端按 rid 关掉上游连接（不再烧 token）并 `killpg` 掉正在跑的工具子进程；已经收到的文字保留在上下文里，可以直接说「继续」。停止不算错误，多标签页互不影响。
- 流式渲染是增量的：已定稿的块（以空行分隔、代码围栏已闭合）只解析并插入一次，每帧只重解析还没写完的那一块。60k 长回复从「整段重绘 20.6ms/帧」降到中位 2.5ms/帧（`python3 test_perf.py` 跑回归）。
- `python3 agent.py --selftest` 检查工具/技能/配置读写；`python3 test_smoke.py` 起 mock `/v1` 跑完整回合 + 配置与技能 API；`python3 test_frontend_probe.py` 生成 `/tmp/agent_probe.html`（清洗逻辑 PASS/FAIL）和 `/tmp/agent_sim.html`（用假 SSE 流跑一遍 `send()`，dump 出轮次分组的 DOM 结构）；`python3 test_perf.py` 生成 `/tmp/agent_perf.html`，量长回复下每帧重绘耗时，标题 `PERF PASS` 才算过。
- 只绑 127.0.0.1；bash 能执行任意命令，别暴露到公网（改 HOST 会被拒绝，真要暴露得加 `SMALL_AGENT_ALLOW_PUBLIC=1`）。

## Web 设置（右上角 ⚙）

- **模型**：BASE_URL / API KEY / MODEL，保存写 `config.json` 并**立即生效**（不用重启），优先级高于 `.env`。API key 只回传 `sk-123…cdef` 形式的提示，留空保存则不修改。
- **循环次数**：一轮 = 一次模型调用；用满就停并提示「已跑满 N 轮」。默认 20，可填 1-200（写 `MAX_ROUNDS`），下次请求生效。
- **技能**：列表 + 新建/编辑/删除，写 `skills/<name>/SKILL.md`，保存后下一次请求即进系统提示。
- **工作区**：点顶部 `📁 目录名` 或设置里的「工作区」，可逐级点选目录或手输路径。选中后 read/write/edit/bash 的相对根就切过去，bash 的 cwd 也是它；持久化到 `config.json`，最近 8 个可选。前端每次对话把当前工作区带上，服务端用线程局部生效——多标签页各干各的不打架。

`config.json`（含 key）已 gitignore。技能目录固定为脚本同级的 `skills/`，不随工作区走。

## 斜杠命令

输入框敲 `/` 弹补全菜单（↑↓ 选，Tab 补全，Enter 跑；直接打完命令名回车也跑）。纯前端，不发请求：

| 命令 | 作用 |
|---|---|
| `/new` | 新开会话：清空上下文和消息列表 |
| `/ws` | 查看当前工作区；`/ws <路径>` 直接切过去 |
| `/model` | 打开模型设置 |
| `/skills` | 打开技能设置 |
| `/help` | 列出所有命令 |

加新命令就往前端 `CMDS` 数组里加一条 `{c:'/xxx', d:'说明', run:()=>…}`，`--selftest` 会检查菜单项漏写 handler。

## 技能（skills）

`skills/<目录>/SKILL.md`，格式同 Hermes skill：

```markdown
---
name: code-review
description: 需要 review 代码改动时使用
---

# 步骤
1. bash: git diff
2. ...
```

- 启动目录下的 `skills/*/SKILL.md` 会被扫描，`name: description` 注入系统提示，模型用 `skill(name)` 工具取全文按步骤执行。
- 每次请求重新扫描，所以**改技能、加技能不用重启**（agent 自己用 write 写一个新技能也能立刻用上）。
- frontmatter 可省略：名字取目录名，描述取正文第一行。
- `skills/example/` 是可直接复制的模板。

模型接口按 OpenAI 规范调用：`BASE_URL` 指到版本段（如 `.../v1` 或智谱的 `.../v4`），程序拼 `/chat/completions`，需要模型支持 function calling + `stream:true` 增量 `tool_calls`。
