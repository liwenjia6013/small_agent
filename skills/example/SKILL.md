---
name: example
description: 技能格式示例 —— 复制这个目录改名即可新增技能
---

# 技能格式

`skills/<目录>/SKILL.md`，YAML frontmatter 只有两个有用字段：

- `name`：调用 `skill(name)` 用的名字，缺省取目录名
- `description`：一句话触发条件，会出现在系统提示的技能列表里

正文是给模型看的完整步骤，写完立刻生效（每次请求重新扫描），不需要重启服务。

## 示例步骤

1. 先用 `bash` 确认当前分支：`git status -sb`
2. 用 `read` 看清要改的文件
3. 用 `edit` 做最小改动，改完用 `bash` 跑一次验证
