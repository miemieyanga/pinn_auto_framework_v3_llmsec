# agents/safety_agent.py
from __future__ import annotations

import json
import re
from typing import Dict, List
from openai import OpenAI

# ---- LLM 基本设置 ----
MODEL = "gpt-5-mini"   # 不传 temperature 等参数
_client = OpenAI()

# ---- 真正需要阻断的危险模式（本地强规则）----
BLOCKING_REGEX: List[str] = [
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"__import__\s*\(",
    r"open\(.+['\"]w['\"]\)",  # 文件写入/覆盖
]

# ---- 仅提示的告警类型（不阻断执行）----
WARNING_TAGS = {
    "resource_exhaustion",
    "memory_growth",
    "runtime_error_risk",
    "performance_risk",
}

# ---- LLM 提示词（分级为 blockers / warnings）----
_LLM_SYSTEM = (
    "You are a Python safety triage assistant. "
    "Classify findings as 'blockers' (must stop) or 'warnings' (informational). "
    "Only these patterns are blockers: os.system, subprocess, eval/exec, __import__, "
    "and file writes via open(...,'w'). Everything else (long training loops, memory growth, "
    "runtime risks, performance concerns) is a warning. "
    "Return ONLY JSON: {\"blockers\": [...], \"warnings\": [...]} "
    "Each item object must include: type, location, severity, description, recommendation."
)

_LLM_USER_TMPL = """Code to review:
```python
{code}
Rules:

Blockers: os.system, subprocess, eval/exec, import, open(...,'w').

Warnings: resource_exhaustion, memory_growth, runtime_error_risk, performance_risk, etc.

Output ONLY minimal machine-readable JSON as specified.
"""

class SafetyAgent:
    @staticmethod
    def run(code: str) -> Dict:
        """
        返回:
        {
        "ok": bool,
        "blockers": [ {type, location, severity, description, recommendation}, ... ],
        "warnings": [ {type, location, severity, description, recommendation}, ... ],
        "issues": blockers + warnings # 兼容旧字段
        }
        """
        blockers: List[Dict] = []
        warnings: List[Dict] = []

        # 1) 先用 LLM 做一次分级（尽力而为，失败也不影响）
        try:
            resp = _client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user", "content": _LLM_USER_TMPL.format(code=code)},
                ],
            )
            txt = resp.choices[0].message.content
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                data = json.loads(m.group(0))
                for item in (data.get("blockers") or []):
                    item.setdefault("severity", "high")
                    blockers.append(item)
                for item in (data.get("warnings") or []):
                    # 规范化 warning 类型
                    t = str(item.get("type", "performance_risk")).lower()
                    if t not in WARNING_TAGS:
                        item["type"] = "performance_risk"
                    item.setdefault("severity", "low")
                    warnings.append(item)
        except Exception:
            # LLM 不可用时忽略，靠本地规则兜底
            pass

        # 2) 本地强校验（命中即一律阻断）
        for pat in BLOCKING_REGEX:
            if re.search(pat, code):
                blockers.append({
                    "type": "blocking_pattern",
                    "location": "source",
                    "severity": "high",
                    "description": f"Matched pattern: {pat}",
                    "recommendation": "Remove dangerous calls (os.system/subprocess/eval/exec/__import__/write).",
                })

        # 3) 汇总与 ok 判定
        ok = (len(blockers) == 0)
        result = {
            "ok": ok,
            "blockers": blockers,
            "warnings": warnings,
            "issues": [*blockers, *warnings],  # 兼容旧字段
        }
        return result
