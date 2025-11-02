
import json, re
from typing import Dict, Any, List
from openai import OpenAI

DEFAULT_POLICY = {
    "forbidden_imports": [
        "requests","urllib","socket","subprocess","ftplib","paramiko","pickle"
    ],
    "forbidden_calls": [
        "os.system","eval","exec","subprocess.Popen","subprocess.run","subprocess.call"
    ],
    "forbidden_patterns": [
        r"open\([\"']/(etc|proc|sys|dev)",
        r"__import__\(",
        r"globals\(\)|locals\(\)"
    ],
    "max_file_io_bytes": 10_000_000  # guardrail hint, LLM should flag huge writes
}

SYSTEM_PROMPT = """You are a meticulous secure code reviewer for Python. 
Task: given code, judge SAFETY for offline execution in a restricted environment. 
Focus ONLY on security risks, not style or accuracy of math.
Return STRICT JSON with fields:
- ok: boolean
- grade: one of ["PASS","WARN","BLOCK"]
- issues: array of short strings, each a single problem
- evidence: array of exact code snippets or line refs
- suggestions: array of actionable remediations
- suggested_patch: string with a MINIMAL patch (optional), or empty string
Rules:
- BLOCK on: network access; shelling out; dynamic code execution (eval/exec); importing dangerous modules; reading system files; credential access; path traversal; big file I/O attempts; infinite loops for resource exhaustion; attempts to change process limits; environment variable exfiltration.
- WARN on: unnecessary file I/O; unclear error handling; excessive logging of data; missing resource timeouts.
- PASS only if none of the above.
Keep JSON compact.
"""

USER_PROMPT_TMPL = """Security policy (JSON):
{policy}

Review this Python code (max 4000 chars slice shown):
```
{code}
```

Answer ONLY as compact JSON following the required fields.
"""

class LLMSecurityAgent:
    def __init__(self, client: OpenAI, policy: Dict[str, Any] = None, model: str = "gpt-5-mini"):
        self.client = client
        self.policy = policy or DEFAULT_POLICY
        self.model = model

    def review(self, code: str) -> Dict[str, Any]:
        # Pre-check lightweight heuristics (defense in depth; helps steer LLM too)
        heuristics_issues: List[str] = []
        for mod in self.policy["forbidden_imports"]:
            if re.search(rf"(^|\n)\s*import\s+{re.escape(mod)}\b", code):
                heuristics_issues.append(f"Forbidden import detected: {mod}")
        for call in self.policy["forbidden_calls"]:
            name = call.split(".")[-1]
            if re.search(rf"\b{name}\s*\(", code):
                heuristics_issues.append(f"Suspicious call: {call}")
        for pat in self.policy["forbidden_patterns"]:
            if re.search(pat, code):
                heuristics_issues.append(f"Pattern matched: {pat}")

        # Trim code to a safe window for prompt
        code_slice = code[:4000]
        prompt = USER_PROMPT_TMPL.format(policy=json.dumps(self.policy), code=code_slice)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": prompt}
            ]
        )
        text = resp.choices[0].message.content.strip()

        # Try parse JSON; if fails, build BLOCK result
        try:
            report = json.loads(text)
        except Exception:
            return {
                "ok": False, "grade": "BLOCK",
                "issues": ["LLM did not return valid JSON"],
                "evidence": [], "suggestions": ["Re-run security review"],
                "suggested_patch": "", "llm_raw": text, "heuristics": heuristics_issues
            }

        # Merge heuristic hints
        if heuristics_issues and report.get("ok", True):
            # downgrade to WARN if LLM missed something
            report["ok"] = False
            report["grade"] = "WARN"
            report.setdefault("issues", []).extend(heuristics_issues)
        report["heuristics"] = heuristics_issues
        return report
