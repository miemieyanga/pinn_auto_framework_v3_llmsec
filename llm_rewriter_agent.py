
import json, ast
from typing import Dict, Any
from openai import OpenAI

SYSTEM_PROMPT = """You are a Python code rewriting expert for Physics-Informed Neural Networks (PINNs).
Rewrite any possibly broken or incomplete code into a correct, executable Python script.
Follow these safety and logic rules:
- Keep all logic, imports, and numerical computations intact.
- Allow saving results to 'results/*.npz' or '*.json' files, but ALWAYS wrap all file I/O in try/except.
- Before saving, check total bytes < 10MB. If too large, skip save and print a warning.
- Never use eval(), exec(), os.system(), or subprocess calls.
- Always create directories and files with safe relative paths.
- Ensure class Model with __init__ and forward exists.
- Ensure forward returns output tensor.
- Ensure HYPERPARAMS is a valid Python dict with quoted keys.
- Output STRICT JSON: {"ok": true, "issues": [..], "code": "<complete valid Python file>"} ONLY.
"""

USER_TMPL = """Rewrite this Python file into a full, valid, safe version.
```
{code}
```
Return STRICT JSON with keys ok, issues, code."""

class LLMRewriterAgent:
    def __init__(self, client: OpenAI, model: str = "gpt-5-mini"):
        self.client = client
        self.model = model

    def rewrite(self, code: str) -> Dict[str, Any]:
        snippet = code[:9000]
        prompt = USER_TMPL.format(code=snippet)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        txt = resp.choices[0].message.content.strip()
        try:
            out = json.loads(txt)
        except Exception:
            return {"ok": False, "issues": ["LLM returned invalid JSON"], "code": code, "raw": txt}
        # Verify AST
        try:
            ast.parse(out.get("code") or "")
            out["ok"] = True
        except Exception as e:
            out["ok"] = False
            out.setdefault("issues", []).append(f"AST parse failed: {e}")
        return out
