
import json, re, ast
from typing import Dict, Any
from openai import OpenAI

SYSTEM_PROMPT = """You are a precise Python code formatter and repair assistant.
Your job: Fix indentation and formatting **only**, without changing semantics.
You must ensure the code is valid Python (passes AST parse).
Return STRICT JSON: {"ok": bool, "issues": [str], "code": "<full corrected file>"}
Rules:
- Preserve all imports, class/function names and bodies; only adjust whitespace/indentation, stray markers, and trivial formatting.
- If you see leftover template markers like <<<FORWARD_DEF>>> etc., remove them.
- Do NOT introduce new imports or external calls.
- Ensure class and function blocks are properly indented; if an empty block would remain, insert a single 'pass' on a properly-indented line.
- Ensure HYPERPARAMS is a valid Python dict literal (quoted keys).
- Return only JSON, no markdown fences or commentary.
"""

USER_TMPL = """Given this Python file content (truncated to 7000 chars if necessary):
```
{code}
```
Return STRICT JSON with keys ok, issues, code (the full corrected file). Do not add any text besides JSON.
"""

class LLMFormatAgent:
    def __init__(self, client: OpenAI, model: str = "gpt-5-mini"):
        self.client = client
        self.model = model

    def repair(self, code: str) -> Dict[str, Any]:
        # Trim to keep prompt size sane while preserving completeness for small files
        code_slice = code[:7000]
        prompt = USER_TMPL.format(code=code_slice)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system", "content": SYSTEM_PROMPT},
                {"role":"user", "content": prompt}
            ]
        )
        txt = resp.choices[0].message.content.strip()
        try:
            data = json.loads(txt)
        except Exception:
            return {"ok": False, "issues": ["LLM returned non-JSON"], "code": code, "raw": txt}

        # Final AST validation on the returned code
        new_code = data.get("code") or code
        try:
            ast.parse(new_code)
            data["ok"] = True
        except Exception as e:
            data["ok"] = False
            data.setdefault("issues", []).append(f"AST parse failed after LLM format: {e}")
            data["code"] = code  # fall back
        return data
