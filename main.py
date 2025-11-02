\
import os, json, argparse, copy, sys, time, random
from dataclasses import dataclass
import pathlib
import re
from typing import Dict, Any
from openai import OpenAI

from sandbox import run_in_docker, run_local
from safety_check import check_file
from llm_security import LLMSecurityAgent
from format_agent import format_and_validate
from llm_rewriter_agent import LLMRewriterAgent
from llm_format_agent import LLMFormatAgent

TEMPLATE_PATH = "pinn_template.py"
GEN_PATH = "generated_impl.py"

# ---------------------- Prompts ----------------------
SYSTEM_PROMPT = """You are a cautious code generator for PINNs.
You will produce ONLY the content for 3 insertion blocks to fill a safe template:
1) <<<MODEL_DEF>>>
2) <<<FORWARD_DEF>>>
3) <<<HYPERPARAMS>>>
Strict rules:
- Use only torch/torch.nn/torch.optim/math/numpy constructs that are CPU-safe.
- Absolutely DO NOT import network libs, do not call os.system/subprocess/eval/exec, do not read env vars.
- Model should be a small MLP with Tanh/ReLU for 1D input and 1D output.
- HYPERPARAMS may include: epochs, lr, collocation, bc_weight, verbose_every, hidden_layers, hidden_units.
- Keep it concise.
"""

USER_PROMPT = """Task: Solve 1D Poisson with PINN using the provided template.
Goal: Minimize relative L2 vs. exact solution within given budget.
Constraints: Insertion must be plain code fragments, no imports. No prints except inside template training prints.
History (last best metrics and hyperparams):
{history}

Please propose new MODEL_DEF / FORWARD_DEF / HYPERPARAMS that improve results.
"""

# ---------------------- Agents ----------------------
@dataclass
class ExecResult:
    ok: bool
    reason: str
    stdout: str = ""
    stderr: str = ""
    metrics: Dict[str, Any] = None
    fragments: Dict[str, str] = None


def _sanitize_hparams_fragment(frag: str):
    """Parse a loose fragment into a safe Python dict.
    Strategy: extract known numeric keys via regex; merge with safe defaults; clamp caps.
    """
    defaults = {
        "epochs": 500,
        "lr": 1e-3,
        "collocation": 200,
        "bc_weight": 100.0,
        "verbose_every": 200,
    }
    # regex extraction (very forgiving)
    def _num(pattern, cast=float, default=None):
        m = re.search(pattern, frag, flags=re.IGNORECASE)
        if not m: 
            return default
        try:
            return cast(m.group(1))
        except Exception:
            return default
    ep = _num(r"epochs\s*[:=]\s*(\d+)", cast=int, default=None)
    lr = _num(r"lr\s*[:=]\s*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", cast=float, default=None)
    co = _num(r"collocation\s*[:=]\s*(\d+)", cast=int, default=None)
    bw = _num(r"bc[_\s]*weight\s*[:=]\s*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", cast=float, default=None)
    ve = _num(r"verbose[_\s]*every\s*[:=]\s*(\d+)", cast=int, default=None)
    out = defaults.copy()
    if ep is not None: out["epochs"] = int(max(1, min(ep, 2000)))
    if lr is not None: out["lr"] = float(max(1e-6, min(lr, 1e-1)))
    if co is not None: out["collocation"] = int(max(10, min(co, 5000)))
    if bw is not None: out["bc_weight"] = float(max(1.0, min(bw, 1e4)))
    if ve is not None: out["verbose_every"] = int(max(50, min(ve, 1000)))
    return out
class ExecAgent:
    def __init__(self, client: OpenAI, use_docker: bool, image: str = "pinn-sandbox"):
        self.client = client
        self.use_docker = use_docker
        self.image = image

    def ask_llm_for_fragments(self, history: str) -> Dict[str,str]:
        resp = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":USER_PROMPT.format(history=history)}
            ]
        )
        text = resp.choices[0].message.content.strip()

        # Expect three fenced blocks with tags or simple markers.
        # To keep robust, split by markers:
        def extract(tag, txt):
            start = txt.find(tag)
            if start<0: return ""
            start += len(tag)
            end = txt.find(f"<<<END_{tag}>>>")
            if end<0: return txt[start:].strip()
            return txt[start:end].strip()

        # Simple protocol:
        # <<<MODEL_DEF>>>
        # ...code...
        # <<<END_<<<MODEL_DEF>>>>  (or not, we handle missing end)
        # similar for <<<FORWARD_DEF>>> and <<<HYPERPARAMS>>>
        frags = {}
        for tag in ["<<<MODEL_DEF>>>","<<<FORWARD_DEF>>>","<<<HYPERPARAMS>>>"]:
            frags[tag] = extract(tag, text)
        return frags

    def render_template(self, fragments: Dict[str,str]) -> str:
        tpl = open(TEMPLATE_PATH,"r",encoding="utf-8").read()
        # defaults if LLM didn't provide
        default_model = """self.net = nn.Sequential(
    nn.Linear(1, 32), nn.Tanh(),
    nn.Linear(32, 32), nn.Tanh(),
    nn.Linear(32, 1)
)"""
        default_forward = "return self.net(x)"
        model_def = fragments.get("<<<MODEL_DEF>>>", "").strip() or default_model
        forward_def = fragments.get("<<<FORWARD_DEF>>>", "").strip() or default_forward
        # ensure forward returns something sensible
        if "return" not in forward_def:
            forward_def = default_forward

        # Replace model/forward blocks
        out = tpl.replace("        # <<<MODEL_DEF>>>", model_def)
        out = out.replace("        # <<<FORWARD_DEF>>>", forward_def)

        # Sanitize hyperparams
        raw_hp = fragments.get("<<<HYPERPARAMS>>>", "")
        safe_hp = _sanitize_hparams_fragment(raw_hp if isinstance(raw_hp, str) else "")
        # replace marker with valid Python literal
        hp_literal = ",".join([f'"{k}": {repr(v)}' for k,v in safe_hp.items()])
        out = out.replace("# __HYPERPARAMS_MARKER__", hp_literal)

        # Quick AST pre-parse to ensure validity before writing
        import ast
        try:
            ast.parse(out, filename="generated_impl.py")
        except Exception as e:
            # fallback to defaults entirely
            out = tpl.replace("        # <<<MODEL_DEF>>>", default_model)\
                     .replace("        # <<<FORWARD_DEF>>>", default_forward)\
                     .replace("# __HYPERPARAMS_MARKER__", '"epochs": 500, "lr": 1e-3, "collocation": 200, "bc_weight": 100.0, "verbose_every": 200')
        open(GEN_PATH,"w",encoding="utf-8").write(out)
        return out

    def run_once(self, history: str) -> ExecResult:
        frags = self.ask_llm_for_fragments(history)
        # safety check on fragments (text only)
        combined = "\n".join(frags.values())
        # AST guard will run on rendered file; LLM security will inspect text fragments later
        hits = []
        if hits:
            return ExecResult(ok=False, reason=f"Safety check failed (forbidden patterns): {hits}", fragments=frags)

        # render
        code = self.render_template(frags)

        llm_rewriter = LLMRewriterAgent(self.client)
        rw = llm_rewriter.rewrite(code)
        if rw.get("ok"):
            code = rw["code"]
            open(GEN_PATH, "w", encoding="utf-8").write(code)
        else:
            return ExecResult(ok=False, reason=f"LLM Rewriter failed: {rw.get('issues')}", fragments=frags)

        # safety check rendered file
        ast_result = check_file(pathlib.Path(GEN_PATH)) if 'pathlib' in globals() else __import__('safety_check').check_file(__import__('pathlib').Path(GEN_PATH))
        if not ast_result.get('ok', False):
            return ExecResult(ok=False, reason=f"AST safety check failed: {ast_result}", fragments=frags)
        # LLM-based security review on the rendered code
        llm_sec = LLMSecurityAgent(self.client)
        llm_report = llm_sec.review(code)
        if llm_report.get('grade') == 'BLOCK':
            return ExecResult(ok=False, reason=f"LLM security blocked: {llm_report}", fragments=frags)


        # execute (sandbox or local)
        if self.use_docker:
            rc, out, err = run_in_docker(self.image, ".", GEN_PATH, timeout=180)
        else:
            rc, out, err = run_local(GEN_PATH, timeout=180)

        metrics = None
        # try to read metrics file
        try:
            with open("results/metrics_latest.json","r",encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

        if rc!=0:
            return ExecResult(ok=False, reason=f"Runtime error (rc={rc})", stdout=out, stderr=err, metrics=metrics, fragments=frags)
        return ExecResult(ok=True, reason="ok", stdout=out, stderr=err, metrics=metrics, fragments=frags)

class EvalAgent:
    def summarize(self, exec_result: ExecResult) -> Dict[str,Any]:
        m = exec_result.metrics or {}
        final_loss = m.get("final_loss", None)
        rel_l2 = m.get("rel_l2", None)
        return {"final_loss": final_loss, "rel_l2": rel_l2}

class SecurityAgent:
    def check(self, fragments: Dict[str,str]) -> bool:
        txt = "\n".join(fragments.values())
        hits = check_text(txt)
        return len(hits)==0

# ---------------------- Orchestrator ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--use-docker", type=lambda x: x.lower()=="true", default=False)
    ap.add_argument("--image", type=str, default="pinn-sandbox")
    args = ap.parse_args()

    client = OpenAI()

    exec_agent = ExecAgent(client, use_docker=args.use_docker, image=args.image)
    eval_agent = EvalAgent()
    sec_agent = SecurityAgent()

    history_list = []
    best = {"rel_l2": 1e9, "config": None}

    for t in range(1, args.iters+1):
        history = json.dumps(history_list[-3:], ensure_ascii=False)  # last 3
        print(f"\n=== Iteration {t} (docker={args.use_docker}) ===")
        result = exec_agent.run_once(history)

        # log
        with open(f"results/log_iter_{t}.json","w",encoding="utf-8") as f:
            json.dump({
                "ok": result.ok, "reason": result.reason,
                "stdout": result.stdout, "stderr": result.stderr,
                "metrics": result.metrics, "fragments": result.fragments
            }, f, ensure_ascii=False, indent=2)

        if not result.ok:
            print("Execution failed:", result.reason)
            continue

        summary = eval_agent.summarize(result)
        print("Metrics:", summary)
        history_list.append(summary)

        # track best
        rel = summary.get("rel_l2", None)
        if rel is not None and rel < best["rel_l2"]:
            best["rel_l2"] = rel
            best["config"] = result.fragments
            # copy artifacts
            try:
                with open("results/metrics_latest.json","r",encoding="utf-8") as f:
                    m = json.load(f)
                with open("results/best_metrics.json","w",encoding="utf-8") as f:
                    json.dump(m, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
            try:
                import shutil
                shutil.copyfile("results/pred_latest.npz", "results/pred_best.npz")
            except Exception:
                pass

    # save final best
    if best["config"]:
        with open("results/best_config.json","w",encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
        print("\nBest rel_l2:", best["rel_l2"], "\nSaved to results/best_config.json")

if __name__ == "__main__":
    main()
