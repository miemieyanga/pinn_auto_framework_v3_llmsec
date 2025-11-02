# PINN Auto-Tuner (LLM Orchestrated)

This is a minimal, safe-ish framework that uses an LLM to automatically **write/adjust a PINN** (Poisson 1D example), 
**check safety**, **execute in a sandbox (Docker optional)**, and **evaluate metrics** (loss and relative L2).

## Structure
- `main.py` — Orchestrator with three agents:
  - **ExecAgent**: asks the LLM to produce a `model_spec` + hyperparameters and **generates code** into `generated_impl.py` using a template.
  - **SecurityAgent**: static checks the generated code (no network, no subprocess, etc.).
  - **EvalAgent**: runs the code, reads JSON metrics, and suggests the next hyperparameters.
- `pinn_template.py` — Safe training template the LLM plugs its `Model` and hyperparameters into.
- `safety_check.py` — Very basic static scanning (regex) before executing generated code.
- `sandbox.py` — Optional Docker-based sandbox runner. Falls back to local execution if Docker is not available.
- `task_config.json` — Task configuration (Poisson 1D); you can extend to other PDEs.
- `results/` — Iteration logs + best config & artifacts.

## Quick Start
1) (Optional) Create and activate a virtual environment.
2) Install deps:
```
pip install -r requirements.txt
```
3) Set your OpenAI API key in env:
```
$env:OPENAI_API_KEY="sk-..."   # PowerShell
# or
export OPENAI_API_KEY="sk-..." # bash
```
4) (Optional, recommended) Build a sandbox image once:
```
docker build -t pinn-sandbox -f Dockerfile.sandbox .
```
5) Run tuning for 5 iterations:
```
python main.py --iters 5 --use-docker false
```
(If you've built the sandbox image, set `--use-docker true` to enable isolated execution.)

## Notes
- The LLM is constrained by prompts to only emit safe code segments (model + hyperparams) which we insert into a vetted template.
- The template **never** imports network libs, never shells out, and writes results to `./results` only.
- `safety_check.py` is conservative; extend it if necessary (e.g., Bandit, AST-based checks).
- You can adapt `pinn_template.py` to Burgers' equation or other PDEs; just keep the interfaces stable.


## LLM-based Security Agent
- We added `llm_security.py` which uses the LLM to **grade** rendered code as PASS/WARN/BLOCK with reasons and suggested minimal patches.
- Orchestrator logic:
  1) **AST guard** (`safety_check.py`) blocks hard violations (imports/calls).
  2) **LLM review** flags subtler risks (exfiltration attempts, dynamic code, path traversal), returns a JSON report.
  3) Local runs will block on WARN by default (you can flip that policy in `main.py`). Docker runs proceed on WARN but still block on BLOCK.
