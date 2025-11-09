# main.py
import json
import types
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Optional

from template import PINN_TEMPLATE
from agents.hyperparam_agent import HyperparamAgent
from agents.format_agent import FormatAgent
from agents.safety_agent import SafetyAgent
from agents.execution_agent import ExecutionAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reflection_agent import ReflectionAgent

@dataclass
class PipelineOutput:
    hp: Dict[str, Any]
    code: str
    safety: Dict[str, Any]
    evaluation: Dict[str, Any]
    reflection: str
    plan: Dict[str, Any]

class Runner:
    @staticmethod
    def run_module(code: str, seed: int = 42) -> Dict[str, Any]:
        mod = types.ModuleType("generated_pinn")
        exec(code, mod.__dict__)
        return mod.train_and_evaluate(seed=seed)

TASK_SPEC = (
    "Solve 1D Poisson: y''(x) = -pi^2 * sin(pi x) on x in [0,1], "
    "BC: y(0)=0, y(1)=0. True solution is sin(pi x). "
    "Train a PINN with physics + boundary loss and evaluate MSE/MAE."
)

def run_pipeline(task: str, feedback: Optional[str] = None) -> PipelineOutput:
    # 1) 超参（LLM）
    hp = HyperparamAgent.run(task, feedback)

    # 2) 代码生成（LLM + 本地兜底）
    code = FormatAgent.run(hp, template=PINN_TEMPLATE)

    # 3) 安检：只拦截 blockers，warnings 仅提示
    safety = SafetyAgent.run(code)
    if safety.get("blockers"):
        raise RuntimeError(f"Safety blockers: {json.dumps(safety['blockers'], ensure_ascii=False, indent=2)}")
    if safety.get("warnings"):
        print("\n[Safety] Warnings (not blocking):")
        print(json.dumps(safety["warnings"], ensure_ascii=False, indent=2))

    # 3.5) 执行规划（LLM）
    plan = ExecutionAgent.plan(hp)
    if not plan.get("proceed", True):
        evaluation = {"metrics": {}, "llm_comment": f"Execution skipped: {plan.get('notes','')}"}
        reflection = ReflectionAgent.run(hp, evaluation)
        return PipelineOutput(hp, code, safety, evaluation, reflection, plan)

    # 4) 本地执行
    try:
        metrics = Runner.run_module(code, seed=plan.get("seed", 42))
    except Exception as e:
        metrics = {"error": str(e), "trace": traceback.format_exc()}

    # 5) 评估（LLM）
    evaluation = EvaluationAgent.run(metrics)

    # 6) 反思（LLM）
    reflection = ReflectionAgent.run(hp, evaluation)

    return PipelineOutput(hp, code, safety, evaluation, reflection, plan)

if __name__ == "__main__":
    # ===== Round 1 =====
    print("\n================ ROUND 1 ================\n")
    round1 = run_pipeline(TASK_SPEC)
    print("\n--- Round 1 Hyperparams ---\n", json.dumps(round1.hp, indent=2))
    print("\n--- Round 1 Metrics ---\n", json.dumps(round1.evaluation.get("metrics", {}), indent=2))
    print("\n--- Round 1 Reflection ---\n", round1.reflection)

    # 准备反馈文本
    feedback_text = (
        f"Metrics: {json.dumps(round1.evaluation.get('metrics', {}), indent=2)}\n\n"
        f"LLM Evaluation: {round1.evaluation.get('llm_comment','')}\n\n"
        f"Reflection: {round1.reflection}"
    )

    # ===== Round 2 =====
    print("\n================ ROUND 2 (Feedback Guided) ================\n")
    round2 = run_pipeline(TASK_SPEC, feedback=feedback_text)
    print("\n--- Round 2 Hyperparams ---\n", json.dumps(round2.hp, indent=2))
    print("\n--- Round 2 Metrics ---\n", json.dumps(round2.evaluation.get("metrics", {}), indent=2))
    print("\n--- Round 2 Reflection ---\n", round2.reflection)

    print("\n✅ Two rounds completed.")
