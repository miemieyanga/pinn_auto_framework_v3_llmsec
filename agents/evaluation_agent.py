# agents/evaluation_agent.py
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-5-mini"

class EvaluationAgent:
    @staticmethod
    def run(metrics: dict):
        sys = "You are a concise evaluator. Describe performance in one short paragraph."
        msg = f"Metrics: {metrics}"
        r = client.chat.completions.create(model=MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":msg}])
        comment = r.choices[0].message.content.strip()
        return {"metrics": metrics, "llm_comment": comment}
