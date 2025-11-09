# agents/reflection_agent.py
import json
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-5-mini"

class ReflectionAgent:
    @staticmethod
    def run(hp: dict, eval_out: dict):
        sys = "You are a PINN coach. Suggest 4-6 bullet points for improving performance."
        msg = f"Hyperparameters:\n{json.dumps(hp,indent=2)}\n\nEvaluation:\n{json.dumps(eval_out,indent=2)}"
        r = client.chat.completions.create(model=MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":msg}])
        return r.choices[0].message.content.strip()
