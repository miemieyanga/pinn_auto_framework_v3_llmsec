# agents/execution_agent.py
import json, re
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-5-mini"

class ExecutionAgent:
    @staticmethod
    def plan(hp: dict):
        sys = "You are an execution planner. Return JSON {\"proceed\": bool, \"seed\": int, \"notes\": str}"
        user = f"Given hyperparameters:\n{json.dumps(hp,indent=2)}"
        out = client.chat.completions.create(model=MODEL, messages=[
            {"role":"system","content":sys},
            {"role":"user","content":user}
        ])
        txt = out.choices[0].message.content
        m = re.search(r"\{.*\}", txt, re.S)
        plan = json.loads(m.group(0)) if m else {"proceed": True, "seed": 42, "notes": ""}
        return plan
