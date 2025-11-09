# agents/format_agent.py
import json, ast, re
from openai import OpenAI
from template import PINN_TEMPLATE

client = OpenAI()
MODEL = "gpt-5-mini"

def ask(system, user):
    r = client.chat.completions.create(model=MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return r.choices[0].message.content

def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.+?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()

class FormatAgent:
    @staticmethod
    def deterministic_fill(hp: dict, template: str = PINN_TEMPLATE):
        code = template
        for k, v in hp.items(): code = code.replace("{"+k+"}", str(v))
        return code

    @staticmethod
    def run(hp: dict, template: str = PINN_TEMPLATE):
        sys_prompt = ("You are a precise code formatter. Replace placeholders like {hidden_layers} "
                      "with JSON hyperparameters. Return FULL Python code only.")
        user_msg = f"JSON:\n{json.dumps(hp, indent=2)}\n\nTemplate:\n```\n{template}\n```"
        try:
            raw = ask(sys_prompt, user_msg)
            code = extract_code(raw)
            for k, v in hp.items():
                if "{"+k+"}" in code:
                    code = code.replace("{"+k+"}", str(v))
            ast.parse(code)
        except Exception:
            code = FormatAgent.deterministic_fill(hp, template)
            ast.parse(code)
        return code.rstrip()+"\n"
