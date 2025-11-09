# agents/hyperparam_agent.py
import json, re
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-5-mini"

HP_CHOICES = {
    "hidden_layers": [1, 2, 3, 4],
    "hidden_width": [16, 32, 64, 128],
    "activation": ["tanh", "relu", "gelu"],
    "optimizer": ["adam", "lbfgs"],
    "lr": [1e-3, 5e-4, 1e-4],
    "epochs": [500, 1000, 2000],
    "pde_collocation": [64, 128, 256, 512],
    "bc_weight": [0.1, 0.5, 1.0, 2.0]
}

def ask(system, user):
    r = client.chat.completions.create(model=MODEL, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
    return r.choices[0].message.content

class HyperparamAgent:
    @staticmethod
    def run(task: str, feedback: str = None):
        sys = "You are a cautious PINN hyperparameter selector. Output ONLY valid JSON."
        msg = f"Task:\n{task}\nFeedback:{feedback or 'None'}\nChoices:\n{json.dumps(HP_CHOICES,indent=2)}"
        out = ask(sys, msg)
        m = re.search(r"\{.*\}", out, re.S)
        hp = json.loads(m.group(0)) if m else {}
        for k, v in HP_CHOICES.items():
            if hp.get(k) not in v:
                hp[k] = v[0]
        return hp
