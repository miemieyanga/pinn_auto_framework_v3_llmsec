\
# GENERATED IMPLEMENTATION (by ExecAgent)
# This file is regenerated each iteration. It must remain safe: no network, no shell, no system calls.

import os, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------- Problem Definition (Poisson 1D) ----------------------
pi = math.pi

def f(x):
    # u(x) = sin(pi x) => u'' = -pi^2 sin(pi x), so bring to residual u'' + pi^2 sin(pi x) = 0
    return (pi**2) * torch.sin(pi * x)

def u_exact(x):
    return torch.sin(pi * x)

# ---------------------- MODEL (inserted by LLM) ------------------------------
# The ExecAgent must fill the following class and HYPERPARAMS dict safely.
# Allowed: torch, torch.nn, torch.optim, math, numpy.
# Forbidden: disallowed risky libs & system calls (enforced by safety checker).
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # <<<MODEL_DEF>>>

    def forward(self, x):
        # <<<FORWARD_DEF>>>
        pass

HYPERPARAMS = {
    # <<<HYPERPARAMS>>>
}

# ---------------------- PINN core --------------------------------------------
def physics_residual(model, x_collocation):
    x_collocation = x_collocation.requires_grad_(True)
    u = model(x_collocation)
    u_x = torch.autograd.grad(u, x_collocation, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_collocation, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    res = u_xx + f(x_collocation)
    return res

def train_and_eval(seed=42):
    # defaults with fallback
    epochs = int(HYPERPARAMS.get("epochs", 1500))
    lr = float(HYPERPARAMS.get("lr", 1e-3))
    collocation = int(HYPERPARAMS.get("collocation", 200))
    bc_weight = float(HYPERPARAMS.get("bc_weight", 100.0))
    verbose_every = int(HYPERPARAMS.get("verbose_every", 200))

    torch.manual_seed(seed)
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    losses = []

    for epoch in range(1, epochs+1):
        x_coll = torch.rand(collocation, 1, dtype=torch.float32)
        res = physics_residual(model, x_coll)
        loss_phys = torch.mean(res**2)

        xb = torch.tensor([[0.0],[1.0]], dtype=torch.float32)
        ub = model(xb)
        loss_b = torch.mean(ub**2)

        loss = loss_phys + bc_weight * loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        if verbose_every>0 and (epoch % verbose_every == 0 or epoch==1 or epoch==epochs):
            with torch.no_grad():
                xt = torch.linspace(0,1,200).unsqueeze(1)
                up = model(xt).detach().numpy().squeeze()
                ut = u_exact(xt).numpy().squeeze()
                l2 = float(np.linalg.norm(up - ut) / np.linalg.norm(ut))
            print(f"[epoch={epoch}] loss={loss.item():.3e} relL2={l2:.3e}")

    # final eval
    xt = torch.linspace(0,1,200).unsqueeze(1)
    up = model(xt).detach().numpy().squeeze()
    ut = u_exact(xt).numpy().squeeze()
    rel_l2 = float(np.linalg.norm(up - ut) / np.linalg.norm(ut))

    # save artifacts
    os.makedirs("results", exist_ok=True)
    np.savez("results/pred_latest.npz", x=xt.numpy().squeeze(), u_pred=up, u_exact=ut)
    with open("results/metrics_latest.json","w",encoding="utf-8") as f:
        json.dump({"final_loss": float(losses[-1]), "rel_l2": rel_l2, "loss_curve": losses[-200:]}, f)

    return {"final_loss": float(losses[-1]), "rel_l2": rel_l2}

if __name__ == "__main__":
    m = train_and_eval()
    print(json.dumps(m))
