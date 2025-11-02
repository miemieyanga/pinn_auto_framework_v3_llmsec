import os
import io
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------- Problem Definition (Poisson 1D) ----------------------
pi = math.pi

def f(x):
    # u(x) = sin(pi x) => u'' = -pi^2 sin(pi x), so bring to residual u'' + pi^2 sin(pi x) = 0
    return (pi ** 2) * torch.sin(pi * x)


def u_exact(x):
    return torch.sin(pi * x)

# ---------------------- MODEL ----------------------------------------------
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # simple 3-layer MLP
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # expect x shape (N,1)
        return self.net(x)

HYPERPARAMS = {
    "epochs": 500,
    "lr": 1e-3,
    "collocation": 200,
    "bc_weight": 100.0,
    "verbose_every": 200
}

# ---------------------- PINN core -----------------------------------------
def physics_residual(model, x_collocation):
    # ensure we have a tensor that requires grad
    x = x_collocation.clone().detach().requires_grad_(True)
    u = model(x)
    # u has shape (N,1); compute gradient wrt x
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # grad_u has shape (N,1); compute second derivative
    grad_u_x = torch.autograd.grad(
        outputs=grad_u,
        inputs=x,
        grad_outputs=torch.ones_like(grad_u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    res = grad_u_x + f(x)
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

    for epoch in range(1, epochs + 1):
        # collocation points in interior (0,1)
        x_coll = torch.rand(collocation, 1, dtype=torch.float32)
        res = physics_residual(model, x_coll)
        loss_phys = torch.mean(res ** 2)

        # boundary conditions u(0)=0, u(1)=0
        xb = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        ub = model(xb)
        loss_b = torch.mean(ub ** 2)

        loss = loss_phys + bc_weight * loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        if verbose_every > 0 and (epoch % verbose_every == 0 or epoch == 1 or epoch == epochs):
            with torch.no_grad():
                xt = torch.linspace(0, 1, 200).unsqueeze(1)
                up = model(xt).detach().numpy().squeeze()
                ut = u_exact(xt).detach().numpy().squeeze()
                l2 = float(np.linalg.norm(up - ut) / np.linalg.norm(ut))
            print(f"[epoch={epoch}] loss={loss.item():.3e} relL2={l2:.3e}")

    # final eval
    xt = torch.linspace(0, 1, 200).unsqueeze(1)
    with torch.no_grad():
        up = model(xt).detach().numpy().squeeze()
        ut = u_exact(xt).detach().numpy().squeeze()
    rel_l2 = float(np.linalg.norm(up - ut) / np.linalg.norm(ut))

    # save artifacts safely with size checks (<10MB)
    results_dir = os.path.join('.', 'results')
    try:
        os.makedirs(results_dir, exist_ok=True)
    except Exception as e:
        print("Warning: could not create results directory:", e)

    # prepare arrays
    x_np = xt.detach().numpy().squeeze()
    u_pred = np.asarray(up)
    u_ex = np.asarray(ut)

    # save .npz in memory first to check size
    try:
        buf = io.BytesIO()
        # numpy will write a zip archive into the buffer
        np.savez(buf, x=x_np, u_pred=u_pred, u_exact=u_ex)
        size_bytes = buf.getbuffer().nbytes
        max_bytes = 10 * 1024 * 1024  # 10MB
        if size_bytes < max_bytes:
            try:
                with open(os.path.join(results_dir, 'pred_latest.npz'), 'wb') as f:
                    f.write(buf.getvalue())
            except Exception as e:
                print("Warning: failed to write pred_latest.npz:", e)
        else:
            print(f"Skipping saving pred_latest.npz (size {size_bytes} bytes >= {max_bytes} bytes)")
    except Exception as e:
        print("Warning: failed to prepare pred_latest.npz in memory:", e)

    # save metrics JSON
    metrics = {
        "final_loss": float(losses[-1]) if len(losses) > 0 else None,
        "rel_l2": rel_l2,
        "loss_curve": losses[-200:]
    }
    try:
        json_bytes = json.dumps(metrics, ensure_ascii=False).encode('utf-8')
        if len(json_bytes) < 10 * 1024 * 1024:
            try:
                with open(os.path.join(results_dir, 'metrics_latest.json'), 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, ensure_ascii=False)
            except Exception as e:
                print("Warning: failed to write metrics_latest.json:", e)
        else:
            print("Skipping saving metrics_latest.json: JSON payload too large")
    except Exception as e:
        print("Warning: failed to prepare metrics JSON:", e)

    print(f"Training completed in {time.time() - t0:.2f} seconds. final relL2={rel_l2:.3e}")
    return {"final_loss": float(losses[-1]) if len(losses) > 0 else None, "rel_l2": rel_l2}


if __name__ == "__main__":
    result = train_and_eval()
    try:
        print(json.dumps(result))
    except Exception:
        print(result)
