# template.py
PINN_TEMPLATE = """\
import math, torch, torch.nn as nn, torch.autograd as autograd

HIDDEN_LAYERS = {hidden_layers}
HIDDEN_WIDTH  = {hidden_width}
ACTIVATION    = "{activation}"
OPTIMIZER     = "{optimizer}"
LR            = {lr}
EPOCHS        = {epochs}
N_COL         = {pde_collocation}
BC_WEIGHT     = {bc_weight}

def _act(name):
    return dict(tanh=nn.Tanh, relu=nn.ReLU, gelu=nn.GELU)[name]()

class PINN(nn.Module):
    def __init__(self, in_dim=1, out_dim=1):
        super().__init__()
        layers = [nn.Linear(in_dim, HIDDEN_WIDTH), _act(ACTIVATION)]
        for _ in range(HIDDEN_LAYERS-1):
            layers += [nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH), _act(ACTIVATION)]
        layers += [nn.Linear(HIDDEN_WIDTH, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def physics_loss(model, device):
    x = torch.rand(N_COL, 1, device=device)
    x.requires_grad_(True)
    y = model(x)
    dy_dx = autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    d2y_dx2 = autograd.grad(dy_dx, x, torch.ones_like(dy_dx), create_graph=True)[0]
    target = -(math.pi**2) * torch.sin(math.pi * x)
    return ((d2y_dx2 - target)**2).mean()

def boundary_loss(model, device):
    x0, x1 = torch.zeros(1,1,device=device), torch.ones(1,1,device=device)
    y0, y1 = model(x0), model(x1)
    return (y0**2 + y1**2).mean()

def train_and_evaluate(seed=42):
    torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=PINN().to(device)
    if OPTIMIZER=="adam":
        opt=torch.optim.Adam(model.parameters(),lr=LR)
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss=physics_loss(model,device)+BC_WEIGHT*boundary_loss(model,device)
            loss.backward(); opt.step()
    else:
        opt=torch.optim.LBFGS(model.parameters(),max_iter=EPOCHS,line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss=physics_loss(model,device)+BC_WEIGHT*boundary_loss(model,device)
            loss.backward(); return loss
        opt.step(closure)
    with torch.no_grad():
        xs=torch.linspace(0,1,200,device=device).unsqueeze(1)
        pred, true=model(xs),torch.sin(math.pi*xs)
        mse=torch.mean((pred-true)**2).item()
        mae=torch.mean(torch.abs(pred-true)).item()
    return {"mse":mse,"mae":mae}
"""
