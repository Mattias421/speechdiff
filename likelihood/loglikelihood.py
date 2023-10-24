import numpy as np
import random
import torch
from torchdiffeq import odeint_adjoint as odeint
import os
import logging
logger = logging.getLogger(__name__)

@torch.no_grad()
def log_likelihood(model, x, t_min, t_max, beta_min, beta_max, atol=1e-3, rtol=1e-3, method='dopri5', seed=1):
    """
    Must use the get_score_model function from GradTTS to get the score model
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device='cuda:0').manual_seed(seed)

    v = torch.randint_like(x, 2) * 2 - 1

    class ODEfunc(torch.nn.Module):
        def __init__(self):
            super(ODEfunc, self).__init__()

        def forward(self, t, x):
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()

                beta_t = beta_min + t * (beta_max - beta_min)
                drift = (0.5 * beta_t[:, None, None] * (self.mu - x))
                diffusion = torch.sqrt(beta_t)

                score = model(x, t)

                drift = drift - diffusion[:, None, None] ** 2 * score * 0.5

                d = drift
                grad = torch.autograd.grad((d * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)

            return d.detach(), d_ll

    x = x.to(dtype=torch.float32)
    x_min = x, x.new_zeros([x.shape[0]])
    t = x.new_tensor([t_min, t_max])
    sol = odeint(ODEfunc().cuda(), x_min, t, atol=atol, rtol=rtol, method=method)
    latent, delta_ll = sol[0][-1], sol[1][-1]
    ll_prior = torch.distributions.Normal(0, t_max).log_prob(latent).flatten(1).sum(1)
    return ll_prior + delta_ll, ll_prior, delta_ll, latent