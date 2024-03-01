import numpy as np
import random
import torch
from torchdiffeq import odeint_adjoint as odeint
import os
import logging
logger = logging.getLogger(__name__)

@torch.no_grad()
def log_likelihood(model, x, t_min, t_max, beta_min, beta_max, atol=1e-3, rtol=1e-3, method='dopri5'):
    """
    Must use the get_score_model function from GradTTS to get the score model
    """
    v = torch.randint_like(x, 2) * 2 - 1

    class ODEfunc(torch.nn.Module):
        def __init__(self):
            super(ODEfunc, self).__init__()
            self.t_upper = t_max if t_max > t_min else t_min
            self.t_lower = t_min if t_min < t_max else t_max

        def forward(self, t, x):
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()

                x = x * model.y_mask
                
                if t > self.t_upper:
                    t = self.t_upper
                elif t < self.t_lower:
                    t = self.t_lower

                t = t * torch.ones(x.size(0)).to(x)
                beta_t = beta_min + t * (beta_max - beta_min)
                drift = (0.5 * beta_t[:, None, None] * (model.mu_y - x))
                diffusion = torch.sqrt(beta_t)

                score = model(x, t)

                drift = drift - diffusion[:, None, None] ** 2 * score * 0.5
                
                d = drift * model.y_mask
                grad = torch.autograd.grad((d * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)

            return d.detach(), d_ll

    x = x * model.y_mask
    x_min = x, x.new_zeros([x.shape[0]])
    t = x.new_tensor([t_min, t_max])
    sol = odeint(ODEfunc().cuda(), x_min, t, atol=atol, rtol=rtol, method=method)
    latent, delta_ll = sol[0][-1], sol[1][-1]
    # ll_prior = torch.distributions.Normal(model.mu_y, t_max).log_prob(latent).flatten(1).sum(1)
    shape = latent.shape
    N = np.prod(shape[1:])
    ll_prior = -N / 2. * np.log(2 * np.pi) - torch.sum((latent - model.mu_y) ** 2, dim=(1, 2)) / 2.
    return ll_prior + delta_ll, ll_prior, delta_ll, latent

@torch.no_grad()
def ode_sample(model, x, t_min, t_max, beta_min, beta_max, atol=1e-3, rtol=1e-3, method='dopri5'):
    """
    Must use the get_score_model function from GradTTS to get the score model
    """

    class ODEfunc(torch.nn.Module):
        def __init__(self):
            super(ODEfunc, self).__init__()

            self.t_upper = t_max if t_max > t_min else t_min
            self.t_lower = t_min if t_min < t_max else t_max

        def forward(self, t, x):
            with torch.enable_grad():
                zeros = x[1]
                x = x[0]

                x = x * model.y_mask

                if t > self.t_upper:
                    t = self.t_upper
                elif t < self.t_lower:
                    t = self.t_lower

                t = t * torch.ones(x.size(0)).to(x)
                beta_t = beta_min + t * (beta_max - beta_min)
                drift = (0.5 * beta_t[:, None, None] * (model.mu_y - x))
                diffusion = torch.sqrt(beta_t)

                score = model(x, t)

                drift = drift - diffusion[:, None, None] ** 2 * (score) * 0.5
                
                d = drift * model.y_mask

            return d.detach(), zeros

    # x = x * model.y_mask
    x_min = x, x.new_zeros([x.shape[0]])
    t = x.new_tensor([t_min, t_max])
    sol = odeint(ODEfunc().cuda(), x_min, t, atol=atol, rtol=rtol, method=method)
    latent = sol[0][-1]
    return latent