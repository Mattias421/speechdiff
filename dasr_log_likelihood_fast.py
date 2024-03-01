import json
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint
import hydra
import logging
logger = logging.getLogger(__name__)

@torch.no_grad()
def log_likelihood(model, x, t_min, t_max, beta_min, beta_max, spk, atol=1e-3, rtol=1e-3, method='dopri5'):
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

                score = model(x, t, spk)


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

def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)



class ScoreModel(torch.nn.Module):
    def __init__(self, estimator, mu, mask):
        super().__init__()
        self.y_mask = mask
        self.mu_y =  mu
        self.estimator = estimator

    def forward(self, x, t, spk):
        x_t = self.estimator(x=x, mask=self.y_mask, mu=self.mu_y, t=t, spk=spk)
        return x_t

def pad_audio(audio):
    audio_len = audio.shape[-1]

    max_len = fix_len_compatibility(audio.shape[-1])
    mask = sequence_mask(torch.LongTensor([[audio.size(-1)]]), 
                         max_len).to(audio)

    pad = torch.zeros((audio.size(0), audio.size(1), max_len)).to(audio)
    pad[:, :, :audio.size(-1)] = audio

    return pad, mask, audio_len

def bits_per_dim(ll, audio_len):
    return -ll / np.log(2) / (audio_len * 80)
    
def ll_speech(text, spk, gtts, cmu, 
              t_max: int = 1, beta_max: int = 20, device = 'cpu'):

    
    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(device)[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
    spk = torch.tensor(spk).to(device)
    gtts = gtts.to(device)

    y_enc, y_dec, attn = gtts.forward(x, x_lengths, n_timesteps=50, spk=spk)

    mu = y_enc
    spec = y_dec
    spec, mask, max_len = pad_audio(spec)
    mu, _, _ = pad_audio(mu)

    model = ScoreModel(gtts.decoder.estimator, mu, mask)
    ll, _, _, _ = log_likelihood(model,
                                  spec,
                                  0,
                                  1.0,
                                  0.05,
                                  beta_max,
                                 gtts.spk_emb(spk),
                                  atol=1e-5, rtol=1e-5)
    audio_len = y_dec.shape[-1]
    return bits_per_dim(ll, audio_len), ll, y_dec.shape

@hydra.main(version_base=None)
def main(cfg):
    logger.info('Preparing to calculate log likelihoods')

    gtts = GradTTS(cfg)
    state_dict = torch.load(cfg.eval.checkpoint, map_location=lambda loc, storage: loc)

    gtts.load_state_dict(state_dict)
    gtts.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gtts.to(device)

    cmu_path = '/fastdata/acq22mc/exp/diff_ll_audio/speechdiff/resources/cmu_dictionary' # hard coded path
    cmu = cmudict.CMUDict(cmu_path)

    results = []
    
    with open(cfg.data.test_filelist_path) as f:
        manifest = [l.rstrip().split('|')[1] for l in f][:50][::8]

    for line in manifest:
        for spk in range(cfg.data.n_spks):
            text = line

            logger.info(f'{text}|{spk}')
            beta_max = float(cfg.model.decoder.beta_max)
            
            ll, ll_raw, shape = ll_speech(text, spk, gtts, cmu, 1.0, beta_max, device)

            result = {'text':text,
                      'spk':spk,
                      'll':ll.item(),
                      'll_raw':ll_raw.item(),
                      'shape':shape}
            results.append(result)

            logger.info(result)

            total = 0
            for r in results:
                total += r['ll']
            logger.info(f'Current average LL is {total/len(results)}')

    with open('results.json', 'w') as f:
        json.dump(results, f)




if __name__ == "__main__":
    main()






