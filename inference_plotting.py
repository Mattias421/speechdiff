
import numpy as np

import torch

from omegaconf import DictConfig, OmegaConf
import hydra

from model import GradTTS
from text.symbols import symbols
import os

# from nemo.collections.tts.models import HifiGanModel

from utils import intersperse, save_plot
from text import text_to_sequence, cmudict

@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig):
    device = torch.device('cpu')

    print(cfg)


    print('Initializing model...')
    model = GradTTS(cfg)
    model.load_state_dict(torch.load(cfg.eval.checkpoint, map_location=lambda loc, storage: loc))
    model.to(device).eval()
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    print('Initializing vocoder...')
    # vocoder = HifiGanModel.from_pretrained(model_name='nvidia/tts_hifigan')

    print(f'Synthesizing text...', end=' ')

    text = 'Now I am become speech, the destroyer of text'

    cmu = cmudict.CMUDict(cfg.data.cmudict_path)

    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(device)[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
    spk = cfg.eval.spk
    spk = torch.tensor(spk).to(device)

    y_enc, y_dec, attn = model.forward(x, x_lengths, n_timesteps=cfg.eval.timesteps, spk=spk)

    # audio = vocoder.convert_spectrogram_to_audio(spec=y_dec)

    print(f'cwd is {os.getcwd()}')
    print(f'Saving plots to {cfg.eval.out_dir}')
    os.mkdir(cfg.eval.out_dir)
    X_T = y_enc[0].cpu() + torch.randn_like(y_enc[0].cpu())
    save_plot(X_T, f'{cfg.eval.out_dir}/input_spec')
    save_plot(y_dec[0].cpu(), f'{cfg.eval.out_dir}/output_spec')


if __name__ == '__main__':
    main()
