
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
import hydra
import os

from model import GradTTS
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols
from model.utils import fix_len_compatibility

from nemo.collections.tts.models import HifiGanModel
from scipy.io.wavfile import write

from utils import intersperse, save_plot
from text import text_to_sequence, cmudict

@hydra.main(version_base=None, config_path='./config')
def main(cfg: DictConfig):
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    device = torch.device(f'cuda:{cfg.training.gpu}')


    print('Initializing model...')
    model = GradTTS(cfg)
    model.load_state_dict(torch.load(cfg.eval.checkpoint, map_location=lambda loc, storage: loc))
    model.to(device).eval()
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    print('Initializing vocoder...')
    vocoder = HifiGanModel.from_pretrained(model_name='nvidia/tts_hifigan')


    print(f'Synthesizing text...', end=' ')

    text = 'Now I am become speech, the destroyer of text'

    cmu = cmudict.CMUDict('./resources/cmu_dictionary')

    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
    spk = cfg.eval.spk

    y_enc, y_dec, attn = model.forward(x, x_lengths, n_timesteps=cfg.eval.timesteps, spk=spk)

    audio = vocoder.convert_spectrogram_to_audio(spec=y_dec)
    audio = audio.squeeze().to('cpu').numpy()

    out_path = f'{cfg.eval.out_dir}/output.wav'

    write(out_path, 22050, audio)
    save_plot(y_dec[0].cpu(), 'output_spec')


if __name__ == '__main__':
    main()