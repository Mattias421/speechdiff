
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

@hydra.main(version_base=None, config_path='./config')
def main(cfg: DictConfig):
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    device = torch.device(f'cuda:{cfg.training.gpu}')

    print('Initializing data loaders...')
    dataset = TextMelSpeakerDataset(cfg.eval.split, cfg)
    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=dataset, batch_size=1,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=cfg.training.num_workers)

    print('Initializing model...')
    print(cfg.model.spk_emb_dim)
    print(cfg.data.n_spks)
    model = GradTTS(cfg)
    model.load_state_dict(torch.load(cfg.eval.checkpoint, map_location=lambda loc, storage: loc))
    model.to(device).eval()
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    print('Initializing vocoder...')
    vocoder = HifiGanModel.from_pretrained(model_name='nvidia/tts_hifigan')

    filelist = []

    with torch.no_grad():
        with tqdm(loader, total=len(dataset)) as progress_bar:
            for i, batch in enumerate(progress_bar):
                x, x_lengths = batch['x'].to(device), batch['x_lengths'].to(device)
                spk = batch['spk'].to(device)

                y_enc, y_dec, attn = model.forward(x, x_lengths, n_timesteps=cfg.eval.timesteps, spk=spk)

                audio = vocoder.convert_spectrogram_to_audio(spec=y_dec)
                audio = audio.squeeze().to('cpu').numpy()

                out_path = f'{cfg.eval.out_dir}/{i}.wav'

                write(out_path, 22050, audio)

                filelist.append(out_path)

                with open(f'{cfg.eval.split}_preds.txt', 'a') as f:
                    f.write(os.path.abspath(out_path) + '\n')


if __name__ == '__main__':
    main()