
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
import soundfile as sf

@hydra.main(version_base=None, config_path='./config')
def main(cfg: DictConfig):
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    device = torch.device(f'cuda:{cfg.training.gpu}')

    print('Initializing logger...')
    log_dir = cfg.training.log_dir
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    dataset = TextMelSpeakerDataset(cfg.eval.split, cfg)
    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=dataset, batch_size=cfg.training.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=cfg.training.num_workers, shuffle=True)

    print('Initializing model...')
    model = GradTTS(cfg)
    model.load_state_dict(torch.load(cfg.eval.checkpoint, map_location=lambda loc, storage: loc))
    model.to(device).eval()
    print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))

    print('Initializing vocoder...')
    vocoder = HifiGanModel.from_pretrained(model_name='nvidia/tts_hifigan')
if __name__ == '__main__':
    main()