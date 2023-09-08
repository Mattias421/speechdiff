import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path  

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.io.wavfile import write
import params_tedlium as params # changed
from model import GradTTS
from data import TextMelZeroSpeakerDataset, TextMelZeroSpeakerBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols

test_filelist_path = params.test_filelist_path
test_spk = params.test_spk
cmudict_path = params.cmudict_path
add_blank = params.add_blank
n_spks = params.n_spks
spk_emb_dim = params.spk_emb_dim

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale

from speechbrain.pretrained import EncoderClassifier, HIFIGAN
import torchaudio
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmpdir", run_opts={'device':'cuda'})



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='path to dir to save generated speech')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    args = parser.parse_args()

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    train_dataset = TextMelZeroSpeakerDataset(test_filelist_path, test_spk, cmudict_path, add_blank,
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max)
    batch_collate = TextMelZeroSpeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=8, shuffle=True)
    
    print('Initializing Grad-TTS...')
    # n_spks should equal -1
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')


    with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for i, batch in enumerate(progress_bar):
                generator.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'], batch['y_lengths'].cuda()
                spk = batch['spk'].cuda()

                y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                       stoc=False, spk=spk, length_scale=1)
                
                for j, mfcc in enumerate(y_dec):
                    length = torch.sum(mfcc != 0, axis=1)[0]
                    audio = hifi_gan.decode_batch(mfcc[:, :length])
                    audio = (audio.cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

                    Path(f'{args.output_dir}/{i}').mkdir(parents=True, exist_ok=True)
                    write(f'{args.output_dir}/{i}/{j}.wav', 16000, audio)

                    save_plot(mfcc[:, :length].cpu(), f'{args.output_dir}/{i}/{j}_gen.png')
                    save_plot(y[j][:, :length], f'{args.output_dir}/{i}/{j}_ref.png')