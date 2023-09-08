import numpy as np
from tqdm import tqdm
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params_tedlium_spk as params # changed
from model import GradTTS
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

import pyworld as pw
import pysptk
import soundfile as sf
from fastdtw import fastdtw
from scipy import spatial

test_filelist_path = params.test_filelist_path
valid_filelist_path = params.valid_filelist_path
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

def fastdtw_distance_on_mels(ref_mel, pred_mel):
    ref_mel = np.asarray(ref_mel)
    pred_mel = np.asarray(pred_mel)
    _, path = fastdtw(ref_mel, pred_mel, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    h_dtw = ref_mel[twf[0]]
    r_dtw = pred_mel[twf[1]]
    dtw_distance = np.sum((h_dtw-r_dtw)**2,1)
    mel_dtw = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * dtw_distance), 0)
    return mel_dtw

def _same_t_in_true_and_est(func):
    def new_func(true_t, true_f, est_t, est_f):
        assert type(true_t) is np.ndarray
        assert type(true_f) is np.ndarray
        assert type(est_t) is np.ndarray
        assert type(est_f) is np.ndarray

        interpolated_f = interp1d(est_t, est_f, bounds_error=False, kind='nearest', fill_value=0)(true_t)
        return func(true_t, true_f, true_t, interpolated_f)

    return new_func

###@_same_t_in_true_and_est
def gross_pitch_error(true_f, est_f):
    """The relative frequency in percent of pitch estimates that are outside a threshold around the true pitch. Only frames that are considered pitched by both the ground truth and the estimator (if
    applicable) are considered.    """

    correct_frames = _true_voiced_frames(true_f, est_f)
    gross_pitch_error_frames = _gross_pitch_error_frames(true_f, est_f)
    return np.sum(gross_pitch_error_frames) / np.sum(correct_frames)

def _gross_pitch_error_frames(true_f, est_f, eps=1e-8):
    voiced_frames = _true_voiced_frames(true_f, est_f)
    true_f_p_eps = [x + eps for x in true_f]
    pitch_error_frames = np.abs(est_f / true_f_p_eps - 1) > 0.2
    return voiced_frames & pitch_error_frames

def _true_voiced_frames(true_f, est_f):
    return (est_f != 0) & (true_f != 0)

def _voicing_decision_error_frames(true_f, est_f):
    return (est_f != 0) != (true_f != 0)

##@_same_t_in_true_and_est
def f0_frame_error(true_f, est_f):
    gross_pitch_error_frames = _gross_pitch_error_frames(true_f, est_f)
    voicing_decision_error_frames = _voicing_decision_error_frames(true_f, est_f)
    return (np.sum(gross_pitch_error_frames) + np.sum(voicing_decision_error_frames)) / (len(true_f))

###@_same_t_in_true_and_est
def voicing_decision_error(true_f, est_f):
    voicing_decision_error_frames = _voicing_decision_error_frames(true_f, est_f)
    return np.sum(voicing_decision_error_frames) / (len(true_f))

def sptk_extract(x: np.ndarray, fs: int, n_fft: int = 512, n_shift: int = 256,  mcep_dim: int = 25,  mcep_alpha: float = 0.41, is_padding: bool = False,) -> np.ndarray:
    """Extract SPTK-based mel-cepstrum.
    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).

    """
    # perform padding
    if is_padding:
        n_pad = n_fft - (len(x) - n_fft) % n_shift
        x = np.pad(x, (0, n_pad), "reflect")
    # get number of frames
    n_frame = (len(x) - n_fft) // n_shift + 1
    # get window function
    win = pysptk.sptk.hamming(n_fft)
    # check mcep and alpha
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    # calculate spectrogram
    mcep = [ pysptk.mcep(x[n_shift * i : n_shift * i + n_fft] * win,  mcep_dim,  mcep_alpha,   eps=1e-6,etype=1,  )  for i in range(n_frame) ]

    return np.stack(mcep)


def world_extract(x: np.ndarray, fs: int, f0min: int = 40, f0max: int = 800, n_fft: int = 512, n_shift: int = 256, mcep_dim: int = 25, mcep_alpha: float = 0.41,) -> np.ndarray:
    """Extract World-based acoustic features.
    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).
        ndarray: F0 sequence (N,).

    """
    # extract features
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(x,fs,f0_floor=f0min, f0_ceil=f0max, frame_period=n_shift / fs * 1000, )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)

    return mcep, f0


def obtainMetrics(pred_x,refFile, pred_mel, ref_mel):
    # pred_x, pred_fs = sf.read(predFile,dtype="float32")  ##int16")
    ref_x, ref_fs = sf.read(refFile,dtype="float32")    ##int16")
    fs = ref_fs
    pred_mcep, pred_f0 = world_extract(x=pred_x, fs=fs, f0min=70, f0max=400, n_fft=512, n_shift=256, mcep_dim=34, mcep_alpha=0.45) ##, mcep_dim=34, mcep_alpha=0.45)
    ref_mcep, ref_f0 = world_extract(x=ref_x, fs=fs, f0min=70, f0max=400, n_fft=512, n_shift=256, mcep_dim=34, mcep_alpha=0.45) ##, mcep_dim=34, mcep_alpha=0.45)
    # DTW
    _, path = fastdtw(pred_mcep, ref_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    pred_f0_dtw = pred_f0[twf[0]]
    ref_f0_dtw = ref_f0[twf[1]]
    ##Get voiced part
    nonzero_idxs = np.where((pred_f0_dtw!=0)&(ref_f0_dtw!=0))[0]
    pred_f0_dtw_voiced = np.log(pred_f0_dtw[nonzero_idxs])
    ref_f0_dtw_voiced = np.log(ref_f0_dtw[nonzero_idxs])
    ##log F0 RMSE
    log_f0_rmse = np.sqrt(np.mean((pred_f0_dtw_voiced-ref_f0_dtw_voiced)**2))
    gen_mcep = sptk_extract(x=pred_x, fs=fs, n_fft=512,n_shift=256,mcep_dim=34,mcep_alpha=0.45)
    gt_mcep = sptk_extract(x=ref_x, fs=fs, n_fft=512,n_shift=256, mcep_dim=34,mcep_alpha=0.45)
    # DTW
    _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    gen_mcep_dtw = gen_mcep[twf[0]]
    gt_mcep_dtw = gt_mcep[twf[1]]
    ##MCD
    diff2sum =  np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
    ###Compute DTW distance between mel-spectrograms
    mel_dtw_distance = 0 # fastdtw_distance_on_mels(ref_mel.squeeze(0).cpu(), pred_mel.squeeze(0).cpu())
    gpe = gross_pitch_error(ref_f0_dtw, pred_f0_dtw)
    vde = voicing_decision_error(ref_f0_dtw, pred_f0_dtw)
    ffe = f0_frame_error(ref_f0_dtw, pred_f0_dtw)

    # print("log_f0_rmse is ", log_f0_rmse, "MCD is ", mcd, "mel-dtw is ", mel_dtw_distance, "gpe is ", gpe, "vde is ", vde, "ffe is ", ffe)

    return  log_f0_rmse, mcd, mel_dtw_distance, gpe, vde, ffe

def main(args):
    device = torch.device(f'cuda:{args.gpu}')

    test_dataset = TextMelSpeakerDataset(test_filelist_path, cmudict_path, add_blank,
                                          n_fft, n_feats, sample_rate, hop_length,
                                          win_length, f_min, f_max)
    batch_collate = TextMelSpeakerBatchCollate()
    loader = DataLoader(dataset=test_dataset, batch_size=1,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=30, shuffle=True)
    
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.to(device).eval()

    HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
    HIFIGAN_CHECKPT = './checkpts/hifigan.pt'
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.to(device).eval()
    vocoder.remove_weight_norm()

    n_evaluations = 50

    results = np.zeros((n_evaluations, 6))

    with torch.no_grad():
            for i in range(n_evaluations):
                batch = test_dataset[i]
                ref_file = test_dataset.filelist[i][0]

                x = batch['x'].to(device)[None]
                x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
                spk = batch['spk'].to(device)
                ref_mel = batch['y']

                y_enc, y_dec, attn = generator(x, x_lengths, n_timesteps=50, spk=spk)
                audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

                result = obtainMetrics(audio, ref_file, y_dec, ref_mel)
                results[i] = result

    print(np.mean(results, axis=0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    # parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    # parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    # parser.add_argument('-o', '--output', type=str, required=True, help='output file')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='Choose which GPU to use')
    args = parser.parse_args()

    main(args)