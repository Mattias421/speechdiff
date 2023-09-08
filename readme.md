# Speech diff, a framework for diffusion applied to speech

The goal of this repository is to make it easy to experiment with audiodatasets and diffusion. This is powered by [hydra](https://hydra.cc/docs/intro/https://hydra.cc/), scorebased generative models (Song et al.) and Grad-TTS (Popov et al.).

At the moment it's currently a revamp of [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)

## Installation

Python 3.11.4

```
pip install -r requirements.txt
```

```
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

## Train a Grad-TTS model

Create a filelist in the form `audio_path|transcription|speaker_id`. Edit `config/data/data_example.yaml` to suit your dataset.

Run
```
python train_multi_speaker.py --config-name=config +data=data_example
```

Edit config as desired, or make use of hydra's [multirun](https://hydra.cc/docs/1.0/tutorials/basic/running_your_app/multi-run/#internaldocs-banner) utility.

## Evaluate a TTS model

Coming soon

## Compute log-likelihoods

Coming soon

## Citation

```
@Misc{Cross2023SpeechDiff,
  author =       {Mattias Cross},
  title =        {Speech diff, a framework for diffusion applied to speech},
  howpublished = {Github},
  year =         {2023},
  url =          {https://github.com/Mattias421/speech-diff}
}
```