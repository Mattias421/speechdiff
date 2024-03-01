import pandas as pd
import torch
import argparse

def bpd(ll, h, w):
    return - ll * torch.log2(torch.exp(1)) * ((h * w)) ** -1

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

data = pd.read_json(args.filename)

for spk in data['spk'].unique():
    avg = data[data['spk'] == spk]['ll'].mean()

    print(avg)

    with open('avg.txt', 'a') as f:
        print(avg, file=f)
