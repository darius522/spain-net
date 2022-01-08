from local import x_umx as m
import torch
from importlib import *

from matplotlib import pyplot as plt
import matplotlib
import os

font = {'family' : 'normal',
    'size'   : 25}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def pos_enc_plot(dir='./paper',filename='pos_encs'):
    
    enc = m.PositionalEncodingAngle()
    x = torch.tensor([[-1,-20,-45,1,20,45]])
    encs = enc(x).squeeze().detach().cpu().numpy()
    fig, axes = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True, figsize=(12,8))
    
    axes[0,0].imshow(encs[0:1,:], aspect='auto')
    axes[0,0].set_title(r'$\alpha = -1$')
    axes[0,1].imshow(encs[1:2,:], aspect='auto')
    axes[0,1].set_title(r'$\alpha = -20$')
    axes[0,2].imshow(encs[2:3,:], aspect='auto')
    axes[0,2].set_title(r'$\alpha = -45$')
    axes[1,0].imshow(encs[3:4,:], aspect='auto')
    axes[1,0].set_title(r'$\alpha = +1$')
    axes[1,1].imshow(encs[4:5,:], aspect='auto')
    axes[1,1].set_title(r'$\alpha = +20$')
    axes[1,2].imshow(encs[5:6,:], aspect='auto')
    axes[1,2].set_title(r'$\alpha = +45$')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(dir,filename)+'.png')
    plt.savefig(os.path.join(dir,filename)+'.pdf')
    
if __name__ == "__main__":
    
    pos_enc_plot()