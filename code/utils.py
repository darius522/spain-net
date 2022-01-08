from asteroid import data
from types import SimpleNamespace

from train import get_disparity_map
from local.x_umx import _STFT, _Spectrogram
import torch
import soundfile as sf

from asteroid.data import MUSDB18Dataset
from local.slakh_dataset import SlakhDataset
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
import yaml
import os
from pathlib import Path


def get_ild_stats(dir, dataset='musdb', split='train', sources=['bass','drums','other','vocals'],
                  means_src=[-35,15,0,-20], stds_src=[5,10,15,20],
                  win=4096, in_chan=4096, nhop=1024, mono=False):
    
    encoder = torch.nn.Sequential(_STFT(window_length=win, n_fft=in_chan, n_hop=nhop),_Spectrogram(spec_power=1.0, mono=(mono==1)))
    
    args = SimpleNamespace()
    args = {'root':dir,'sources':sources, 'targets':sources, 'split':split, 'means':means_src, 'stds':stds_src, 'valid_tracks':100}
    
    if dataset=='musdb':
        train_dataset = MUSDB18Dataset(**args)
    else:
        train_dataset = SlakhDataset(**args)
    
    means, stds = [], []
    for i, (x, y) in tqdm(enumerate(train_dataset)):
        mean_ild, std_ild = get_disparity_map(y.unsqueeze(0), encoder)
        means.append(mean_ild.squeeze().numpy())
        stds.append(std_ild.squeeze().numpy())

    means = np.array(means)
    stds = np.array(stds)
    cs = ['tab:brown','tab:pink','tab:gray','tab:olive']
    for i, s in enumerate(sources):
        plt.scatter(means[:,i], stds[:,i], marker='x',color=cs[i],label='{}_{}_{}'.format(s,means_src[0][i],stds_src[0][i]), alpha=0.7)
        
    plt.xlabel('ILD Mean')
    plt.ylabel('ILD Std')
    plt.title('ILD Stats Distribution for {} - {} Set'.format(dataset, split))
    plt.legend()
    
    plt.savefig('./images/{}{}_stats.png'.format(dataset, split))
    
def build_test_set(dataset_dir, dest_dir, split='test', sources=['guitar1','guitar2','bass','piano'],
                  means_src=[-35,15,0,-20], stds_src=[5,10,15,20], song_len=60.0, test_tracks=50):
    
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    
    args = SimpleNamespace()
    args = {'root':dataset_dir,'sources':sources, 'targets':sources, 'split':split, 'means':means_src, 'stds':stds_src, 'test_tracks':test_tracks*2, 'seed':42}
    
    test_dataset = SlakhDataset(**args)
    
    with open(os.path.join(dest_dir,'config.yml'), 'w') as outfile:
        yaml.dump(test_dataset.get_infos(), outfile, default_flow_style=False)
        
    track_names = [x['paths']['piano'][0].split('/')[-3] for x in list(test_dataset.get_tracks())]
    written_files = 0
    for i, (x, y, z) in tqdm(enumerate(test_dataset), total=len(track_names)):
        x = x.numpy()
        y = y.numpy()
        track_name = track_names[i]
        # create output dir if not exist
        exp_dir = Path(dest_dir,track_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        fname = Path(exp_dir,'mix.wav')
        # Avoid silence song
        m=np.array([0])
        count = 0
        while any(m==0) and count <= 100:
            start = int(np.random.uniform(0, x.shape[-1] - int(song_len*44100)))
            end = start+int(song_len*44100)
            m = np.mean(y[:,:,start:end],(1,2))
            count += 1
            
        if count >= 100: 
            print('Could not write: ', track_name)
            continue
        
        sf.write(fname, x[:,start:end].T,44100)
        for i, s in enumerate(sources):
            fname = Path(exp_dir,s+'.wav')
            sf.write(fname, y[i,:,start:end].T,44100)
            
        angles = {sources[i]:z[i] for i in range(len(sources))}
        np.savez(os.path.join(exp_dir,'angles.npz'),**angles)
        
        written_files += 1
        
        if written_files >= test_tracks:
            break
        
def build_musdb_slakh_split(musdb_dir, slakh_dir, dest_dir, slakh_inst, split='train', test_tracks=50):
    
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    
    args = SimpleNamespace()
    args = {'root':slakh_dir,'sources':[slakh_inst], 'targets':[slakh_inst], 'split':split, 'test_tracks':test_tracks*2, 'seed':42}
    
    dataset = SlakhDataset(**args)
    
    with open(os.path.join(dest_dir,'config.yml'), 'w') as outfile:
        yaml.dump(dataset.get_infos(), outfile, default_flow_style=False)
        
    track_names = [x['paths'][slakh_inst][0].split('/')[-3] for x in list(dataset.get_tracks())]
    written_files = 0
    for i, (x, y, z) in tqdm(enumerate(dataset), total=len(track_names)):
        print(y.shape)
    
        
def compute_results(paths, sources):
    
    excludes = []#['Track01990','Track02004','']
    for p in paths:
        dic = {'SDR':[],'ISR':[],'SAR':[],'SIR':[]}
        d = np.load(os.path.join(p,'results','results.npz'),allow_pickle=True)
        d = d['data']
        for r in d:
            for m in dic:
                if not r['track_name'] in excludes:
                    dic[m].append(r[m])
        print('\n',p,'\n',sources)
        for m in dic:
            print(m,np.round(np.nanmean(dic[m],0),3),'±',np.round(np.nanstd(dic[m],0),3))    
    
if __name__ == "__main__":
    dataset_dir = '../../../../../data/slakh2100'
    dest_dir = '../../../../../data/slakh2100/test_task1'
    dataset='slakh'
    split='train'
    sources=['guitar','strings (continued)','bass','piano']
    means=[-35,15,0,-20],
    stds=[0,0,0,0],
    # get_ild_stats(dataset_dir, dataset='slakh', split='test', sources=sources,means_src=means, stds_src=stds)
    #build_test_set(dataset_dir=dataset_dir, dest_dir=dest_dir, split=split, sources=sources, means_src=means, stds_src=stds)
    #build_musdb_slakh_split(musdb_dir='../../../../../data/musdb', 
                            # slakh_dir=dataset_dir, 
                            # dest_dir='', slakh_inst='piano', split='train', test_tracks=50)
    compute_results(['./exp/paper/train_xumx_f691ecaa_baseline_task1'],'')