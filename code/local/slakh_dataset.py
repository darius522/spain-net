from pathlib import Path
import torch.utils.data
import random
import torch
import tqdm
import soundfile as sf
import yaml
import itertools
import numpy as np
import re
import copy


class SlakhDataset(torch.utils.data.Dataset):

    dataset_name = "Slakh"

    def __init__(
        self,
        root,
        sources=["guitar1", "guitar2", "bass", "piano"],
        means=[-35,15,0,-20],
        stds=[0,0,0,0],
        targets=["guitar1", "guitar2", "bass", "piano"],
        suffix=".wav",
        split="train",
        subset=None,
        segment=None,
        samples_per_track=1,
        random_segments=False,
        random_track_mix=False,
        source_augmentations=lambda audio: audio,
        sample_rate=44100,
        valid_tracks=15,
        test_tracks=50,
        seed=None,
    ):
        if seed:
            torch.manual_seed(seed)
            random.seed(seed)
        self.validation = True if split=='validation' else False
        self.valid_tracks = valid_tracks
        self.test_tracks = test_tracks
        self.means = means if type(means) == list else means[0]
        self.stds = stds if type(stds) == list else stds[0]
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.segment = segment
        self.random_track_mix = random_track_mix
        self.random_segments = random_segments
        self.source_augmentations = source_augmentations
        self.sources = sources
        self.targets = targets
        self.suffix = suffix
        self.subset = subset
        self.samples_per_track = samples_per_track
        self.tracks = list(self.get_tracks())

        if not self.tracks:
            raise RuntimeError("No tracks found.")
                
    def _panner(self, x, angle):
        """
        
        
        pan a mono audio source into stereo
        x: <frames, 2>
        angle: the angle in degrees
        """
        
        angle_r = np.radians(angle)
        left = np.sqrt(2)/2.0 * (np.cos(angle_r) - np.sin(angle_r)) * x[...,0]
        right = np.sqrt(2)/2.0 * (np.cos(angle_r) + np.sin(angle_r)) * x[...,1]
        return np.stack((left,right),-1)

    def __getitem__(self, index):
        # assemble the mixture of target and interferers
        audio_sources = []
        source_angles = []

        # get track_id
        track_id = index // self.samples_per_track
        min_dur = self.tracks[track_id]["min_duration"]
        if self.random_segments:
            start = random.uniform(0, min_dur - self.segment)
        else:
            start = 0

        # load sources
        paths = copy.deepcopy(self.tracks[track_id]["paths"])
        for i, source in enumerate(self.sources):
            
            # If multiple of the same inst exists, remove the leading number (i.e. "guitar1" -> "guitar")
            source = re.sub(r'[0-9]+', '', source)

            # loads the full track duration
            start_sample = int(start * self.sample_rate)
            # check if dur is none
            if self.segment:
                # stop in soundfile is calc in samples, not seconds
                stop_sample = start_sample + int(self.segment * self.sample_rate)
                seg_stop = int(self.segment * self.sample_rate)
            else:
                # set to None for reading complete file
                stop_sample = None
                seg_stop = int(min_dur*self.sample_rate)

            # load actual audio
            random.shuffle(paths[source])
            if source!='_dummy':
                p = random.choice(paths[source])
            else:
                p = paths[source].pop(0)
                
            audio, _ = sf.read(p,always_2d=True, start=start_sample, stop=stop_sample)
            audio = audio.repeat(2, -1)
            angle = np.clip(np.random.uniform(low=-45.0,high=45.0), a_min=-45.0, a_max=45.0)
            noise = np.random.normal(0.0,4.0)
            source_angles.append(np.clip(angle, a_min=-45.0, a_max=45.0))
            audio = self._panner(audio[:seg_stop,:], angle)
            # convert to torch tensor
            audio = torch.tensor(audio.T, dtype=torch.float)
            # apply source-wise augmentations
            #audio = self.source_augmentations(audio)
            audio_sources.append(audio)

        # apply linear mix over source index=0
        audio_mix = torch.stack(audio_sources).sum(0)
        if self.targets:
            audio_sources = torch.stack(audio_sources, dim=0)
        return audio_mix, audio_sources, torch.FloatTensor(source_angles)

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_tracks(self):
        """Loads input and output tracks"""
        p = list(Path(self.root, self.split).iterdir())
        
        if self.validation or self.split=='test':
            tracks = random.sample(p, len(p))
        else:
            tracks = p
            
        sources = list(set([re.sub(r'[0-9]+', '', source) for source in self.sources]))
        include_tracks = 0
        
        for track_path in tqdm.tqdm(tracks):
            if track_path.is_dir():
                
                # Open yaml and check for matching stems
                with open(track_path / "metadata.yaml", 'r') as stream:
                    stems_conf = yaml.safe_load(stream)['stems']

                source_paths = {src:[str(track_path / 'stems' / (s + self.suffix)) for s in stems_conf.keys() if (stems_conf[s]['inst_class'].lower()==src and stems_conf[s]['audio_rendered'])] for src in set(sources)}
                if any(len(x)==0 for x in source_paths.values()):# or len(source_paths['guitar'])<2:
                    print("Exclude track due to non-existing source", track_path)
                    continue
                
                # get metadata
                infos = list(map(sf.info, list(itertools.chain(*source_paths.values()))))
                if not all(i.samplerate == self.sample_rate for i in infos):
                    print("Exclude track due to different sample rate ", track_path)
                    continue
                
                # If we go beyond num. tracks for the split, just stop
                if self.validation and self.valid_tracks <= include_tracks:
                    continue
                elif self.split=='test' and self.test_tracks <= include_tracks:
                    continue

                include_tracks += 1
                min_duration = min(i.duration for i in infos)
                if self.segment is not None:
                    # get minimum duration of track
                    if min_duration > self.segment:
                        yield ({"paths": source_paths, "min_duration": min_duration})
                else:
                    yield ({"paths": source_paths, "min_duration": min_duration})

    def get_infos(self):
        """Get dataset infos (for publishing models).
        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        props = vars(self).copy()
        del props['tracks']
        del props['source_augmentations']
        return props