import torch
import numpy as np
import argparse
import soundfile as sf
import musdb
import museval
import norbert
from pathlib import Path
import scipy.signal
import resampy
from local.x_umx import XUMX as XUMX_Spatial
from local.x_umx_base import XUMX
from train import get_disparity_map
from local.x_umx import _STFT, _Spectrogram
from local.slakh_dataset import SlakhDataset
from asteroid.complex_nn import torch_complex_from_magphase
import os
import warnings
import sys
import yaml
from tqdm import tqdm
from train import MultiDomainLoss


def load_model(model_name, baseline=True, device="cpu"):
    print("Loading model from: {}".format(model_name), file=sys.stderr)
    
    conf = torch.load(model_name)
    
    if not baseline:
        conf["model_args"]['device'] = device
        model = XUMX_Spatial(**conf["model_args"])
    else:
        model = XUMX(**conf["model_args"])
    model.load_state_dict(conf["state_dict"])
    model.eval()
    model = model.to(device)
    print(vars(model))
    return model, model.sources, conf["model_args"]


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2), rate, nperseg=n_fft, noverlap=n_fft - n_hopsize, boundary=True
    )
    return audio


def separate(
    audio,
    stats,
    x_umx_target,
    instruments,
    niter=1,
    softmask=False,
    alpha=1.0,
    residual_model=False,
    device="cpu",
):
    """
    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    x_umx_target: asteroid.models
        X-UMX model used for separating

    instruments: list
        The list of instruments, e.g., ["bass", "drums", "vocals"]

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary with all estimates obtained by the separation model.
    """

    # convert numpy audio to torch
    audio_torch = torch.tensor(audio[None, ...]).float().to(device)
    source_names = []
    V = []

    if stats:
        stats[0] = stats[0].to(device)
        stats[1] = stats[1].to(device)
        stats[2] = stats[2].to(device) 
        masked_tf_rep, _ = x_umx_target(audio_torch, stats)
    else:
        masked_tf_rep, _ = x_umx_target(audio_torch)
        
    # shape: (Sources, frames, batch, channels, fbin)

    for j, target in enumerate(instruments):
        Vj = masked_tf_rep[j, Ellipsis].cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj ** alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, Ellipsis])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    # convert to complex numpy type
    tmp = x_umx_target.encoder(audio_torch)
    X = torch_complex_from_magphase(tmp[0].permute(1, 2, 3, 0), tmp[1])
    X = X.detach().cpu().numpy()
    X = X[0].transpose(2, 1, 0)

    if residual_model or len(instruments) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += ["residual"] if len(instruments) > 1 else ["accompaniment"]

    Y = norbert.wiener(V, X.astype(np.complex128), niter, use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            rate=x_umx_target.sample_rate,
            n_fft=x_umx_target.in_chan,
            n_hopsize=x_umx_target.n_hop,
        )
        estimates[name] = audio_hat.T

    return estimates


def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inf_parser.add_argument(
        "--softmask",
        dest="softmask",
        action="store_true",
        help=(
            "if enabled, will initialize separation with softmask."
            "otherwise, will use mixture phase with spectrogram"
        ),
    )

    inf_parser.add_argument(
        "--niter", type=int, default=1, help="number of iterations for refining results."
    )

    inf_parser.add_argument(
        "--alpha", type=float, default=1.0, help="exponent in case of softmask separation"
    )

    inf_parser.add_argument("--samplerate", type=int, default=44100, help="model samplerate")

    inf_parser.add_argument(
        "--residual-model", action="store_true", help="create a model for the residual"
    )
    return inf_parser.parse_args()


def eval_main(
    root,
    samplerate=44100,
    niter=1,
    alpha=1.0,
    softmask=False,
    residual_model=False,
    model_name="xumx",
    outdir=None,
    start=0.0,
    duration=-1.0,
    no_cuda=False,
    training_stats=False,
    baseline=True
):

    model_name = os.path.abspath(model_name)
    if not (os.path.exists(model_name)):
        outdir = os.path.abspath("./results_using_pre-trained")
        model_name = "r-sawata/XUMX_MUSDB18_music_separation"
    else:
        outdir = os.path.abspath(outdir)
        
    Path(outdir).mkdir(exist_ok=True, parents=True)
    print("Evaluated results will be saved in:\n {}".format(outdir), file=sys.stderr)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model, instruments, model_args = load_model(model_name, baseline=baseline, device=device)

    Path(outdir).mkdir(exist_ok=True, parents=True)
    txtout = os.path.join(outdir, "results.txt")
    fp = open(txtout, "w")
    results = []
    
    tracks = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root,x))]
    for track_name in tqdm(tracks):
        
        x = sf.read(os.path.join(root,track_name,'mix.wav'))[0].T
        y = []
        for s in instruments:
            y.append(sf.read(os.path.join(root,track_name,s+'.wav'))[0].T)
        y = np.stack(np.array(y),0)
        
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        # nbatch, nsources, nchannels, ntimes
        stats = None
        if not baseline:
            if not training_stats:
                encoder = torch.nn.Sequential(_STFT(window_length=model_args['window_length'], n_fft=model_args['in_chan'], n_hop=model_args['n_hop']), 
                                            _Spectrogram(spec_power=model_args['spec_power'], mono=(model_args['nb_channels']!=2)))
                mean_ild, std_ild = get_disparity_map(y.unsqueeze(0), encoder)
            else:
                pass
                stats = np.load(os.path.join(root,'slakh_train_stats.npz'), allow_pickle=True)
                mean_ild = torch.from_numpy(stats['mean'][...,np.newaxis,np.newaxis])
                std_ild = torch.from_numpy(stats['std'][...,np.newaxis,np.newaxis])
                
            angles = np.load(os.path.join(root,track_name,'angles.npz'),allow_pickle=True)
            angles = np.array([angles[s].item() for s in instruments])
            angles += np.random.uniform(0.0,8.0)
            angles = np.clip(angles, a_min=-45.0, a_max=45.0)
            z = torch.from_numpy(angles.astype(np.float32)).unsqueeze(0)
            stats = [mean_ild,std_ild, z]
        angles = np.load(os.path.join(root,track_name,'angles.npz'),allow_pickle=True)
        angles = np.array([angles[s].item() for s in instruments])
        print(angles)
            
        estimates = separate(
            x,
            stats,
            model,
            instruments,
            niter=niter,
            alpha=alpha,
            softmask=softmask,
            residual_model=residual_model,
            device=device,
        )
        
        output_path = Path(os.path.join(outdir, track_name))
        output_path.mkdir(exist_ok=True, parents=True)

        print("Processing... {}".format(track_name), file=sys.stderr)
        print(track_name, file=fp)
        
        for target, estimate in estimates.items():
            sf.write(str(output_path / Path(target).with_suffix(".wav")), estimate, samplerate)
            
        y = y.permute(0,2,1).numpy()
        estimates = np.stack([estimates[i] for i in instruments],0)
            
        SDR, ISR, SIR, SAR = museval.evaluate(y, estimates)
        d = {"track_name": track_name, "SDR": np.nanmean(SDR,-1), "ISR": np.nanmean(ISR,-1), "SIR": np.nanmean(SIR,-1), "SAR": np.nanmean(SAR,-1)}
        results.append(d)
        print(d, file=sys.stderr)
        print(d, file=fp)
        
    np.savez(os.path.join(outdir, "results.npz"),data=results)
    fp.close()

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="OSU Inference", add_help=False)

    parser.add_argument("--root", 
                        type=str,
                        default='../../../../../data/slakh2100/test_task2',
                        help="The path to the MUSDB18 dataset")

    parser.add_argument(
        "--modeldir",
        type=str,
        default="./exp/paper/train_xumx_d96d6838_2048_task2/",
        help="Results path where " "best_model.pth" " is stored",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="./exp/paper/train_xumx_d96d6838_2048_task2/results",
    )

    parser.add_argument("--start", type=float, default=0.0, help="Audio chunk start in seconds")

    parser.add_argument(
        "--duration",
        type=float,
        default=-1.0,
        help="Audio chunk duration in seconds, negative values load full track",
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA inference"
    )
    
    parser.add_argument(
        "--training_stats",
        type=bool,
        default=False,
    )
    
    parser.add_argument(
        "--baseline",
        type=bool,
        default=False,
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    model = os.path.join(args.modeldir, "best_model.pth")
    eval_main(
        root=args.root,
        samplerate=args.samplerate,
        alpha=args.alpha,
        softmask=args.softmask,
        niter=args.niter,
        residual_model=args.residual_model,
        model_name=model,
        outdir=args.outdir,
        start=args.start,
        duration=args.duration,
        no_cuda=args.no_cuda,
        training_stats=args.training_stats,
        baseline=args.baseline
    )
