#!/usr/bin/env python3
"""cifar_flow_matching.py
========================================================
Train and evaluate unconditional Flow-Matching on CIFAR-10 with W&B.
Includes per-epoch train/val loss, FID, and NNL reporting,
plus saving sample grids and checkpoints. Uses torchdiffeq for sampling.
"""
from __future__ import annotations
import argparse
import copy
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from torchdiffeq import odeint
from cleanfid import fid
from torchdyn.core import NeuralODE

import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, repo_root)

try:
    import wandb
except ImportError:
    wandb = None

from torchmetrics.image.fid import FrechetInceptionDistance
from utils.utils_cifar import ema, generate_samples_return, infiniteloop
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['otcfm','icfm','fm','si'], default='otcfm',
                   help='flow matching model to use')
    p.add_argument('--sigma', type=float, default=0.0)
    p.add_argument('--num_channel', type=int, default=128,
                   help='base channel count for UNet')
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)

    # TODO use a smaller, default=400000
    p.add_argument('--total_steps', type=int, default=400000)
    p.add_argument('--warmup', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--ema_decay', type=float, default=0.9999)
    p.add_argument('--parallel', action='store_true')

    # TODO use a smaller, default=20000
    p.add_argument('--save_step', type=int, default=2000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--wandb_project', type=str, default=None)
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--fid_batch', type=int, default=64)
    p.add_argument('--fid_samples', type=int, default=16)

    # IB hyperparameters
    p.add_argument('--use_ib', type="bool", default=False, help='Enable Information Bottleneck regularization')
    p.add_argument('--ib_lambda', type=float, default=1e-1, help='Weight for kinetic-energy penalty')
    p.add_argument('--ib_beta', type=float, default=1e-4, help='Weight for entropy regularizer')
    return p.parse_args()


def warmup_lr(step: int, warmup: int) -> float:
    return min(step, warmup) / warmup


def compute_fid_from_tensor(
    generated: torch.Tensor,
    real_loader: DataLoader,
    device: torch.device
) -> float:
    """
    Compute FID between a batch of generated images (in [-1,1]) and a real data loader,
    without re-sampling inside the function.
    """
    # Ensure input tensor
    if generated is None or not torch.is_tensor(generated):
        raise ValueError(f"Expected generated to be a torch.Tensor, got {type(generated)}")

    # Initialize FID metric
    metric = FrechetInceptionDistance(feature=64).to(device)
    metric.reset()

    # Prepare generated images: map [-1,1]→[0,1], resize & replicate if needed
    gen = (generated + 1) / 2  # [0,1]
    # CIFAR is already 3-channel, so no need to repeat; just resize to Inception size
    gen = F.interpolate(gen.to(device), size=(299, 299), mode="bilinear", align_corners=False)
    # Convert to uint8 in [0,255] as required
    gen_uint8 = (gen * 255).to(torch.uint8)
    metric.update(gen_uint8, real=False)

    # Iterate through real dataset once
    for xr, _ in real_loader:
        xr = (xr.to(device) + 1) / 2
        xr = F.interpolate(xr, size=(299, 299), mode="bilinear", align_corners=False)
        xr_uint8 = (xr * 255).to(torch.uint8)
        metric.update(xr_uint8, real=True)

    return metric.compute().item()


def crude_nnl_estimate(samples: torch.Tensor) -> float:
    p = (samples + 1) / 2
    log_px = p * torch.log(p.clamp_min(1e-7)) + (1-p)*torch.log((1-p).clamp_min(1e-7))
    return -log_px.mean().item() / math.log(2)


def save_samples_grid(samples: torch.Tensor, save_dir: Path, step: int, nrow: int = 8) -> None:
    save_dir.mkdir(exist_ok=True, parents=True)
    grid = make_grid(samples, nrow=nrow, padding=2, normalize=True, value_range=(-1,1))
    img = ToPILImage()(grid)
    img.save(save_dir / f"samples_step{step:06d}.png")

def compute_entropy_loss(pred: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Estimate entropy via per-dimension variance."""
    B = pred.shape[0]
    D = pred.numel() // B
    x = pred.reshape(B, D)
    var = x.var(dim=0, unbiased=False) + eps
    return 0.5 * torch.sum(torch.log(2 * math.pi * math.e * var))

def main() -> Tuple[float,int,float]:
    args = parse_args()
    cfg = vars(args)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset = 'cifar10'
    ib_tag = 'noib'
    checkpoint_dir = Path('checkpoints') / f"{cfg['model']}_{dataset}_{ib_tag}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = checkpoint_dir / 'samples'
    sample_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = args.wandb_project and not args.no_wandb and wandb
    if use_wandb:
        run_name = args.wandb_run_name or checkpoint_dir.name
        print("wandb run name:", run_name)
        wandb.init(project=args.wandb_project, name=run_name, config=cfg)

    device = torch.device(cfg['device'])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=cfg['num_workers'], drop_last=True)
    val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=cfg['fid_batch'], shuffle=False,
                             num_workers=cfg['num_workers'])

    # TODO use a smaller for debug, num_res_blocks=2
    net = UNetModelWrapper(
        dim=(3,32,32), num_res_blocks=2,
        num_channels=cfg['num_channel'], channel_mult=[1,2,2,2],
        num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.1
    ).to(device)
    ema_model = copy.deepcopy(net)
    if args.parallel:
        net = torch.nn.DataParallel(net)
        ema_model = torch.nn.DataParallel(ema_model)

    print(f"Model size: {sum(p.numel() for p in net.parameters())/1e6:.2f}M params")

    FM = {
        'otcfm': ExactOptimalTransportConditionalFlowMatcher,
        'icfm': ConditionalFlowMatcher,
        'fm': TargetConditionalFlowMatcher,
        'si': VariancePreservingConditionalFlowMatcher,
    }[cfg['model']](sigma=cfg['sigma'])

    optim = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.LambdaLR(optim,
            lr_lambda=lambda s: warmup_lr(s, cfg['warmup']))

    final_fid, final_nnl = float('nan'), float('nan')
    step, max_steps = 0, cfg['total_steps']
    pbar = tqdm(total=max_steps, dynamic_ncols=True)
    while step < max_steps:
        for x1,_ in loader:
            if step >= max_steps: break
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)
            optim.zero_grad()

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net(t, xt)
            flow_loss = F.mse_loss(vt, ut)
            if cfg['use_ib']:
                kin = vt.pow(2).mean()
                ent = compute_entropy_loss(vt)
                loss = flow_loss + cfg['ib_lambda'] * kin - cfg['ib_beta'] * ent
            else:
                loss = flow_loss

            # Make a dict of all the pieces you care about
            log_dict = {
                'flow_loss': flow_loss.item(),
                'step_loss': loss.item(),
                'step': step,
            }
            if cfg['use_ib']:
                log_dict.update({
                    'kinetic': kin.item(),
                    'entropy': ent.item(),
                    'ib_term': ib_term.item(),
                    'λ·kinetic': (cfg['ib_lambda'] * kin).item(),
                    'β·entropy': (cfg['ib_beta'] * ent).item(),
                })
            # send them off to wandb (or your logger of choice)
            if use_wandb:
                wandb.log(log_dict)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip'])
            optim.step()
            sched.step()
            ema(net, ema_model, cfg['ema_decay'])

            if (step+1) % cfg['save_step'] == 0:
                # --- validation ---
                net.eval()
                val_loss_sum = 0.0
                n_val = 0
                with torch.no_grad():
                    for x_val,_ in val_loader:
                        x_val = x_val.to(device)
                        x0_val = torch.randn_like(x_val)
                        t_val, xt_val, ut_val = FM.sample_location_and_conditional_flow(x0_val, x_val)
                        v_val = net(t_val, xt_val)
                        val_loss_sum += F.mse_loss(v_val, ut_val).item() * x_val.size(0)
                        n_val += x_val.size(0)
                avg_val_loss = val_loss_sum / n_val
                print(f"Step {step+1}: val_loss={avg_val_loss:.4f}")
                if use_wandb:
                    wandb.log({'val_loss': avg_val_loss, 'step': step+1})

                # sampling and metrics
                samples = generate_samples_return(
                    ema_model, args.parallel,
                )

                # Map from [-1,1] to [0,1] for saving
                to_save = (samples + 1) / 2
                # Save an 8×8 grid (nrow=8) of the 64 images
                outfile = sample_dir / f"samples_step{step:06d}.png"
                save_image(to_save, outfile, nrow=8)

                print(f"→ saved sample grid to {outfile}")

                fid = compute_fid_from_tensor(samples, val_loader, device)

                nnl = crude_nnl_estimate(samples)
                final_fid, final_nnl = fid, nnl
                print(f"Step {step+1}: FID={fid:.2f}, NNL={nnl:.3f}")
                if use_wandb:
                    wandb.log({'FID': fid, 'NNL': nnl, 'step': step+1})

                if (step + 1) % (cfg['save_step'] * 10) == 0:
                    # save checkpoint
                    ckpt = {
                        'step': step+1,
                        'net': net.state_dict(),
                        'ema': ema_model.state_dict(),
                        'optim': optim.state_dict(),
                        'sched': sched.state_dict(),
                    }
                    torch.save(ckpt, checkpoint_dir / f"ckpt_step{step+1:06d}.pt")

            step += 1
            pbar.update(1)
    pbar.close()

    print(f"Final metrics → FID: {final_fid:.2f}, NNL: {final_nnl:.3f} bits")
    if use_wandb:
        wandb.finish()
    return final_fid, max_steps, final_nnl


if __name__ == '__main__':
    main()
