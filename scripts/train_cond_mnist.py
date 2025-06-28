#!/usr/bin/env python3
"""conditional_mnist_flow_matching.py
========================================================
Train and evaluate class-conditional Flow-Matching on MNIST or Fashion-MNIST with W&B.
Includes per-epoch train/val loss, FID, NFE, and NNL reporting,
plus saving sample grids and checkpoints. Uses torchdiffeq for sampling.
Supports optional Information Bottleneck (IB) regularization with per-term reporting.
"""
from __future__ import annotations
import argparse
import math
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from torchdiffeq import odeint
from pytorch_fid import fid_score

try:
    import wandb
except ImportError:
    wandb = None

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
)
from torchcfm.models.unet import UNetModel

# ─────────────────────────────────────────────────────────────────────────────
USE_TORCH_DIFFEQ = True
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['mnist','fashion-mnist'], default='mnist',
                   help='Which 28×28 dataset to train on')
    p.add_argument('--matcher', choices=['cfm','ot','sb','target'], default='cfm')
    p.add_argument('--sigma', type=float, default=0.0)
    p.add_argument('--n_epochs', type=int, default=200000)
    p.add_argument('--save_every', type=int, default=4000,help = 'Save samples & checkpoint every N epochs')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--n_steps', type=int, default=128,
                   help='ODE solver steps for sampling (NFE)')
    p.add_argument('--n_samples', type=int, default=100,
                   help='Number of generated samples (must be multiple of 10)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--wandb_project', type=str, default=None)
    p.add_argument('--wandb_run_name', type=str, default='flow_matching')
    p.add_argument('--no_wandb', action='store_true')
    # IB hyperparameters
    p.add_argument('--use_ib', action='store_true', help='Enable Information Bottleneck regularization')
    p.add_argument('--ib_lambda', type=float, default=1e-1, help='Weight for kinetic-energy penalty')
    p.add_argument('--ib_beta', type=float, default=1e-4, help='Weight for entropy regularizer')
    return p.parse_args()

def _make_run_name(cfg):
    """
    Compose a descriptive W&B run-name.

    Format
        <model>_<dataset>_<IB|noIB>_<YYYYMMDD-HHMMSS>

    If the user supplies --wandb_run_name we respect it.
    """
    if cfg.get('wandb_run_name'):               # user override
        return cfg['wandb_run_name']

    model   = cfg.get('model',   'unknownModel')
    dataset = cfg.get('dataset', 'unknownData')
    ib_tag  = 'IB' if cfg.get('use_ibd', False) else 'noIB'
    stamp   = datetime.now().strftime('%Y%m%d-%H%M%S')

    return f"{model}_{dataset}_{ib_tag}_{stamp}"



def get_data_loaders(dataset: str, batch_size: int, workers: int = 1) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    if dataset == 'mnist':
        DS = datasets.MNIST; root = './data/mnist'
    else:
        DS = datasets.FashionMNIST; root = './data/fashion-mnist'
    train_ds = DS(root, train=True, download=True, transform=tfm)
    val_ds   = DS(root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=workers)
    return train_loader, val_loader


# IB entropy estimate via per-dimension variance

def compute_entropy_loss(pred: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    B = pred.shape[0]
    D = pred.numel() // B
    x = pred.reshape(B, D)
    var = x.var(dim=0, unbiased=False) + eps  # [D]
    H = 0.5 * torch.sum(torch.log(2 * math.pi * math.e * var))
    return H


# FID via pytorch-fid

@torch.no_grad()
def compute_fid(
    generated: torch.Tensor,
    real_loader: DataLoader,
    device: torch.device,
    batch_size: int = 64,
    dims: int = 2048,
    num_workers: int = 1,
) -> float:
    with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as gen_dir:
        to_pil = ToPILImage()
        gen = ((generated + 1)/2).repeat(1,3,1,1)
        gen = F.interpolate(gen, size=(299,299), mode='bilinear', align_corners=False)
        for i, img in enumerate(gen):
            to_pil(img.cpu()).save(f"{gen_dir}/{i:05d}.png")
        idx = 0
        for xr, _ in real_loader:
            xr = ((xr.to(device) + 1)/2).repeat(1,3,1,1)
            xr = F.interpolate(xr, size=(299,299), mode='bilinear', align_corners=False)
            for img in xr:
                to_pil(img.cpu()).save(f"{real_dir}/{idx:05d}.png"); idx += 1
        fid_val = fid_score.calculate_fid_given_paths(
            [real_dir, gen_dir], batch_size, str(device), dims, num_workers
        )
    return fid_val


# Sampling trajectory

@torch.no_grad()
def sample_and_evaluate(
    model: UNetModel,
    n_samples: int,
    n_steps: int,
    device: torch.device,
    use_torchdiffeq: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    method: str = "dopri5",
) -> torch.Tensor:
    cls = torch.arange(10, device=device).repeat(n_samples//10)
    x0 = torch.randn(n_samples,1,28,28,device=device)
    if use_torchdiffeq:
        t_span = torch.linspace(0,1,2,device=device)
        traj = odeint(lambda t,x: model(t,x,cls), x0, t_span,
                      atol=atol, rtol=rtol, method=method)
    else:
        dt = 1/n_steps
        x = x0; t = torch.ones(n_samples,device=device); outs=[x]
        for _ in range(n_steps):
            v = model(t,x,cls)
            x = x - v*dt
            t = t - dt
            outs.append(x)
        traj = torch.stack(outs, dim=0)
    return traj.clamp(-1,1).cpu()


def save_samples_grid(
    samples: torch.Tensor,
    checkpoint_dir: Path,
    epoch: int,
    nrow: int = 10,
    value_range: tuple[float, float] = (-1, 1),
) -> None:
    save_dir = checkpoint_dir / "samples"
    save_dir.mkdir(parents=True, exist_ok=True)
    grid = make_grid(
        samples,
        nrow=nrow,
        padding=0,
        normalize=True,
        value_range=value_range,
    )
    img = ToPILImage()(grid)
    out_path = save_dir / f"mnist_flow_epoch{epoch:03d}.png"
    img.save(out_path)
    print(f"→ saved generated grid to {out_path}")


def main() -> Tuple[float, int, float]:
    args = parse_args()
    cfg = vars(args)

    cfg['wandb_run_name'] = _make_run_name(cfg)

    device = torch.device(cfg['device'])
    use_wandb = (cfg['wandb_project'] is not None) and not cfg['no_wandb'] and wandb
    if use_wandb:
        print("wandb run name:", cfg['wandb_run_name'])
        wandb.init(project=cfg['wandb_project'], name=cfg['wandb_run_name'], config=cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ib_tag = 'ib' if cfg.get('use_ibd', False) else 'noib'
    checkpoint_dir = Path("checkpoints") / f"{cfg['matcher']}_{cfg['dataset']}_{ib_tag}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_data_loaders(cfg['dataset'], cfg['batch_size'])
    # model = UNetModel(
    #     dim=(1,28,28), num_channels=32, num_res_blocks=1,
    #     num_classes=10, class_cond=True
    # ).to(device)
    # TODO use a larger model
    model = UNetModel(
        dim=(1,28,28), num_channels=32, num_res_blocks=2,
        num_classes=10, class_cond=True
    ).to(device)

    score_model: Optional[UNetModel] = None
    if cfg['matcher'] == 'sb':
        score_model = UNetModel(
            dim=(1,28,28), num_channels=32, num_res_blocks=1,
            num_classes=10, class_cond=True
        ).to(device)

    FM = {
        'cfm': ConditionalFlowMatcher,
        'ot': ExactOptimalTransportConditionalFlowMatcher,
        'sb': SchrodingerBridgeConditionalFlowMatcher,
        'target': TargetConditionalFlowMatcher,
    }[cfg['matcher']](sigma=cfg['sigma'])

    opt = optim.Adam(
        list(model.parameters()) + ([] if score_model is None else list(score_model.parameters())),
        lr=cfg['lr']
    )

    final_fid, final_nfe, final_nnl = float('nan'), cfg['n_steps'], float('nan')
    for epoch in range(cfg['n_epochs']):
        # --- training ---
        model.train()
        if score_model:
            score_model.train()

        sum_flow = sum_score = sum_kin = sum_ent = sum_loss = 0.0
        for x1, y in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            x1, y = x1.to(device), y.to(device)
            x0 = torch.randn_like(x1)
            opt.zero_grad()

            # Flow matching target
            if cfg['matcher'] in ['cfm','target']:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                u_pred = model(t, xt, y)
                flow_loss = F.mse_loss(u_pred, ut)
                score_loss = 0.0
            elif cfg['matcher']=='ot':
                t, xt, ut, *_ = FM.guided_sample_location_and_conditional_flow(x0, x1, y1=y)
                u_pred = model(t, xt, y)
                flow_loss = F.mse_loss(u_pred, ut)
                score_loss = 0.0
            else:  # 'sb'
                t, xt, ut, _, y1, eps = FM.guided_sample_location_and_conditional_flow(
                    x0, x1, y1=y, return_noise=True
                )
                λ_t = FM.compute_lambda(t)
                u_pred = model(t, xt, y1)
                s = score_model(t, xt, y1)
                flow_loss = F.mse_loss(u_pred, ut)
                score_loss = F.mse_loss(λ_t.view(-1,1,1,1) * s + eps, torch.zeros_like(eps))
                sum_score += score_loss.item()

            # Base loss including flow + score
            base_loss = flow_loss + score_loss

            # IB regularization
            if cfg['use_ib']:
                kin = u_pred.pow(2).mean()
                ent = compute_entropy_loss(u_pred)
                loss = base_loss + cfg['ib_lambda'] * kin - cfg['ib_beta'] * ent

                sum_kin += cfg['ib_lambda'] * kin.item()
                sum_ent +=  - cfg['ib_beta'] * ent.item()
            else:
                loss = base_loss

            loss.backward()
            opt.step()

            sum_flow += flow_loss.item()
            sum_loss += loss.item()

        # Average metrics
        n_batches = len(train_loader)
        metrics = {
            'epoch': epoch,
            'flow_loss': sum_flow / n_batches,
            'score_loss': sum_score / n_batches,
            'total_loss': sum_loss / n_batches,
        }
        if cfg['use_ib']:
            metrics.update({
                'kinetic': sum_kin / n_batches,
                'entropy': sum_ent / n_batches,
            })
        print(f"Epoch {epoch} Train → {metrics}")
        if use_wandb:
            wandb.log({f"train_{k}":v for k,v in metrics.items()})

        # --- validation ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x1, y in val_loader:
                x1, y = x1.to(device), y.to(device)
                x0 = torch.randn_like(x1)
                if cfg['matcher'] in ['cfm','target']:
                    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                else:
                    t, xt, ut, *_ = FM.guided_sample_location_and_conditional_flow(x0, x1, y1=y)
                v = model(t, xt, y)
                val_loss_sum += F.mse_loss(v, ut).item() * x1.size(0)
        avg_val_loss = val_loss_sum / len(val_loader.dataset)
        print(f"Epoch {epoch} Val → val_loss: {avg_val_loss:.4f}")
        if use_wandb:
            wandb.log({'val_loss': avg_val_loss})

        # --- sampling & FID ---
        traj = sample_and_evaluate(
            model, cfg['n_samples'], cfg['n_steps'], device,
            use_torchdiffeq=USE_TORCH_DIFFEQ
        )
        samples = traj[-1]
        fid = compute_fid(samples, val_loader, device)
        nnl = -( (samples+1)/2 * torch.log(((samples+1)/2).clamp_min(1e-7)) ).mean().item()/math.log(2)
        nfe = cfg['n_steps']
        final_fid, final_nfe, final_nnl = fid, nfe, nnl
        print(f"Epoch {epoch} Sample → FID: {fid:.2f}, NFE: {nfe}, NNL: {nnl:.3f}")
        if use_wandb:
            wandb.log({'FID': fid, 'NFE': nfe, 'NNL': nnl})

        # --- save samples & checkpoint ---
        if (epoch + 1) % cfg['save_every'] == 0:
            save_samples_grid(samples, checkpoint_dir, epoch)
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'cfg': cfg
            }
            ckpt_path = checkpoint_dir / f"ckpt_epoch{epoch:03d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"→ saved checkpoint to {ckpt_path}\n")

    if use_wandb:
        wandb.finish()

    print(f"Final metrics → FID: {final_fid:.2f}, NFE: {final_nfe}, NNL: {final_nnl:.3f} bits")
    return final_fid, final_nfe, final_nnl


if __name__ == '__main__':
    main()
