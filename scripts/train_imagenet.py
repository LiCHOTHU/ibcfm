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
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToPILImage

from tqdm import tqdm
from torchdiffeq import odeint
from torchmetrics.image.fid import FrechetInceptionDistance

import sys
# add repo root to path
torchcfm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, torchcfm_root)

try:
    import wandb
except ImportError:
    wandb = None

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from utils.utils_cifar import ema, generate_samples_return, compute_entropy_loss
from torchcfm.models.unet.unet import UNetModel
from utils.dataset import get_imagenet64_loaders

import pdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['otcfm','icfm','fm','si'], default='otcfm')
    p.add_argument('--sigma', type=float, default=0.0)
    p.add_argument('--num_channel', type=int, default=128)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--total_steps', type=int, default=800000)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--ema_decay', type=float, default=0.9999)
    p.add_argument('--save_step', type=int, default=25000)
    p.add_argument('--data_root', type=str, default="/storage/home/hcoda1/8/lwang831/p-agarg35-0/workspace/ibcfm/data/imagenet64_png")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--wandb_project', type=str, default=None)
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--no_wandb', action='store_true')
    # IB
    p.add_argument('--use_ib', action='store_true')
    p.add_argument('--ib_lambda', type=float, default=1e-1)
    p.add_argument('--ib_beta', type=float, default=1e-4)
    return p.parse_args()


def warmup_lr(step: int, warmup: int) -> float:
    return min(step, warmup) / warmup


# Sampling trajectory

@torch.no_grad()
def sample_and_evaluate(
    model: UNetModel,
    n_samples: int,
    n_steps: int,
    device: torch.device,
    image_size: int,
    image_channel: int,
    use_torchdiffeq: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    method: str = "dopri5",
) -> torch.Tensor:
    cls = torch.arange(10, device=device).repeat(n_samples//10)
    x0 = torch.randn(n_samples,image_channel,image_size,image_size,device=device)
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
    out_path = save_dir / f"image_net_flow_epoch{epoch:03d}.png"
    img.save(out_path)
    print(f"→ saved generated grid to {out_path}")

def compute_fid_from_tensor(
    generated: torch.Tensor,
    real_loader: DataLoader,
    device: torch.device
) -> float:
    metric = FrechetInceptionDistance(feature=64).to(device)
    metric.reset()
    gen = (generated + 1) / 2  # [0,1]
    gen = F.interpolate(gen.to(device), size=(299, 299), mode="bilinear", align_corners=False)
    gen_uint8 = (gen * 255).to(torch.uint8)
    metric.update(gen_uint8, real=False)

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


def save_samples_grid(samples: torch.Tensor, save_dir: Path, step: int):
    save_dir.mkdir(exist_ok=True, parents=True)
    grid = make_grid(samples, nrow=8, padding=2, normalize=True, value_range=(-1,1))
    img = ToPILImage()(grid)
    img.save(save_dir / f"samples_step{step:06d}.png")


def main() -> Tuple[float,int,float]:
    args = parse_args()
    cfg = vars(args)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = Path('checkpoints') / f"{cfg['model']}_imagenet64_{timestamp}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    sample_dir = checkpoint_dir / 'samples'
    sample_dir.mkdir(exist_ok=True, parents=True)

    use_wandb = args.wandb_project and not args.no_wandb and wandb
    if use_wandb:
        run_name = args.wandb_run_name or checkpoint_dir.name
        wandb.init(project=args.wandb_project, name=run_name, config=cfg)

    device = torch.device(cfg['device'])
    train_loader, val_loader = get_imagenet64_loaders(
        data_root=cfg['data_root'], batch_size=cfg['batch_size'], num_workers=cfg['num_workers']
    )

    # Conditional UNet
    net = UNetModel(
        dim=(3,64,64), num_channels=cfg['num_channel'], num_res_blocks=2,
        num_classes=10, class_cond=True,
        channel_mult=[1,2,2,2], num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.1
    ).to(device)

    # # TODO use a smaller model to debug
    # net = UNetModelWrapper(
    #     dim=(3, 64, 64), num_res_blocks=1,
    #     num_channels=cfg['num_channel'], channel_mult=[1, 2],
    #     num_heads=1, num_head_channels=8,
    #     attention_resolutions='16', dropout=0.1
    # ).to(device)

    ema_model = copy.deepcopy(net)

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
    pbar = tqdm(total=max_steps)
    while step < max_steps:
        for x1,_ in train_loader:
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
                loss = flow_loss + cfg['ib_lambda']*kin - cfg['ib_beta']*ent
            else:
                loss = flow_loss

            # Make a dict of all the pieces you care about
            log_dict = {
                'flow_loss': flow_loss.item(),
                'step_loss': loss.item(),
            }
            if cfg['use_ib']:
                log_dict.update({
                    'kinetic': kin.item(),
                    'entropy': ent.item(),
                    'ib_term': (cfg['ib_lambda'] * kin - cfg['ib_beta'] * ent).item(),
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
                # validation
                net.eval()
                val_loss_sum, n_val = 0.0, 0
                with torch.no_grad():
                    for x_val,_ in val_loader:
                        x_val = x_val.to(device)
                        x0v = torch.randn_like(x_val)
                        tv, xtv, utv = FM.sample_location_and_conditional_flow(x0v, x_val)
                        vv = net(tv, xtv)
                        val_loss_sum += F.mse_loss(vv, utv).item()*x_val.size(0)
                        n_val += x_val.size(0)
                avg_val = val_loss_sum/n_val
                print(f"Step {step+1}: val_loss={avg_val:.4f}")
                if use_wandb: wandb.log({'val_loss': avg_val, 'step': step+1})

                # sampling
                sample_steps = 256

                USE_TORCH_DIFFEQ = True
                traj = sample_and_evaluate(
                    net, 100, sample_steps, device, image_size=64,
                    image_channel=3,
                    use_torchdiffeq=USE_TORCH_DIFFEQ
                )
                samples = traj[-1]

                # samples = generate_samples_return(ema_model, False, 64, sample_steps)
                save_samples_grid(samples, sample_dir, step+1)
                fid = compute_fid_from_tensor(samples, val_loader, device)
                nnl = crude_nnl_estimate(samples)
                print(f"Step {step+1}: FID={fid:.2f}, NNL={nnl:.3f}")
                if use_wandb:
                    wandb.log({'FID': fid, 'NNL': nnl, 'step': step+1})
            step += 1
            pbar.update(1)

            if (step + 1) % (cfg['save_step']) == 0:
                # save checkpoint
                ckpt = {
                    'step': step + 1,
                    'net': net.state_dict(),
                    'ema': ema_model.state_dict(),
                    'optim': optim.state_dict(),
                    'sched': sched.state_dict(),
                }
                torch.save(ckpt, checkpoint_dir / f"ckpt_step{step + 1:06d}.pt")
    pbar.close()

    print(f"Done → FID: {final_fid:.2f}, NNL: {final_nnl:.3f}")
    if use_wandb: wandb.finish()

if __name__ == '__main__':
    main()
