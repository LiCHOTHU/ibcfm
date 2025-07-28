import argparse
import copy
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import wandb
import pdb
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from torchdiffeq import odeint
from torchmetrics.image.fid import FrechetInceptionDistance

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
)
from torchcfm.models.unet import UNetModel

import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, repo_root)
from utils.utils_cifar import ema, generate_samples_return, infiniteloop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Model selection now called matcher for consistency
    p.add_argument('--matcher', choices=['cfm', 'ot', 'sb', 'target'], default='cfm')
    p.add_argument(
        '--sigma',
        type=float,
        default=0.0,
        help='noise scale sigma'
    )
    p.add_argument(
        '--num_channel',
        type=int,
        default=128,
        help='base channel count for UNet'
    )

    p.add_argument(
        '--image_size',
        type=int,
        default=32,
        help='Image Size'
    )

    p.add_argument(
        '--image_channel',
        type=int,
        default=3,
        help='Image Channel'
    )
    p.add_argument(
        '--lr',
        type=float,
        default=5e-5,
        help='learning rate'
    )
    p.add_argument(
        '--grad_clip',
        type=float,
        default=1.0,
        help='gradient clipping norm'
    )
    p.add_argument(
        '--total_steps',
        type=int,
        default=400000,
        help='total training steps'
    )
    p.add_argument(
        '--warmup',
        type=int,
        default=500,
        help='learning rate warmup steps'
    )
    p.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='training batch size'
    )
    p.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='number of DataLoader workers'
    )
    p.add_argument(
        '--ema_decay',
        type=float,
        default=0.9999,
        help='EMA decay rate'
    )
    p.add_argument(
        '--parallel',
        action='store_true',
        help='use DataParallel'
    )
    p.add_argument(
        '--save_step',
        type=int,
        default=8000,
        help='steps between saving samples and checkpoints'
    )

    p.add_argument(
        '--val_step',
        type=int,
        default=100,
        help='steps between saving samples and checkpoints'
    )
    p.add_argument(
        '--fid_batch',
        type=int,
        default=64,
        help='batch size for FID computation'
    )

    p.add_argument('--n_steps', type=int, default=128,
                   help='ODE solver steps for sampling (NFE)')
    p.add_argument('--n_samples', type=int, default=100,
                   help='Number of generated samples (must be multiple of 10)')

    p.add_argument(
        '--fid_samples',
        type=int,
        default=100,
        help='number of samples for FID evaluation'
    )
    p.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='compute device'
    )
    p.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='Weights & Biases project name'
    )
    p.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='Weights & Biases run name override'
    )
    p.add_argument(
        '--no_wandb',
        action='store_true',
        help='disable Weights & Biases logging'
    )
    p.add_argument(
        '--use_ib',
        action='store_true',
        help='enable Information Bottleneck regularization'
    )
    p.add_argument(
        '--ib_lambda',
        type=float,
        default=5e-2,
        help='weight for IB kinetic penalty'
    )
    p.add_argument(
        '--ib_beta',
        type=float,
        default=2e-5,
        help='weight for IB entropy regularizer'
    )

    return p.parse_args()


def warmup_lr(step: int, warmup: int) -> float:
    return min(step, warmup) / warmup


def crude_nnl_estimate(samples: torch.Tensor) -> float:
    p = (samples + 1) / 2
    log_px = p * torch.log(p.clamp_min(1e-7)) + (1-p)*torch.log((1-p).clamp_min(1e-7))
    return -log_px.mean().item() / math.log(2)


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

@torch.no_grad()
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
    out_path = save_dir / f"cifar10_flow_epoch{epoch:03d}.png"
    img.save(out_path)
    print(f"→ saved generated grid to {out_path}")


def compute_entropy_loss(pred: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    B = pred.shape[0]
    D = pred.numel() // B
    x = pred.reshape(B, D)
    var = x.var(dim=0, unbiased=False) + eps
    return 0.5 * torch.sum(torch.log(2 * math.pi * math.e * var))


def main() -> Tuple[float,int,float]:
    args = parse_args()
    cfg = vars(args)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ib_tag = 'ib' if cfg.get('use_ib', False) else 'noib'
    checkpoint_dir = Path('checkpoints') / f"{cfg['matcher']}_cifar10_{ib_tag}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = checkpoint_dir / 'samples'
    sample_dir.mkdir(exist_ok=True, parents=True)

    use_wandb = args.wandb_project and not args.no_wandb and hasattr(__import__('wandb'), 'log')
    if use_wandb:
        run_name = args.wandb_run_name or checkpoint_dir.name
        wandb.init(project=args.wandb_project, name=run_name, config=cfg)

    device = torch.device(cfg['device'])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=cfg['num_workers'], drop_last=True)
    val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=cfg['fid_batch'], shuffle=False,
                             num_workers=cfg['num_workers'])

    # Conditional UNet
    net = UNetModel(
        dim=(3,32,32), num_channels=cfg['num_channel'], num_res_blocks=2,
        num_classes=10, class_cond=True,
        channel_mult=[1,2,2,2], num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.1
    ).to(device)

    # # TODO just for debug
    # net = UNetModel(
    #     dim=(3,32,32), num_channels=cfg['num_channel'], num_res_blocks=1,
    #     num_classes=10, class_cond=True,
    #     channel_mult=[1], num_heads=1, num_head_channels=4,
    #     attention_resolutions='16', dropout=0.1
    # ).to(device)

    ema_model = copy.deepcopy(net)
    if cfg.get('parallel'):
        net = torch.nn.DataParallel(net)
        ema_model = torch.nn.DataParallel(ema_model)

    score_model: Optional[UNetModel] = None
    if cfg['matcher']=='sb':
        score_model = UNetModel(
            dim=(3,32,32), num_channels=cfg['num_channel'], num_res_blocks=1,
            num_classes=10, class_cond=True,
            channel_mult=[1,2,2,2], num_heads=4, num_head_channels=64,
            attention_resolutions='16', dropout=0.1
        ).to(device)

    FM = {
        'cfm': ConditionalFlowMatcher,
        'ot': ExactOptimalTransportConditionalFlowMatcher,
        'sb': SchrodingerBridgeConditionalFlowMatcher,
        'target': TargetConditionalFlowMatcher,
    }[cfg['matcher']]()

    optimizer = optim.Adam(
        list(net.parameters()) + ([] if score_model is None else list(score_model.parameters())),
        lr=cfg['lr']
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                lr_lambda=lambda s: warmup_lr(s, cfg['warmup']))

    step, max_steps = 0, cfg['total_steps']
    pbar = tqdm(total=max_steps, dynamic_ncols=True)
    while step < max_steps:
        for x1, y in loader:
            if step >= max_steps:
                break
            x1, y = x1.to(device), y.to(device)
            x0 = torch.randn_like(x1)
            optimizer.zero_grad()

            # choose sampler
            if cfg['matcher'] in ['cfm','target']:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                u_pred = net(t, xt, y)
                score_loss = 0.0
            elif cfg['matcher'] == 'ot':
                t, xt, ut, _, y1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y1=y)
                # TODO lets try y1
                u_pred = net(t, xt, y1)
                score_loss = 0.0
            # else:  # 'sb'
            #     t, xt, ut, _, y1, eps = FM.guided_sample_location_and_conditional_flow(
            #         x0, x1, y1=y, return_noise=True
            #     )
            #     u_pred = net(t, xt, y1)
            #     lambda_t = FM.compute_lambda(t).view(-1,1,1,1)
            #     s = score_model(t, xt, y1)
            #     score_loss = F.mse_loss(lambda_t * s + eps, torch.zeros_like(eps))

            else:
                # assume y in [0..num_classes)
                num_classes = 10
                xt_chunks, ut_chunks, y1_chunks, eps_chunks, t_chunks = [], [], [], [], []
                for c in range(num_classes):
                    idx = (y == c).nonzero(as_tuple=True)[0]
                    if idx.numel() == 0:
                        continue
                    x0_c, x1_c, y_c = x0[idx], x1[idx], y[idx]
                    # 1) draw SB‐coupling *only* within this class
                    x0i, x1j, y0i, y1j = FM.ot_sampler.sample_plan_with_labels(
                        x0_c, x1_c, y1=y_c
                    )
                    # 2) sample the flow‐matching pair for these points
                    t_c, xt_c, ut_c, eps_c = FM.sample_location_and_conditional_flow(
                        x0i, x1j, return_noise=True
                    )
                    xt_chunks.append(xt_c)
                    ut_chunks.append(ut_c)
                    y1_chunks.append(y1j)
                    eps_chunks.append(eps_c)
                    t_chunks.append(t_c)

                # 3) recombine
                xt = torch.cat(xt_chunks, dim=0)
                ut = torch.cat(ut_chunks, dim=0)
                eps = torch.cat(eps_chunks, dim=0)
                t = torch.cat(t_chunks, dim=0)
                y1 = torch.cat(y1_chunks, dim=0)

                # 4) forward + losses
                u_pred = net(t, xt, y1)
                lambda_t = FM.compute_lambda(t).view(-1, 1, 1, 1)
                s = score_model(t, xt, y1)
                score_loss = F.mse_loss(lambda_t * s + eps, torch.zeros_like(eps))

            flow_loss = F.mse_loss(u_pred, ut)
            if cfg['use_ib']:
                kin = u_pred.pow(2).mean()
                ent = compute_entropy_loss(u_pred)
                loss = flow_loss + cfg['ib_lambda'] * kin - cfg['ib_beta'] * ent + score_loss
            else:
                loss = flow_loss + score_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip'])
            optimizer.step()
            scheduler.step()
            ema(net, ema_model, cfg['ema_decay'])

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
                    'ib_term': (cfg['ib_lambda'] * kin - cfg['ib_beta'] * ent).item(),
                    'λ·kinetic': (cfg['ib_lambda'] * kin).item(),
                    'β·entropy': -(cfg['ib_beta'] * ent).item(),
                })
            if use_wandb:
                wandb.log(log_dict)


            if (step + 1) % cfg['val_step'] == 0:
                # validation pass
                net.eval()
                val_loss_sum, n_val = 0.0, 0
                with torch.no_grad():
                    for xv, yv in val_loader:
                        xv, yv = xv.to(device), yv.to(device)
                        x0v = torch.randn_like(xv)
                        if cfg['matcher'] in ['cfm','target']:
                            tv, xtv, utv = FM.sample_location_and_conditional_flow(x0v, xv)
                        else:
                            tv, xtv, utv, *_ = FM.guided_sample_location_and_conditional_flow(x0v, xv, y1=yv)
                        v_pred = net(tv, xtv, yv)
                        val_loss_sum += F.mse_loss(v_pred, utv).item() * xv.size(0)
                        n_val += xv.size(0)
                print(f"Step {step+1}: val_loss={val_loss_sum/n_val:.4f}")
                avg_val_loss = val_loss_sum / n_val
                if use_wandb:
                    wandb.log({'val_loss': avg_val_loss, 'step': step+1})

                # --- sampling & FID ---
                USE_TORCH_DIFFEQ = True
                traj = sample_and_evaluate(
                    net, cfg['n_samples'], cfg['n_steps'], device, image_size=cfg['image_size'], image_channel=cfg['image_channel'],
                    use_torchdiffeq=USE_TORCH_DIFFEQ
                )
                samples = traj[-1]

                fid = compute_fid_from_tensor(samples, val_loader, device)
                nnl = -((samples + 1) / 2 * torch.log(((samples + 1) / 2).clamp_min(1e-7))).mean().item() / math.log(2)
                nfe = cfg['n_steps']
                final_fid, final_nfe, final_nnl = fid, nfe, nnl
                print(f"Epoch {step} Sample → FID: {fid:.2f}, NFE: {nfe}, NNL: {nnl:.3f}")
                if use_wandb:
                    wandb.log({'FID': fid, 'NFE': nfe, 'NNL': nnl})

                if (step + 1) % (cfg['save_step']) == 0:
                    save_samples_grid(samples, checkpoint_dir, step)
                    # save checkpoint
                    ckpt = {
                        'step': step+1,
                        'net': net.state_dict(),
                        'ema': ema_model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'sched': scheduler.state_dict(),
                    }
                    torch.save(ckpt, checkpoint_dir / f"ckpt_step{step+1:06d}.pt")


            step += 1
            pbar.update(1)
    pbar.close()

    return 0.0, max_steps, 0.0


if __name__ == '__main__':
    main()
