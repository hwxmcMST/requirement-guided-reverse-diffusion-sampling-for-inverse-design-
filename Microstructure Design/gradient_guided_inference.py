"""Single-run gradient-guided diffusion for Mechanical MNIST R1 and R2."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from data_utils import DISPLACEMENTS, energy_path, load_energy_curves
from diffusion import extract, load_diffusion_model, to_zero_one
from inference import load_surrogate, pick_device, predict_curves
from requirement import evaluate_curves, hard_satisfaction, parse_requirement


def requirement_loss(curves, req):
    if req.kind == 'final_energy_range':
        y = curves[:, -1]
        width = req.psi_max - req.psi_min
        return (torch.relu(req.psi_min-y).square() +
                torch.relu(y-req.psi_max).square()) / width**2
    if req.kind != 'low_early_energy':
        raise ValueError(f'Unknown requirement kind: {req.kind}')
    mask = torch.as_tensor(DISPLACEMENTS <= req.early_u_max, device=curves.device)
    y = curves[:, mask].amax(dim=1)
    return torch.relu(y-req.c1).square() / max(abs(req.c1), 1e-8)**2


def gradient_sample(model, surrogate, scaler, req, schedule, n_steps, args, device):
    mean = torch.tensor(scaler.mean, dtype=torch.float32, device=device)
    std = torch.tensor(scaler.std, dtype=torch.float32, device=device)
    x = torch.randn(args.n_samples, 1, 28, 28, device=device)
    for step in reversed(range(n_steps)):
        x = x.detach().requires_grad_(True)
        t = torch.full((len(x),), step, dtype=torch.long, device=device)
        eps = model(x, t, n_steps)
        beta = extract(schedule['betas'], t, x)
        alpha = extract(schedule['alphas'], t, x)
        sqrt_omab = extract(schedule['sqrt_one_minus_alpha_bar'], t, x)
        clean = (x - sqrt_omab*eps) / extract(schedule['sqrt_alpha_bar'], t, x)
        curves = surrogate(to_zero_one(clean)) * std + mean
        loss = requirement_loss(curves, req)
        grad = torch.autograd.grad(loss.sum(), x)[0]
        if not torch.isfinite(grad).all():
            raise RuntimeError(f'Nonfinite gradient at step {step}')
        if args.grad_clip > 0:
            norm = grad.flatten(1).norm(dim=1).clamp(min=1e-12)
            scale = (args.grad_clip/norm).clamp(max=1).view(-1, 1, 1, 1)
            grad = grad * scale
        with torch.no_grad():
            x = (x - beta*eps/sqrt_omab) / alpha.sqrt()
            x -= args.guidance_scale * beta * grad
            if step > 0:
                x += extract(schedule['posterior_var'], t, x).sqrt() * torch.randn_like(x)
    return to_zero_one(x).cpu().numpy().astype(np.float32)


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--requirement', choices=['R1', 'R2'], required=True)
    parser.add_argument('--diffusion-checkpoint', type=Path, default=project/'outputs/diffusion/diffusion_best.pt')
    parser.add_argument('--surrogate-checkpoint', type=Path, default=project/'outputs/surrogate/surrogate_best.pt')
    parser.add_argument('--output-dir', type=Path, default=project/'outputs/gradient')
    parser.add_argument('--n-samples', type=int, default=30)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--psi-min', type=float)
    parser.add_argument('--psi-max', type=float)
    parser.add_argument('--c1', type=float)
    parser.add_argument('--early-u-max', type=float, default=1.0)
    parser.add_argument('--r1-q-min', type=float, default=0.6)
    parser.add_argument('--r1-q-max', type=float, default=0.8)
    parser.add_argument('--r2-q', type=float, default=0.3)
    args = parser.parse_args()
    if args.n_samples < 1 or not np.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        parser.error('n-samples must be positive; guidance-scale must be finite and nonnegative')
    if not np.isfinite(args.grad_clip) or args.grad_clip < 0:
        parser.error('grad-clip must be finite and nonnegative (0 disables clipping)')
    req = parse_requirement(args, load_energy_curves(energy_path('train')))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    model, schedule, n_steps, _ = load_diffusion_model(args.diffusion_checkpoint, device)
    surrogate, scaler = load_surrogate(args.surrogate_checkpoint, device)
    for net in (model, surrogate):
        net.requires_grad_(False)
    print(f'{args.requirement}: device={device}, diffusion_steps={n_steps}, n_samples={args.n_samples}', flush=True)
    images = gradient_sample(model, surrogate, scaler, req, schedule, n_steps, args, device)
    curves = predict_curves(surrogate, scaler, images, device, 256)
    metrics = evaluate_curves(curves, req)
    out = args.output_dir / args.requirement
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/'samples.npy', images)
    np.save(out/'pred_curves.npy', curves)
    np.savetxt(out/'scores.csv', np.column_stack((np.arange(len(images)), hard_satisfaction(curves, req))),
               delimiter=',', header='sample_id,hard_score', comments='', fmt='%d')
    summary = {'requirement': asdict(req), 'seed': args.seed, 'n_samples': args.n_samples,
               'diffusion_steps': n_steps, 'guidance_scale': args.guidance_scale,
               'grad_clip': args.grad_clip, 'n_satisfied': metrics['n_feasible'],
               'satisfaction_rate': metrics['feasible_rate']}
    (out/'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
