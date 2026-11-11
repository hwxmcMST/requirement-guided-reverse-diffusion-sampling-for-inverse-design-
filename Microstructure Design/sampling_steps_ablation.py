"""Compare reverse sampling step counts without changing the training schedule."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data_utils import energy_path, load_energy_curves
from diffusion import load_diffusion_model, to_zero_one
from inference import (load_surrogate, pick_device, predict_curves,
                       estimate_x0, requirement_distance)
from requirement import parse_requirement, hard_satisfaction
from gradient_guided_inference import requirement_loss


def time_pairs(total, steps):
    if not 1 <= steps <= total:
        raise ValueError(f'sampling steps must be between 1 and {total}')
    times = np.rint(np.linspace(0, total - 1, steps)).astype(int)[::-1]
    if steps == 1:
        times[0] = total - 1
    return list(zip(times.tolist(), times[1:].tolist() + [-1]))


def transition(model, x, t, previous, schedule, total):
    a = schedule['alpha_bar'][t]
    b = schedule['alpha_bar'][previous] if previous >= 0 else x.new_tensor(1.)
    if previous == t - 1:
        beta = schedule['betas'][t]
        variance = schedule['posterior_var'][t]
    else:
        beta = 1 - a / b
        variance = beta * (1 - b) / (1 - a)
    eps = model(x, torch.full((len(x),), t, device=x.device, dtype=torch.long), total)
    clean = (x - (1 - a).sqrt() * eps) / a.sqrt()
    mean = (x - beta * eps / (1 - a).sqrt()) / (1 - beta).sqrt()
    return mean, clean, beta, variance


def sample(method, model, surrogate, scaler, schedule, total, steps, req, args, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x = torch.randn(args.n_samples, 1, 28, 28, device=device)
    mean_y = torch.tensor(scaler.mean, dtype=x.dtype, device=device)
    std_y = torch.tensor(scaler.std, dtype=x.dtype, device=device)
    for t, previous in time_pairs(total, steps):
        if method == 'gradient_guided' and args.guidance_scale > 0:
            x = x.detach().requires_grad_(True)
            mean, clean, beta, variance = transition(model, x, t, previous, schedule, total)
            curves = surrogate(to_zero_one(clean)) * std_y + mean_y
            grad = torch.autograd.grad(requirement_loss(curves, req).sum(), x)[0]
            if not torch.isfinite(grad).all():
                raise RuntimeError(f'Nonfinite gradient at timestep {t}')
            if args.grad_clip > 0:
                norm = grad.flatten(1).norm(dim=1).clamp(min=1e-12)
                grad = grad * (args.grad_clip / norm).clamp(max=1).view(-1, 1, 1, 1)
            with torch.no_grad():
                x = mean - args.guidance_scale * beta * grad
                if previous >= 0:
                    x += variance.sqrt() * torch.randn_like(x)
        else:
            with torch.no_grad():
                k = args.k_candidates if method == 'guided' else 1
                candidates, _, _, variance = transition(model, x.repeat_interleave(k, 0), t, previous, schedule, total)
                if previous >= 0:
                    candidates += variance.sqrt() * torch.randn_like(candidates)
                if method == 'guided' and k > 1:
                    clean = to_zero_one(estimate_x0(model, candidates, previous, schedule, total))
                    curves = predict_curves(surrogate, scaler, clean.cpu().numpy(), device, args.batch_size)
                    flags = hard_satisfaction(curves, req).reshape(args.n_samples, k)
                    distances = requirement_distance(curves, req).reshape(args.n_samples, k)
                    choices = []
                    for i in range(args.n_samples):
                        ok = np.flatnonzero(flags[i])
                        choices.append(int(rng.choice(ok)) if len(ok) else int(distances[i].argmin()))
                    indices = torch.arange(args.n_samples, device=device) * k + torch.tensor(choices, device=device)
                    x = candidates[indices]
                else:
                    x = candidates
        if not torch.isfinite(x).all():
            raise RuntimeError(f'Nonfinite samples at timestep {t}')
    return to_zero_one(x.detach()).cpu().numpy().astype(np.float32)


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requirements', nargs='+', choices=['R1', 'R2'], default=['R1', 'R2'])
    p.add_argument('--sampling-steps', nargs='+', type=int, default=[100, 250, 500, 1000])
    p.add_argument('--methods', nargs='+', choices=['unguided', 'guided', 'gradient_guided'], default=['unguided', 'guided', 'gradient_guided'])
    p.add_argument('--n-samples', type=int, default=30)
    p.add_argument('--k-candidates', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--guidance-scale', type=float, default=1.)
    p.add_argument('--grad-clip', type=float, default=10.)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    p.add_argument('--psi-min', type=float)
    p.add_argument('--psi-max', type=float)
    p.add_argument('--c1', type=float)
    p.add_argument('--early-u-max', type=float, default=1.)
    p.add_argument('--r1-q-min', type=float, default=0.6)
    p.add_argument('--r1-q-max', type=float, default=0.8)
    p.add_argument('--r2-q', type=float, default=0.3)
    p.add_argument('--diffusion-checkpoint', type=Path, default=root/'outputs/diffusion/diffusion_best.pt')
    p.add_argument('--surrogate-checkpoint', type=Path, default=root/'outputs/surrogate/surrogate_best.pt')
    p.add_argument('--output-dir', type=Path, default=root/'outputs/sampling_steps')
    args = p.parse_args()
    if min(args.n_samples, args.k_candidates, args.batch_size) < 1:
        p.error('sample, candidate and batch counts must be positive')
    if any(not np.isfinite(v) or v < 0 for v in [args.guidance_scale, args.grad_clip]):
        p.error('guidance-scale and grad-clip must be finite and nonnegative')
    reference = load_energy_curves(energy_path('train'))
    requirements = []
    for name in args.requirements:
        args.requirement = name
        requirements.append(parse_requirement(args, reference))
    device = pick_device(args.device)
    model, schedule, total, _ = load_diffusion_model(args.diffusion_checkpoint, device)
    surrogate, scaler = load_surrogate(args.surrogate_checkpoint, device)
    for net in (model, surrogate):
        net.requires_grad_(False)
    pairs = {str(s): time_pairs(total, s) for s in args.sampling_steps}
    out = args.output_dir / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(training_steps=total, timesteps=pairs, requirements=[asdict(r) for r in requirements],
                  sampler='respaced_ddpm_posterior_variance')
    (out/'config.json').write_text(json.dumps(config, indent=2))
    rows = []
    for steps in args.sampling_steps:
        for req in requirements:
            for method in args.methods:
                print(f'{req.name}: steps={steps}/{total}, method={method}, device={device}', flush=True)
                images = sample(method, model, surrogate, scaler, schedule, total, steps, req, args, device)
                curves = predict_curves(surrogate, scaler, images, device, args.batch_size)
                flags = hard_satisfaction(curves, req)
                dest = out/f'steps_{steps}'/req.name/method
                dest.mkdir(parents=True, exist_ok=True)
                np.save(dest/'samples.npy', images)
                np.save(dest/'pred_curves.npy', curves)
                row = dict(requirement=req.name, method=method, sampling_steps=steps, training_steps=total,
                           seed=args.seed, n_samples=args.n_samples, n_satisfied=int(flags.sum()),
                           satisfaction_rate=float(flags.mean()))
                (dest/'metrics.json').write_text(json.dumps(dict(row, requirement_settings=asdict(req)), indent=2))
                rows.append(row)
                with (out/'summary.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
