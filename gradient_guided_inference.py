"""Gradient-guided acoustic diffusion. Reuses the existing project checkpoints.

Guidance uses a piecewise-differentiable interval-containment penalty.
Final feasibility always uses the original requirement.py criterion.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def predict_response(x, model, dc, sc, device):
    from data_utils import SQRT2, STRUT_MIN, STRUT_MAX, CENTER_MAX, CORNER_MAX
    def tensor(value):
        return torch.as_tensor(value, dtype=x.dtype, device=device)
    gs = dc['geometry_scaler']
    raw = x * tensor(gs['std']) + tensor(gs['mean'])
    # Match project_designs, while retaining gradients.
    r = raw[:, 0].clamp(STRUT_MIN, STRUT_MAX)
    raw = torch.stack((r, torch.maximum(raw[:, 1].clamp(max=CENTER_MAX),
                                       SQRT2 * r + 1e-6),
                       torch.maximum(raw[:, 2].clamp(max=CORNER_MAX),
                                     (SQRT2 + 1) * r + 1e-6)), dim=1)
    modes = tensor(dc['metadata']['modes'])
    waves = tensor(dc['metadata']['wavevectors'])
    m, w = torch.meshgrid(modes, waves, indexing='ij')
    q = torch.stack((m.flatten(), w.flatten()), dim=1)
    query = torch.cat((raw[:, None, :].expand(-1, len(q), -1),
                       q[None].expand(len(x), -1, -1)), dim=2)
    xs, ys = sc['x_scaler'], sc['y_scaler']
    pred = model(((query - tensor(xs['mean'])) / tensor(xs['std'])).reshape(-1, 5))
    pred = pred * tensor(ys['std']) + tensor(ys['mean'])
    return pred.reshape(len(x), len(modes), len(waves))


def requirement_loss(response, waves, requirement):
    from requirement import get_requirement, SEGMENTS, wavevector_path_coordinates
    waves = wavevector_path_coordinates(waves)
    req = get_requirement(requirement)
    losses = []
    for kind in ('stopbands', 'passbands'):
        for segment, low, high in req[kind]:
            a, b = SEGMENTS[segment]
            start, end = sorted((int(np.argmin(abs(waves - a))),
                                 int(np.argmin(abs(waves - b)))))
            selected = response[:, :, start:end]  # Same endpoint convention as original.
            starts = selected.amin(dim=2)
            ends = selected.amax(dim=2)
            order = starts.argsort(dim=1)
            starts, ends = starts.gather(1, order), ends.gather(1, order)
            if kind == 'stopbands':
                # Potential gaps include gaps below the original 1 kHz threshold.
                # This relaxation supplies gradients before a gap becomes feasible.
                left = torch.cat((torch.zeros_like(ends[:, :1]), ends), dim=1)
                right = torch.cat((starts, torch.full_like(starts[:, :1], 60)), dim=1)
                right = right.clamp(max=60)
                cost = (torch.relu(left - low).square() +
                        torch.relu(high - right).square() +
                        torch.relu(1.000001 - (right - left)).square())
                losses.append(cost.amin(dim=1) / (high - low)**2)
            else:
                per_sample = []
                for s, e in zip(starts, ends):
                    # Merge overlapping mode intervals as in the original criterion.
                    intervals = []
                    lo, hi = s[0], e[0]
                    for j in range(1, len(s)):
                        if bool((s[j] > hi).detach()):
                            intervals.append((lo, hi))
                            lo, hi = s[j], e[j]
                        else:
                            hi = torch.maximum(hi, e[j])
                    intervals.append((lo, hi))
                    costs = torch.stack([torch.relu(lo-low).square() +
                                         torch.relu(high-hi).square()
                                         for lo, hi in intervals])
                    per_sample.append(costs.min() / (high-low)**2)
                losses.append(torch.stack(per_sample))
    return torch.stack(losses, dim=1).sum(dim=1)


def sample(model, surrogate, schedule, dc, sc, args, device):
    x = torch.randn(args.n_samples, 3, device=device)
    waves = np.asarray(dc['metadata']['wavevectors'])
    for step in reversed(range(schedule.steps)):
        x = x.detach().requires_grad_(True)
        t = torch.full((len(x),), step, dtype=torch.long, device=device)
        beta, alpha, abar = schedule.betas[step], schedule.alphas[step], schedule.alpha_bars[step]
        noise = model(x, t)
        clean = (x - (1-abar).sqrt()*noise) / abar.sqrt()
        response = predict_response(clean, surrogate, dc, sc, device)
        loss = requirement_loss(response, waves, args.requirement)
        grad = torch.autograd.grad(loss.sum(), x)[0]
        if not torch.isfinite(grad).all():
            raise RuntimeError(f'Nonfinite gradient at step {step}')
        if args.grad_clip > 0:
            norm = grad.norm(dim=1, keepdim=True).clamp(min=1e-12)
            grad = grad * (args.grad_clip / norm).clamp(max=1)
        with torch.no_grad():
            x = (x - beta / (1-abar).sqrt()*noise) / alpha.sqrt()
            x -= args.guidance_scale * beta * grad
            if step > 0:
                x += beta.sqrt() * torch.randn_like(x)
    return x.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--requirement', default='T1', choices=[f'T{i}' for i in range(1, 11)])
    parser.add_argument('--n-samples', type=int, default=30)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.n_samples < 1 or args.guidance_scale < 0:
        parser.error('n-samples must be positive; guidance-scale must be nonnegative')
    sys.path.insert(0, str(args.project_dir.resolve()))
    from inference import load_diffusion, load_surrogate, evaluate_samples
    device = torch.device(args.device)
    model, schedule, dc = load_diffusion(args.project_dir / 'outputs/diffusion/diffusion_best.pt', device)
    surrogate, sc = load_surrogate(args.project_dir / 'outputs/surrogate/surrogate_best.pt', device)
    for net in (model, surrogate):
        net.requires_grad_(False)
    out = args.output_dir or args.project_dir / 'outputs' / f'gradient_{args.requirement}'
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    samples = sample(model, surrogate, schedule, dc, sc, args, device)
    result = evaluate_samples('gradient_guided', samples, dc, surrogate, sc,
                              args.requirement, device, 16384)
    result['projected_designs'].to_csv(out / f'designs_seed_{args.seed}.csv', index=False)
    result['scores'].to_csv(out / f'scores_seed_{args.seed}.csv', index=False)
    summary = {'requirement': args.requirement, 'n_samples': args.n_samples,
               'seed': args.seed, 'diffusion_steps': schedule.steps,
               'guidance_scale': args.guidance_scale, 'grad_clip': args.grad_clip,
               'n_satisfied': result['metrics']['n_satisfied'],
               'satisfaction_rate': result['metrics']['satisfaction_rate'],
               'note': 'Surrogate-based feasibility; guidance scale requires separate tuning.'}
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
