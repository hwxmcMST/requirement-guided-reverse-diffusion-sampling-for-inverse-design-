"""Sampling-step sensitivity using the trained schedule and respaced DDPM."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffusion import get_device
from inference import (load_diffusion, load_surrogate, decode_designs,
                       predict_band_structure, choose_candidates, evaluate_samples)
from gradient_guided_inference import predict_response, requirement_loss
from requirement import list_requirements, requirement_overlap_ratio


def time_pairs(total, steps):
    if not 1 <= steps <= total:
        raise ValueError(f'sampling steps must be between 1 and {total}')
    times = np.rint(np.linspace(0, total - 1, steps)).astype(int)[::-1]
    if steps == 1:
        times[0] = total - 1
    return list(zip(times.tolist(), times[1:].tolist() + [-1]))


def transition(model, x, t, previous, schedule):
    """Fixed-large DDPM variance, matching the original sampler at full steps."""
    a = schedule.alpha_bars[t]
    b = schedule.alpha_bars[previous] if previous >= 0 else x.new_tensor(1.)
    alpha = a / b
    # Use the original beta directly for adjacent steps to avoid cancellation.
    beta = schedule.betas[t] if previous == t - 1 else 1 - alpha
    alpha = 1 - beta
    noise = model(x, torch.full((len(x),), t, device=x.device, dtype=torch.long))
    clean = (x - (1 - a).sqrt() * noise) / a.sqrt()
    mean = (x - beta / (1 - a).sqrt() * noise) / alpha.sqrt()
    return mean, clean, beta


def sample(method, model, surrogate, schedule, dc, sc, args, steps, req, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x = torch.randn(args.n_samples, 3, device=device)
    waves = np.asarray(dc['metadata']['wavevectors'])
    modes = np.asarray(dc['metadata']['modes'])
    for t, previous in time_pairs(schedule.steps, steps):
        if method == 'gradient_guided' and args.guidance_scale > 0:
            x = x.detach().requires_grad_(True)
            mean, clean, beta = transition(model, x, t, previous, schedule)
            loss = requirement_loss(predict_response(clean, surrogate, dc, sc, device), waves, req)
            grad = torch.autograd.grad(loss.sum(), x)[0]
            if not torch.isfinite(grad).all():
                raise RuntimeError(f'Nonfinite gradient at timestep {t}')
            if args.grad_clip > 0:
                grad = grad * (args.grad_clip / grad.norm(dim=1, keepdim=True).clamp(min=1e-12)).clamp(max=1)
            with torch.no_grad():
                x = mean - args.guidance_scale * beta * grad
                if previous >= 0:
                    x += beta.sqrt() * torch.randn_like(x)
        else:
            with torch.no_grad():
                k = args.k_candidates if method == 'guided' else 1
                flat = x.repeat_interleave(k, dim=0)
                candidates, _, beta = transition(model, flat, t, previous, schedule)
                if previous >= 0:
                    candidates += beta.sqrt() * torch.randn_like(candidates)
                if method == 'guided' and k > 1:
                    # -1 is clean; index 0 still has noise and must be denoised.
                    clean = transition(model, candidates, previous, -1, schedule)[1] if previous >= 0 else candidates
                    _, designs = decode_designs(clean.cpu().numpy(), dc['geometry_scaler'], dc['metadata']['geometry_bounds'])
                    response, _ = predict_band_structure(surrogate, designs, modes, waves,
                                                        sc['x_scaler'], sc['y_scaler'], device,
                                                        args.eval_batch_size, False)
                    scores = requirement_overlap_ratio(response, waves, req).reshape(args.n_samples, k)
                    choices = choose_candidates(scores >= 1 - 1e-6, scores, rng)
                    x = candidates.reshape(args.n_samples, k, 3)[torch.arange(args.n_samples, device=device), torch.as_tensor(choices, device=device)]
                else:
                    x = candidates
        if not torch.isfinite(x).all():
            raise RuntimeError(f'Nonfinite samples at timestep {t}')
    return x.detach().cpu().numpy().astype(np.float32)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirements', nargs='+', choices=list_requirements(), default=list_requirements())
    parser.add_argument('--sampling-steps', nargs='+', type=int, default=[10, 25, 50, 100])
    parser.add_argument('--methods', nargs='+', choices=['unguided', 'guided', 'gradient_guided'], default=['unguided', 'guided', 'gradient_guided'])
    parser.add_argument('--n-samples', type=int, default=30)
    parser.add_argument('--k-candidates', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--guidance-scale', type=float, default=1.)
    parser.add_argument('--grad-clip', type=float, default=10.)
    parser.add_argument('--eval-batch-size', type=int, default=16384)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--diffusion-checkpoint', type=Path, default=root/'outputs/diffusion/diffusion_best.pt')
    parser.add_argument('--surrogate-checkpoint', type=Path, default=root/'outputs/surrogate/surrogate_best.pt')
    parser.add_argument('--output-dir', type=Path, default=root/'outputs/sampling_steps')
    args = parser.parse_args()
    if min(args.n_samples, args.k_candidates, args.eval_batch_size) < 1:
        parser.error('sample count, candidate count and batch size must be positive')
    if any(not np.isfinite(v) or v < 0 for v in [args.guidance_scale, args.grad_clip]):
        parser.error('guidance scale and grad clip must be finite and nonnegative')
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    model, schedule, dc = load_diffusion(args.diffusion_checkpoint, device)
    surrogate, sc = load_surrogate(args.surrogate_checkpoint, device)
    for net in (model, surrogate):
        net.requires_grad_(False)
    for steps in args.sampling_steps:
        time_pairs(schedule.steps, steps)
    out = args.output_dir / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(training_steps=schedule.steps, sampler='respaced_ddpm_fixed_large',
                  timesteps={str(s): time_pairs(schedule.steps, s) for s in args.sampling_steps},
                  note='Original surrogate feasibility criterion; clean estimation includes noisy index 0. Single-seed sensitivity, not a statistical robustness test.')
    (out/'config.json').write_text(json.dumps(config, indent=2))
    rows = []
    for steps in args.sampling_steps:
        for req in args.requirements:
            for method in args.methods:
                print(f'{req}: steps={steps}/{schedule.steps}, method={method}, device={device}', flush=True)
                designs = sample(method, model, surrogate, schedule, dc, sc, args, steps, req, device)
                result = evaluate_samples(method, designs, dc, surrogate, sc, req, device, args.eval_batch_size)
                dest = out/f'steps_{steps}'/req/method
                dest.mkdir(parents=True, exist_ok=True)
                np.save(dest/'samples_normalized.npy', designs)
                result['projected_designs'].to_csv(dest/'designs.csv', index=False)
                result['scores'].to_csv(dest/'scores.csv', index=False)
                row = dict(requirement=req, method=method, sampling_steps=steps, training_steps=schedule.steps,
                           seed=args.seed, n_samples=args.n_samples, k_candidates=args.k_candidates if method == 'guided' else 1,
                           n_satisfied=result['metrics']['n_satisfied'], satisfaction_rate=result['metrics']['satisfaction_rate'])
                (dest/'metrics.json').write_text(json.dumps(row, indent=2))
                rows.append(row)
                pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
