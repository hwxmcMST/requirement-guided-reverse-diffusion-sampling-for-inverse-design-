"""Independent FEniCS verification of saved Proposed designs (no PyTorch needed).

Physics adapted from elejeune11/Mechanical-MNIST:
generate_dataset/Uniaxial_Extension_FEA_test_FEniCS.py
https://github.com/elejeune11/Mechanical-MNIST

MIT License -- Copyright (c) 2019 elejeune11
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

DISPLACEMENTS = np.array([0, .001, .01, .1, .5, 1, 2, 4, 6, 8, 10, 12, 14])


def solve_energy(image):
    import dolfin as df
    from mshr import Rectangle, generate_mesh

    # Same mesh, quadrature, pixel orientation and continuous material mapping
    # as the original solver; image is normalized, so E = 1 + 99 * image.
    df.parameters['form_compiler']['cpp_optimize'] = True
    df.parameters['form_compiler']['representation'] = 'uflacs'
    df.parameters['form_compiler']['quadrature_degree'] = 2
    mesh = generate_mesh(Rectangle(df.Point(0, 0), df.Point(28, 28)), 140)
    space = df.VectorFunctionSpace(mesh, 'Lagrange', 2)
    coords = df.SpatialCoordinate(mesh)
    young = 1.0
    for col in range(28):
        for row in range(28):
            inside = (df.conditional(coords[0] >= col, 1, 0)
                      + df.conditional(coords[0] < col + 1, 1, 0)
                      + df.conditional(coords[1] >= row, 1, 0)
                      + df.conditional(coords[1] < row + 1, 1, 0))
            young += 99 * float(image[27 - row, col]) * df.conditional(inside > 3, 1, 0)
    nu = .3
    mu = young / (2 * (1 + nu))
    lam = young * nu / ((1 + nu) * (1 - 2 * nu))
    displacement = df.Function(space)
    trial, test = df.TrialFunction(space), df.TestFunction(space)
    deformation = df.Identity(2) + df.grad(displacement)
    jacobian = df.det(deformation)
    energy_density = (.5 * mu * (df.inner(deformation, deformation) - 3 - 2 * df.ln(jacobian))
                      + .5 * lam * (.5 * (jacobian**2 - 1) - df.ln(jacobian)))
    integral = energy_density * df.Measure('dx', domain=mesh)
    residual = df.derivative(integral, displacement, test)
    tangent = df.derivative(residual, displacement, trial)
    bottom = df.CompiledSubDomain('near(x[1], 0.0)')
    top = df.CompiledSubDomain('near(x[1], 28.0)')
    bottom_bc = df.DirichletBC(space, df.Constant((0., 0.)), bottom)
    energies = []
    for value in DISPLACEMENTS:
        top_bc = df.DirichletBC(space, df.Constant((0., float(value))), top)
        df.solve(residual == 0, displacement, [bottom_bc, top_bc], J=tangent,
                 solver_parameters={'newton_solver': {'error_on_nonconvergence': True}})
        energies.append(float(df.assemble(integral)))
        print(f'  displacement={value:g}: solved', flush=True)
    energies = np.asarray(energies)
    return energies - energies[0]


def satisfied(curves, req):
    if req['name'] == 'R1':
        lo, hi = req['psi_min'], req['psi_max']
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError('Invalid saved R1 bounds')
        return (curves[:, -1] >= lo) & (curves[:, -1] <= hi)
    if req['name'] == 'R2':
        limit, c1 = req['early_u_max'], req['c1']
        mask = DISPLACEMENTS <= limit
        if not np.isfinite([limit, c1]).all() or not mask.any():
            raise ValueError('Invalid saved R2 thresholds')
        return curves[:, mask].max(axis=1) <= c1
    raise ValueError('Expected R1 or R2 in metrics.json')


def load_inputs(folder, maximum=None):
    # Accept inference.py outputs and sampling_steps_ablation.py guided outputs.
    regular = (folder/'guided_samples.npy').exists()
    images = np.load(folder/('guided_samples.npy' if regular else 'samples.npy'))
    predicted = np.load(folder/('guided_pred_curves.npy' if regular else 'pred_curves.npy'))
    metadata = json.loads((folder/'metrics.json').read_text())
    if not regular and metadata.get('method') != 'guided':
        raise ValueError('Select the guided (Proposed) output folder')
    req = metadata.get('requirement_settings', metadata.get('requirement'))
    if not isinstance(req, dict):
        raise ValueError('metrics.json must contain the original requirement thresholds')
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) == 0:
        raise ValueError('Expected images of shape [N,1,28,28] or [N,28,28]')
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError('Images must be finite and normalized to [0,1]')
    if predicted.shape != (len(images), 13) or not np.isfinite(predicted).all():
        raise ValueError('Expected matching finite surrogate curves of shape [N,13]')
    satisfied(predicted, req)  # Validate thresholds before starting FEA.
    if maximum is not None:
        images, predicted = images[:maximum], predicted[:maximum]
    return images, predicted, req


def validate(images, predicted, req, out):
    out.mkdir(parents=True, exist_ok=True)
    curves = np.full((len(images), 13), np.nan)
    pred_flags = satisfied(predicted, req)
    rows = []
    for i, image in enumerate(images):
        print(f'{req["name"]}: sample {i + 1}/{len(images)}', flush=True)
        row = dict(sample_id=i, status='failed', surrogate_satisfied=int(pred_flags[i]),
                   fea_satisfied='', mae='', error='')
        try:
            curve = np.asarray(solve_energy(image), dtype=float)
            if curve.shape != (13,) or not np.isfinite(curve).all():
                raise RuntimeError('Invalid FEA energy curve')
            curves[i] = curve
            row.update(status='ok', fea_satisfied=int(satisfied(curve[None], req)[0]),
                       mae=float(np.abs(curve - predicted[i]).mean()))
        except Exception as exc:
            row['error'] = str(exc)
            print(f'  FAILED: {exc}', flush=True)
        rows.append(row)
        np.save(out/'fea_curves.npy', curves)
        with (out/'comparison.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
    ok = np.isfinite(curves).all(axis=1)
    flags = satisfied(curves[ok], req) if ok.any() else np.array([], dtype=bool)
    n_ok, n_all = int(ok.sum()), len(images)
    result = dict(requirement=req, n_samples=n_all, n_solved=n_ok, n_failed=n_all - n_ok,
                  surrogate_satisfaction_rate=float(pred_flags.mean()),
                  fea_n_satisfied=int(flags.sum()),
                  fea_satisfaction_rate=float(flags.mean()) if n_ok == n_all else None,
                  fea_confirmed_fraction_all=float(flags.sum() / n_all),
                  surrogate_false_positive_count=int((pred_flags[ok] & ~flags).sum()),
                  surrogate_false_negative_count=int((~pred_flags[ok] & flags).sum()),
                  curve_mae=float(np.abs(curves[ok] - predicted[ok]).mean()) if n_ok else None,
                  note='Misclassification counts and MAE use solved samples only; failures are unresolved. Confirmed fraction uses all samples.')
    (out/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--input-dir', type=Path, required=True, help='Folder containing saved Proposed samples, predictions and metrics.json')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--max-samples', type=int, help='Optional first N samples for a smoke test')
    args = p.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        p.error('max-samples must be positive')
    images, predicted, req = load_inputs(args.input_dir, args.max_samples)
    try:
        import dolfin
        import mshr
    except ImportError:
        p.error('Run with legacy FEniCS (dolfin + mshr) and NumPy installed; FEniCSx alone is not compatible.')
    if dolfin.MPI.size(dolfin.MPI.comm_world) != 1:
        p.error('Run as a single process, without mpirun')
    out = args.output_dir or args.input_dir/'physics_validation'
    if out.resolve() == args.input_dir.resolve():
        p.error('output-dir must differ from input-dir')
    out.mkdir(parents=True, exist_ok=True)
    config = dict(input_dir=str(args.input_dir.resolve()), requirement=req, n_samples=len(images),
                  dolfin_version=dolfin.__version__, displacements=DISPLACEMENTS.tolist(),
                  mesh_resolution=140, poisson_ratio=.3, young_modulus_range=[1, 100],
                  source='https://github.com/elejeune11/Mechanical-MNIST/blob/master/generate_dataset/Uniaxial_Extension_FEA_test_FEniCS.py')
    (out/'config.json').write_text(json.dumps(config, indent=2))
    validate(images, predicted, req, out)


if __name__ == '__main__':
    main()
