"""Run RPTM on Reuters, ALOI, COIL20, MSRC_v1, or Mfeat."""
import argparse
import csv
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from clustering import clustering_metrics, fit_clusters
from data import _cell_views, _read_mat, make_mask, materialise_view, preprocess
from model import RPTM, freeze_and_cache, rademacher
from train import generator, set_seed, train_representation, write_json


DATASETS = {
    "mfeat": {
        "file": "Mfeat.mat", "clusters": 10, "epochs": 80, "latent": 64,
        "ridge": 5e-3, "hidden": (512,), "recovery_hidden": 512,
        "lambda_corr": 1.0, "lambda_nbr": 0.3, "tau_n": 0.2,
        "keys": ("X", "Y"),
    },
    "msrc_v1": {
        "file": "MSRC_v1.mat", "clusters": 7, "epochs": 300, "latent": 64,
        "ridge": 2e-3, "hidden": (512,), "recovery_hidden": 512,
        "lambda_corr": 1.0, "lambda_nbr": 0.1, "tau_n": 0.2,
        "keys": None,
    },
    "reuters": {
        "file": "Reuters.mat", "clusters": 6, "epochs": 20, "latent": 64,
        "ridge": 5e-3, "hidden": (512,), "recovery_hidden": 512,
        "lambda_corr": 1.0, "lambda_nbr": 0.3, "tau_n": 0.2,
        "keys": ("X", "y"), "svd": 100, "tfidf": True,
        "preprocess": "l2", "pullback_backend": "cached",
    },
    "coil20": {
        "file": "COIL20.mat", "keys": ("X", "y"), "clusters": 20,
        "epochs": 200, "latent": 64, "hidden": (512,), "recovery_hidden": 512,
        "lr": 1e-3, "batch_size": 256, "preprocess": "zscore",
        "alpha_g": 1.0, "ridge": 5e-3, "mu_min": 1e-8,
        "trace_probes": 8, "r_max": math.pi / 2,
        "eps_w": 1e-8, "eps_ang": 1e-6, "eps_f": 1e-8,
        "cg_max_iter": 64, "cg_rtol": 1e-5, "cg_atol": 1e-9,
        "lambda_corr": 1.0, "lambda_nbr": 0.3, "tau_n": 0.2,
        "rho_cluster": 0.4, "tau_cluster": 0.1,
        "teacher_kmeans_iters": 20, "teacher_kmeans_restarts": 3,
        "lambda_cos": 0.7, "lambda_tan": 0.3, "beta": 16.0,
        "cluster_max_iter": 100, "cluster_tol": 1e-6,
        "backtracking_steps": 30, "backtracking_factor": 0.5,
        "match_batch_size": 1024, "n_streams": 8, "selection_tol": 0.02,
        "pullback_backend": "original",
    },
    "aloi": {
        "file": "ALOI.mat", "clusters": 100, "epochs": 80, "latent": 128,
        "ridge": 5e-3, "hidden": (256, 128), "recovery_hidden": 128,
        "lambda_corr": 0.1, "lambda_nbr": 1.0, "tau_n": 0.2,
        "keys": ("X", "y"),
    },
}


@dataclass
class Config:
    clusters: int = 10
    latent_dim: int = 64
    hidden: tuple = (512,)
    recovery_hidden: int = 512
    epochs: int = 80
    lr: float = 1e-3
    batch_size: int = 256
    preprocess: str = "zscore"
    alpha_g: float = 1.0
    ridge: float = 5e-3
    mu_min: float = 1e-8
    trace_probes: int = 8
    r_max: float = math.pi / 2
    eps_w: float = 1e-8
    eps_ang: float = 1e-6
    eps_f: float = 1e-8
    cg_max_iter: int = 64
    cg_rtol: float = 1e-5
    cg_atol: float = 1e-9
    lambda_corr: float = 1.0
    lambda_nbr: float = 0.1
    tau_n: float = 0.2
    rho_cluster: float = 0.4
    tau_cluster: float = 0.1
    teacher_kmeans_iters: int = 20
    teacher_kmeans_restarts: int = 3
    lambda_cos: float = 0.7
    lambda_tan: float = 0.3
    beta: float = 16.0
    cluster_max_iter: int = 100
    cluster_tol: float = 1e-6
    backtracking_steps: int = 30
    backtracking_factor: float = 0.5
    match_batch_size: int = 1024
    n_streams: int = 8
    selection_tol: float = 0.02
    pullback_backend: str = "original"


def resolve_data_file(name):
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_absolute():
        raise FileNotFoundError(f"Dataset file not found: {candidate}")
    roots = [Path(__file__).resolve().parent / "data", Path(__file__).resolve().parent,
             Path.cwd() / "data", Path.cwd()]
    for root in roots:
        path = root / candidate
        if path.is_file():
            return path.resolve()
        path = root / candidate.name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Could not find {candidate.name}. Place it in the project's data/ folder "
        "or pass its path with --data-file."
    )


def _find_key(mat, name):
    if name in mat:
        return name
    lookup = {key.lower(): key for key in mat}
    return lookup.get(name.lower())


def _label_key(mat):
    for name in ("y", "Y", "truth", "label", "labels", "gt"):
        key = _find_key(mat, name)
        if key is not None:
            return key
    raise ValueError("Could not find a label field in the MAT file.")


def _msrc_views(mat):
    key_map = {key.lower(): key for key in mat}
    for prefix in ("msr", "x"):
        keys = [key_map.get(f"{prefix}{i}") for i in range(1, 6)]
        if all(key is not None for key in keys):
            return [np.asarray(mat[key], dtype=np.float64) for key in keys]
    x_key = _find_key(mat, "X")
    if x_key is not None:
        views = _cell_views(mat[x_key])
        if views is not None and len(views) == 5:
            return [np.asarray(view.toarray() if hasattr(view, "toarray") else view,
                               dtype=np.float64) for view in views]
    available = sorted(key for key in mat if not key.startswith("__"))
    raise ValueError(
        "MSRC_v1 expects five view fields (msr1..msr5 or X1..X5), "
        f"or a five-view X cell array. Found fields: {available}"
    )


def load_dataset(name, data_file=None):
    spec = DATASETS[name]
    path = resolve_data_file(data_file or spec["file"])
    mat = _read_mat(path)
    if spec["keys"] is None:
        views = _msrc_views(mat)
        y_key = _label_key(mat)
    else:
        x_name, y_name = spec["keys"]
        x_key = _find_key(mat, x_name)
        y_key = _find_key(mat, y_name)
        if x_key is None or y_key is None:
            raise ValueError(
                f"{name} requires fields {x_name!r} and {y_name!r}; "
                f"available fields: {sorted(mat)}"
            )
        cells = _cell_views(mat[x_key])
        if cells is None:
            raise ValueError(f"Field {x_key!r} must contain a cell array of views.")
        cache_dir = Path(__file__).resolve().parent / "data_cache"
        if spec.get("svd"):
            cache_dir.mkdir(exist_ok=True)
        svd_dim = spec.get("svd")
        suffix = f".svd{svd_dim}" + (".tfidf" if spec.get("tfidf") else "")
        views = [
            materialise_view(
                view, svd_dim,
                cache_dir / f"{name}.v{index}{suffix}.npy" if svd_dim else None,
                tfidf=bool(spec.get("tfidf")),
            )
            for index, view in enumerate(cells)
        ]
    raw_y = np.asarray(mat[y_key]).squeeze()
    if raw_y.ndim != 1:
        raise ValueError(f"Label field {y_key!r} must be a one-dimensional vector.")
    n = raw_y.size
    oriented_views = []
    for index, view in enumerate(views):
        view = np.asarray(view, dtype=np.float64)
        if view.ndim != 2:
            raise ValueError(f"View {index + 1} must be a 2D matrix; got {view.shape}.")
        if view.shape[0] != n and view.shape[1] == n:
            view = view.T
        if view.shape[0] != n:
            raise ValueError(
                f"View {index + 1} has {view.shape[0]} samples but labels have {n}."
            )
        oriented_views.append(view)
    views = oriented_views
    _, labels = np.unique(raw_y, return_inverse=True)
    n_classes = len(np.unique(labels))
    if n_classes != spec["clusters"]:
        raise ValueError(f"{name} expects {spec['clusters']} classes; found {n_classes}.")
    info = {
        "name": name,
        "path": str(path),
        "n": int(n),
        "views": len(views),
        "clusters": int(n_classes),
        "dims": [int(view.shape[1]) for view in views],
    }
    return views, labels.astype(np.int64), info


def build_config(name, overrides):
    spec = DATASETS[name]
    preset = {field.name: spec[field.name] for field in fields(Config)
              if field.name in spec}
    preset.update(clusters=spec["clusters"], latent_dim=spec["latent"],
                  hidden=tuple(spec["hidden"]),
                  recovery_hidden=spec["recovery_hidden"])
    base = Config(**preset)
    cfg = replace(base, **overrides)
    if not (0 < cfg.lambda_cos <= 1 and
            0 <= cfg.lambda_tan <= 1 - cfg.lambda_cos + 1e-10):
        raise ValueError("Invalid Eq. (42) coefficients.")
    if cfg.epochs < 1 or cfg.latent_dim < 2 or cfg.ridge <= 0 or cfg.n_streams < 1:
        raise ValueError("Invalid configuration values.")
    return cfg


def run_one(raw_views, labels, info, cfg, mr, seed, device, root, log_every,
            model_seed=0):
    run_dir = root / f"MR_{mr:g}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    set_seed(model_seed)
    mask_seed = int(round(mr * 10000)) * 100003 + seed
    mask = make_mask(info["n"], info["views"], mr, device,
                     generator(device, mask_seed))
    xs, _ = preprocess(
        [torch.as_tensor(view, device=device, dtype=torch.float32) for view in raw_views],
        mask, cfg.preprocess,
    )
    model = RPTM(info["dims"], cfg).to(device=device, dtype=torch.float32)
    write_json(run_dir / "config.json", {
        "config": asdict(cfg), "missing_rate": mr, "seed": seed,
        "model_seed": model_seed, "data": info, "device": str(device),
        "torch_version": torch.__version__,
    })
    print(
        f"\n{info['name']} | MR={mr:g} seed={seed} | N={info['n']} "
        f"V={info['views']} K={cfg.clusters} complete={int(mask.all(1).sum())} "
        f"latent={cfg.latent_dim} epochs={cfg.epochs}", flush=True,
    )
    start = time.perf_counter()
    history = train_representation(
        model, xs, mask, cfg, generator(device, model_seed + 2000003), log_every
    )
    train_seconds = time.perf_counter() - start
    probes = rademacher(cfg, device, torch.float32,
                        generator(device, model_seed + 3000001))
    cache = freeze_and_cache(model, xs, mask, probes)
    start = time.perf_counter()
    fitted = fit_clusters(cache, mask, cfg, n_streams=cfg.n_streams)
    cluster_seconds = time.perf_counter() - start
    true_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    scores = clustering_metrics(true_labels, fitted["labels"])
    scores.update(
        missing_rate=mr, seed=seed, model_seed=model_seed,
        objective=fitted["objective"], converged=fitted["converged"],
        stop_reason=fitted["stop_reason"], iterations=fitted["iterations"],
        train_seconds=train_seconds, cluster_seconds=cluster_seconds,
        reconstruction_loss=float(history[-1]["rec"]),
        selected_seed_mode=fitted["selected_seed_mode"],
        n_streams=cfg.n_streams,
    )
    write_json(run_dir / "metrics.json", scores)
    np.save(run_dir / "labels.npy", fitted["labels"].detach().cpu().numpy())
    print(
        f"  ACC={scores['ACC']:.4f} NMI={scores['NMI']:.4f} "
        f"ARI={scores['ARI']:.4f} converged={scores['converged']} "
        f"train={train_seconds:.1f}s cluster={cluster_seconds:.1f}s",
        flush=True,
    )
    return scores


def summarize(root, results):
    rows = []
    for mr in sorted({result["missing_rate"] for result in results}):
        group = [result for result in results if result["missing_rate"] == mr]
        row = {"missing_rate": mr, "runs": len(group),
               "all_converged": all(result["converged"] for result in group)}
        for metric in ("ACC", "NMI", "ARI"):
            values = [result[metric] for result in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(row)
    write_json(root / "summary.json", rows)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="RPTM incomplete multi-view clustering")
    parser.add_argument("--dataset", default="msrc_v1", choices=sorted(DATASETS))
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--mr", nargs="+", type=float, default=[0.1,0.3,0.5,0.7])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--latent", type=int)
    parser.add_argument("--hidden", type=int, nargs="+")
    parser.add_argument("--recovery-hidden", type=int)
    parser.add_argument("--ridge", type=float)
    parser.add_argument("--lambda-corr", type=float)
    parser.add_argument("--lambda-nbr", type=float)
    parser.add_argument("--tau-n", type=float)
    parser.add_argument("--rho-cluster", type=float)
    parser.add_argument("--alpha-g", type=float)
    parser.add_argument("--lambda-cos", type=float)
    parser.add_argument("--lambda-tan", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--probes", type=int)
    parser.add_argument("--streams", type=int)
    parser.add_argument("--selection-tol", type=float)
    parser.add_argument("--preprocess", choices=["none", "l2", "zscore", "zscore_l2", "zscore_unit"])
    parser.add_argument("--pullback-backend", choices=["original", "cached"])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    overrides = {}
    fields = (
        ("epochs", "epochs"), ("latent", "latent_dim"), ("ridge", "ridge"),
        ("lambda_corr", "lambda_corr"), ("lambda_nbr", "lambda_nbr"),
        ("tau_n", "tau_n"), ("rho_cluster", "rho_cluster"),
        ("alpha_g", "alpha_g"), ("lambda_cos", "lambda_cos"),
        ("lambda_tan", "lambda_tan"), ("beta", "beta"),
        ("probes", "trace_probes"), ("streams", "n_streams"),
        ("selection_tol", "selection_tol"), ("preprocess", "preprocess"),
        ("pullback_backend", "pullback_backend"),
        ("recovery_hidden", "recovery_hidden"),
    )
    for argument, field in fields:
        value = getattr(args, argument)
        if value is not None:
            overrides[field] = value
    if args.hidden:
        overrides["hidden"] = tuple(args.hidden)
    cfg = build_config(args.dataset, overrides)
    if len(set(args.mr)) != len(args.mr) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Missing rates and seeds must be unique.")
    views, labels, info = load_dataset(args.dataset, args.data_file)
    if args.output:
        root = Path(args.output).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        root = Path(f"results_{args.dataset}_{timestamp}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any((root / f"MR_{mr:g}" / f"seed_{seed}").exists()
           for mr in args.mr for seed in args.seeds):
        raise FileExistsError(f"Output directory already contains one or more requested runs: {root}")
    print(
        f"Dataset={info['name']} N={info['n']} V={info['views']} "
        f"K={info['clusters']} dims={info['dims']} device={device} "
        f"latent={cfg.latent_dim} epochs={cfg.epochs} "
        f"lambda_corr={cfg.lambda_corr:g} lambda_nbr={cfg.lambda_nbr:g} "
        f"tau_n={cfg.tau_n:g} streams={cfg.n_streams}",
        flush=True,
    )
    results = []
    for mr in args.mr:
        for seed in args.seeds:
            results.append(run_one(views, labels, info, cfg, mr, seed, device,
                                   root, args.log_every, args.model_seed))
            summarize(root, results)
    print(f"Results saved to: {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
