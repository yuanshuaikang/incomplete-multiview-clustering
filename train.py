"""Stage-I optimization and label-free cluster-teacher construction."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from data import calibration_masks
from model import rademacher, representation_loss, unit


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


@torch.no_grad()
def _spherical_kmeans(z, k, n_iter, n_restarts, gen):
    n = len(z)
    if k < 2 or n < k:
        return None
    z = unit(z.detach())
    best_centers, best_objective = None, float("inf")
    eps = torch.finfo(z.dtype).eps
    for _ in range(max(1, int(n_restarts))):
        chosen = [int(torch.randint(n, (1,), device=z.device, generator=gen).item())]
        min_dist2 = (2.0 - 2.0 * (z @ z[chosen[0]])).clamp_min(0)
        while len(chosen) < k:
            weights = min_dist2.clone()
            if not bool(torch.isfinite(weights).all()) or float(weights.sum()) <= eps:
                unused = [i for i in range(n) if i not in set(chosen)]
                if not unused:
                    return best_centers
                next_idx = unused[0]
            else:
                next_idx = int(torch.multinomial(weights, 1, generator=gen).item())
                if next_idx in chosen:
                    unused = [i for i in range(n) if i not in set(chosen)]
                    if unused:
                        next_idx = unused[0]
            chosen.append(next_idx)
            d2 = (2.0 - 2.0 * (z @ z[next_idx])).clamp_min(0)
            min_dist2 = torch.minimum(min_dist2, d2)
        centers = z[torch.tensor(chosen, device=z.device, dtype=torch.long)].clone()
        for _ in range(max(1, int(n_iter))):
            sim = z @ centers.T
            labels = sim.argmax(dim=1)
            nearest_sim = sim.max(dim=1).values
            new_centers, reseeded = [], set()
            farthest_order = torch.argsort(nearest_sim).tolist()
            for c in range(k):
                members = z[labels == c]
                if len(members):
                    new_centers.append(unit(members.mean(dim=0, keepdim=True))[0])
                else:
                    idx = next((j for j in farthest_order if j not in reseeded),
                               farthest_order[0])
                    reseeded.add(idx)
                    new_centers.append(z[idx])
            updated = torch.stack(new_centers, dim=0)
            delta = (updated - centers).norm()
            centers = updated
            if float(delta) <= 1e-6:
                break
        sim = z @ centers.T
        objective = float((1.0 - sim.max(dim=1).values).mean().item())
        if objective < best_objective:
            best_objective, best_centers = objective, centers.detach().clone()
    return best_centers


@torch.no_grad()
def estimate_cluster_teacher(model, xs, mask, cfg, gen, epoch=1):
    teacher_ids = mask.all(1).nonzero(as_tuple=True)[0]
    k = int(cfg.clusters)
    if k < 2 or len(teacher_ids) < k:
        return None
    was_training = model.training
    model.eval()
    chunks = []
    for start in range(0, len(teacher_ids), cfg.batch_size):
        ids = teacher_ids[start:start + cfg.batch_size]
        chunks.append(model.encode([x[ids] for x in xs], mask[ids]))
    z_full = unit(torch.cat(chunks, dim=0))
    model.train(was_training)
    seed = (int(gen.initial_seed()) + int(epoch) * 104729) % (2**63 - 1)
    teacher_gen = torch.Generator(device=mask.device).manual_seed(seed)
    centers = _spherical_kmeans(z_full, k, cfg.teacher_kmeans_iters,
                                cfg.teacher_kmeans_restarts, teacher_gen)
    if centers is None:
        return None
    tau = max(float(cfg.tau_cluster), 1e-6)
    probs = torch.softmax((z_full @ centers.T) / tau, dim=1)
    all_probs = z_full.new_zeros((len(mask), k))
    all_probs[teacher_ids] = probs
    return all_probs.detach()


def train_representation(model, xs, mask, cfg, gen, log_every=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0)
    history = []
    n = len(mask)
    for epoch in range(1, cfg.epochs + 1):
        started = time.perf_counter()
        teacher = (estimate_cluster_teacher(model, xs, mask, cfg, gen, epoch)
                   if cfg.rho_cluster > 0 else None)
        order = torch.randperm(n, device=mask.device, generator=gen)
        accum = xs[0].new_zeros(4)
        batches = 0
        for start in range(0, n, cfg.batch_size):
            ids = order[start:start + cfg.batch_size]
            current_mask = mask[ids]
            complete = int(current_mask.all(1).sum())
            calibration_mask = (
                calibration_masks(current_mask.shape[1], complete, mask.device)
                if complete else current_mask.new_empty((0, current_mask.shape[1]))
            )
            probes = rademacher(cfg, mask.device, xs[0].dtype, gen)
            model.prepare_update()
            optimizer.zero_grad(set_to_none=True)
            batch_teacher = teacher[ids] if teacher is not None else None
            loss, stats = representation_loss(
                model, [x[ids] for x in xs], current_mask,
                calibration_mask, probes, batch_teacher
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"Non-finite gradient at epoch {epoch}.")
            optimizer.step()
            accum += stats
            batches += 1
        row = dict(zip(("loss", "rec", "corr", "nbr"), (accum / batches).tolist()))
        row.update(epoch=epoch, seconds=time.perf_counter() - started)
        history.append(row)
        if epoch == 1 or epoch % max(log_every, 1) == 0 or epoch == cfg.epochs:
            print(f"  epoch {epoch:4d}/{cfg.epochs} loss={row['loss']:.5f} "
                  f"rec={row['rec']:.5f} corr={row['corr']:.5f} "
                  f"nbr={row['nbr']:.5f} ({row['seconds']:.1f}s)", flush=True)
    return history
