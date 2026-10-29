"""Tangent-Cosine clustering and spherical prototype optimization."""
import torch

from model import Pullback, tangent, unit


@torch.no_grad()
def rectangular_assignment(cost):
    """Minimum-cost assignment with distinct columns (rows <= columns, Eq.44)."""
    if cost.ndim != 2 or cost.shape[0] > cost.shape[1] or cost.shape[0] == 0:
        raise ValueError("assignment needs 0 < rows <= columns")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("nonfinite assignment cost")
    cost = cost.to(torch.float64)
    nr, nc = cost.shape
    u = cost.new_zeros(nr + 1)
    v = cost.new_zeros(nc + 1)
    p = torch.zeros(nc + 1, dtype=torch.long, device=cost.device)
    way = torch.zeros_like(p)
    for i in range(1, nr + 1):
        p[0] = i
        minv = cost.new_full((nc + 1,), float("inf"))
        used = torch.zeros(nc + 1, dtype=torch.bool, device=cost.device)
        j0 = 0
        while True:
            used[j0] = True
            i0 = int(p[j0])
            cur = cost[i0 - 1] - u[i0] - v[1:]
            improve = (~used[1:]) & (cur < minv[1:])
            minv[1:] = torch.where(improve, cur, minv[1:])
            way[1:] = torch.where(improve, j0, way[1:])
            available = minv[1:].masked_fill(used[1:], float("inf"))
            delta, next_col = available.min(0)
            cols = used.nonzero(as_tuple=True)[0]
            u.index_add_(0, p[cols], delta.expand(len(cols)))
            v[cols] -= delta
            minv[~used] -= delta
            j0 = int(next_col) + 1
            if int(p[j0]) == 0:
                break
        while j0 != 0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    cols = (p[1:] > 0).nonzero(as_tuple=True)[0] + 1
    result = torch.empty(nr, dtype=torch.long, device=cost.device)
    result[p[cols] - 1] = cols - 1
    return result


@torch.no_grad()
def nonempty_assignment(cost):
    """Eqs.(45)-(46): best distinct representatives plus baseline labels."""
    n, k = cost.shape
    if not 1 <= k <= n:
        raise ValueError("need 1 <= K <= N")
    baseline, labels = cost.min(1)
    representatives = rectangular_assignment((cost - baseline[:, None]).T)
    labels[representatives] = torch.arange(k, device=cost.device)
    return labels, representatives


def tangent_cosine(op, prototypes):
    """Eq.(42) cost; prototypes is [K,d] or paired [B,1,d]."""
    cfg, q = op.cfg, op.z
    p = prototypes[None] if prototypes.ndim == 2 else prototypes
    c = (q[:, None] * p).sum(-1).clamp(-1, 1)
    directions = p - c[..., None] * q[:, None]
    a = op.raw_quadratic(directions) / op.mu[:, None]
    dcos = (1 - c) / 2
    dtan = a / (a + cfg.beta)
    return cfg.lambda_cos * dcos + cfg.lambda_tan * ((1 + c) / 2) * dtan


class ClusterIndex:
    """Sample indices per cluster, sorted once per assignment."""

    def __init__(self, labels, clusters):
        self.ends = torch.cumsum(torch.bincount(labels, minlength=int(clusters)), 0).tolist()
        self.order = torch.argsort(labels, stable=True)

    def indices(self, k):
        end = self.ends[k]
        return self.order[0 if k == 0 else self.ends[k - 1]:end]


class Matcher:
    """Evaluates Eq.(42) on a frozen cache; the pullback operator is built once."""

    def __init__(self, cache, mask, cfg):
        self.cache, self.mask, self.cfg = cache, mask, cfg
        n = len(mask)
        step = int(cfg.match_batch_size)
        self._ops = []
        for start in range(0, n, step):
            idx = torch.arange(start, min(start + step, n), device=mask.device)
            self._ops.append((idx, Pullback(cache["weights"], cache["q"][idx], mask[idx],
                                            cfg, mu=cache["mu"][idx])))

    @torch.no_grad()
    def costs(self, prototypes):
        return torch.cat([tangent_cosine(op, prototypes) for _, op in self._ops])

    @torch.no_grad()
    def cluster_sums(self, prototypes, labels, only=None):
        """Per-cluster cost sums; only the `only` clusters are scored when given."""
        total = prototypes.new_zeros(len(prototypes))
        if only is not None and len(only) == 0:
            return total
        if only is None:
            for idx, op in self._ops:
                values = tangent_cosine(op, prototypes[labels[idx], None]).squeeze(1)
                total.index_add_(0, labels[idx], values)
            return total
        index = ClusterIndex(labels, len(prototypes))
        pieces = [x for x in (index.indices(int(k)) for k in only.tolist()) if len(x)]
        if not pieces:
            return total
        joined = torch.cat(pieces)
        step = int(self.cfg.match_batch_size)
        for start in range(0, len(joined), step):
            idx = joined[start:start + step]
            op = Pullback(self.cache["weights"], self.cache["q"][idx], self.mask[idx],
                          self.cfg, mu=self.cache["mu"][idx])
            values = tangent_cosine(op, prototypes[labels[idx], None]).squeeze(1)
            total.index_add_(0, labels[idx], values)
        return total

    def gradient(self, prototypes, labels):
        with torch.enable_grad():
            variable = prototypes.detach().clone().requires_grad_(True)
            gradient = torch.zeros_like(variable)
            for idx, op in self._ops:
                value = tangent_cosine(op, variable[labels[idx], None]).sum()
                gradient += torch.autograd.grad(value, variable)[0]
        return tangent(prototypes, gradient)


@torch.no_grad()
def prototype_step(matcher, prototypes, labels, max_trials=30, shrink=0.5,
                   min_angle=1e-6):
    """Eqs.(47)-(48) with a per-cluster backtracking line search."""
    gradient = matcher.gradient(prototypes, labels)
    before = matcher.cluster_sums(prototypes, labels)
    steps = prototypes.new_full((len(prototypes),), 1.0)
    pending = gradient.square().sum(1) > 0
    result = prototypes.clone()
    angles = prototypes.new_zeros(len(prototypes))
    trials = 0
    while trials < int(max_trials) and bool(pending.any()):
        trials += 1
        candidate = unit(prototypes - steps[:, None] * gradient)
        angle = torch.acos((candidate * prototypes).sum(1).clamp(-1.0, 1.0))
        stalled = pending & (angle <= float(min_angle))
        pending &= ~stalled
        if bool(pending.any()):
            after = matcher.cluster_sums(candidate, labels,
                                         only=pending.nonzero(as_tuple=True)[0])
            good = pending & torch.isfinite(after) & (after <= before)
        else:
            after = prototypes.new_zeros(len(prototypes))
            good = torch.zeros_like(pending)
        result[good] = candidate[good]
        before[good] = after[good]
        angles[good] = angle[good]
        pending &= ~good
        steps[pending] = steps[pending] * float(shrink)
    return result, angles


def fit_from(matcher, cfg, prototypes, angle_tol=None):
    """Run alternating assignment and spherical prototype updates."""
    costs = matcher.costs(prototypes)
    labels, _ = nonempty_assignment(costs)
    rows = torch.arange(len(labels), device=labels.device)
    objective = float(costs[rows, labels].double().sum())
    n = len(labels)
    converged, reason = False, "max_iter"
    changed, relative, max_angle = n, 0.0, 0.0
    iteration = 0
    for iteration in range(1, int(cfg.cluster_max_iter) + 1):
        new_p, angles = prototype_step(
            matcher, prototypes, labels, max_trials=cfg.backtracking_steps,
            shrink=cfg.backtracking_factor)
        new_costs = matcher.costs(new_p)
        new_labels, _ = nonempty_assignment(new_costs)
        new_objective = float(new_costs[rows, new_labels].double().sum())
        if new_objective > objective + 2e-6 * max(1.0, abs(objective)):
            raise RuntimeError("Objective increased beyond numerical tolerance.")
        changed = int((labels != new_labels).sum())
        relative = abs(objective - new_objective) / max(1.0, abs(objective))
        max_angle = float(angles.max())
        prototypes, labels, objective = new_p, new_labels, new_objective
        angle_ok = True if angle_tol is None else max_angle <= angle_tol
        if changed == 0 and relative <= cfg.cluster_tol and angle_ok:
            converged, reason = True, "converged"
            break
        reason = ("no_label_change_but_moving"
                  if changed == 0 and relative <= cfg.cluster_tol else "relabelling")
    if not converged and iteration >= int(cfg.cluster_max_iter):
        reason = "max_iter"
    return {"labels": labels, "prototypes": prototypes, "objective": objective,
            "converged": converged, "stop_reason": reason, "iterations": iteration}


@torch.no_grad()
def quantile_stream(q, k, offset=0.05, inner=0.17):
    """Medoid anchor + fixed-quantile D2 walk (no RNG)."""
    qn = unit(q)
    n = len(qn)
    first = int((qn @ unit(qn.mean(0, keepdim=True)).t()).argmax())
    chosen = [first]
    d2 = (1.0 - qn @ qn[first]).clamp_min(0.0)
    for t in range(1, int(k)):
        p = d2.clone()
        p[torch.tensor(chosen, device=qn.device)] = 0.0
        cumulative = torch.cumsum(p, 0)
        total = float(cumulative[-1])
        if total <= 0.0:
            j = next(i for i in range(n) if i not in chosen)
        else:
            q_t = (offset + inner * (t - 1)) % 1.0
            j = int((cumulative >= q_t * total).nonzero()[0])
            while j in chosen:
                j = (j + 1) % n
        chosen.append(j)
        d2 = torch.minimum(d2, (1.0 - qn @ qn[j]).clamp_min(0.0))
    return unit(qn[torch.tensor(chosen, device=qn.device)].clone())


def quantile_streams(q, k, n_streams=8, base=0.05, spacing=0.073, inner=0.17):
    return [quantile_stream(q, k, base + spacing * s, inner=inner)
            for s in range(int(n_streams))]


@torch.no_grad()
def size_balance(labels, clusters):
    counts = torch.bincount(labels, minlength=int(clusters)).to(torch.float64)
    share = (counts / counts.sum().clamp_min(1.0)).clamp_min(1e-12)
    return float(-(share * share.log()).sum())


def select_band(results, clusters, tol):
    """Keep the objective band, then the most balanced cluster sizes (no labels)."""
    entropies = [size_balance(r["labels"], clusters) for r in results]
    objectives = [float(r["objective"]) for r in results]
    best_index = min(range(len(results)), key=lambda i: objectives[i])
    best_obj = objectives[best_index]
    threshold = best_obj * (1.0 + tol) if tol > 0 else best_obj
    eligible = [i for i, o in enumerate(objectives) if o <= threshold]
    chosen_index = max(eligible, key=lambda i: entropies[i]) if tol > 0 else best_index
    chosen = results[chosen_index]
    chosen["selected_restart"] = chosen_index
    return chosen



@torch.no_grad()
def clustering_metrics(y_true, y_pred):
    """ACC / NMI / ARI on device; evaluation only."""
    _, truth = torch.unique(y_true, sorted=True, return_inverse=True)
    _, predicted = torch.unique(y_pred, sorted=True, return_inverse=True)
    n = len(truth)
    a, b = int(truth.max()) + 1, int(predicted.max()) + 1
    count = torch.bincount(truth * b + predicted, minlength=a * b).reshape(a, b).double()
    size = max(a, b)
    square = count.new_zeros((size, size))
    square[:a, :b] = count
    columns = rectangular_assignment(-square)
    acc = square[torch.arange(size, device=count.device), columns].sum() / n
    row, col = count.sum(1), count.sum(0)
    positive = count > 0
    ratio = (count * n) / (row[:, None] * col[None]).clamp_min(1)
    mi = ((count[positive] / n) * ratio[positive].log()).sum()
    pr, pc = row / n, col / n
    entropy = -(pr[pr > 0] * pr[pr > 0].log()).sum() - (pc[pc > 0] * pc[pc > 0].log()).sum()
    nmi = 2 * mi / entropy if float(entropy) > 0 else count.new_tensor(1.)
    choose2 = lambda x: (x * (x - 1) / 2).sum()
    pair_total = n * (n - 1) / 2
    same, rows, cols = choose2(count), choose2(row), choose2(col)
    expected = rows * cols / max(pair_total, 1)
    denominator = (rows + cols) / 2 - expected
    ari = (same - expected) / denominator if float(denominator) != 0 else count.new_tensor(1.)
    return {"ACC": float(acc), "NMI": float(nmi), "ARI": float(ari)}


def fit_clusters(cache, mask, cfg, n_streams=8):
    """Initialize prototypes, optimize each candidate, and select without labels."""
    matcher = Matcher(cache, mask, cfg)
    results = []
    streams = quantile_streams(cache["q"], cfg.clusters, n_streams)
    for index, prototypes in enumerate(streams):
        result = fit_from(matcher, cfg, prototypes, angle_tol=None)
        result["seed_mode"] = f"q{0.05 + 0.073 * index:.3f}"
        results.append(result)
    best = select_band(results, cfg.clusters, float(cfg.selection_tol))
    best["selected_seed_mode"] = results[best["selected_restart"]]["seed_mode"]
    return best
