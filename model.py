"""RPTM representation stage, Eqs.(11)-(38): pullback correction + three-term loss."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import jvp, vjp, vmap


def unit(x, eps=1e-12):
    norm = x.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(x)
    fallback[..., 0] = 1
    return torch.where(norm > eps, x / norm.clamp_min(eps), fallback)


def tangent(z, a):
    if a.ndim == z.ndim + 1:
        z = z.unsqueeze(-2)
    return a - (a * z).sum(-1, keepdim=True) * z


def rademacher(cfg, device, dtype, generator):
    return (2 * torch.randint(2, (cfg.trace_probes, cfg.latent_dim),
                              device=device, generator=generator) - 1).to(dtype)


class Decoder(nn.Module):
    """Smooth tanh MLP; the pullback metric is built from exactly this map."""

    def __init__(self, widths):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(i, o) for i, o in zip(widths[:-1], widths[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers):
                x = torch.tanh(x)
        return x

    def snapshot(self):
        return tuple((layer.weight.detach().clone(), layer.bias.detach().clone())
                     for layer in self.layers)


class RPTM(nn.Module):
    """View encoders, view decoders and the recovery network phi (Eq.16)."""

    def __init__(self, dims, cfg):
        super().__init__()
        self.cfg = cfg
        self.dims = tuple(dims)
        latent = cfg.latent_dim
        self.encoders = nn.ModuleList()
        for dv in dims:
            widths = [dv, *cfg.hidden, latent]
            layers = []
            for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
                layers.append(nn.Linear(a, b))
                if i < len(widths) - 2:
                    layers.append(nn.ReLU())
            self.encoders.append(nn.Sequential(*layers))
        self.decoders = nn.ModuleList(
            Decoder([latent, *reversed(cfg.hidden), dv]) for dv in dims)
        self.recovery = nn.Sequential(
            nn.Linear(latent + len(dims), cfg.recovery_hidden),
            nn.ReLU(), nn.Linear(cfg.recovery_hidden, latent + 1))

    def encode(self, xs, mask):
        """Eq.34 input: z = Norm(mean over observed views of unit(E_v(x)))."""
        z = xs[0].new_zeros((len(mask), self.cfg.latent_dim))
        for v, encoder in enumerate(self.encoders):
            ids = mask[:, v].nonzero(as_tuple=True)[0]
            if len(ids):
                z = z.index_add(0, ids, unit(encoder(xs[v][ids])))
        count = mask.sum(1, keepdim=True)
        if bool((count == 0).any()):
            raise ValueError("every sample must keep at least one view")
        return unit(z / count)

    def signal(self, z, mask):
        predicted = self.recovery(torch.cat([z, mask.to(z.dtype)], dim=1))
        return (tangent(z, predicted[:, :-1]),
                self.cfg.r_max * torch.sigmoid(predicted[:, -1]))

    def correct(self, z, mask, weights=None, probes=None):
        op = Pullback(weights, z, mask, self.cfg, probes)
        return predict_correction(self, z, mask, op)[0]

    def prepare_update(self):
        self.train()

    def snapshot(self):
        return tuple(decoder.snapshot() for decoder in self.decoders)


class Pullback:
    """Frozen coefficient operator A = P(I + alpha*G)P + zz^T at one base point."""

    def __init__(self, snapshot, z, mask, cfg, probes=None, mu=None):
        self.weights = snapshot
        self.z = z.detach().clone()
        self.mask = mask.detach().clone()
        self.cfg = cfg
        self.count = self.mask.sum(1).to(z.dtype)
        self.backend = str(getattr(cfg, "pullback_backend", "original")).lower()
        if self.backend == "original":
            _, self._vjp = vjp(self.phi, self.z)
        elif self.backend == "cached":
            self._precompute_derivatives()
        else:
            raise ValueError(f"unknown pullback_backend {self.backend!r}")
        if mu is not None:
            self.mu = mu.detach().clone()
        else:
            if probes is None:
                raise ValueError("trace probes must be provided")
            a = tangent(self.z, probes[None].expand(len(z), -1, -1))
            with torch.no_grad():
                self.mu = (self.raw_quadratic(a).mean(1) / (z.shape[1] - 1)).clamp_min(
                    cfg.mu_min).detach()

    def phi(self, z):
        """Eq.(11): concat_v D_v(z) / sqrt(|Omega| * d_v)."""
        blocks = []
        for v, layers in enumerate(self.weights):
            x = z
            for i, (weight, bias) in enumerate(layers):
                x = F.linear(x, weight, bias)
                if i + 1 < len(layers):
                    x = torch.tanh(x)
            scale = self.mask[:, v].to(z.dtype) / (self.count * x.shape[-1]).sqrt()
            blocks.append(x * scale[:, None])
        return torch.cat(blocks, dim=1)

    def _precompute_derivatives(self):
        self._derivatives, self._scales = [], []
        with torch.no_grad():
            for v, layers in enumerate(self.weights):
                x = self.z
                derivatives = []
                for i, (w, b) in enumerate(layers):
                    if i < len(layers) - 1:
                        x = torch.tanh(F.linear(x, w, b))
                        derivatives.append((1.0 - x.square()).detach())
                self._derivatives.append(derivatives)
                dv = layers[-1][0].shape[0]
                self._scales.append((self.mask[:, v].to(self.z.dtype)
                                     / (self.count * dv).sqrt()).detach())

    def _cached_jvp(self, a):
        extra = a.ndim == 3
        blocks = []
        for layers, derivatives, scale in zip(self.weights, self._derivatives,
                                              self._scales):
            t = a
            for i, (w, _) in enumerate(layers):
                t = torch.matmul(t, w.t())
                if i < len(derivatives):
                    t = t * (derivatives[i][:, None, :] if extra else derivatives[i])
            blocks.append(t * (scale[:, None, None] if extra else scale[:, None]))
        return torch.cat(blocks, dim=-1)

    def _cached_vjp(self, b):
        pieces, start = [], 0
        for layers, derivatives, scale in zip(self.weights, self._derivatives,
                                              self._scales):
            dv = layers[-1][0].shape[0]
            t = b[:, start:start + dv] * scale[:, None]
            start += dv
            for i in range(len(layers) - 1, -1, -1):
                t = torch.matmul(t, layers[i][0])
                if i > 0:
                    t = t * derivatives[i - 1]
            pieces.append(t)
        return sum(pieces[1:], pieces[0])

    def jvp(self, a):
        if self.backend == "cached":
            return self._cached_jvp(a)
        if a.ndim == 3:
            return vmap(lambda t: jvp(self.phi, (self.z,), (t,))[1],
                        in_dims=1, out_dims=1)(a)
        return jvp(self.phi, (self.z,), (a,))[1]

    def raw_quadratic(self, a):
        return self.jvp(a).square().sum(-1) + self.cfg.ridge * a.square().sum(-1)

    def G(self, a):
        jv = self.jvp(a)
        jt = self._cached_vjp(jv) if self.backend == "cached" else self._vjp(jv)[0]
        return (jt + self.cfg.ridge * a) / self.mu[:, None]

    def A(self, a):
        projected = tangent(self.z, a)
        return (projected + self.cfg.alpha_g * tangent(self.z, self.G(projected))
                + self.z * (self.z * a).sum(-1, keepdim=True))


def conjugate_gradient(op, rhs):
    """Batched SPD conjugate gradient, Eqs.(17)-(18)."""
    cfg = op.cfg
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rr = r.square().sum(1)
    threshold = (cfg.cg_atol + cfg.cg_rtol * rhs.norm(dim=1)).square()
    active = rr > threshold
    tiny = torch.finfo(rhs.dtype).tiny
    for _ in range(cfg.cg_max_iter):
        if not bool(active.any()):
            break
        ap = op.A(p)
        pap = (p * ap).sum(1)
        if bool(((pap <= 0) & active).any()):
            raise FloatingPointError("CG found a nonpositive curvature")
        step = torch.where(active, rr / pap.clamp_min(tiny), torch.zeros_like(rr))
        x = x + step[:, None] * p
        r = r - step[:, None] * ap
        next_rr = r.square().sum(1)
        next_active = next_rr > threshold
        beta = torch.where(active & next_active, next_rr / rr.clamp_min(tiny),
                           torch.zeros_like(rr))
        p = torch.where(next_active[:, None], r + beta[:, None] * p, torch.zeros_like(p))
        rr, active = next_rr, next_active
    residual = (op.A(x) - rhs).norm(dim=1)
    allowed = 5 * (cfg.cg_atol + cfg.cg_rtol * rhs.norm(dim=1))
    if not bool(torch.isfinite(x).all()) or bool((residual > allowed).any()):
        raise RuntimeError("CG residual check failed")
    return x


class _StoppedSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs, op):
        ctx.op = op
        with torch.no_grad():
            return conjugate_gradient(op, rhs)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            return conjugate_gradient(ctx.op, grad_output), None


def solve(op, rhs):
    return _StoppedSolve.apply(rhs, op)


def exponential_correction(z, w, radius, eps_w):
    """Eq.(21): spherical step of the predicted radius along w."""
    norm = w.norm(dim=1, keepdim=True)
    direction = w / norm.clamp_min(eps_w)
    q = radius.cos()[:, None] * z + radius.sin()[:, None] * direction
    return torch.where(norm > eps_w, q, z)


def predict_correction(model, z, mask, op):
    f, radius = model.signal(z, mask)
    w = solve(op, f)
    return exponential_correction(z, w, radius, model.cfg.eps_w), f, radius


@torch.no_grad()
def calibration_targets(z, reference, op):
    """Eqs.(25)-(30), including coincident and near-antipodal pairs."""
    cfg = op.cfg
    z, reference = z.detach(), reference.detach()
    c = (z * reference).sum(1).clamp(-1, 1)
    f = reference - c[:, None] * z
    s = f.norm(dim=1)
    nondegenerate = s > cfg.eps_ang
    valid = nondegenerate | (c >= 0)
    f = torch.where(nondegenerate[:, None], f, torch.zeros_like(f))
    w = conjugate_gradient(op, f)
    u = w / w.norm(dim=1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
    radius = torch.atan2((reference * u).sum(1), c).clamp(0, cfg.r_max)
    radius = torch.where(nondegenerate, radius, torch.zeros_like(radius))
    q = radius.cos()[:, None] * z + radius.sin()[:, None] * u
    return {"f": f, "r": radius, "q": q, "valid": valid,
            "nondegenerate": nondegenerate}


def reconstruction_sum(model, z, xs, mask):
    """Eq.(33): mean squared error per observed sample-view."""
    total = z.sum() * 0
    terms = 0
    for v, decoder in enumerate(model.decoders):
        ids = mask[:, v].nonzero(as_tuple=True)[0]
        if len(ids):
            total = total + (decoder(z[ids]) - xs[v][ids]).square().mean(1).sum()
            terms += len(ids)
    return total, terms


def correction_loss(q, f, radius, target, cfg):
    valid, nondegenerate = target["valid"], target["nondegenerate"]
    loss = q.sum() * 0
    if bool(valid.any()):
        loss = (1 - (q[valid] * target["q"][valid]).sum(1)
                + ((radius[valid] - target["r"][valid]) / cfg.r_max).square()).mean()
    if bool(nondegenerate.any()):
        a = f[nondegenerate]
        b = target["f"][nondegenerate]
        a = a / a.norm(dim=1, keepdim=True).clamp_min(cfg.eps_f)
        b = b / b.norm(dim=1, keepdim=True).clamp_min(cfg.eps_f)
        loss = loss + (1 - (a * b).sum(1)).mean()
    return loss


def neighborhood_loss(q, reference, valid, cfg, teacher_probs=None):
    """Eq.(37) local KL, with the optional label-free cluster-aware target."""
    q, reference = q[valid], reference[valid].detach()
    if teacher_probs is not None:
        teacher_probs = teacher_probs[valid].detach()
    n = len(q)
    if n < 2:
        return q.sum() * 0
    keep = ~torch.eye(n, dtype=torch.bool, device=q.device)
    logits_p = ((reference @ reference.T) / cfg.tau_n)[keep].reshape(n, n - 1)
    p_local = logits_p.softmax(dim=1).detach()
    logits_q = ((q @ q.T) / cfg.tau_n)[keep].reshape(n, n - 1)
    log_q = logits_q.log_softmax(dim=1)
    if teacher_probs is None or teacher_probs.ndim != 2 or teacher_probs.shape[1] < 2:
        p_target = p_local
    else:
        pi = teacher_probs.clamp_min(0)
        pi = pi / pi.sum(dim=1, keepdim=True).clamp_min(1e-12)
        affinity = (pi @ pi.T)[keep].reshape(n, n - 1)
        affinity_sum = affinity.sum(dim=1, keepdim=True)
        p_cls = affinity / affinity_sum.clamp_min(1e-12)
        top2 = pi.topk(2, dim=1).values
        confidence = ((top2[:, 0] - top2[:, 1]) / 0.3).clamp(0.0, 1.0)
        hard = pi.argmax(dim=1)
        same_cluster = (hard[:, None] == hard[None, :]) & keep
        usable = (affinity_sum.squeeze(1) > 1e-12) & same_cluster.any(dim=1)
        weight = float(cfg.rho_cluster) * confidence * usable.to(confidence.dtype)
        p_cls = torch.where(usable[:, None], p_cls, p_local)
        p_target = (1.0 - weight[:, None]) * p_local + weight[:, None] * p_cls
        p_target = p_target / p_target.sum(dim=1, keepdim=True).clamp_min(1e-12)
        p_target = p_target.detach()
    log_p = p_target.clamp_min(1e-12).log()
    return (p_target * (log_p - log_q)).sum(dim=1).mean()


def representation_loss(model, xs, mask, calibration_mask, probes, teacher_probs=None):
    """Eqs.(32)-(38): reconstruction + recovery calibration + neighbourhood KL."""
    cfg = model.cfg
    z = model.encode(xs, mask)
    weights = model.snapshot()
    complete = mask.all(1)
    incomplete = (~complete).nonzero(as_tuple=True)[0]
    data_sum, data_terms = reconstruction_sum(model, z, xs, mask)
    if len(incomplete):
        with torch.no_grad():
            q = model.correct(z[incomplete], mask[incomplete], weights, probes)
        part, count = reconstruction_sum(model, q.detach(),
                                         [x[incomplete] for x in xs], mask[incomplete])
        data_sum, data_terms = data_sum + part, data_terms + count
    data_rec = data_sum / data_terms
    cal_rec = corr = nbr = z.sum() * 0
    ids = complete.nonzero(as_tuple=True)[0]
    if len(ids):
        if len(calibration_mask) != len(ids):
            raise ValueError("calibration mask count differs from complete rows")
        cal_x = [x[ids] for x in xs]
        z_cal = model.encode(cal_x, calibration_mask)
        reference = z[ids].detach()
        op = Pullback(weights, z_cal, calibration_mask, cfg, probes)
        target = calibration_targets(z_cal, reference, op)
        q_cal, f, radius = predict_correction(model, z_cal, calibration_mask, op)
        corr = correction_loss(q_cal, f, radius, target, cfg)
        batch_teacher = teacher_probs[ids] if teacher_probs is not None else None
        nbr = neighborhood_loss(q_cal, reference, target["valid"], cfg, batch_teacher)
        full_mask = torch.ones_like(calibration_mask)
        cal_sum, cal_terms = reconstruction_sum(model, z_cal, cal_x, full_mask)
        valid = target["valid"].nonzero(as_tuple=True)[0]
        if len(valid):
            part, count = reconstruction_sum(model, target["q"][valid],
                                             [x[valid] for x in cal_x],
                                             full_mask[valid])
            cal_sum, cal_terms = cal_sum + part, cal_terms + count
        cal_rec = cal_sum / cal_terms
    rec = data_rec + cal_rec
    loss = rec + cfg.lambda_corr * corr + cfg.lambda_nbr * nbr
    return loss, torch.stack([loss.detach(), rec.detach(), corr.detach(), nbr.detach()])


@torch.no_grad()
def freeze_and_cache(model, xs, mask, probes):
    """Freeze the model and cache corrected representations and geometry weights."""
    model.eval()
    model.requires_grad_(False)
    weights = model.snapshot()
    all_q, all_mu = [], []
    for start in range(0, len(mask), model.cfg.batch_size):
        sl = slice(start, start + model.cfg.batch_size)
        m = mask[sl]
        z = model.encode([x[sl] for x in xs], m)
        q = z.clone()
        ids = (~m.all(1)).nonzero(as_tuple=True)[0]
        if len(ids):
            q[ids] = model.correct(z[ids], m[ids], weights, probes)
        op = Pullback(weights, q, m, model.cfg, probes)
        all_q.append(q)
        all_mu.append(op.mu)
    return {"q": torch.cat(all_q), "mu": torch.cat(all_mu), "weights": weights}
