"""Data loading, incomplete-view masks and observed-only preprocessing."""
from decimal import Decimal, ROUND_FLOOR
import numpy as np
import torch


def _read_mat(path):
    from scipy.io import loadmat
    try:
        return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except (NotImplementedError, ValueError, OSError) as exc:
        try:
            import h5py
        except ImportError:
            raise RuntimeError("MAT file needs h5py for v7.3 format") from exc
        if not h5py.is_hdf5(path):
            raise exc
        with h5py.File(path, "r") as f:
            def read(node):
                if isinstance(node, h5py.Group):
                    return {k: read(v) for k, v in node.items()}
                a = node[()]
                if h5py.check_dtype(ref=node.dtype) is not None:
                    out = np.empty(a.shape, dtype=object)
                    for idx in np.ndindex(a.shape):
                        out[idx] = read(f[a[idx]])
                    return out.T
                return a.T if a.ndim >= 2 else a
            return {k: read(v) for k, v in f.items() if not k.startswith("#")}


def _cell_views(value):
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.size == 1:
            return _cell_views(value.flat[0])
        return [x if hasattr(x, "toarray") else np.asarray(x)
                for x in value.ravel(order="F")]
    if isinstance(value, (list, tuple)):
        return [x if hasattr(x, "toarray") else np.asarray(x) for x in value]
    return None


def materialise_view(view, svd_dim=None, cache=None, tfidf=False):
    """Convert one view to a dense array with optional TF-IDF and truncated SVD."""
    if not hasattr(view, "toarray"):
        return np.asarray(view, dtype=np.float64)
    if not svd_dim:
        dense = view.toarray().astype(np.float64)
        return _tfidf(dense) if tfidf else dense
    if cache is not None and cache.exists():
        return np.load(cache)
    from scipy.sparse.linalg import svds
    weighted = view.astype(np.float64)
    if tfidf:
        weighted = weighted.tocsr()
        doc_freq = np.asarray((weighted > 0).sum(axis=0)).ravel()
        idf = np.log((1.0 + weighted.shape[0]) / (1.0 + doc_freq)) + 1.0
        weighted = weighted.multiply(idf[None, :]).tocsr()
    u, s, _ = svds(weighted, k=int(svd_dim))
    order = np.argsort(-s)
    reduced = u[:, order] * s[order]
    if cache is not None:
        np.save(cache, reduced)
    return reduced


def _tfidf(dense):
    doc_freq = (dense > 0).sum(axis=0)
    idf = np.log((1.0 + dense.shape[0]) / (1.0 + doc_freq)) + 1.0
    return dense * idf[None, :]


def make_mask(n, v, mr, device, generator):
    """Paper IV-A.2: floor(MR*N) incomplete rows, 1..V-1 views removed at random."""
    if not 0 < mr < 1 or v < 2:
        raise ValueError("need 0 < MR < 1 and V >= 2")
    m = int((Decimal(str(mr)) * int(n)).to_integral_value(rounding=ROUND_FLOOR))
    if not 0 < m < n:
        raise ValueError("need both incomplete and complete rows")
    mask = torch.ones((n, v), device=device, dtype=torch.bool)
    selected = torch.randperm(n, device=device, generator=generator)[:m]
    counts = torch.randint(1, v, (m,), device=device, generator=generator)
    order = torch.rand((m, v), device=device, generator=generator).argsort(1)
    missing = torch.zeros((m, v), device=device, dtype=torch.bool)
    missing.scatter_(1, order, torch.arange(v, device=device)[None] < counts[:, None])
    mask[selected] = ~missing
    return mask


def preprocess(views, mask, mode="zscore"):
    """Per-view normalisation fitted on observed rows only; missing entries stay zero."""
    if mode not in {"none", "l2", "zscore", "zscore_l2", "zscore_unit"}:
        raise ValueError(mode)
    output, stats = [], []
    for v, x in enumerate(views):
        observed = mask[:, v]
        if not bool(observed.any()):
            raise ValueError(f"view {v + 1} has no observed row")
        a = x[observed]
        mean = torch.zeros_like(a[0])
        scale = torch.ones_like(a[0])
        if mode in {"zscore", "zscore_l2", "zscore_unit"}:
            mean = a.mean(0)
            std = a.std(0, unbiased=False)
            scale = torch.where(std > 1e-8, std, torch.ones_like(std))
            a = torch.where(std > 1e-8, (a - mean) / scale, torch.zeros_like(a))
        if mode == "zscore_unit":
            a = a / float(x.shape[1]) ** 0.5
        if mode in {"l2", "zscore_l2"}:
            a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
        clean = torch.zeros_like(x)
        clean[observed] = a
        output.append(clean)
        stats.append({"mean": mean, "scale": scale, "mode": mode})
    return output, stats


def calibration_masks(views, count, device):
    """Generate calibration masks by cycling through missing-view combinations."""
    import itertools
    rows = []
    sizes = list(range(1, int(views)))
    combos = {s: list(itertools.combinations(range(int(views)), s)) for s in sizes}
    for i in range(int(count)):
        s = sizes[i % len(sizes)]
        combo = combos[s][(i // len(sizes)) % len(combos[s])]
        row = torch.zeros(int(views), dtype=torch.bool, device=device)
        row[list(combo)] = True
        rows.append(row)
    return torch.stack(rows)
