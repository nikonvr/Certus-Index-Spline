#!/usr/bin/env python3


# -*- coding: utf-8 -*-


"""Spline spectral objective (masked grid, PWL nk, sigma codecs)."""


from __future__ import annotations


from typing import Any


import numpy as np


from certus_core import N_MIN_LIMIT, N_MAX_LIMIT


from certus_index_utils import _ratio_theoretical_from_nk, spectral_rmse_weights


import functools

# Thread-safe LRU cache for weights, avoiding global thrashing
@functools.lru_cache(maxsize=16)
def _cached_spectral_rmse_weights_inner(key: bytes) -> np.ndarray:
    lam_f = np.frombuffer(key, dtype=np.float64)
    return spectral_rmse_weights(lam_f).astype(np.float64, copy=False)

def _cached_spectral_rmse_weights(lam_f: np.ndarray) -> np.ndarray:
    """Trapezoidal ln lambda weights for ``lam_f``; avoids ~N identical calls."""
    key = np.asarray(lam_f, dtype=np.float64).ravel().tobytes()
    return _cached_spectral_rmse_weights_inner(key)


from certus_physics import (

    calculate_reflection_array,

    calculate_reflection_single,

    calculate_transmission_array,

    calculate_T_substrate_array,


)


from certus_index_spline_core import (

    DataType,

    SplineOptConfig,

    K_MIN_PHYS,

    SIGMA_KNOTS_MIN_SEP_REL,

    _reflectance_absolute_backside_from_nk,

    _to_fraction_T,

    n_lambda_rising_with_wavelength_penalty,

    physical_nodes_to_x_slice_n,

    x_slice_n_to_physical_nodes,


)


def sigma_knots_encode(sk: np.ndarray, s_lo: float, s_hi: float) -> np.ndarray:

    """Encodes sigma knots as log-proportions (M = K-1 components) for free optimization."""

    sks = np.sort(np.asarray(sk, dtype=np.float64).ravel())

    ds = np.diff(np.clip(sks, s_lo, s_hi))

    ds = np.clip(ds, 1e-12, None)

    ds /= np.sum(ds)

    return np.log(np.clip(ds, 1e-12, None))


def sigma_knots_decode(

    raw: np.ndarray, s_lo: float, s_hi: float, eps_s: float | None = None


) -> np.ndarray:

    """Decodes log-proportions -> sorted sigma knots in [s_lo, s_hi]."""

    ww = np.exp(np.clip(raw, -20.0, 20.0))

    sw = np.sum(ww)

    if not np.isfinite(sw) or sw <= 0.0:

        ww = np.ones_like(raw)

        sw = float(ww.size)

    if eps_s is None:

        eps_s = max(1e-10, SIGMA_KNOTS_MIN_SEP_REL * max(s_hi - s_lo, 1e-12))

    ds = (ww / sw) * (s_hi - s_lo)

    c = s_lo + np.cumsum(ds)

    sk = np.concatenate(([s_lo], c[:-1], [s_hi]))

    sk[1:] = np.maximum(sk[1:], sk[:-1] + eps_s)

    sk[-1] = s_hi

    return sk


def _interpolate_along_sigma(

    sig: np.ndarray,

    sigma_knots: np.ndarray,

    values_at_knots: np.ndarray,

    profile_interp: str,


) -> np.ndarray:

    """Interpolates values at sigma nodes along sigma: PWL or cubic spline (not-a-knot, >=4 nodes)."""

    sk = np.asarray(sigma_knots, dtype=np.float64).ravel()

    v = np.asarray(values_at_knots, dtype=np.float64).ravel()

    sig_a = np.asarray(sig, dtype=np.float64)

    k = int(sk.size)

    if k < 2:

        raise ValueError("_interpolate_along_sigma: at least 2 nodes required")

    mode = str(profile_interp or "smooth").strip().lower()

    if mode not in ("pwl", "smooth"):

        mode = "smooth"

    if mode == "smooth" and k >= 4:
        global _CubicSpline
        if '_CubicSpline' not in globals():
            from scipy.interpolate import CubicSpline
            _CubicSpline = CubicSpline
            
        sig_c = np.clip(sig_a, float(sk[0]), float(sk[-1]))
        sp = _CubicSpline(sk, v, bc_type="not-a-knot")
        return np.asarray(sp(sig_c), dtype=np.float64)

    return np.interp(sig_a, sk, v)


def nk_from_x_pwlnk(

    x: np.ndarray,

    lam_nm: np.ndarray,

    sigma_knots: np.ndarray,

    k_clip_lo: float,

    k_clip_hi: float,

    sig_pre: np.ndarray | None = None,

    *,

    n_mono_band_nm: tuple[float, float] | None = None,

    profile_interp: str = "smooth",


) -> tuple[np.ndarray, np.ndarray]:

    """x = [d, n_0..n_{K-1}, L_0..L_{K-1}] with L = ln k at knots. ``sig_pre`` = 1/lambda precomputed.

    If ``n_mono_band_nm`` = (lambda_lo, lambda_hi) nm: n components of the vector are reparameterized

    with n(sigma) non-decreasing on sigma segments overlapping [lambda_lo, lambda_hi].

    ``profile_interp``: ``"pwl"`` = linear in sigma between nodes; ``"smooth"`` = cubic spline

    (not-a-knot) if K>=4, otherwise fallback to PWL.

    """

    x = np.asarray(x, dtype=np.float64, order="C").ravel()

    sigma_knots = np.asarray(sigma_knots, dtype=np.float64).ravel()

    k = int(sigma_knots.size)

    xi_or_n = x[1 : 1 + k]

    n_n = x_slice_n_to_physical_nodes(xi_or_n, sigma_knots, n_mono_band_nm)

    L_n = x[1 + k : 1 + 2 * k]

    if sig_pre is None:

        lam = np.asarray(lam_nm, dtype=np.float64).ravel()

        sig = 1.0 / np.maximum(lam, 1e-9)

    else:

        sig = sig_pre

    mode = str(profile_interp or "smooth").strip().lower()

    if mode not in ("pwl", "smooth"):

        mode = "smooth"

    n_lam = _interpolate_along_sigma(sig, sigma_knots, n_n, mode)

    L_lam = _interpolate_along_sigma(sig, sigma_knots, L_n, mode)

    k_lam = np.exp(L_lam)

    k_lo_eff = max(float(k_clip_lo), K_MIN_PHYS)

    k_hi_eff = float(k_clip_hi)

    np.clip(k_lam, k_lo_eff, k_hi_eff, out=k_lam)

    np.clip(n_lam, N_MIN_LIMIT, N_MAX_LIMIT, out=n_lam)

    return n_lam, k_lam


def build_segment_optimizer_x_vector(

    out: dict[str, Any], cfg: SplineOptConfig


) -> tuple[np.ndarray, np.ndarray] | None:

    """Optimization vector [d, n..., L...] and sigma knots from segmental result."""

    sk = np.asarray(out.get("sigma_knots"), dtype=np.float64).ravel()

    k = int(sk.size)

    if k < 2:

        return None

    xa = out.get("x")

    if xa is not None:

        xa = np.asarray(xa, dtype=np.float64).ravel()

        if xa.size == 1 + 2 * k:

            return xa.copy(), sk.copy()

    n_phys = np.asarray(out.get("n_nodes_physical"), dtype=np.float64).ravel()

    L_n = np.asarray(out.get("L_nodes"), dtype=np.float64).ravel()

    if n_phys.size != k or L_n.size != k:

        return None

    d_nm = float(out.get("d_nm", float("nan")))

    if not np.isfinite(d_nm):

        return None

    xi = physical_nodes_to_x_slice_n(n_phys, sk, cfg.n_mono_band_nm)

    xb = np.concatenate(

        (

            np.asarray([d_nm], dtype=np.float64),

            np.asarray(xi, dtype=np.float64).ravel(),

            np.asarray(L_n, dtype=np.float64).ravel(),

        )

    )

    return xb, sk


def _spline_objective_lam_mask(cfg: SplineOptConfig) -> np.ndarray:

    """Same lambda mask as ``SplinePWLObjective`` (points used in the loss).

    Stages SOL3 / SOL3b / Deltan_sub refinement: ``build_spline_objective_masked_grid`` and

    ``spline_objective_mse_on_masked_grid`` use the same spectral criterion."""

    lam = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    mask = np.isfinite(lam) & np.isfinite(np.asarray(cfg.n_sub, dtype=np.float64).ravel())

    if cfg.t_exp is not None:

        mask &= np.isfinite(np.asarray(cfg.t_exp, dtype=np.float64).ravel())

    if cfg.r_exp is not None and cfg.data_type != DataType.TRANSMISSION:

        mask &= np.isfinite(np.asarray(cfg.r_exp, dtype=np.float64).ravel())

    if cfg.rmse_fit_lambda_nm is not None:

        lo = float(min(cfg.rmse_fit_lambda_nm[0], cfg.rmse_fit_lambda_nm[1]))

        hi = float(max(cfg.rmse_fit_lambda_nm[0], cfg.rmse_fit_lambda_nm[1]))

        mask &= (lam >= lo) & (lam <= hi)

    return mask


def objective_lam_mask_on_target_grid(

    cfg: SplineOptConfig,

    lam_target: np.ndarray,


) -> np.ndarray:

    """

    Same definition as ``_spline_objective_lam_mask(cfg)``, sampled on ``lam_target``.

    If the grids coincide (within tolerance), returns the direct mask; otherwise linear interp 0/1

    on sorted ascending ``lam``.

    """

    lam_tgt = np.asarray(lam_target, dtype=np.float64).ravel()

    lam_src = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    m_src = _spline_objective_lam_mask(cfg)

    if lam_tgt.size == 0:

        return np.zeros(0, dtype=bool)

    if lam_src.size == 0:

        return np.zeros(lam_tgt.size, dtype=bool)

    if lam_tgt.shape == lam_src.shape and np.allclose(lam_tgt, lam_src, rtol=0.0, atol=1e-3):

        return m_src.astype(bool, copy=False)

    order = np.argsort(lam_src, kind="mergesort")

    ls = lam_src[order]

    ms = m_src[order].astype(np.float64)

    if ls.size < 2:

        v = float(ms[0]) if ls.size == 1 else 0.0

        return np.full(lam_tgt.size, v >= 0.5, dtype=bool)

    interp_vals = np.interp(lam_tgt, ls, ms, left=float(ms[0]), right=float(ms[-1]))

    return interp_vals >= 0.5


def build_spline_objective_masked_grid(

    cfg: SplineOptConfig,


) -> tuple[

    np.ndarray,

    np.ndarray,

    np.ndarray,

    np.ndarray,

    float,

    np.ndarray | None,

    np.ndarray | None,


] | None:

    """

    Same spectral reduction as ``SplinePWLObjective``: ``lam_f``, ``sig_f``, weights, exp, ``inv_npix``.

    Returns ``None`` if no points in the objective mask.

    """

    lam = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    mask = _spline_objective_lam_mask(cfg)

    lam_f = lam[mask]

    if lam_f.size == 0:

        return None

    n_sub_f = np.asarray(cfg.n_sub, dtype=np.float64).ravel()[mask]

    t_exp_f = (

        _to_fraction_T(np.asarray(cfg.t_exp, float).ravel()[mask])

        if cfg.t_exp is not None

        else None

    )

    r_exp_f = (

        _to_fraction_T(np.asarray(cfg.r_exp, float).ravel()[mask])

        if cfg.r_exp is not None

        else None

    )

    sig_f = 1.0 / np.maximum(lam_f, 1e-9)

    n_pix = int(sig_f.size)

    inv_npix = 1.0 / max(n_pix, 1)

    w = _cached_spectral_rmse_weights(lam_f)

    return lam_f, sig_f, n_sub_f, w, inv_npix, t_exp_f, r_exp_f


def spline_objective_mse_on_masked_grid(

    cfg: SplineOptConfig,

    *,

    lam_f: np.ndarray,

    n_sub_f: np.ndarray,

    w: np.ndarray,

    inv_npix: float,

    t_exp_f: np.ndarray | None,

    r_exp_f: np.ndarray | None,

    n_l: np.ndarray,

    k_l: np.ndarray,

    d: float,


) -> float:

    """

    Weighted average MSE identical to ``SplinePWLObjective.__call__`` (same T/R formulas).

    ``n_sub_f`` is the point-by-point effective substrate (e.g. + Delta n_sub).

    """

    loss = 0.0

    wsum = 0.0

    wt = float(cfg.weight_t)

    wr = float(cfg.weight_r)

    t_sub_cache = None

    if cfg.t_is_ratio:

        use_t = (

            cfg.data_type in (DataType.TRANSMISSION, DataType.BOTH)

            and t_exp_f is not None

            and wt > 0.0

        )

        use_r = (

            cfg.data_type in (DataType.REFLECTION, DataType.BOTH)

            and r_exp_f is not None

            and wr > 0.0

        )

        if use_t or use_r:

            t_sub_cache = np.asarray(

                calculate_T_substrate_array(lam_f, n_sub_f), dtype=np.float64

            )

    if cfg.data_type in (DataType.TRANSMISSION, DataType.BOTH) and t_exp_f is not None and wt > 0:

        if cfg.t_is_ratio and t_sub_cache is not None:

            t_th = _ratio_theoretical_from_nk(lam_f, n_l, k_l, n_sub_f, d)

        else:

            t_th = calculate_transmission_array(lam_f, n_l, k_l, d, n_sub_f)

        e = t_exp_f - t_th

        loss += wt * (float(np.dot(w, e * e)) * inv_npix)

        wsum += wt

    if cfg.data_type in (DataType.REFLECTION, DataType.BOTH) and r_exp_f is not None and wr > 0:

        if cfg.t_is_ratio and t_sub_cache is not None:

            r_th = _reflectance_absolute_backside_from_nk(lam_f, n_l, k_l, d, n_sub_f)

        else:

            r_th = calculate_reflection_array(lam_f, n_l, k_l, d, n_sub_f)

        e = r_exp_f - r_th

        loss += wr * (float(np.dot(w, e * e)) * inv_npix)

        wsum += wr

    if wsum <= 0.0:

        return 1e30

    return float(loss / wsum)


def spectral_mse_rmse_masked_from_nk(

    cfg: SplineOptConfig,

    out_meta: dict[str, Any],

    lam_full: np.ndarray,

    n_lam: np.ndarray,

    k_lam: np.ndarray,

    d_nm: float,


) -> tuple[float, float]:

    """

    MSE / RMSE on the same spectral mask as the spline objective (RMSE window, T/R, weights),

    with effective n_sub = nominal (no Deltan_sub).

    """

    mgf = build_spline_objective_masked_grid(cfg)

    if mgf is None:

        return float("nan"), float("nan")

    lam_f, _sig_f, n_sub_f_mg, w_f, inv_npix, t_exp_f, r_exp_f = mgf

    n_sub_eff_f = np.asarray(n_sub_f_mg, dtype=np.float64)

    lam_full = np.asarray(lam_full, dtype=np.float64).ravel()

    n_l = np.asarray(n_lam, dtype=np.float64).ravel()

    k_l = np.asarray(k_lam, dtype=np.float64).ravel()

    if lam_full.size < 2 or n_l.size != lam_full.size or k_l.size != lam_full.size:

        return float("nan"), float("nan")

    # O(1) Mask extraction if the incoming lam_full matches the configuration (no binary search overhead)
    if lam_full.shape == np.asarray(cfg.lam_nm, dtype=np.float64).ravel().shape:
        mask = _spline_objective_lam_mask(cfg)
        if mask.size == lam_full.size and np.sum(mask) == lam_f.size:
            nlf = n_l[mask]
            klf = k_l[mask]
        else:
            nlf = np.interp(lam_f, lam_full, n_l)
            klf = np.interp(lam_f, lam_full, k_l)
    else:
        nlf = np.interp(lam_f, lam_full, n_l)
        klf = np.interp(lam_f, lam_full, k_l)

    m = spline_objective_mse_on_masked_grid(

        cfg,

        lam_f=lam_f,

        n_sub_f=n_sub_eff_f,

        w=w_f,

        inv_npix=inv_npix,

        t_exp_f=t_exp_f,

        r_exp_f=r_exp_f,

        n_l=nlf,

        k_l=klf,

        d=float(d_nm),

    )

    if not np.isfinite(m) or m >= 1e29:

        return float("nan"), float("nan")

    return float(m), float(np.sqrt(max(m, 0.0)))


def spline_spectral_mse_from_xy_nk(

    cfg: "SplineOptConfig",

    lam_full: np.ndarray,

    n_lam_full: np.ndarray,

    k_lam_full: np.ndarray,

    d_nm: float,


) -> float | None:

    """Spectral MSE only (without auxiliary penalties), on the objective mask."""

    mg = build_spline_objective_masked_grid(cfg)

    if mg is None:

        return None

    lam_f, _sf, n_sub_f, w, inv_npix, t_exp_f, r_exp_f = mg

    lam0 = np.asarray(lam_full, dtype=np.float64).ravel()

    n0 = np.asarray(n_lam_full, dtype=np.float64).ravel()

    k0 = np.asarray(k_lam_full, dtype=np.float64).ravel()

    if n0.size != lam0.size or k0.size != lam0.size:

        return None

    o = np.argsort(lam0, kind="mergesort")

    ls = lam0[o]

    nlf = np.interp(lam_f, ls, n0[o])

    klf = np.interp(lam_f, ls, k0[o])

    if not np.all(np.isfinite(nlf) & np.isfinite(klf)):

        return None

    return float(

        spline_objective_mse_on_masked_grid(

            cfg,

            lam_f=lam_f,

            n_sub_f=n_sub_f,

            w=w,

            inv_npix=inv_npix,

            t_exp_f=t_exp_f,

            r_exp_f=r_exp_f,

            n_l=nlf,

            k_l=klf,

            d=float(d_nm),

        )

    )


def decompose_spline_pwl_objective(

    cfg: SplineOptConfig,

    sigma_knots: np.ndarray,

    x: np.ndarray,


) -> tuple[float, float, float]:

    """Decomposes the cost like ``SplinePWLObjective.__call__``: (mse_spectral, penalties, total).

    Useful to explain a RMSE jump between two sigma meshes (e.g. K=12 in dialog vs K=14 canonical):

    the total is ``mse_spectral + pen``; the displayed RMSE is ``sqrt(mse_spectral)`` (penalties are not RMSE).

    If ``cfg.spline_pure_spectral_objective`` is True, ``pen`` is 0 and ``total == mse_spectral``.

    """

    x = np.asarray(x, dtype=np.float64).ravel()

    sk = np.asarray(sigma_knots, dtype=np.float64).ravel()

    k_nodes = int(sk.size)

    if x.size != 1 + 2 * k_nodes:

        return float("nan"), float("nan"), float("nan")

    lam = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    mask = _spline_objective_lam_mask(cfg)

    lam_f = lam[mask]

    if lam_f.size == 0:

        return 1e30, 0.0, 1e30

    n_sub_f = np.asarray(cfg.n_sub, dtype=np.float64).ravel()[mask]

    t_exp_f = (

        _to_fraction_T(np.asarray(cfg.t_exp, float).ravel()[mask]) if cfg.t_exp is not None else None

    )

    r_exp_f = (

        _to_fraction_T(np.asarray(cfg.r_exp, float).ravel()[mask]) if cfg.r_exp is not None else None

    )

    sig_f = 1.0 / np.maximum(lam_f, 1e-9)

    inv_npix = 1.0 / max(int(lam_f.size), 1)

    w = _cached_spectral_rmse_weights(lam_f)

    d = float(x[0])

    n_l, k_l = nk_from_x_pwlnk(

        x,

        lam_f,

        sk,

        float(cfg.k_clip_lo),

        float(cfg.k_clip_hi),

        sig_pre=sig_f,

        n_mono_band_nm=cfg.n_mono_band_nm,

        profile_interp=cfg.nk_profile_interp,

    )

    n_n = x_slice_n_to_physical_nodes(x[1 : 1 + k_nodes], sk, cfg.n_mono_band_nm)

    if bool(getattr(cfg, "spline_pure_spectral_objective", False)):

        pen = 0.0

    else:

        pen = float(n_lambda_rising_with_wavelength_penalty(cfg, sk, n_n))

    mse_sp = float(

        spline_objective_mse_on_masked_grid(

            cfg,

            lam_f=lam_f,

            n_sub_f=n_sub_f,

            w=w,

            inv_npix=inv_npix,

            t_exp_f=t_exp_f,

            r_exp_f=r_exp_f,

            n_l=n_l,

            k_l=k_l,

            d=d,

        )

    )

    if not np.isfinite(mse_sp):

        mse_sp = 1e30

    tot = float(mse_sp + pen)

    return mse_sp, pen, tot


class SplinePWLObjective:

    """Pre-allocated objective for PWL spline (state-of-the-art, avoids closure allocations)."""

    def __init__(self, cfg: SplineOptConfig, sigma_knots: np.ndarray):

        lam = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

        mask = _spline_objective_lam_mask(cfg)

        self.lam_f = lam[mask]

        self.n_sub_f = np.asarray(cfg.n_sub, dtype=np.float64).ravel()[mask]

        self.t_exp_f = (

            _to_fraction_T(np.asarray(cfg.t_exp, float).ravel()[mask]) if cfg.t_exp is not None else None

        )

        self.r_exp_f = (

            _to_fraction_T(np.asarray(cfg.r_exp, float).ravel()[mask]) if cfg.r_exp is not None else None

        )

        self.sig_f = 1.0 / np.maximum(self.lam_f, 1e-9)

        self.n_pix = int(self.sig_f.size)

        self.inv_npix = 1.0 / max(self.n_pix, 1)

        self._cached_weights = _cached_spectral_rmse_weights(self.lam_f)

        self.w_t = self._cached_weights

        self.w_r = self._cached_weights  # Sharing spectral weights (same lam_f grid)

        self.sigma_k = np.asarray(sigma_knots, dtype=np.float64).ravel()

        self.cfg = cfg

        self.wt = float(cfg.weight_t)

        self.wr = float(cfg.weight_r)

        self.k_lo = float(cfg.k_clip_lo)

        self.k_hi = float(cfg.k_clip_hi)

        self._cache = None

    def _get_cached(self, x: np.ndarray):

        c = self._cache

        if c is None or c["x"].shape != x.shape or not np.array_equal(c["x"], x):

            return None

        return c

    def __call__(self, xv: np.ndarray) -> float:

        x = np.asarray(xv, dtype=np.float64, order="C").ravel()

        cached = self._get_cached(x)

        if cached is not None and cached["cost"] is not None:

            return cached["cost"]

        d = float(x[0])

        n_l, k_l = nk_from_x_pwlnk(

            x,

            self.lam_f,

            self.sigma_k,

            self.k_lo,

            self.k_hi,

            sig_pre=self.sig_f,

            n_mono_band_nm=self.cfg.n_mono_band_nm,

            profile_interp=self.cfg.nk_profile_interp,

        )

        k_nodes = int(self.sigma_k.size)

        n_n = x_slice_n_to_physical_nodes(

            x[1 : 1 + k_nodes], self.sigma_k, self.cfg.n_mono_band_nm

        )

        if bool(getattr(self.cfg, "spline_pure_spectral_objective", False)):

            pen = 0.0

        else:

            pen = n_lambda_rising_with_wavelength_penalty(self.cfg, self.sigma_k, n_n)

        mse_sp = spline_objective_mse_on_masked_grid(

            self.cfg,

            lam_f=self.lam_f,

            n_sub_f=self.n_sub_f,

            w=self.w_t,

            inv_npix=self.inv_npix,

            t_exp_f=self.t_exp_f,

            r_exp_f=self.r_exp_f,

            n_l=n_l,

            k_l=k_l,

            d=d,

        )

        if not np.isfinite(mse_sp):

            mse_sp = 1e30

        final_cost = float(mse_sp) + float(pen)

        self._cache = {"x": x.copy(), "cost": final_cost}

        return final_cost

    def analytic_gradient(self, xv: np.ndarray) -> np.ndarray | None:

        """Gradient of ``__call__`` when ``spline_pwl_analytic_grad_supported(self.cfg)``."""

        x = np.asarray(xv, dtype=np.float64, order="C").ravel()

        return compute_spline_pwl_objective_analytic_gradient(

            self.cfg,

            self.sigma_k,

            self.lam_f,

            self.sig_f,

            self.n_sub_f,

            self.w_t,

            self.inv_npix,

            self.t_exp_f,

            self.r_exp_f,

            x,

            nk_profile_interp=str(self.cfg.nk_profile_interp or "smooth"),

            include_n_lambda_rising_penalty=not bool(

                getattr(self.cfg, "spline_pure_spectral_objective", False)

            ),

        )


def spline_pwl_analytic_grad_supported(cfg: SplineOptConfig) -> bool:

    """True if the analytic gradient (T + penalties; R by FD/lambda) is consistent with the objective."""

    if cfg.n_mono_band_nm is not None:

        return False

    if bool(cfg.t_is_ratio):

        return False



    return True


def _lambda_rising_penalty_grad_nn(

    sk: np.ndarray, nn: np.ndarray, cfg: SplineOptConfig


) -> np.ndarray:

    """∂pen/∂n_aux_nodes (physical n at knots), aligned with ``n_lambda_rising_with_wavelength_penalty``."""

    wpen = float(getattr(cfg, "n_lambda_rising_penalty_weight", 0.0) or 0.0)

    band = getattr(cfg, "n_lambda_rising_penalty_band_nm", None)

    slack = float(getattr(cfg, "n_lambda_rising_penalty_slack", 0.0) or 0.0)

    if slack < 0.0:

        slack = 0.0

    sk = np.asarray(sk, dtype=np.float64).ravel()

    nn = np.asarray(nn, dtype=np.float64).ravel()

    ksz = int(sk.size)

    g = np.zeros(ksz, dtype=np.float64)

    if wpen <= 0.0 or band is None or ksz < 2 or nn.size != ksz:

        return g

    lam_lo = float(min(float(band[0]), float(band[1])))

    lam_hi = float(max(float(band[0]), float(band[1])))

    eps = 1e-12

    for j in range(ksz - 1):

        s0, s1 = float(sk[j]), float(sk[j + 1])

        if s1 <= s0 + eps:

            continue

        lam_min_seg = 1.0 / s1

        lam_max_seg = 1.0 / s0

        seg_lo = min(lam_min_seg, lam_max_seg)

        seg_hi = max(lam_min_seg, lam_max_seg)

        if seg_hi < lam_lo or seg_lo > lam_hi:

            continue

        viol = float(nn[j]) - float(nn[j + 1])

        if viol <= eps:

            continue

        ve = max(0.0, viol - slack)

        if ve <= 0.0:

            continue

        dv = 2.0 * wpen * ve

        g[j] += dv

        g[j + 1] -= dv

    return g


def _spectral_wsum_and_channels(

    cfg: SplineOptConfig,

    t_exp_f: np.ndarray | None,

    r_exp_f: np.ndarray | None,


) -> tuple[float, bool, bool]:

    wt = float(cfg.weight_t)

    wr = float(cfg.weight_r)

    use_t = (

        cfg.data_type in (DataType.TRANSMISSION, DataType.BOTH)

        and t_exp_f is not None

        and wt > 0.0

    )

    use_r = (

        cfg.data_type in (DataType.REFLECTION, DataType.BOTH)

        and r_exp_f is not None

        and wr > 0.0

    )

    wsum = 0.0

    if use_t:

        wsum += wt

    if use_r:

        wsum += wr

    return wsum, use_t, use_r


def compute_spline_pwl_objective_analytic_gradient(

    cfg: SplineOptConfig,

    sigma_k: np.ndarray,

    lam_f: np.ndarray,

    sig_f: np.ndarray,

    n_sub_f: np.ndarray,

    w: np.ndarray,

    inv_npix: float,

    t_exp_f: np.ndarray | None,

    r_exp_f: np.ndarray | None,

    x: np.ndarray,

    *,

    nk_profile_interp: str = "pwl",

    include_n_lambda_rising_penalty: bool = True,


) -> np.ndarray | None:

    """

    Gradient of (masked spectral MSE + n↗lambda penalty) / wsum with respect to x = [d, n₀…, L₀…].

    - T: ``_compute_single_layer_sensitivity_kernel`` (consistent with ``calculate_transmission_*``).

    - R: ``calculate_reflection_single`` (front side only, like ``calculate_reflection_array``);

      ∂R/∂n, ∂R/∂k, ∂R/∂d derivatives by central differences **per spectral point** (fast vs FD on full x).

    """

    if not spline_pwl_analytic_grad_supported(cfg):

        return None



    from _certus_physics_impl import _compute_single_layer_sensitivity_kernel

    x = np.asarray(x, dtype=np.float64, order="C").ravel()

    sk = np.asarray(sigma_k, dtype=np.float64).ravel()

    k_nodes = int(sk.size)

    dim = 1 + 2 * k_nodes

    if x.size != dim or k_nodes < 2:

        return None

    lam_f = np.asarray(lam_f, dtype=np.float64).ravel()

    sig_f = np.asarray(sig_f, dtype=np.float64).ravel()

    n_sub_f = np.asarray(n_sub_f, dtype=np.float64).ravel()

    w = np.asarray(w, dtype=np.float64).ravel()

    n_pix = int(lam_f.size)

    if n_pix == 0 or sig_f.size != n_pix or n_sub_f.size != n_pix or w.size != n_pix:

        return None

    wsum, use_t, use_r = _spectral_wsum_and_channels(cfg, t_exp_f, r_exp_f)
    
    mode = str(nk_profile_interp or "smooth").strip().lower()
    if mode == "smooth" and k_nodes >= 4:
        global _CubicSpline
        if '_CubicSpline' not in globals():
            from scipy.interpolate import CubicSpline as _CubicSpline
        sp = _CubicSpline(sk, np.eye(k_nodes), bc_type="not-a-knot")
        sig_c = np.clip(sig_f, float(sk[0]), float(sk[-1]))
        S_mat = sp(sig_c) # shape: (n_pix, k_nodes)
    else:
        mode = "pwl"
        S_mat = None
        
    grad_n_lam = np.zeros(n_pix, dtype=np.float64)
    grad_k_lam = np.zeros(n_pix, dtype=np.float64)

    if wsum <= 0.0:

        return None

    d_nm = float(x[0])

    n_l, k_l = nk_from_x_pwlnk(

        x,

        lam_f,

        sk,

        float(cfg.k_clip_lo),

        float(cfg.k_clip_hi),

        sig_pre=sig_f,

        n_mono_band_nm=cfg.n_mono_band_nm,

        profile_interp=nk_profile_interp,

    )

    t_th = (

        calculate_transmission_array(lam_f, n_l, k_l, d_nm, n_sub_f)

        if use_t

        else None

    )

    r_th = (

        calculate_reflection_array(lam_f, n_l, k_l, d_nm, n_sub_f)

        if use_r

        else None

    )

    grad = np.zeros(dim, dtype=np.float64)

    scale = 1.0 / wsum

    fd_eps = 1e-7

    n_n = x_slice_n_to_physical_nodes(x[1 : 1 + k_nodes], sk, cfg.n_mono_band_nm)

    if include_n_lambda_rising_penalty:

        grad[1 : 1 + k_nodes] += _lambda_rising_penalty_grad_nn(sk, n_n, cfg)

    for i in range(n_pix):

        wl = float(lam_f[i])

        ns = float(n_sub_f[i])

        wi = float(w[i])

        nr = float(n_l[i])

        ni = float(k_l[i])

        s = float(sig_f[i])

        if s <= float(sk[0]):

            j = 0

            w0, w1 = 1.0, 0.0

        elif s >= float(sk[-1]):

            j = k_nodes - 2

            w0, w1 = 0.0, 1.0

        else:

            j = int(np.searchsorted(sk, s, side="right") - 1)

            j = max(0, min(j, k_nodes - 2))

            denom = float(sk[j + 1] - sk[j])

            if denom <= 1e-18:

                w1 = 0.0

                w0 = 1.0

            else:

                w1 = (s - float(sk[j])) / denom

                w0 = 1.0 - w1

        L_n = x[1 + k_nodes : 1 + 2 * k_nodes]

        n_unc = float(w0 * float(n_n[j]) + w1 * float(n_n[j + 1]))

        L_lam = float(w0 * float(L_n[j]) + w1 * float(L_n[j + 1]))

        k_unc = float(np.exp(L_lam))

        k_lo = float(max(float(cfg.k_clip_lo), float(K_MIN_PHYS)))

        k_hi = float(cfg.k_clip_hi)

        dn_dnk = 0.0 if (n_unc <= float(N_MIN_LIMIT) or n_unc >= float(N_MAX_LIMIT)) else 1.0

        dk_dkunc = 0.0 if (k_unc <= k_lo or k_unc >= k_hi) else 1.0

        dTdn = dTdk = dTdd = 0.0

        dRdn = dRdk = dRdd = 0.0

        if use_t:

            dTdn, dTdk, _dRdn_s, _dRdk_s, dTdd, _dRdd_s = _compute_single_layer_sensitivity_kernel(

                wl, nr, ni, d_nm, ns

            )

        if use_r:

            rp = calculate_reflection_single(wl, nr + fd_eps, ni, d_nm, ns)

            rm = calculate_reflection_single(wl, nr - fd_eps, ni, d_nm, ns)

            dRdn = (rp - rm) / (2.0 * fd_eps)

            rp2 = calculate_reflection_single(wl, nr, ni + fd_eps, d_nm, ns)

            rm2 = calculate_reflection_single(wl, nr, ni - fd_eps, d_nm, ns)

            if ni < fd_eps:

                dRdk = (rp2 - float(r_th[i])) / fd_eps

            else:

                dRdk = (rp2 - rm2) / (2.0 * fd_eps)

            rp3 = calculate_reflection_single(wl, nr, ni, d_nm + fd_eps, ns)

            rm3 = calculate_reflection_single(wl, nr, ni, d_nm - fd_eps, ns)

            dRdd = (rp3 - rm3) / (2.0 * fd_eps)

        clip_t = 0.0 if (use_t and t_th is not None and (t_th[i] <= 1e-14 or t_th[i] >= 1.0 - 1e-14)) else 1.0

        clip_r = 0.0 if (use_r and r_th is not None and (r_th[i] <= 1e-14 or r_th[i] >= 1.0 - 1e-14)) else 1.0

        gn_i = 0.0

        gk_i = 0.0

        if use_t and t_exp_f is not None and t_th is not None:

            e_t = float(t_exp_f[i]) - float(t_th[i])

            gn_i += float(cfg.weight_t) * inv_npix * wi * (-2.0 * e_t * dTdn) * clip_t

            gk_i += float(cfg.weight_t) * inv_npix * wi * (-2.0 * e_t * dTdk) * clip_t

        if use_r and r_exp_f is not None and r_th is not None:

            e_r = float(r_exp_f[i]) - float(r_th[i])

            gn_i += float(cfg.weight_r) * inv_npix * wi * (-2.0 * e_r * dRdn) * clip_r

            gk_i += float(cfg.weight_r) * inv_npix * wi * (-2.0 * e_r * dRdk) * clip_r

        gn_i *= scale

        gk_i *= scale

        grad[0] += scale * (

            (

                float(cfg.weight_t) * inv_npix * wi * (-2.0 * (float(t_exp_f[i]) - float(t_th[i])) * dTdd) * clip_t

            )

            if (use_t and t_exp_f is not None and t_th is not None)

            else 0.0

        )

        grad[0] += scale * (

            (

                float(cfg.weight_r) * inv_npix * wi * (-2.0 * (float(r_exp_f[i]) - float(r_th[i])) * dRdd) * clip_r

            )

            if (use_r and r_exp_f is not None and r_th is not None)

            else 0.0

        )

        chain_n = gn_i * dn_dnk
        chain_k = gk_i * dk_dkunc * k_unc

        if mode == "pwl":
            grad[1 + j] += chain_n * w0
            grad[1 + j + 1] += chain_n * w1
            grad[1 + k_nodes + j] += chain_k * w0
            grad[1 + k_nodes + j + 1] += chain_k * w1
        else:
            grad_n_lam[i] = chain_n
            grad_k_lam[i] = chain_k

    if mode == "smooth":
        grad[1 : 1 + k_nodes] += S_mat.T @ grad_n_lam
        grad[1 + k_nodes : 1 + 2 * k_nodes] += S_mat.T @ grad_k_lam

    return grad
