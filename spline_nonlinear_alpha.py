#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Post-pass: joint local polish of alpha≈1 together with thickness and spline nodes."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from threading import Event
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from certus_index_spline_core import SplineOptConfig, _bounds_x0_for_sigma_knots, n_lambda_rising_with_wavelength_penalty
from certus_physics import clip_to_bounds
from spline_objective import (
    build_spline_objective_masked_grid,
    nk_from_x_pwlnk,
    spectral_mse_rmse_masked_from_nk,
    spline_objective_mse_on_masked_grid,
    x_slice_n_to_physical_nodes,
)

log = logging.getLogger("CERTUS")

ALPHA_NL_LO: float = 0.995
ALPHA_NL_HI: float = 1.005
ALPHA_NL_STEP: float = 0.0005
ALPHA_NL_SIGMA_PRIOR: float = 0.0015

_NL_KEYS_CLEAR: tuple[str, ...] = (
    "nl_alpha_opt", "nl_lam_nm", "n_lam_nl", "k_lam_nl", "d_nm_nl",
    "nl_rmse_vs_meas_orig", "nl_rmse_vs_meas_scaled", "nl_rmse_reference_best",
    "nl_profile_mode", "nl_optim_ok", "nl_optim_message", "nl_objective_final",
    "nl_second_pass_applied", "nl_alpha_grid_n", "nl_alpha_grid_step",
    "nl_alpha_budget_mode", "nl_second_pass_maxfun", "nl_alpha_identifiable",
    "nl_alpha_identifiability_note", "nl_alpha_budget_maxfun_hits",
    "nl_alpha_raw_rmse_span", "nl_alpha_steps_evaluated", "nl_alpha_scan_early_stopped",
    "nl_alpha_adaptive_applied", "nl_alpha_selection_criterion",
    "nl_alpha_best_by_objective", "nl_alpha_best_by_raw_rmse", "nl_alpha_best_by_scaled_rmse",
    "nl_alpha_selection_diverges_from_raw_rmse", "nl_alpha_identifiability_thr_flat",
    "nl_alpha_identifiability_thr_budget_hits",
)


def clear_nl_result_fields(out: dict[str, Any]) -> None:
    for k in _NL_KEYS_CLEAR:
        out.pop(k, None)


def nl_alpha_grid_values() -> np.ndarray:
    n = int(round((ALPHA_NL_HI - ALPHA_NL_LO) / ALPHA_NL_STEP)) + 1
    return np.linspace(ALPHA_NL_LO, ALPHA_NL_HI, max(2, n), dtype=np.float64)


def nl_alpha_scan_order_values() -> np.ndarray:
    alphas = nl_alpha_grid_values()
    if alphas.size <= 1:
        return alphas.copy()
    idx0 = int(np.argmin(np.abs(alphas - 1.0)))
    hi = int(alphas.size - 1)
    order_idx: list[int] = [idx0]
    j = 1
    while True:
        got = False
        li, ri = idx0 - j, idx0 + j
        if li >= 0:
            order_idx.append(li)
            got = True
        if ri <= hi:
            order_idx.append(ri)
            got = True
        if not got:
            break
        j += 1
    return np.asarray(alphas[np.array(order_idx, dtype=np.intp)], dtype=np.float64)


def nonlinear_alpha_lbfgs_maxfun_per_step(cfg: Any) -> tuple[int, str]:
    mode = str(getattr(cfg, "nl_alpha_budget_mode", "slow") or "slow").strip().lower()
    p_max = int(getattr(cfg, "polish_maxfun", 8000) or 8000)
    if mode == "fast":
        return int(max(600, min(max(1500, p_max), 20000))), "fast"
    return int(max(4000, min(max(int(round(1.5 * p_max)), p_max), 100000))), "slow"


def nonlinear_alpha_second_pass_maxfun_effective(cfg: Any, mf_per: int) -> int:
    v = getattr(cfg, "nl_alpha_second_pass_maxfun", None)
    if v is not None and int(v) > 0:
        return int(v)
    return int(max(mf_per, 5000))


def _lbfgsb_exit_kind(success: bool, msg: str) -> str:
    m = str(msg).upper()
    if "TOTAL NO. OF F" in m or "MAXFUN" in m:
        return "budget_maxfun"
    if success or "CONVERGED" in m:
        return "converged"
    return "other_error"


def _alpha_from_u(u: float) -> float:
    half_span = 0.5 * (ALPHA_NL_HI - ALPHA_NL_LO)
    return float(1.0 + half_span * np.tanh(float(u)))


def _u_from_alpha(alpha: float) -> float:
    half_span = 0.5 * (ALPHA_NL_HI - ALPHA_NL_LO)
    if not np.isfinite(alpha) or half_span <= 0.0:
        return 0.0
    y = float((float(alpha) - 1.0) / half_span)
    y = float(np.clip(y, -0.999999, 0.999999))
    return float(np.arctanh(y))


class _NLJointObjective:
    def __init__(
        self,
        cfg: SplineOptConfig,
        sk: np.ndarray,
        lam_f: np.ndarray,
        sig_f: np.ndarray,
        n_sub_f: np.ndarray,
        w: np.ndarray,
        inv_npix: float,
        t_exp_f: np.ndarray | None,
        r_exp_f: np.ndarray | None,
        profile: str,
        stop_event: Event | None,
        *,
        alpha_sigma_prior: float,
    ) -> None:
        self.cfg = cfg
        self.sk = np.asarray(sk, dtype=np.float64).ravel()
        self.lam_f = np.asarray(lam_f, dtype=np.float64).ravel()
        self.sig_f = np.asarray(sig_f, dtype=np.float64).ravel()
        self.n_sub_f = np.asarray(n_sub_f, dtype=np.float64).ravel()
        self.w = np.asarray(w, dtype=np.float64).ravel()
        self.inv_npix = float(inv_npix)
        self.t_exp_f = None if t_exp_f is None else np.asarray(t_exp_f, dtype=np.float64).ravel()
        self.r_exp_f = None if r_exp_f is None else np.asarray(r_exp_f, dtype=np.float64).ravel()
        self.profile = str(profile)
        self.stop_event = stop_event
        self.alpha_sigma_prior = float(max(alpha_sigma_prior, 1e-9))

    def unpack(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        zv = np.asarray(z, dtype=np.float64).ravel()
        return _alpha_from_u(float(zv[0])), np.asarray(zv[1:], dtype=np.float64).ravel()

    def __call__(self, z: np.ndarray) -> float:
        if self.stop_event is not None and self.stop_event.is_set():
            return 1e30
        alpha, x = self.unpack(z)
        n_l, k_l = nk_from_x_pwlnk(
            x,
            self.lam_f,
            self.sk,
            float(self.cfg.k_clip_lo),
            float(self.cfg.k_clip_hi),
            sig_pre=self.sig_f,
            n_mono_band_nm=self.cfg.n_mono_band_nm,
            profile_interp=self.profile,
        )
        t_s = None if self.t_exp_f is None else (self.t_exp_f * alpha)
        r_s = None if self.r_exp_f is None else (self.r_exp_f * alpha)
        mse = spline_objective_mse_on_masked_grid(
            self.cfg,
            lam_f=self.lam_f,
            n_sub_f=self.n_sub_f,
            w=self.w,
            inv_npix=self.inv_npix,
            t_exp_f=t_s,
            r_exp_f=r_s,
            n_l=n_l,
            k_l=k_l,
            d=float(x[0]),
        )
        k_nodes = int(self.sk.size)
        n_n = x_slice_n_to_physical_nodes(x[1 : 1 + k_nodes], self.sk, self.cfg.n_mono_band_nm)
        pen_n = 0.0 if bool(getattr(self.cfg, "spline_pure_spectral_objective", False)) else n_lambda_rising_with_wavelength_penalty(self.cfg, self.sk, n_n)
        pen_alpha = ((float(alpha) - 1.0) / self.alpha_sigma_prior) ** 2
        return float(mse + pen_n + pen_alpha)


def apply_nonlinear_alpha_refinement(
    cfg: SplineOptConfig,
    out: dict[str, Any],
    stop_event: Event | None = None,
    progress_cb: Callable[[int | float, str], None] | None = None,
) -> dict[str, Any]:
    """Joint local polish on ``alpha`` + ``[d, n_nodes, L_nodes]`` near the final solution."""

    mg = build_spline_objective_masked_grid(cfg)
    if mg is None:
        return {"nl_optim_ok": False, "nl_optim_message": "masked_grid_empty"}

    lam_f, sig_f, n_sub_f, w, inv_npix, t_exp_f, r_exp_f = mg
    lam_full = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    sk = np.asarray(out.get("sigma_knots", []), dtype=np.float64).ravel()
    x0_raw = out.get("x_seg_spline_sigma", out.get("x"))
    x0 = np.asarray(x0_raw, dtype=np.float64).ravel() if x0_raw is not None else np.array([], dtype=np.float64)

    if sk.size < 2 or x0.size != 1 + 2 * int(sk.size):
        return {"nl_optim_ok": False, "nl_optim_message": "missing_initial_state"}

    profile = str(getattr(cfg, "nk_profile_interp", "smooth") or "smooth").strip().lower()
    mf_per, budget_mode = nonlinear_alpha_lbfgs_maxfun_per_step(cfg)
    mf_pass2 = nonlinear_alpha_second_pass_maxfun_effective(cfg, mf_per)
    alpha_sigma_prior = float(getattr(cfg, "nl_alpha_sigma_prior", ALPHA_NL_SIGMA_PRIOR) or ALPHA_NL_SIGMA_PRIOR)

    bounds_x, _, _, _ = _bounds_x0_for_sigma_knots(cfg, sk)
    lo_x = np.asarray(bounds_x[:, 0], dtype=np.float64)
    hi_x = np.asarray(bounds_x[:, 1], dtype=np.float64)
    x0 = clip_to_bounds(np.asarray(x0, dtype=np.float64).copy(), lo_x, hi_x)

    obj = _NLJointObjective(
        replace(cfg, spline_pure_spectral_objective=bool(getattr(cfg, "nl_alpha_pure_spectral", True))),
        sk,
        lam_f,
        sig_f,
        n_sub_f,
        w,
        inv_npix,
        t_exp_f,
        r_exp_f,
        profile,
        stop_event,
        alpha_sigma_prior=alpha_sigma_prior,
    )

    z0 = np.concatenate(([float(_u_from_alpha(1.0))], x0))
    bounds_z = [(-4.0, 4.0)] + [(float(lo_x[i]), float(hi_x[i])) for i in range(int(lo_x.size))]

    best_z = z0.copy()
    best_fun = float(obj(best_z))
    prog = {"nfev": 0, "nit": 0, "t0": time.monotonic(), "last_emit": 0.0}

    def _emit_progress(force: bool = False, prefix: str = "NL alpha joint polish") -> None:
        if progress_cb is None:
            return
        now = time.monotonic()
        frac = min(1.0, float(prog["nfev"]) / float(max(1, mf_pass2)))
        pct = 100.0 * frac
        if force or now - float(prog["last_emit"]) >= 5.0:
            prog["last_emit"] = now
            alpha_now, _x_now = obj.unpack(best_z)
            progress_cb(
                pct,
                f"{prefix}: evals={int(prog['nfev'])}/{mf_pass2} | nit={int(prog['nit'])} | alpha={float(alpha_now):.6f} | elapsed={now - float(prog['t0']):.0f}s",
            )

    def _obj_tracked(z: np.ndarray) -> float:
        nonlocal best_fun, best_z
        if stop_event is not None and stop_event.is_set():
            return 1e30
        prog["nfev"] += 1
        val = float(obj(z))
        if np.isfinite(val) and val < best_fun - 1e-18:
            best_fun = float(val)
            best_z = np.asarray(z, dtype=np.float64).ravel().copy()
        _emit_progress(force=False)
        return val

    def _cb(_zk: np.ndarray) -> None:
        prog["nit"] += 1
        _emit_progress(force=False)

    log.info(
        "INDEX_SPLINE [NL alpha] Joint local polish | alpha in [%.5f, %.5f] | sigma_prior=%.4g | mode=%s | maxfun=%d",
        float(ALPHA_NL_LO),
        float(ALPHA_NL_HI),
        float(alpha_sigma_prior),
        budget_mode,
        int(mf_pass2),
    )

    _emit_progress(force=True)

    try:
        res = minimize(
            _obj_tracked,
            z0,
            method="L-BFGS-B",
            bounds=bounds_z,
            options={"maxfun": int(mf_pass2), "ftol": 1e-12, "gtol": 1e-8},
            callback=_cb,
        )
        rz = np.asarray(getattr(res, "x", best_z), dtype=np.float64).ravel()
        rv = float(obj(rz))
        if np.isfinite(rv) and rv < best_fun - 1e-18:
            best_fun = float(rv)
            best_z = rz.copy()
        exit_kind = _lbfgsb_exit_kind(bool(getattr(res, "success", False)), str(getattr(res, "message", "")))
        nl_msg = str(getattr(res, "message", ""))
        budget_hit = exit_kind == "budget_maxfun"
    except Exception as ex:
        log.warning("INDEX_SPLINE [NL alpha] joint polish failed (%s)", ex, exc_info=True)
        exit_kind = "other_error"
        nl_msg = str(ex)
        budget_hit = False

    prog["nfev"] = max(int(prog["nfev"]), int(mf_pass2 if budget_hit else prog["nfev"]))
    _emit_progress(force=True, prefix="NL alpha joint polish done")

    alpha_opt, x_best = obj.unpack(best_z)
    x_best = clip_to_bounds(np.asarray(x_best, dtype=np.float64).copy(), lo_x, hi_x)

    sig_full = 1.0 / np.maximum(lam_full, 1e-9)
    n_nl, k_nl = nk_from_x_pwlnk(
        x_best,
        lam_full,
        sk,
        float(cfg.k_clip_lo),
        float(cfg.k_clip_hi),
        sig_pre=sig_full,
        n_mono_band_nm=cfg.n_mono_band_nm,
        profile_interp=profile,
    )
    d_nl = float(x_best[0])

    _, rm_orig = spectral_mse_rmse_masked_from_nk(cfg, out, lam_full, n_nl, k_nl, d_nl)
    t_fin = None if cfg.t_exp is None else (np.asarray(cfg.t_exp, dtype=np.float64).ravel() * float(alpha_opt))
    r_fin = None if cfg.r_exp is None else (np.asarray(cfg.r_exp, dtype=np.float64).ravel() * float(alpha_opt))
    cfg_fin = replace(cfg, t_exp=t_fin, r_exp=r_fin)
    _, rm_scaled = spectral_mse_rmse_masked_from_nk(cfg_fin, out, lam_full, n_nl, k_nl, d_nl)

    raw_span = float(abs(float(rm_orig) - float(rm_scaled))) if np.isfinite(float(rm_orig)) and np.isfinite(float(rm_scaled)) else float("nan")
    alpha_identifiable = bool(np.isfinite(raw_span) and raw_span > 1e-7)
    alpha_note = (
        "alpha joint polish produced a measurable masked-RMSE difference"
        if alpha_identifiable
        else "alpha remains very close to 1 or has negligible masked-RMSE impact"
    )

    log.info(
        "INDEX_SPLINE [NL alpha] Joint polish result | alpha=%.6f | RMSE(raw)=%.8f | RMSE(alpha-scaled)=%.8f | objective=%.6e | exit=%s",
        float(alpha_opt),
        float(rm_orig) if np.isfinite(float(rm_orig)) else float("nan"),
        float(rm_scaled) if np.isfinite(float(rm_scaled)) else float("nan"),
        float(best_fun),
        exit_kind,
    )

    return {
        "nl_alpha_opt": float(alpha_opt),
        "nl_lam_nm": lam_full.copy(),
        "n_lam_nl": np.asarray(n_nl, dtype=np.float64).copy(),
        "k_lam_nl": np.asarray(k_nl, dtype=np.float64).copy(),
        "d_nm_nl": float(d_nl),
        "x_nl": np.asarray(x_best, dtype=np.float64).copy(),
        "nl_rmse_vs_meas_orig": float(rm_orig) if np.isfinite(float(rm_orig)) else None,
        "nl_rmse_vs_meas_scaled": float(rm_scaled) if np.isfinite(float(rm_scaled)) else None,
        "nl_rmse_reference_best": out.get("spectral_rmse_best_value"),
        "nl_optim_ok": True,
        "nl_optim_message": str(nl_msg),
        "nl_profile_mode": profile,
        "nl_objective_final": float(best_fun),
        "nl_second_pass_applied": False,
        "nl_alpha_grid_n": 1,
        "nl_alpha_grid_step": None,
        "nl_alpha_budget_mode": str(budget_mode),
        "nl_second_pass_maxfun": int(mf_pass2),
        "nl_alpha_identifiable": bool(alpha_identifiable),
        "nl_alpha_identifiability_note": str(alpha_note),
        "nl_alpha_budget_maxfun_hits": int(1 if budget_hit else 0),
        "nl_alpha_raw_rmse_span": raw_span,
        "nl_alpha_steps_evaluated": int(prog["nfev"]),
        "nl_alpha_scan_early_stopped": False,
        "nl_alpha_adaptive_applied": False,
        "nl_alpha_selection_criterion": "joint_objective_alpha_plus_x",
        "nl_alpha_best_by_objective": float(alpha_opt),
        "nl_alpha_best_by_raw_rmse": float(alpha_opt),
        "nl_alpha_best_by_scaled_rmse": float(alpha_opt),
        "nl_alpha_selection_diverges_from_raw_rmse": False,
        "nl_alpha_identifiability_thr_flat": 1e-7,
        "nl_alpha_identifiability_thr_budget_hits": 1,
    }
