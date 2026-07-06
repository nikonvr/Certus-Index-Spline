#!/usr/bin/env python3


# -*- coding: utf-8 -*-


"""Main spline pipeline: JSON logging, RMSE snapshots, worker orchestration."""


from __future__ import annotations


import logging


import time


from dataclasses import replace


from threading import Event


from typing import Any, Callable


import numpy as np


from certus_index_spline_core import (

    DataType,

    K_MIN_PHYS,

    SplineOptConfig,

    _canonical_knots_min_lambda_kw,

    canonical_spline_sigma_knots,

    corridor_profile_refit_maxfun,

    _bounds_x0_for_sigma_knots,

    _log_index_spline_best_config,

    _log_spectral_mesh_polish_rmse_block,

    _log_spline_pipeline_json,

    _reflectance_absolute_backside_from_nk,

    apply_rmse_fit_window_nk_nan_to_result,

    enforce_k_floor_on_nodes,

    snapshot_result_with_rmse_fit_meta,


)


from certus_index_utils import (

    _ratio_theoretical_from_nk,

    _reflectance_ratio_theoretical_from_nk,


)


from certus_physics import (

    calculate_reflection_array,

    calculate_transmission_array,

    clip_to_bounds,


)


from spline_objective import (

    build_segment_optimizer_x_vector,

    build_spline_objective_masked_grid,

    spectral_mse_rmse_masked_from_nk,

    spline_objective_mse_on_masked_grid,

    x_slice_n_to_physical_nodes,


)


from spline_finalize import (

    _finalize_spectral_rmse_mesh_polish_and_best,

    _log_skipped_knot_insertion_fixed_mesh,

    _spectral_polish_node_mesh_profile,


)


from spline_profile_corridors import (

    ProfileCorridorConfig,

    compute_profiled_corridors_by_d,

    widen_corridor_envelope_to_include_nk_in_result,

    log_coaching_corridor_pipeline_skip_empty,

    log_coaching_uncertainty_parameter_guide,


)


def enforce_local_optimization_policy(cfg: SplineOptConfig) -> None:
    """INDEX-SPLINE policy: optimization is local-only (L-BFGS-B)."""
    cfg.spline_local_only = True


class _WorkerProgressCoordinator:

    """

    Monotonic gauge for the UI: maps local sub-phases (0-100) onto global ranges,

    without backtracking (avoids jumps like 93 % -> 5 % between SOL2 and SOL3).

    Values sent to ``root_cb`` are **centi-percent** integers in ``0..10000`` (i.e. ``5423``

    means ``54.23 %``), except completion which emits ``10000`` for a full bar.

    """

    __slots__ = ("_root", "_last")

    def __init__(self, root_cb: Callable[[int, str], None]):

        self._root = root_cb

        self._last = 0.0
    def emit(self, g: float, msg: str) -> None:
        g = float(np.clip(g, 0.0, 99.99))
        if g < self._last:
            g = self._last
        self._last = g
        # Log percentage for linearity analysis (INFO for transitions, DEBUG for iterations)
        _logger = logging.getLogger("CERTUS")
        if "iterations=" in msg or "Polish L-BFGS-B" in msg:
            _logger.debug("[%5.2f%%] %s", g, msg)
        else:
            _logger.info("[%5.2f%%] %s", g, msg)
        self._root(int(min(10000, max(0, round(g * 100.0)))), msg)

    def scoped(self, lo: float, hi: float) -> Callable[[float | int, str], None]:
        lo = float(np.clip(lo, 0.0, 100.0))
        hi = float(max(lo, min(100.0, float(hi))))
        span = max(1e-9, hi - lo)
        coord = self

        def inner(p_local: float | int, msg: str) -> None:
            pl = float(np.clip(float(p_local), 0.0, 100.0))
            g = lo + span * (pl / 100.0)
            coord.emit(g, msg)

        return inner

    def finish(self, msg: str) -> None:
        self._last = 100.0
        logging.getLogger("CERTUS").info("[100.00%%] %s", msg)
        self._root(10000, msg)


def _spl_rmse_improves_meaningfully(rmse_ref: float, rmse_cand: float) -> bool:
    """True if candidate RMSE beats the reference by more than numerical / tie noise."""
    if not (np.isfinite(rmse_cand) and np.isfinite(rmse_ref)):
        return False
    rr = float(rmse_ref)
    rc = float(rmse_cand)
    if rc >= rr:
        return False
    min_gain = max(1e-7, 1e-4 * max(abs(rr), 1e-12))
    return (rr - rc) > min_gain


def _select_corridor_base_result_for_profile(
    cfg: SplineOptConfig,
    out: dict,
    solver_snapshot: dict,
) -> tuple[dict, str]:
    """Selects the "base" dict for corridor profiling (starting n,k + consistent RMSE).

    The corridor is a **local sensitivity**: small steps in *d* around optimum, refit
    n/L only, with an **RMSE tolerance** (alpha or Delta). Starting point must be the
    **same model** as the one providing the **best spectral RMSE** reference: with
    ``best_polished``, we copy polished curves **and**, if available,
    ``x_seg_spline_sigma`` -> ``x`` / ``n_nodes_physical`` / ``L_nodes`` so that the
    central refit seed is aligned with this model (otherwise threshold and refit mismatch).

    If ``corridor_profile_d_base_source`` attribute is missing (minimal cfg), default =
    ``best_polished`` comme ``SplineOptConfig``.
    """
    mode = str(
        getattr(cfg, "corridor_profile_d_base_source", "best_polished") or "best_polished"
    ).strip().lower()
    if bool(getattr(cfg, "gui_use_nl_alpha_for_corridors", False)):
        alpha_nl = out.get("nl_alpha_opt")
        n_nl = out.get("n_lam_nl")
        k_nl = out.get("k_lam_nl")
        d_nl = out.get("d_nm_nl")
        x_nl = out.get("x_nl")
        sk_nl = np.asarray(out.get("sigma_knots", solver_snapshot.get("sigma_knots", [])), dtype=np.float64).ravel()
        if (
            alpha_nl is not None
            and np.isfinite(float(alpha_nl))
            and n_nl is not None
            and k_nl is not None
            and d_nl is not None
            and x_nl is not None
            and sk_nl.size >= 2
        ):
            xa = np.asarray(x_nl, dtype=np.float64).ravel()
            k_sig = int(sk_nl.size)
            if xa.size == 1 + 2 * k_sig:
                b = dict(out)
                b["n_lam"] = np.asarray(n_nl, dtype=np.float64).copy()
                b["k_lam"] = np.asarray(k_nl, dtype=np.float64).copy()
                b["d_nm"] = float(d_nl)
                b["rmse"] = float(out.get("nl_rmse_vs_meas_scaled", out.get("rmse", float("nan"))))
                b["x"] = xa.copy()
                b["x_nl"] = xa.copy()
                b["sigma_knots"] = sk_nl.copy()
                b["x_encoding"] = "corridor_base_nl_alpha_joint"
                n_slice = xa[1 : 1 + k_sig]
                L_slice = xa[1 + k_sig : 1 + 2 * k_sig]
                b["L_nodes"] = np.asarray(L_slice, dtype=np.float64).copy()
                if cfg.n_mono_band_nm is None:
                    b["n_nodes_physical"] = np.asarray(n_slice, dtype=np.float64).copy()
                else:
                    from spline_objective import x_slice_n_to_physical_nodes

                    b["n_nodes_physical"] = x_slice_n_to_physical_nodes(
                        np.asarray(n_slice, dtype=np.float64),
                        sk_nl,
                        cfg.n_mono_band_nm,
                    )
                return b, "nl_alpha_joint"
    if mode == "dict":
        return out, "dict"
    if mode == "best_polished":
        rs = out.get("spectral_rmse_seg_spline_sigma")
        if isinstance(rs, (int, float)) and np.isfinite(float(rs)):
            if "n_lam_seg_spline_sigma" in out and "k_lam_seg_spline_sigma" in out:
                b = dict(solver_snapshot)
                b["n_lam"] = np.asarray(out["n_lam_seg_spline_sigma"], dtype=np.float64).copy()
                b["k_lam"] = np.asarray(out["k_lam_seg_spline_sigma"], dtype=np.float64).copy()
                b["d_nm"] = float(out.get("d_nm_seg_spline_sigma", b.get("d_nm", float("nan"))))
                b["rmse"] = float(rs)
                b["x_encoding"] = "corridor_base_best_polished_spline_sigma"
                best_label = out.get("spectral_rmse_best_label")
                best_val = out.get("spectral_rmse_best_value")
                if best_label is not None:
                    b["spectral_rmse_best_label"] = best_label
                if isinstance(best_val, (int, float)) and np.isfinite(float(best_val)):
                    b["spectral_rmse_best_value"] = float(best_val)
                b["spectral_rmse_seg_spline_sigma"] = float(rs)
                if "n_lam_seg_spline_sigma" in out:
                    b["n_lam_seg_spline_sigma"] = np.asarray(
                        out["n_lam_seg_spline_sigma"], dtype=np.float64
                    ).copy()
                if "k_lam_seg_spline_sigma" in out:
                    b["k_lam_seg_spline_sigma"] = np.asarray(
                        out["k_lam_seg_spline_sigma"], dtype=np.float64
                    ).copy()
                if "d_nm_seg_spline_sigma" in out and np.isfinite(
                    float(out.get("d_nm_seg_spline_sigma"))
                ):
                    b["d_nm_seg_spline_sigma"] = float(out["d_nm_seg_spline_sigma"])
                # Corridor seed: same nodes / x as sigma-mesh polish (otherwise RMSE_ref vs refit inconsistent).
                sk_b = np.asarray(
                    out.get("sigma_knots", b.get("sigma_knots")),
                    dtype=np.float64,
                ).ravel()
                if sk_b.size < 2:
                    lam_cfg = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()
                    if lam_cfg.size:
                        sk_b = np.asarray(
                            canonical_spline_sigma_knots(
                                float(np.min(lam_cfg)),
                                float(np.max(lam_cfg)),
                                **_canonical_knots_min_lambda_kw(cfg),
                            ),
                            dtype=np.float64,
                        ).ravel()
                if sk_b.size >= 2:
                    b["sigma_knots"] = sk_b.copy()
                x_seg = out.get("x_seg_spline_sigma")
                if x_seg is not None and sk_b.size >= 2:
                    xa = np.asarray(x_seg, dtype=np.float64).ravel()
                    k_sig = int(sk_b.size)
                    if xa.size == 1 + 2 * k_sig:
                        b["x_seg_spline_sigma"] = xa.copy()
                        b["x"] = xa.copy()
                        n_slice = xa[1 : 1 + k_sig]
                        L_slice = xa[1 + k_sig : 1 + 2 * k_sig]
                        b["L_nodes"] = np.asarray(L_slice, dtype=np.float64).copy()
                        if cfg.n_mono_band_nm is None:
                            b["n_nodes_physical"] = np.asarray(n_slice, dtype=np.float64).copy()
                        else:
                            from spline_objective import x_slice_n_to_physical_nodes

                            b["n_nodes_physical"] = x_slice_n_to_physical_nodes(
                                np.asarray(n_slice, dtype=np.float64),
                                sk_b,
                                cfg.n_mono_band_nm,
                            )
                        return b, "best_polished"
                return b, "best_polished"
        return dict(solver_snapshot), "solver"

    return dict(solver_snapshot), "solver"


def _resolve_corridor_mode(cfg: SplineOptConfig) -> tuple[str, str, float]:
    """Map ``corridor_profile_d_mode`` to ``(walk_mode, threshold_mode, tol_abs)``."""
    raw = str(getattr(cfg, "corridor_profile_d_mode", "alpha") or "alpha").strip().lower()
    tol = float(getattr(cfg, "corridor_profile_d_rmse_abs_tolerance", 0.001) or 0.001)
    if raw == "lr":
        return "lr", "alpha", tol
    if raw == "abs_delta":
        return "alpha", "abs_delta", tol
    if raw == "alpha_plus_delta":
        return "alpha", "alpha_plus_delta", tol
    return "alpha", "alpha", tol


def _build_profile_corridor_config(
    cfg: SplineOptConfig,
    p_mode: str,
    rm_thr_mode: str,
    tol_abs: float,
) -> ProfileCorridorConfig:
    """Build a :class:`ProfileCorridorConfig` from ``SplineOptConfig`` corridor fields."""
    return ProfileCorridorConfig(
        enabled=True,
        rmse_alpha=float(getattr(cfg, "corridor_profile_d_rmse_alpha", 1.05) or 1.05),
        mode=str(p_mode),
        step_nm=float(getattr(cfg, "corridor_profile_d_step_nm", 1.0) or 1.0),
        step_nm_initial=float(getattr(cfg, "corridor_profile_d_step_nm_initial", 1.0) or 1.0),
        step_growth=float(getattr(cfg, "corridor_profile_d_step_growth", 1.4) or 1.4),
        step_nm_max=float(getattr(cfg, "corridor_profile_d_step_nm_max", 4.0) or 4.0),
        max_span_nm=float(getattr(cfg, "corridor_profile_d_max_span_nm", 15.0) or 15.0),
        min_valid_each_side=int(getattr(cfg, "corridor_profile_d_min_valid_each_side", 1) or 1),
        lr_conf_level=float(getattr(cfg, "corridor_profile_d_lr_conf_level", 0.95) or 0.95),
        sigma_t=getattr(cfg, "corridor_profile_d_sigma_t", None),
        sigma_r=getattr(cfg, "corridor_profile_d_sigma_r", None),
        n_starts=int(getattr(cfg, "corridor_profile_d_n_starts", 1) or 1),
        fit_auto_n_starts=bool(getattr(cfg, "corridor_profile_d_fit_auto_n_starts", False)),
        fit_max_n_starts=int(getattr(cfg, "corridor_profile_d_fit_max_n_starts", 2) or 2),
        fit_retry_maxfun_scale=float(getattr(cfg, "corridor_profile_d_fit_retry_maxfun_scale", 1.5) or 1.5),
        jitter_n=float(getattr(cfg, "corridor_profile_d_jitter_n", 0.02) or 0.02),
        jitter_L=float(getattr(cfg, "corridor_profile_d_jitter_L", 0.15) or 0.15),
        seed_gate_keep_nominal_if_refit_worse=bool(
            getattr(cfg, "corridor_profile_d_seed_gate_keep_nominal_if_refit_worse", True)
        ),
        seed_gate_tol_rel=float(getattr(cfg, "corridor_profile_d_seed_gate_tol_rel", 0.0)),
        seed_gate_tol_abs=float(getattr(cfg, "corridor_profile_d_seed_gate_tol_abs", 1e-5)),
        rng_seed=int(getattr(cfg, "corridor_profile_d_rng_seed", 0) or 0),
        sigma_hetero_residual=bool(getattr(cfg, "corridor_profile_d_sigma_hetero", False)),
        sigma_hetero_scale=float(getattr(cfg, "corridor_profile_d_sigma_hetero_scale", 1.0) or 1.0),
        threshold_basis=str(getattr(cfg, "corridor_profile_d_threshold_basis", "max") or "max"),
        threshold_ratio_guard=float(getattr(cfg, "corridor_profile_d_threshold_ratio_guard", 1.25) or 1.25),
        auto_relax_threshold_to_include_center=bool(
            getattr(cfg, "corridor_profile_d_auto_relax_threshold", True)
        ),
        auto_relax_epsilon=float(getattr(cfg, "corridor_profile_d_auto_relax_epsilon", 0.002) or 0.002),
        auto_relax_max_factor=float(getattr(cfg, "corridor_profile_d_auto_relax_max_factor", 1.5) or 1.5),
        rmse_threshold_mode=str(rm_thr_mode),
        rmse_abs_tolerance=float(tol_abs),
        scientific_nominal_corridor=bool(getattr(cfg, "corridor_scientific_nominal_enabled", True)),
    )


def _run_corridor_profile_block(
    cfg: SplineOptConfig,
    out: dict,
    solver_snapshot: dict,
    log: logging.Logger,
) -> None:
    """Run d-profiling corridors if enabled; modifies ``out`` in-place.

    Guard: returns immediately if ``corridor_profile_d_enabled`` is False.
    """
    if not bool(getattr(cfg, "corridor_profile_d_enabled", False)):
        return
    try:
        cfg_eff = cfg
        if bool(getattr(cfg, "gui_use_nl_alpha_for_corridors", False)):
            alpha_nl = out.get("nl_alpha_opt")
            if alpha_nl is not None and np.isfinite(float(alpha_nl)):
                a_nl = float(alpha_nl)
                t_exp_eff = None if cfg.t_exp is None else (np.asarray(cfg.t_exp, dtype=np.float64).ravel() * a_nl)
                r_exp_eff = None if cfg.r_exp is None else (np.asarray(cfg.r_exp, dtype=np.float64).ravel() * a_nl)
                cfg_eff = replace(cfg, t_exp=t_exp_eff, r_exp=r_exp_eff)
        base_corridor_result, base_source_effective = _select_corridor_base_result_for_profile(
            cfg_eff, out, solver_snapshot
        )
        log_coaching_uncertainty_parameter_guide()
        _pmf = int(corridor_profile_refit_maxfun(cfg_eff))
        _raw_corr = str(getattr(cfg_eff, "corridor_profile_d_mode", "alpha") or "alpha").strip().lower()
        _p_mode, _rm_thr_mode, _tol_abs = _resolve_corridor_mode(cfg_eff)
        log.info(
            "PIPELINE [CORRIDORS d] Corridor profiling (acceptance envelope) | base_source=%s | mode=%s (walk=%s thr=%s) "
            "alpha=%.3f Delta_rmse=%.6f conf=%.3f sigma=%s | step=%.4g nm span=%.4g nm | refit_maxfun=%d (polish run=%d) | starts=%d jitter_n=%.4g jitter_L=%.4g seed=%d",
            str(base_source_effective),
            str(_raw_corr),
            str(_p_mode),
            str(_rm_thr_mode),
            float(getattr(cfg_eff, "corridor_profile_d_rmse_alpha", 1.05) or 1.05),
            float(_tol_abs),
            float(getattr(cfg_eff, "corridor_profile_d_lr_conf_level", 0.95) or 0.95),
            str(getattr(cfg_eff, "corridor_profile_d_sigma_t", None)),
            float(getattr(cfg_eff, "corridor_profile_d_step_nm", 1.0) or 1.0),
            float(getattr(cfg_eff, "corridor_profile_d_max_span_nm", 15.0) or 15.0),
            int(_pmf),
            int(getattr(cfg_eff, "polish_maxfun", 0) or 0),
            int(getattr(cfg_eff, "corridor_profile_d_n_starts", 1) or 1),
            float(getattr(cfg_eff, "corridor_profile_d_jitter_n", 0.02) or 0.02),
            float(getattr(cfg_eff, "corridor_profile_d_jitter_L", 0.15) or 0.15),
            int(getattr(cfg_eff, "corridor_profile_d_rng_seed", 0) or 0),
        )
        if _rm_thr_mode == "alpha_plus_delta":
            log.info(
                "PIPELINE [CORRIDORS d] Scientific corridor (alpha_plus_delta): threshold = %.3f × RMSE_ref + %.4f; nominal included natively.",
                float(getattr(cfg_eff, "corridor_profile_d_rmse_alpha", 1.05) or 1.05),
                float(_tol_abs),
            )
        elif _rm_thr_mode == "abs_delta":
            if bool(getattr(cfg, "corridor_scientific_nominal_enabled", True)):
                log.info(
                    "PIPELINE [CORRIDORS d] Scientific corridor: RMSE_ref = best polished RMSE "
                    "(spectral_rmse_best_value); threshold = RMSE_ref + Delta; nominal included natively.",
                )
            else:
                log.info(
                    "PIPELINE [CORRIDORS d] Corridor threshold (legacy abs_delta): RMSE <= RMSE_ref (base curves) + Delta.",
                )
        else:
            log.info(
                "PIPELINE [CORRIDORS d] Reminder: RMSE for refits at fixed d can exceed spectral_rmse_segments "
                "(local n,L re-optimization, corridor budget) - compare to INDEX_SPLINE [CORRIDORS d] logs "
                "('Reminder', 'Center: OK', 'RMSE threshold fallback').",
            )
        pconf = _build_profile_corridor_config(cfg_eff, _p_mode, _rm_thr_mode, _tol_abs)
        cfg_prof = replace(cfg_eff, nk_profile_interp="smooth")
        if str(getattr(cfg_eff, "nk_profile_interp", "smooth") or "smooth").strip().lower() != "smooth":
            log.info(
                "PIPELINE [CORRIDORS d] nk_profile_interp forced to 'smooth' for profiling refits.",
            )
        extra = compute_profiled_corridors_by_d(cfg_prof, base_corridor_result, pconf=pconf)
        extra["profile_d_base_source_effective"] = str(base_source_effective)
        extra["profile_d_base_x_encoding"] = str(base_corridor_result.get("x_encoding", "") or "")
        try:
            extra["profile_d_base_rmse_seed"] = float(base_corridor_result.get("rmse", float("nan")))
        except (TypeError, ValueError):
            extra["profile_d_base_rmse_seed"] = float("nan")
        out.update(extra)
        if not bool(extra.get("profile_d_scientific_nominal", False)):
            widen_corridor_envelope_to_include_nk_in_result(
                out,
                np.asarray(out.get("n_lam", []), dtype=np.float64),
                np.asarray(out.get("k_lam", []), dtype=np.float64),
            )
        # V2.3: ln(k) regularization weight sensitivity scan.
        if bool(getattr(cfg, "corridor_reg_sensitivity_enabled", False)):
            try:
                from spline_profile_corridors import compute_reg_sensitivity_scan

                base_w = float(max(getattr(cfg, "lnk_spline_reg_weight", 0.0) or 0.0, 0.0))
                decades = int(max(0, getattr(cfg, "corridor_reg_sensitivity_decades", 2) or 2))
                npt = int(max(2, getattr(cfg, "corridor_reg_sensitivity_points", 5) or 5))
                if base_w <= 0.0:
                    base_w = 1e-3
                exps = np.linspace(-decades, decades, npt, dtype=np.float64)
                w_grid = base_w * (10.0 ** exps)
                log.info(
                    "PIPELINE [CORRIDORS d] REG-SENS start | base=%.6g decades=%d points=%d | grid=%s",
                    float(base_w),
                    int(decades),
                    int(npt),
                    np.array2string(np.asarray(w_grid), precision=3),
                )
                extra2 = compute_reg_sensitivity_scan(cfg_prof, out, pconf=pconf, weights=w_grid)
                out.update(extra2)
            except Exception:
                log.exception("PIPELINE [CORRIDORS d] REG-SENS failed (ignored).")
        # V2.4: parametric bootstrap (bands).
        if bool(getattr(cfg, "corridor_bootstrap_enabled", False)):
            try:
                from spline_profile_corridors import compute_bootstrap_corridors_by_d

                B = int(max(0, getattr(cfg, "corridor_bootstrap_n", 40) or 40))
                p = float(getattr(cfg, "corridor_bootstrap_percentile", 0.95) or 0.95)
                seedB = int(getattr(cfg, "corridor_bootstrap_seed", 0) or 0)
                sigT = getattr(cfg, "corridor_bootstrap_sigma_t", None)
                sigR = getattr(cfg, "corridor_bootstrap_sigma_r", None)
                modeB = str(getattr(cfg, "corridor_bootstrap_mode", "parametric") or "parametric")
                blkB = int(max(1, getattr(cfg, "corridor_bootstrap_block_len", 1) or 1))
                qref = (
                    int(max(0, getattr(cfg, "corridor_bootstrap_quick_refit_maxfun", 0) or 0))
                    if bool(getattr(cfg, "corridor_bootstrap_quick_refit", False))
                    else None
                )
                log.info(
                    "PIPELINE [CORRIDORS d] BOOT start | mode=%s blk=%d | B=%d p=%.3f seed=%d sigma_T=%s sigma_R=%s | quick_refit_maxfun=%s",
                    str(modeB), int(blkB), int(B), float(p), int(seedB),
                    str(sigT), str(sigR), str(qref),
                )
                # Solver snapshot after mesh polish: align n/k, sigma, spectral_rmse_segments with seed.
                extra3 = compute_bootstrap_corridors_by_d(
                    cfg_prof,
                    solver_snapshot,
                    pconf=pconf,
                    n_boot=B,
                    percentile=p,
                    seed=seedB,
                    sigma_t=sigT,
                    sigma_r=sigR,
                    mode=modeB,
                    block_len=int(blkB),
                    quick_refit_maxfun=qref,
                )
                out.update(extra3)
            except Exception:
                log.exception("PIPELINE [CORRIDORS d] BOOT failed (ignored).")
        if "profile_d_interval_nm" in extra:
            try:
                a0, a1 = extra["profile_d_interval_nm"]
                log.info(
                    "PIPELINE [CORRIDORS d] OK | d_interval=[%.6f, %.6f] nm | n_valid=%d",
                    float(a0),
                    float(a1),
                    int(np.asarray(extra.get("profile_d_values_nm", [])).size),
                )
            except (TypeError, ValueError, KeyError):
                log.info("PIPELINE [CORRIDORS d] OK (interval not parsable)")
        else:
            log.info("PIPELINE [CORRIDORS d] SKIP/EMPTY (no corridor returned).")
            log_coaching_corridor_pipeline_skip_empty()
    except Exception:
        log.exception("Profiled d-corridors: failed (ignored).")


def worker_run_corridor_profile_after_nl_choice(
    cfg: SplineOptConfig,
    result: dict,
    stop_event: Event,
    progress_cb,
) -> dict | None:
    log = logging.getLogger("CERTUS")
    if not isinstance(result, dict):
        return None
    out = dict(result)
    solver_snapshot = out.get("gui_solver_snapshot_for_corridors")
    if not isinstance(solver_snapshot, dict):
        solver_snapshot = dict(out)
    if stop_event.is_set():
        return out
    progress_cb(5, "Corridors: preparing post-NL profiling...")
    _run_corridor_profile_block(cfg, out, dict(solver_snapshot), log)
    if stop_event.is_set():
        return out
    apply_rmse_fit_window_nk_nan_to_result(out, cfg.rmse_fit_lambda_nm)
    progress_cb(100, "Corridors: completed.")
    return out


def worker_spline_optimization(

    cfg: SplineOptConfig, stop_event: Event, progress_cb, live_cb=None


) -> dict | None:

    """Orchestrates the full spline pipeline (SOL2 -> SOL3 -> post-processing).

    **Inputs**

        ``cfg``: spectral configuration and budgets (see ``SplineOptConfig``); ``x0`` / Smart Init

        already integrated by ``make_bounds_and_x0`` during internal worker steps.

        ``stop_event``: cooperative cancellation (threading ``Event``).

        ``progress_cb``: ``Callable[[int, str], None]`` - global progress in **centi-percent**

        ``0..10000`` (``10000`` = 100 %) and UI label (see ``_WorkerProgressCoordinator``).

        ``live_cb``: optional; receives ``dict`` snapshots (see ``snapshot_result_with_rmse_fit_meta``)

        for real-time curves / metrics refresh.

    **Output**

        Final result ``dict`` (``rmse``, ``mse``, ``n_lam``, ``k_lam``, ``d_nm``,

        cubic sigma-spline mesh polish ``*_seg_spline_sigma``, ``log10k_*`` corridor, etc.) or ``None``.

    **Steps** (summary): SOL2 (local L-BFGS-B), SOL3 free nodes, fixed mesh,

    ``k`` floor, sigma-mesh polish (cubic spline), spectral RMSE synthesis.

    """

    from spline_workers import (
        _run_single_spline_stage,

        _run_split_free_knot_stage,

    )

    log = logging.getLogger("CERTUS")

    enforce_local_optimization_policy(cfg)

    coord = _WorkerProgressCoordinator(progress_cb)

    t_worker = time.perf_counter()

    # Watermark: tracked the best observed RMSE across all pipeline stages.

    _wm_best_rmse = float("inf")

    _wm_best_stage = "init"

    def _wm_update(rmse_val: float, stage_name: str) -> None:

        nonlocal _wm_best_rmse, _wm_best_stage

        if np.isfinite(rmse_val) and rmse_val < _wm_best_rmse:

            _wm_best_rmse = float(rmse_val)

            _wm_best_stage = str(stage_name)

    lam_arr = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    # Effective K_sigma (12 or 14): must match ``canonical_spline_sigma_knots`` so progress messages,

    # JSON metadata, and declared dimensions stay aligned with ``make_bounds_and_x0`` /

    # ``SplinePWLObjective`` (avoid a fixed "25 vars" while K=14).

    if lam_arr.size:

        _sk_mesh = canonical_spline_sigma_knots(

            float(np.min(lam_arr)),

            float(np.max(lam_arr)),

            **_canonical_knots_min_lambda_kw(cfg),

        )

        _k_mesh = int(np.asarray(_sk_mesh).size)

    else:

        _k_mesh = 12

    _nseg_mesh = max(1, _k_mesh - 1)

    # SOL2: x = [d, n_0...n_{K-1}, L_0...L_{K-1}] -> 1 + 2K components (sigma at fixed knots).

    _dim_sol2 = 1 + 2 * _k_mesh

    # SOL3 (split sigma_n / sigma_L): see ``spline_workers._run_free_knot_stage`` - z = [d] + enc(sigma_n) + enc(sigma_L) + n + L

    # with enc(.) of length K-1 each -> 1 + 2(K-1) + 2K = 4K - 1.

    _dim_sol3 = 4 * _k_mesh - 1

    _log_spline_pipeline_json(

        log,

        "worker_start",

        seq="01",

        n_lambda=int(lam_arr.size),

        lam_nm_min=float(np.min(lam_arr)) if lam_arr.size else None,

        lam_nm_max=float(np.max(lam_arr)) if lam_arr.size else None,

        d_bounds=(float(cfg.d_lo), float(cfg.d_hi)),

        data_type=str(getattr(cfg.data_type, "name", cfg.data_type)),

        polish_maxfun=int(cfg.polish_maxfun),

        optimization_mode="local_only",

        lnk_spline_stage_enabled=bool(getattr(cfg, "lnk_spline_stage_enabled", True)),

        node_mesh_spectral_polish_enabled=bool(

            getattr(cfg, "node_mesh_spectral_polish_enabled", True)

        ),

        t_is_ratio=bool(cfg.t_is_ratio),

        weight_t=float(cfg.weight_t),

        weight_r=float(cfg.weight_r),

        K_sigma_mesh=int(_k_mesh),

        n_seg_mesh=int(_nseg_mesh),

        nonlinear_alpha_budget_mode=str(getattr(cfg, "nonlinear_alpha_budget_mode", "slow") or "slow"),

        nonlinear_alpha_second_pass_enabled=bool(

            getattr(cfg, "nonlinear_alpha_second_pass_enabled", True)

        ),

    )

    log.info(

        "PIPELINE [01/09] Starting INDEX_SPLINE worker | %d points lambda | d in [%.4f, %.4f] | %s",

        int(lam_arr.size),

        float(cfg.d_lo),

        float(cfg.d_hi),

        str(getattr(cfg.data_type, "name", cfg.data_type)),

    )

    # SOL 2: local L-BFGS-B fixed knots (dim = d + K*n + K*L)

    coord.emit(0, f"SOL 2: local optimization ({_dim_sol2} vars, fixed knots, K_sigma={_k_mesh})...")

    # Do not clear ``smart_init_manual_force_restart`` here: ``_run_single_spline_stage`` reads it

    # for FACTUAL tracing (dialog RMSE vs first worker cost) and resets it to False after use.

    if getattr(cfg, "smart_init_manual_force_restart", False):

        log.info(

            "PIPELINE [02/09] Manual Smart Init: SOL2 - FACTUAL tracing on worker side (comparison of declared / recalculated RMSE)."

        )

    res2, _ = _run_single_spline_stage(

        cfg,

        stop_event,

        coord.scoped(0, 26),

        live_cb=live_cb,

        pipeline_seq="02_SOL2",

        fatal_finish=coord.finish,

    )

    if res2 is None:

        coord.finish("Stop  no usable sample")

        return None

    _wm_update(float(res2["rmse"]), "SOL2")

    # Log summary concisely
    log.info("PIPELINE [02/09] SOL2 (Fixed Knots, %d vars, K=%d) finished | RMSE=%.6f | d=%.4f nm",
             _dim_sol2, _k_mesh, float(res2["rmse"]), float(res2["d_nm"]))

    _log_spline_pipeline_json(log, "worker_after_sol2", seq="02",
        rmse=float(res2["rmse"]), d_nm=float(res2["d_nm"]),
        x_encoding=str(res2.get("x_encoding", "")),
        stage_repli_local=bool(res2.get("stage_repli_local", False)),
        nfev_local=int(res2.get("nit_polish", 0) or 0))

    _log_index_spline_best_config(log, res2, res2["rmse"], title="SOL 2 (Fixed Knots)")

    if live_cb is not None:

        live_cb(snapshot_result_with_rmse_fit_meta(cfg, res2))

    if stop_event.is_set():

        coord.finish("User stop after SOL 2.")

        return snapshot_result_with_rmse_fit_meta(cfg, res2)

    # Visual pause: live_cb already refreshed the plot; UI pacing is handled in _on_live_update.

    # Worker continues without blocking here.

    # SOL 3: L-BFGS-B free knots (dim = 1 + 2(K-1) + 2K = 4K - 1)

    coord.emit(

        26,

        f"SOL 2 completed - SOL 3 ({_dim_sol3} vars, free knots, K_sigma={_k_mesh})...",

    )

    _log_spline_pipeline_json(

        log,

        "worker_sol3_enter",

        seq="03",

        rmse_sol2=float(res2["rmse"]),

        d_sol2=float(res2["d_nm"]),

    )

    log.info(

        "PIPELINE [03/09] SOL3 (%d vars, free sigma_n/sigma_L, K_sigma=%d) - start | SOL2 ref. RMSE=%.6f",

        _dim_sol3,

        _k_mesh,

        float(res2["rmse"]),

    )

    res3 = _run_split_free_knot_stage(

        cfg, res2, stop_event, coord.scoped(26, 68), live_cb=live_cb

    )

    if res3 is not None and _spl_rmse_improves_meaningfully(
        float(res2.get("rmse", float("inf"))),
        float(res3.get("rmse", float("inf"))),
    ):
        log.info("PIPELINE [03/09] SOL3 (Free Knots, %d vars, K=%d) finished | RMSE=%.6f | d=%.4f nm",
                 _dim_sol3, _k_mesh, res3["rmse"], res3["d_nm"])

        _log_spline_pipeline_json(

            log,

            "worker_sol3_accepted",

            seq="03",

            rmse_sol2=float(res2["rmse"]),

            rmse_sol3=float(res3["rmse"]),

            delta_rmse=float(res3["rmse"] - res2["rmse"]),

            d_sol3=float(res3["d_nm"]),

            x_encoding=str(res3.get("x_encoding", "")),

        )

        log.info(

            "PIPELINE [03/09] SOL3 accepted | RMSE %.6f -> %.6f (Delta=%+.6f)",

            float(res2["rmse"]),

            float(res3["rmse"]),

            float(res3["rmse"] - res2["rmse"]),

        )

        _log_index_spline_best_config(log, res3, res3["rmse"], title="SOL 3 (Free Knots)")

        _wm_update(float(res3["rmse"]), "SOL3")

        sol_spline = res3

    else:

        r3r = float(res3["rmse"]) if res3 is not None else float("nan")
        r2r = float(res2.get("rmse", float("nan")))
        sol3_rej_reason = "sol3_none_or_worse_than_sol2"
        if res3 is not None and np.isfinite(r3r) and np.isfinite(r2r) and r3r < r2r:
            sol3_rej_reason = "sol3_negligible_rmse_gain"

        _log_spline_pipeline_json(
            log,
            "worker_sol3_rejected",
            seq="03",
            rmse_sol2=float(res2["rmse"]),
            rmse_sol3_candidate=r3r if np.isfinite(r3r) else None,
            reason=sol3_rej_reason,
        )

        if sol3_rej_reason == "sol3_negligible_rmse_gain":
            log.info(
                "PIPELINE [03/09] SOL3 rejected - candidate RMSE=%.6f slightly below SOL2=%.6f "
                "but gain below acceptance floor (keeping SOL2, %d vars).",
                r3r,
                r2r,
                _dim_sol3,
            )
        else:
            log.info(
                "PIPELINE [03/09] SOL3 rejected or unavailable - keeping SOL2 "
                "(SOL3 RMSE=%s vs SOL2 %.6f). Poor convergence here = %d vars "
                "landscape or unsuitable warm-start. If phase 1 logged « EXCEEDS LIMIT », "
                "increase sol3_phase1_maxfun (GUI advanced panel or SplineOptConfig).",
                f"{r3r:.6f}" if res3 is not None and np.isfinite(r3r) else "n/a",
                float(res2["rmse"]),
                _dim_sol3,
            )

        log.info("SOL 3 does not improve SOL 2, keeping SOL 2 (RMSE=%.6f).", res2["rmse"])

        sol_spline = res2

    if live_cb is not None:

        live_cb(snapshot_result_with_rmse_fit_meta(cfg, sol_spline))

    if stop_event.is_set():

        coord.finish("User stop after SOL 3.")

        return snapshot_result_with_rmse_fit_meta(cfg, sol_spline)

    coord.emit(68, "SOL 3 completed - continuing with fixed mesh finalization.")

    _log_spline_pipeline_json(

        log,

        "worker_after_sol3",

        seq="04",

        rmse_before=float(sol_spline["rmse"]),

    )

    # Final SOL: fixed sigma mesh (no knot insertion)

    _log_spline_pipeline_json(

        log,

        "worker_final_insert_enter",

        seq="05",

        rmse_before=float(sol_spline["rmse"]),

        fixed_mesh=True,

        K_nodes=int(_k_mesh),

        n_seg_fixed=int(_nseg_mesh),

    )

    log.info(

        "PIPELINE [05/09] Post-SOL3 - fixed mesh (K=%d sigma knots, %d segments), insertion disabled | RMSE=%.6f",

        int(_k_mesh),

        int(_nseg_mesh),

        float(sol_spline["rmse"]),

    )

    coord.emit(68, "Final spline: fixed mesh (no knot insertion)...")

    _log_skipped_knot_insertion_fixed_mesh(

        cfg, sol_spline, stop_event, coord.scoped(68, 72), live_cb=live_cb

    )

    coord.emit(72, "Final fixed mesh - k floor / sigma mesh polish...")

    out = sol_spline

    _log_spline_pipeline_json(

        log,

        "worker_after_final_insert",

        seq="06",

        rmse_sol_spline=float(sol_spline["rmse"]),

        rmse_chosen=float(out["rmse"]),

        used_insert_result=False,

    )

    log.info(

        "PIPELINE [06/09] After final stage (insertion disabled) | kept RMSE=%.6f "

        "(current spline solution; unchanged if SOL3/3b rejected)",

        float(out["rmse"]),

    )

    coord.emit(76, "k floor and spectral polish...")

    _rmse_before_kfloor = float(out.get("rmse", float("nan")))

    # --- Enforce k >= k_clip_lo with optional knot insertion ---

    _sk_out = np.asarray(out.get("sigma_knots_L", out.get("sigma_knots", [])), dtype=np.float64).ravel()

    _LL_out = np.asarray(out.get("L_nodes", []), dtype=np.float64).ravel()

    if _sk_out.size >= 2 and _LL_out.size == _sk_out.size:

        _sk_f, _LL_f, _kf_mod = enforce_k_floor_on_nodes(

            _sk_out, _LL_out, k_floor=float(cfg.k_clip_lo)

        )

        if _kf_mod:

            log.info(

                "k_floor enforce (final): %d nodes -> %d nodes (k_floor=%.1e)",

                int(_sk_out.size), int(_sk_f.size), float(cfg.k_clip_lo),

            )

            out["L_nodes"] = _LL_f

            if _sk_f.size != _sk_out.size:

                # Re-interpolate n(sigma) onto new sigma_L; source grid = sigma_knots_n if present

                _n_phys_old = np.asarray(out.get("n_nodes_physical", []), dtype=np.float64).ravel()

                _skn_src = np.asarray(

                    out.get("sigma_knots_n", out.get("sigma_knots", _sk_out)),

                    dtype=np.float64,

                ).ravel()

                if _n_phys_old.size == _skn_src.size:

                    out["n_nodes_physical"] = np.interp(_sk_f, _skn_src, _n_phys_old)

                else:

                    log.warning(

                        "k_floor insert: n_nodes_physical size (%d) != sigma_knots_n (%d) - n not re-sampled",

                        int(_n_phys_old.size),

                        int(_skn_src.size),

                    )

                if "sigma_knots_L" in out:

                    out["sigma_knots_L"] = _sk_f

                else:

                    out["sigma_knots"] = _sk_f

                # x vector invalid if knot count changed

                out.pop("x", None)

            # Recompute k_lam and spectra (any _kf_mod: clamp-only or insert)

            lam_out = np.asarray(out.get("lam_nm", cfg.lam_nm), dtype=np.float64).ravel()

            sig_out = 1.0 / np.maximum(lam_out, 1e-30)

            L_lam_out = np.interp(sig_out, _sk_f, _LL_f)

            k_lam_out = np.exp(L_lam_out)

            lo_k = max(float(cfg.k_clip_lo), K_MIN_PHYS)

            np.clip(k_lam_out, lo_k, float(cfg.k_clip_hi), out=k_lam_out)

            out["k_lam"] = k_lam_out

            n_lam_out = np.asarray(out.get("n_lam", []), dtype=np.float64).ravel()

            if n_lam_out.size == lam_out.size:

                d_o = float(out.get("d_nm", float("nan")))

                n_sub_o = np.asarray(cfg.n_sub, dtype=np.float64).ravel()

                if n_sub_o.size == lam_out.size and np.isfinite(d_o):

                    if cfg.t_is_ratio:

                        out["t_theo"] = _ratio_theoretical_from_nk(

                            lam_out, n_lam_out, k_lam_out, n_sub_o, d_o

                        )

                    else:

                        out["t_theo"] = calculate_transmission_array(

                            lam_out, n_lam_out, k_lam_out, d_o, n_sub_o

                        )

                    if cfg.r_exp is not None:

                        if cfg.t_is_ratio:

                            out["r_theo"] = _reflectance_ratio_theoretical_from_nk(

                                lam_out, n_lam_out, k_lam_out, n_sub_o, d_o

                            )

                        else:

                            out["r_theo"] = _reflectance_absolute_backside_from_nk(

                                lam_out, n_lam_out, k_lam_out, d_o, n_sub_o

                            )

                    mgf = build_spline_objective_masked_grid(cfg)

                    if mgf is not None:

                        lam_f, _sf, n_sub_f_mg, w_f, inv_npix, t_exp_f, r_exp_f = mgf

                        n_sub_eff_mg = np.asarray(n_sub_f_mg, dtype=np.float64)

                        nlf = np.interp(lam_f, lam_out, n_lam_out)

                        klf = np.interp(lam_f, lam_out, k_lam_out)

                        m_new = spline_objective_mse_on_masked_grid(

                            cfg,

                            lam_f=lam_f,

                            n_sub_f=n_sub_eff_mg,

                            w=w_f,

                            inv_npix=inv_npix,

                            t_exp_f=t_exp_f,

                            r_exp_f=r_exp_f,

                            n_l=nlf,

                            k_l=klf,

                            d=d_o,

                        )

                        out["mse"] = float(m_new)

                        out["rmse"] = float(np.sqrt(max(m_new, 0.0)))

            _log_spline_pipeline_json(

                log,

                "worker_k_floor_enforced",

                seq="07b",

                K_nodes_before=int(_sk_out.size),

                K_nodes_after=int(_sk_f.size),

                k_floor=float(cfg.k_clip_lo),

                rmse_before_kfloor=_rmse_before_kfloor

                if np.isfinite(_rmse_before_kfloor)

                else None,

                rmse_after=float(out["rmse"]),

            )

            log.info(

                "PIPELINE [07b] k_floor enforce | nodes %d->%d | RMSE %.6f -> %.6f",

                int(_sk_out.size),

                int(_sk_f.size),

                _rmse_before_kfloor

                if np.isfinite(_rmse_before_kfloor)

                else float("nan"),

                float(out["rmse"]),

            )

    # --- 07c Polish spectral sur le maillage sigma : spline cubique uniquement ---

    for _k in (

        "spectral_rmse_seg_spline_sigma",

        "n_lam_seg_spline_sigma",

        "k_lam_seg_spline_sigma",

        "d_nm_seg_spline_sigma",

        "x_seg_spline_sigma",

        # States saved before rework: remove to avoid mixing with current output.

        "spectral_rmse_seg_pwl",

        "n_lam_seg_pwl",

        "k_lam_seg_pwl",

        "d_nm_seg_pwl",

        "x_seg_pwl",

    ):

        out.pop(_k, None)

    log.info(

        "PIPELINE [07c] sigma-mesh polish - single cubic sigma-spline pass (same x0 as segmented solver)."

    )

    if not bool(getattr(cfg, "node_mesh_spectral_polish_enabled", True)):

        log.info(

            "PIPELINE [07c] SKIP: node_mesh_spectral_polish_enabled=False. "

            "SMART COACHING: The final mesh polish (L-BFGS-B cubic smoothing) was skipped. "

            "If your final metric (RMSE) is poor but intermediate models were good, enable 'node_mesh_spectral_polish_enabled' for a final holistic optimization."

        )

        coord.emit(82, "sigma mesh polish disabled - continuing.")

    else:

        xb_sk = build_segment_optimizer_x_vector(out, cfg)

        if xb_sk is None:

            log.warning(

                "PIPELINE [07c] POLISH FAILED: Unable to build [d, n nodes, ln k nodes] optimization vector. "

                "SMART COACHING: Your intermediate spline failed to resolve physically valid 'n_nodes_physical'. "

                "Check for extremely restrictive d bounds or divergent thickness targets."

            )

            coord.emit(82, "Mesh polish impossible (x vector) - continuing.")

        elif stop_event is not None and stop_event.is_set():

            log.info(

                "PIPELINE [07c] SKIP: mesh polish aborted because UI cancel event was triggered."

            )

            coord.emit(82, "Stop before mesh polish - continuing.")

        else:

            xb, sk_pol = xb_sk

            xa_dbg = out.get("x")

            if xa_dbg is not None:

                xa_a = np.asarray(xa_dbg, dtype=np.float64).ravel()

                if xa_a.size == 1 + 2 * int(sk_pol.size):

                    _src_x0 = "x vector from segmental solver (direct reuse)"

                else:

                    _src_x0 = "reconstruction from d_nm, n_nodes_physical, L_nodes (x solver size != 1+2K)"

            else:

                _src_x0 = "reconstruction from d_nm, n_nodes_physical, L_nodes (no x output)"

            bounds_b, _, _, _ = _bounds_x0_for_sigma_knots(cfg, sk_pol)

            x0c = clip_to_bounds(

                np.asarray(xb, dtype=np.float64).copy(),

                bounds_b[:, 0],

                bounds_b[:, 1],

            )

            nm_mf = getattr(cfg, "node_model_spectral_polish_maxfun", None)

            mf_pol = int(nm_mf) if nm_mf is not None else int(cfg.polish_maxfun)

            mf_pol = max(300, mf_pol)

            log.info(

                "PIPELINE [07c] Common start | K=%d sigma knots | source=%s | d(started)=%.6f nm | "

                "node_model_spectral_polish_maxfun->effective maxfun=%d (floor 300) | stop_event=%s",

                int(sk_pol.size),

                _src_x0,

                float(x0c[0]),

                mf_pol,

                "active" if (stop_event is not None and stop_event.is_set()) else "inactive",

            )

            coord.emit(76, "Spectral mesh polish: cubic spline sigma...")

            suf = "seg_spline_sigma"

            x_polish = np.asarray(x0c, dtype=np.float64).copy()

            pack = _spectral_polish_node_mesh_profile(

                cfg,

                out,

                x_polish,

                sk_pol,

                bounds_b,

                stop_event=stop_event,

                maxfun=mf_pol,

                progress_cb=coord.scoped(76, 92),

            )

            if pack is not None:

                out[f"n_lam_{suf}"] = pack["n_lam"]

                out[f"k_lam_{suf}"] = pack["k_lam"]

                out[f"d_nm_{suf}"] = float(pack["d_nm"])

                out[f"x_{suf}"] = pack["x_best"]

                out[f"spectral_rmse_{suf}"] = pack["spectral_rmse"]

                log.info(

                    "PIPELINE [07c] Storing result %s | spectral_rmse_%s=%s | d_nm_%s=%.6f nm",

                    suf,

                    suf,

                    f"{float(pack['spectral_rmse']):.8f}"

                    if pack.get("spectral_rmse") is not None

                    and np.isfinite(float(pack["spectral_rmse"]))

                    else "n/a",

                    suf,

                    float(pack["d_nm"]),

                )

            else:

                log.info(

                    "PIPELINE [07c] Spline sigma pass: no packets returned (empty mask, stop or internal error).",

                )

            coord.emit(92, "sigma mesh polish completed - RMSE synthesis.")

    lam_seg = np.asarray(out.get("lam_nm", cfg.lam_nm), dtype=np.float64).ravel()

    n_seg_arr = np.asarray(out.get("n_lam", []), dtype=np.float64).ravel()

    k_seg_arr = np.asarray(out.get("k_lam", []), dtype=np.float64).ravel()

    d_seg = float(out.get("d_nm", float("nan")))

    mse_seg_ref, rmse_seg_ref = spectral_mse_rmse_masked_from_nk(

        cfg, out, lam_seg, n_seg_arr, k_seg_arr, d_seg

    )

    if np.isfinite(rmse_seg_ref):

        out["spectral_rmse_segments"] = float(rmse_seg_ref)

        out["spectral_mse_segments"] = float(mse_seg_ref)

    else:

        out["spectral_rmse_segments"] = None

        out["spectral_mse_segments"] = None

    log.info(

        "PIPELINE [07c->reference] spectral_rmse_segments = masked RMSE on current n(lambda),k(lambda) 'solver' "

        "(out.n_lam / out.k_lam / out.d_nm), without sigma mesh polish - value = %s. "

        "Compare with spectral_rmse_seg_spline_sigma after cubic spline polish.",

        f"{float(rmse_seg_ref):.8f}" if np.isfinite(rmse_seg_ref) else "n/a",

    )

    _log_spline_pipeline_json(

        log,

        "worker_mesh_polish_summary_enter",

        seq="08",

        rmse_dict_before=float(out.get("rmse", float("nan")))

        if np.isfinite(float(out.get("rmse", float("nan"))))

        else None,

        spectral_rmse_segments=out.get("spectral_rmse_segments"),

        spectral_rmse_seg_spline_sigma=out.get("spectral_rmse_seg_spline_sigma"),

    )

    coord.emit(93, "Spectral sigma mesh polish RMSE synthesis...")

    log.info(

        "PIPELINE [08/09] After sigma mesh polish | dict RMSE=%s | reference solver RMSE (mesh)=%s",

        f"{float(out['rmse']):.6f}"

        if np.isfinite(float(out.get("rmse", float("nan"))))

        else "n/a",

        f"{float(rmse_seg_ref):.8f}" if np.isfinite(rmse_seg_ref) else "n/a",

    )

    # Shallow copy of ``out`` after sigma-mesh polish and spectral_rmse_segments recalculation,

    # before best_* synthesis and before NL alpha - used if corridor base = solver (main curves).

    solver_snapshot = dict(out)

    _finalize_spectral_rmse_mesh_polish_and_best(out)

    from spline_nonlinear_alpha import apply_nonlinear_alpha_refinement, clear_nl_result_fields

    clear_nl_result_fields(out)

    if bool(getattr(cfg, "nonlinear_alpha_refinement_enabled", True)):

        try:

            coord.emit(94, "Non-linearity: alpha x measurements re-optimization (L-BFGS-B)...")

            extra_nl = apply_nonlinear_alpha_refinement(

                cfg,

                out,
                stop_event,

                progress_cb=coord.scoped(94, 98),

            )

            out.update(extra_nl)

        except Exception:

            log.exception("INDEX_SPLINE [NL alpha] failure (ignored).")

            out["nl_optim_ok"] = False

            out["nl_optim_message"] = "exception"

    out["gui_solver_snapshot_for_corridors"] = dict(solver_snapshot)

    out["rmse_fit_lambda_nm"] = cfg.rmse_fit_lambda_nm

    # n/k corridors + d interval via d profiling (optional, off by default).
    if not bool(getattr(cfg, "gui_defer_corridor_profile_after_nl", False)):
        _run_corridor_profile_block(cfg, out, solver_snapshot, log)

    apply_rmse_fit_window_nk_nan_to_result(out, cfg.rmse_fit_lambda_nm)

    # --- Watermark RMSE: final check ---

    _wm_update(float(out.get("rmse", float("inf"))), "final")

    out["pipeline_best_rmse_watermark"] = float(_wm_best_rmse) if np.isfinite(_wm_best_rmse) else None

    out["pipeline_best_rmse_stage"] = _wm_best_stage

    _rmse_final = float(out.get("rmse", float("nan")))

    if np.isfinite(_rmse_final) and np.isfinite(_wm_best_rmse) and _rmse_final > _wm_best_rmse + 1e-8:

        wm_expected = False

        out["pipeline_watermark_degradation_expected"] = False

        log_fn = log.warning

        tag = ""

        log_fn(

            "PIPELINE [WATERMARK] SMART COACHING: Final returned RMSE (%.8f) is worse than "

            "the mathematical best RMSE found earlier (%.8f at stage '%s', loss = %+.8f). "

            "ACTION: This indicates a post-processing step (like Nonlinear Alpha smoothing or k_floor enforcement) "

            "overrode the solver's raw optimum to enforce physical constraints. "

            "If you prefer pure mathematical fidelity over physical smoothness, disable those post-processes.%s",

            _rmse_final,

            _wm_best_rmse,

            _wm_best_stage,

            _rmse_final - _wm_best_rmse,

            tag,

        )

        log_fn(

            "PIPELINE [WATERMARK] Mesh polish: spectral_rmse_seg_spline_sigma, "

            "spectral_rmse_best_label / spectral_rmse_best_value."

        )

        _log_spline_pipeline_json(

            log,

            "watermark_degradation",

            seq="09",

            rmse_final=_rmse_final,

            watermark_best_rmse=_wm_best_rmse,

            watermark_best_stage=_wm_best_stage,

            delta=float(_rmse_final - _wm_best_rmse),

            degradation_expected=bool(wm_expected),

        )

    _log_spline_pipeline_json(

        log,

        "worker_complete",

        seq="09",

        rmse_final=float(out.get("rmse", float("nan")))

        if np.isfinite(float(out.get("rmse", float("nan"))))

        else None,

        x_encoding=str(out.get("x_encoding", "")),

        elapsed_worker_s=float(time.perf_counter() - t_worker),

    )

    log.info(

        "PIPELINE [09/09] Worker completed | dict RMSE (final n/k vs masked spectrum) = %s | duration %.2fs | x_encoding=%s",

        f"{float(out['rmse']):.6f}"

        if np.isfinite(float(out.get("rmse", float("nan"))))

        else "n/a",

        float(time.perf_counter() - t_worker),

        str(out.get("x_encoding", "?")),

    )

    if np.isfinite(float(_wm_best_rmse)):

        log.info(

            "PIPELINE [09/09] Watermark reminder: best RMSE observed during run = %.8f (stage %s) - "

            "pipeline_best_rmse_watermark / pipeline_best_rmse_stage dict fields.",

            float(_wm_best_rmse),

            _wm_best_stage,

        )

    log.info(

        "PIPELINE [09/09] UI delivery: a result dict is returned to the GUI (rmse, mse, n_lam, k_lam, d_nm, "

        "polish metadata). Curve display uses these n_lam/k_lam."

    )

    coord.finish(f"Completed | RMSE={out.get('rmse', float('nan')):.6f}")

    return snapshot_result_with_rmse_fit_meta(cfg, out)
