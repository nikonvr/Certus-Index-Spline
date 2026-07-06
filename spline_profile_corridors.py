#!/usr/bin/env python3


# -*- coding: utf-8 -*-


"""n/k corridors by thickness *d* profiling (continuation).


Design intent (local sensitivity around the best spectral fit):

  - The corridor is **not** a new global inversion: it is a **narrow strip** of models obtained by

    **small variations of thickness** *d* around the optimum, re-equilibrating **only** the spline

    nodes (n and L = ln k) at each *d* with the **same** objective / masks / penalties as the main

    spline fit. Think **differential-style sensitivity** or **profile likelihood on *d*** with a

    **fixed RMSE tolerance band** (alpha×RMSE_ref or RMSE_ref+Delta): every accepted point stays inside that

    tube in spectral space.

  - Operationally, the **center** must be the solution that carries the **best (lowest) masked

    spectral RMSE** you export as reference - in the pipeline this is typically

    ``corridor_profile_d_base_source="best_polished"`` (mesh-polished sigma splines) so that *d_opt*,

    ``n_lam``/``k_lam``, ``x``/nodes, and ``RMSE_ref`` all describe the **same** state before any

    corridor refit.


Idea:

  - Start from an existing spline solution (sigma_knots, d_opt, n/L at knots).

  - Fix *d* at d_target and re-optimize only the nodes (n and L=ln k)

    with EXACTLY the same guards / penalties / RMSE convention as the spline objective.

  - Repeat in "continuation" on both sides of d_opt; solutions whose RMSE

    stays below a threshold define a plausible interval [d_min, d_max] and an envelope

    (corridor) on n(lambda) and k(lambda).


This module does not touch the UI; it only produces fields to merge into the result dict.


RMSE reference (alpha mode, and automatic sigma in LR):

  - if ``spectral_rmse_segments`` is present in the pipeline dict (>0, finite), it is **preferred**

    (same convention as the n/k spline « solver » optimization);

  - otherwise ``rmse`` from the dict, then direct spectral RMSE recomputed from extracted nodes.

  The key ``profile_d_rmse_ref_source`` records which was used.


Automatic lift (alpha mode, default): if the refit at ``d_opt`` exceeds ``alpha×RMSE_ref``,


the effective threshold is raised to ``RMSE_center×(1+ε)`` to avoid empty profiling while still


logging ``profile_d_rmse_thresh_nominal`` vs ``profile_d_rmse_thresh`` and ``profile_d_auto_relaxed_threshold``.


**RMSE_ref vs refit RMSE (do not confuse in logs)**:

  - ``RMSE_ref`` (often ``spectral_rmse_segments``) = spectrum for the frozen **solver** solution

    (no corridor re-optimization of nodes).

  - Each « best-of » / refit step = L-BFGS-B on **n and L** only at fixed ``d``, budget

    ``corridor_profile_d_polish_maxfun``, jitter / multi-starts allowed -> RMSE can be

    **much larger** than ``RMSE_ref`` (local minima, insufficient budget) **without a bug**;

    the code then adjusts the threshold (``center_refit`` fallback, automatic lift).


**Scientific nominal mode** (``scientific_nominal_corridor`` + ``abs_delta``):

  ``RMSE_ref = spectral_rmse_best_value``; the nominal polished n(lambda),k(lambda) is the **first** member of the

  accepted family; ``corridor_reference_*`` is a copy of that nominal curve (not the center-d refit).

  No post-hoc widening toward a separate « main » ``n_lam``/``k_lam`` export.


**Legacy UI widening** (when scientific mode is off):

  After min/max over refits, the envelope can be **expanded** so reported ``n_lam``/``k_lam`` lie inside

  ``[corridor_*_lo, corridor_*_hi]`` at each lambda.


"""


from __future__ import annotations


import logging


import os


import time


from concurrent.futures import ThreadPoolExecutor


from dataclasses import dataclass, replace


from typing import Any


import numpy as np


from scipy.optimize import minimize


from scipy.stats import chi2 as _chi2


from certus_physics import clip_to_bounds


from certus_index_spline_core import (

    N_MAX_LIMIT,

    N_MIN_LIMIT,

    DataType,

    SplineOptConfig,

    corridor_profile_refit_maxfun,


)


from spline_objective import (

    SplinePWLObjective,

    nk_from_x_pwlnk,

    spectral_mse_rmse_masked_from_nk,

    x_slice_n_to_physical_nodes,


)


log = logging.getLogger("CERTUS")


_LOG_PREFIX = "INDEX_SPLINE [CORRIDORS d]"


def _expand_corridor_envelope_with_reported_nk(

    n_lo: np.ndarray,

    n_hi: np.ndarray,

    k_lo: np.ndarray,

    k_hi: np.ndarray,

    n_nom: np.ndarray,

    k_nom: np.ndarray,


) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """Widen [lo, hi] so reported ``n_lam`` / ``k_lam`` lie inside the envelope (per lambda)."""

    n_lo = np.asarray(n_lo, dtype=np.float64).copy()

    n_hi = np.asarray(n_hi, dtype=np.float64).copy()

    k_lo = np.asarray(k_lo, dtype=np.float64).copy()

    k_hi = np.asarray(k_hi, dtype=np.float64).copy()

    nn = np.asarray(n_nom, dtype=np.float64).ravel()

    kn = np.asarray(k_nom, dtype=np.float64).ravel()

    sz = int(n_lo.size)

    if sz == 0 or nn.size < sz or kn.size < sz:

        return n_lo, n_hi, k_lo, k_hi

    nn = nn[:sz]

    kn = kn[:sz]

    m_n = np.isfinite(n_lo) & np.isfinite(n_hi) & np.isfinite(nn)

    n_lo[m_n] = np.minimum(n_lo[m_n], nn[m_n])

    n_hi[m_n] = np.maximum(n_hi[m_n], nn[m_n])

    m_k = np.isfinite(k_lo) & np.isfinite(k_hi) & np.isfinite(kn) & (kn >= 0.0)

    k_lo[m_k] = np.minimum(k_lo[m_k], kn[m_k])

    k_hi[m_k] = np.maximum(k_hi[m_k], kn[m_k])

    return n_lo, n_hi, k_lo, k_hi


@dataclass(frozen=True)


class ProfileCorridorConfig:

    """Parameters for profiling on *d*.

    rmse_alpha:

      - threshold = rmse_alpha * rmse_opt (heuristic, non-probabilistic).

      - keep low (typically 1.02-1.08) to avoid overly permissive profiles.

    """

    enabled: bool = True

    # "alpha" mode: RMSE threshold <= alpha * RMSE_opt (heuristic).

    rmse_alpha: float = 1.05

    # "lr" (Likelihood Ratio): Deltaχ² threshold <= χ²_{1,conf}. Constant sigma_T, sigma_R.

    mode: str = "alpha"  # "alpha" | "lr"

    lr_conf_level: float = 0.95

    # Constant sigma (same units as data: T as fraction, R as fraction).

    # None => auto = rmse_opt (same value for T and R).

    sigma_t: float | None = None

    sigma_r: float | None = None

    # Alpha-mode threshold policy: "nominal" | "center_refit" | "max"

    threshold_basis: str = "max"

    # Guard: if RMSE_refit_center / RMSE_ref exceeds this ratio, explicit threshold fallback.

    threshold_ratio_guard: float = 1.25

    # Continuation step (nm). The code may shrink automatically on convergence failure.

    step_nm: float = 1.0

    # Adaptive march: initial step, growth factor, step cap.

    step_nm_initial: float = 1.0

    step_growth: float = 1.0

    step_nm_max: float = 4.0

    # Max exploration span around d_opt (runtime safety), in nm.

    max_span_nm: float = 15.0

    # Max fits (safety) per direction.

    max_steps_each_side: int = 250

    # Stop tolerance: if not enough valid points.

    min_valid_points: int = 3

    # Require valid points on both sides of d_opt for a usable corridor.

    min_valid_each_side: int = 1

    # If True: include d_opt solution in "valid" list even if rmse_opt is NaN (rare).

    include_center_even_if_nan: bool = True

    # Boundary refinement (bisection) around first *d* that exceeds the threshold.

    refine_boundary: bool = True

    refine_max_iter: int = 18

    refine_tol_nm: float = 0.2

    # Multi-start (robustness to local minima):

    # - 1 => continuation only (fast)

    # - >1 => try several initializations per *d*, keep best metric (RMSE or χ²).

    n_starts: int = 1

    # If True, auto-increase n_starts when a fit fails (up to fit_max_n_starts).

    fit_auto_n_starts: bool = False

    fit_max_n_starts: int = 2

    # On STOP maxfun, optional retry with maxfun*scale.

    fit_retry_maxfun_scale: float = 1.5

    # Gaussian jitter (std dev) on x0 (in x space: n_slice or ξ, and L=ln k).

    # Jitters are in parameter units (n or ξ; and ln k).

    jitter_n: float = 0.02

    jitter_L: float = 0.15

    # Seed gate for fixed-d refits:
    # keep incoming seed if refit RMSE worsens beyond the dedicated refit tolerance.
    # This tolerance is intentionally distinct from corridor acceptance slack (rmse_abs_tolerance).
    seed_gate_keep_nominal_if_refit_worse: bool = True
    seed_gate_tol_rel: float = 0.0
    seed_gate_tol_abs: float = 1e-5

    # RNG seed for reproducibility.

    rng_seed: int = 0

    # V2.5 LR: heteroscedastic spectral sigma on masked grid (sigma_i = max(floor, scale×|residual_i|)).

    sigma_hetero_residual: bool = False

    sigma_hetero_scale: float = 1.0

    # Alpha mode: if refit at d_opt exceeds alpha×RMSE_ref (e.g. segments << "nodes-only" error),

    # raise threshold to RMSE_center×(1+ε) to keep center admissible and continue the march.

    auto_relax_threshold_to_include_center: bool = True

    auto_relax_epsilon: float = 0.002

    auto_relax_max_factor: float = 1.5

    # When ``mode`` is "alpha" (not LR): "alpha" = alpha×RMSE_ref (legacy) ;

    # "abs_delta" = accept refit iff RMSE <= RMSE_ref(base n,k on mask) + ``rmse_abs_tolerance``.

    rmse_threshold_mode: str = "alpha"  # "alpha" | "abs_delta"

    # Absolute RMSE slack on the same masked spectral objective as refits (T/R fractions).

    rmse_abs_tolerance: float = 0.001

    # Si True avec ``abs_delta`` : RMSE_ref = ``spectral_rmse_best_value``, nominale = best polish ;

    # pas d’élargissement d’enveloppe correctif ; ``corridor_reference_*`` = nominale (pas le refit central).

    scientific_nominal_corridor: bool = True

    # When True (default): corridor refits use a **pure spectral objective** (spline_pure_spectral_objective=True),
    # disabling the n_lambda_rising penalty (weight ~3000) and lnk curvature regularization during fixed-d refits.
    # These penalties cause L-BFGS-B to flee the nominal solution, always returning higher spectral RMSE than
    # the seed → seed-gate keeps nominal n,k for every d → zero-width corridor.
    # Setting True ensures refits genuinely explore n,k space at each target d.
    refit_pure_spectral: bool = True


def widen_corridor_envelope_to_include_nk_in_result(

    out: dict[str, Any],

    n_nom: np.ndarray,

    k_nom: np.ndarray,


) -> None:

    """In-place widen corridor_* lo/hi so ``n_nom``/``k_nom`` lie inside (per lambda)."""

    lam = np.asarray(out.get("lam_nm"), dtype=np.float64).ravel()

    if lam.size == 0:

        return

    keys = (

        "corridor_n_lo",

        "corridor_n_hi",

        "corridor_k_lo",

        "corridor_k_hi",

    )

    if not all(k in out and out[k] is not None for k in keys):

        return

    n_lo, n_hi, k_lo, k_hi = _expand_corridor_envelope_with_reported_nk(

        np.asarray(out["corridor_n_lo"], dtype=np.float64),

        np.asarray(out["corridor_n_hi"], dtype=np.float64),

        np.asarray(out["corridor_k_lo"], dtype=np.float64),

        np.asarray(out["corridor_k_hi"], dtype=np.float64),

        np.asarray(n_nom, dtype=np.float64),

        np.asarray(k_nom, dtype=np.float64),

    )

    out["corridor_n_lo"] = n_lo

    out["corridor_n_hi"] = n_hi

    out["corridor_k_lo"] = k_lo

    out["corridor_k_hi"] = k_hi


def log_coaching_uncertainty_parameter_guide() -> None:

    """Static reminder: when reading logs, map UI / cfg settings to interpretation."""

    log.info("%s ━━━ Parameter guide (acceptance envelope, profiling in d) ━━━", _LOG_PREFIX)

    log.info(

        "%s • RMSE_ref + Delta : avec corridor **scientifique** (défaut), RMSE_ref = **spectral_rmse_best_value** "

        "+ courbe nominale polie ; seuil = RMSE_ref + Delta ; pas d’élargissement vers la courbe solveur. "

        "Without scientific mode: RMSE_ref = masked spectrum of base curves + legacy widening possible.",

        _LOG_PREFIX,

    )

    log.info(

        "%s • alpha (mode heuristique): RMSE(refit à d fixé) <= alpha × RMSE_opt (réf. segments / dict). "

        "Closer to 1 -> narrower d interval. Typ. 1.02-1.10.",

        _LOG_PREFIX,

    )

    log.info(

        "%s • max_span_nm: max |d - d_opt| exploré de chaque côté. Si les deux marches butent sur la limite, "

        "élargir span ; en mode alpha resserrer alpha près de d_opt ; en mode RMSE_ref+Delta augmenter Delta si le seuil est trop strict.",

        _LOG_PREFIX,

    )

    log.info(

        "%s • step_nm: continuation step. Too large -> risk of \"skipping\" the threshold boundary; "

        "too small -> more refits (time). Bisection refine (refine_boundary) helps if step > 0.5 nm.",

        _LOG_PREFIX,

    )

    log.info(

        "%s • Référence RMSE: en mode absolu -> toujours recalcul sur les courbes base (voir profile_d_rmse_ref_source). "

        "En mode alpha / auto-sigma LR -> préférence ``spectral_rmse_segments`` puis dict ``rmse``.",

        _LOG_PREFIX,

    )

    log.info(

        "%s • **Refit à d fixé** : les lignes « RMSE après refit » mesurent une ré-optimisation n,L (budget corridor). "

        "Elles peuvent dépasser la référence selon le mode ; en mode RMSE_ref+Delta la référence est la courbe nominale.",

        _LOG_PREFIX,

    )

    log.info(

        "%s • Relèvement auto du seuil (**mode heuristique alpha uniquement**, si activé): si le refit central dépasse "

        "alpha×RMSE_ref, le seuil effectif peut monter - voir profile_d_auto_relaxed_threshold. **Inactif** en RMSE_ref+Delta.",

        _LOG_PREFIX,

    )

    log.info(

        "%s • LR mode + sigma_T/sigma_R: interpretation close to a likelihood-ratio test if sigma reflects instrumental noise "

        "(T, R fractions). Auto sigma uses the same RMSE reference as the alpha threshold (segments then dict).",

        _LOG_PREFIX,

    )

    log.info(

        "%s • Residual sigma(lambda) (LR): weights points by |residual|; useful for heteroscedastic noise. "

        "``scale`` controls amplitude; floor depends on RMSE_opt.",

        _LOG_PREFIX,

    )

    log.info(

        "%s • n_starts / jitter: if refits often fail (okfits=0) or RMSE looks wrong, raise n_starts or jitter "

        "to escape local minima in fixed-d optimizations.",

        _LOG_PREFIX,

    )

    log.info(

        "%s • Refit budget: ``corridor_profile_d_polish_maxfun`` (INDEX-SPLINE: 2500 default in UI); "

        "None or <=0 = same as run polish_maxfun. Increase if « STOP: … EXCEEDS LIMIT »; "

        "decrease for a faster local d scan.",

        _LOG_PREFIX,

    )

    log.info(

        "%s • Bootstrap: larger B stabilizes quantiles; bootstrap sigma should match real data noise. "

        "block_len > 1 for spectral correlation; quick_refit is faster but can bias if maxfun is too low.",

        _LOG_PREFIX,

    )

    log.info(

        "%s • Regularization scan (REG-SENS): if corridor width varies strongly with ln(k) weight, "

        "uncertainty on n,k is very sensitive to regularization - report a range, not a single value.",

        _LOG_PREFIX,

    )

    log.info("%s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", _LOG_PREFIX)


def log_coaching_corridor_pipeline_skip_empty() -> None:

    """When the pipeline got no corridor: short reminder to tie to [PROFILE] logs / cfg."""

    log.info(

        "%s [COACH] SKIP/EMPTY: voir plus haut les lignes [CORRIDORS d] (centre_fail, too_few_valid, rmse_meta_invalid). "

        "Pistes: mode RMSE_ref+Delta -> augmenter Delta ou budget refit ; mode alpha -> assouplir alpha ; LR -> conf/sigma ; "

        "max_span_nm, step_nm, n_starts/jitter, polish_maxfun.",

        _LOG_PREFIX,

    )


def _log_coaching_corridor_outcome(

    *,

    pconf: ProfileCorridorConfig,

    use_lr: bool,

    use_abs_delta: bool = False,

    d0: float,

    d_arr: np.ndarray,

    rm_arr: np.ndarray,

    rmse_opt: float,

    rmse_thresh: float,

    polish_maxfun: int,

    base_result: dict,


) -> None:

    """Interpret profiling outcome and suggest settings (INFO logs)."""

    _ = polish_maxfun  # reserved for future messages (budget shown elsewhere)

    log.info("%s ━━━ Smart Coaching: Result Analysis & Feedback ━━━", _LOG_PREFIX)

    npt = int(d_arr.size)

    dmin = float(np.min(d_arr)) if npt else float("nan")

    dmax = float(np.max(d_arr)) if npt else float("nan")

    width = float(dmax - dmin) if npt and np.isfinite(dmin) and np.isfinite(dmax) else float("nan")

    max_sp = float(pconf.max_span_nm)

    eps_nm = max(0.05, 0.02 * max_sp)

    _rmse_boundary = float(np.nanmax(rm_arr)) if (npt and rm_arr.size) else float("nan")
    _rmse_reached = (
        np.isfinite(_rmse_boundary)
        and np.isfinite(rmse_thresh)
        and rmse_thresh > 0.0
        and _rmse_boundary >= 0.85 * rmse_thresh
    )

    touch_lo = bool(

        npt and np.isfinite(d0) and np.isfinite(dmin) and (d0 - dmin) >= max_sp - eps_nm
        and not _rmse_reached

    )

    touch_hi = bool(

        npt and np.isfinite(d0) and np.isfinite(dmax) and (dmax - d0) >= max_sp - eps_nm
        and not _rmse_reached

    )

    if touch_lo and touch_hi:

        log.info(

            "%s -> THICKNESS CORRIDOR CAPPED: Interval hit max_span_nm limits (+/-%.3g nm) on both sides. "

            "ACTION: Your model strongly lacks thickness sensitivity. The n/k curves adapt almost perfectly to any subset "

            "of d. Consider physically fixing d via external measurement, or heavily decrease K-nodes (decrease degrees of freedom).",

            _LOG_PREFIX,

            max_sp,

        )

    elif (not touch_lo) and (not touch_hi) and npt >= 5 and (not use_lr):

        log.info(

            "%s -> THICKNESS SENSITIVE: Both boundaries are threshold-limited, span ~ %.4g nm over %d evaluated points. "

            "ACTION: Good physical constraint. To squeeze the envelope, decrease 'rmse_abs_tolerance'. To explore broader local minima, increase it.",

            _LOG_PREFIX,

            width,

            npt,

        )

    if (not use_lr) and npt <= 2:

        log.info(

            "%s -> FEW VALID POINTS (%d): Often only d_opt stays inside the RMSE tube; lateral refits may be rejected by the threshold, or reverted to the nominal seed when spectral RMSE would degrade after the fixed-d refit.",

            _LOG_PREFIX,

            npt,

        )

    elif npt <= int(max(3, pconf.min_valid_points + 1)) and (not use_lr):

        log.info(

            "%s -> HIGHLY CONSTRAINED (Only %d valid points): The threshold rejected almost all refits. "

            "ACTION: Either your global minimum is very sharp (excellent spectral data), or the refits are getting stuck "

            "in numeric artifacts. If the model seems noisy, increase 'corridor_profile_d_maxfun_override' to >= 3000 to allow deeper local L-BFGS-B relaxation.",

            _LOG_PREFIX,

            npt,

        )

    if npt > 40 and width >= 2.0 * max_sp - 2 * eps_nm:

        log.info(

            "%s -> WIDE DEGENERATIVE VALLEY: Many admissible points across the full range. "

            "ACTION: The inversion is mathematically degenerate. n(lam) and d are perfectly coupled. "

            "You cannot determine d and n simultaneously with certainty on this subset. Force d externally.",

            _LOG_PREFIX,

        )

    if rm_arr.size and npt >= 3:

        rm_spread = float(np.nanmax(rm_arr) - np.nanmin(rm_arr))

        if np.isfinite(rm_spread) and rm_spread < 1e-6:

            log.info(

                "%s -> FLAT COST FUNCTION: RMSE varies by < 1e-6 over the sampled thickness window. "

                "ACTION: Severe decoupling issue. Substrate variations or backside inaccuracies are dominating the cost function "

                "making the film thickness invisible to the solver.",

                _LOG_PREFIX,

            )

    if not use_abs_delta:

        rs = base_result.get("spectral_rmse_segments")

        rm_dict = float(base_result.get("rmse", float("nan")))

        if (

            np.isfinite(rmse_opt)

            and rs is not None

            and np.isfinite(float(rs))

            and np.isfinite(rm_dict)

            and float(rs) > 0

        ):

            rel = abs(rm_dict - float(rs)) / float(rs)

            if rel > 0.4:

                log.info(

                    "%s -> RMSE DEVIATION WARNING: Dict RMSE (%.6g) and solver internal RMSE (%.6g) diverged significantly (|Delta|/solver ~ %.2f). "

                    "ACTION: Disable aggressive post-processing steps (like nonlinear alpha) prior to corridor generation to restore consistency.",

                    _LOG_PREFIX,

                    rm_dict,

                    float(rs),

                    rel,

                )

    log.info(

        "%s • NOTE: The n(lambda) and k(lambda) ribbons are deterministic max/min envelopes obtained strictly via multi-thickness L-BFGS-B relaxation. "

        "Strict bayesian interpretation requires empirical noise mapping.",

        _LOG_PREFIX,

    )

    log.info("%s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", _LOG_PREFIX)


def _log_coaching_corridor_failure(

    *,

    reason: str,

    pconf: ProfileCorridorConfig,

    use_lr: bool,

    rmse_opt: float,

    rmse_thresh: float,

    d0: float,


) -> None:

    _ = pconf, use_lr, rmse_opt, rmse_thresh  # reserved context

    log.info("%s ━━━ Smart Coaching: Failure Analysis (%s) ━━━", _LOG_PREFIX, reason)

    if reason == "rmse_meta_invalid":

        log.info(

            "%s -> INVALID METRIC: RMSE_opt or threshold is NaN. "

            "ACTION: Your primary spline optimization mathematically crashed or diverged prior to running corridors. "

            "Check for impossible targets (like T_exp < 0 or > 1) or structurally invalid n_sub.",

            _LOG_PREFIX,

        )

    elif reason == "centre_fail":

        log.info(

            "%s -> CENTER DENIED: The optimal thickness (d_opt) fell outside its own calculated threshold after a local L-BFGS-B pass. "

            "ACTION: This means your threshold is too tight (statistical noise dominates) or the refit hit a local trap. "

            "To fix: increase 'corridor_profile_d_maxfun_override', slightly relax 'rmse_abs_tolerance', or ensure 'n_starts' >= 2 to break out of traps.",

            _LOG_PREFIX,

        )

    elif reason == "too_few_valid":

        log.info(

            "%s -> TOO FEW VALID POINTS: The optimization walked away from the center (d_opt ~ %.4f nm) but immediately hit a wall. "

            "ACTION: The cost valley is incredibly steep or the search step is too large. "

            "To fix: reduce 'corridor_profile_d_step_nm' to capture the extremely narrow valley, or gracefully accept that this thickness constraint is laser-sharp.",

            _LOG_PREFIX,

            float(d0),

        )

    log.info("%s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", _LOG_PREFIX)


def _log_coaching_bootstrap_outcome(

    *,

    B: int,

    n_ok: int,

    p: float,

    mode: str,

    qref: int,

    d_lo_q: float,

    d_hi_q: float,


) -> None:

    _ = mode

    log.info("%s ━━━ Smart Coaching: Bootstrap Uncertainty ━━━", _LOG_PREFIX)

    frac = float(n_ok) / float(max(B, 1))

    if frac < 0.5:

        log.info(

            "%s -> LOW BOOTSTRAP YIELD (%.0f%% OK): Most synthetic samples failed to converge. "

            "ACTION: Your model is highly brittle to noise. To increase stability: check if the applied 'sigma_T' / 'sigma_R' noise "

            "is drastically overestimating actual spectrometer noise. Also, heavily increase 'corridor_bootstrap_quick_refit_maxfun' to give synthetic fits more breathing room.",

            _LOG_PREFIX,

            100.0 * frac,

        )

    if qref > 0 and qref < 2000:

        log.info(

            "%s -> SHALLOW REFIT (maxfun=%d): The quick refit budget is extremely modest. "

            "ACTION: Bootstrap might be generating artificially wide uncertainty bounds due to premature stopping. "

            "Raise 'corridor_bootstrap_quick_refit_maxfun' to >= 4000 to ensure synthetic samples reach their true physical minima.",

            _LOG_PREFIX,

            int(qref),

        )

    spread = float(d_hi_q) - float(d_lo_q)

    if np.isfinite(spread) and spread > 1e-6:

        log.info(

            "%s -> DISPERSION RESULT (p=%.2f): [%s, %s] nm (span ~ %.3g nm). "

            "ACTION: This represents the conditional uncertainty given the assumed spline constraints. If the span is huge, "

            "your spectrum simply lacks enough interference fringes to resolve thickness securely against optical index.",

            _LOG_PREFIX,

            float(p),

            f"{d_lo_q:.4f}",

            f"{d_hi_q:.4f}",

            spread,

        )

    log.info("%s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", _LOG_PREFIX)


def _log_coaching_reg_sensitivity_outcome(

    *,

    weights: np.ndarray,

    d_lo: np.ndarray,

    d_hi: np.ndarray,

    nw: np.ndarray,

    kw: np.ndarray,


) -> None:

    log.info("%s ━━━ Smart Coaching: Regularization Profile ━━━", _LOG_PREFIX)

    if weights.size < 2:

        return

    w = np.asarray(weights, dtype=np.float64).ravel()

    lo = np.asarray(d_lo, dtype=np.float64).ravel()

    hi = np.asarray(d_hi, dtype=np.float64).ravel()

    nwa = np.asarray(nw, dtype=np.float64).ravel()

    kwa = np.asarray(kw, dtype=np.float64).ravel()

    if lo.size == w.size and hi.size == w.size:

        widths = hi - lo

        m = np.isfinite(widths)

        if np.any(m):

            wmin = float(np.nanmin(widths[m]))

            wmax = float(np.nanmax(widths[m]))

            rw = float(wmax / max(wmin, 1e-30))

            if rw > 3.0:

                log.info(

                    "%s -> REGULARIZATION DEPENDENCE (Varies by ~%.2f×): Your thickness uncertainty is heavily linked to "

                    "the smoothness penalty applied to k. "

                    "ACTION: Do not report a single value. You MUST report thickness bounds conditionally based on expected physical smoothness.",

                    _LOG_PREFIX,

                    rw,

                )

    if nwa.size == w.size and np.any(np.isfinite(nwa)):

        pos = nwa[np.isfinite(nwa) & (nwa > 0)]

        if pos.size:

            rn = float(np.nanmax(nwa) / max(float(np.nanmin(pos)), 1e-30))

            if rn > 2.5:

                log.info(

                    "%s -> n(lambda) CORRIDOR DEPENDS ON REGULARIZATION (Varies by ~%.2f×). "

                    "ACTION: The width of your index envelope expands massively if ln(k) is heavily smoothed. Review physical consistency.",

                    _LOG_PREFIX,

                    rn,

                )

    if kwa.size == w.size and np.any(np.isfinite(kwa)):

        posk = kwa[np.isfinite(kwa) & (kwa > 0)]

        if posk.size:

            rk = float(np.nanmax(kwa) / max(float(np.nanmin(posk)), 1e-30))

            if rk > 2.5:

                log.info(

                    "%s -> k(lambda) CORRIDOR EXTREMELY DEPENDENT ON REGULARIZATION (Varies by ~%.2f×). "

                    "ACTION: The extinction bounds are mostly an artifact of the regularization term, not your actual data topology. "

                    "Proceed with extreme caution when interpreting k-confidence intervals.",

                    _LOG_PREFIX,

                    rk,

                )

    log.info("%s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", _LOG_PREFIX)


def _pick_rmse_reference_for_profile(

    cfg: SplineOptConfig,

    base_result: dict,

    sk: np.ndarray,

    d0: float,

    x_nodes0: np.ndarray,


) -> tuple[float, str]:

    """Pick reference RMSE for alpha×RMSE threshold and auto sigma (LR).

    Order: ``spectral_rmse_segments`` (solver-consistent) -> dict ``rmse`` -> recomputed spectral RMSE.

    """

    rs = base_result.get("spectral_rmse_segments")

    try:

        rsv = float(rs) if rs is not None else float("nan")

    except (TypeError, ValueError):

        rsv = float("nan")

    if np.isfinite(rsv) and rsv > 0:

        return rsv, "spectral_rmse_segments"

    rm = float(base_result.get("rmse", float("nan")))

    if np.isfinite(rm):

        return rm, "dict_rmse"

    ska = np.asarray(sk, dtype=np.float64).ravel()

    lam_full = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    x_full0 = np.concatenate((np.asarray([float(d0)], dtype=np.float64), np.asarray(x_nodes0, dtype=np.float64).ravel()))

    n0, k0 = nk_from_x_pwlnk(

        x_full0,

        lam_full,

        ska,

        cfg.k_clip_lo,

        cfg.k_clip_hi,

        sig_pre=None,

        n_mono_band_nm=cfg.n_mono_band_nm,

        profile_interp=str(cfg.nk_profile_interp or "smooth"),

    )

    _m0, rr = spectral_mse_rmse_masked_from_nk(cfg, base_result, lam_full, n0, k0, float(d0))

    return rr, "recalc_objective"


def _extract_knots_and_nodes_from_result(

    result: dict,


) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:

    """Fetch sigma_knots_L, n and L on that grid, d_nm, and a diagnostic dict (logs / traceability).

    If ``sigma_knots_n`` is present and differs from ``sigma_knots_L`` (sizes or sigma values),

    ``n_nodes_physical`` is interpolated onto the sigma_L grid (same convention as L_nodes / nk PWL).

    """

    sk_L = np.asarray(result.get("sigma_knots_L", result.get("sigma_knots", [])), dtype=np.float64).ravel()

    has_sk_n_key = "sigma_knots_n" in result and result.get("sigma_knots_n") is not None

    sk_n = (

        np.asarray(result.get("sigma_knots_n"), dtype=np.float64).ravel()

        if has_sk_n_key

        else sk_L.copy()

    )

    n_nodes = np.asarray(result.get("n_nodes_physical", []), dtype=np.float64).ravel()

    L_nodes = np.asarray(result.get("L_nodes", []), dtype=np.float64).ravel()

    d_nm = float(result.get("d_nm", float("nan")))

    meta: dict[str, Any] = {

        "x_encoding": str(result.get("x_encoding", "") or ""),

        "sigma_knots_n_key_present": bool(has_sk_n_key),

    }

    if sk_L.size < 2 or not np.isfinite(d_nm):

        raise ValueError("profile corridors: incomplete result (sigma_knots_L / d_nm).")

    if L_nodes.size != sk_L.size:

        raise ValueError(

            f"profile corridors: len(L_nodes)={L_nodes.size} != K_sigma_L={sk_L.size}."

        )

    if n_nodes.size != sk_n.size:

        raise ValueError(

            f"profile corridors: len(n_nodes_physical)={n_nodes.size} != len(sigma_knots_n)={sk_n.size}."

        )

    sig_span = float(np.ptp(sk_L)) if sk_L.size else 0.0

    atol_sig = max(1e-14, 1e-9 * sig_span) if np.isfinite(sig_span) and sig_span > 0 else 1e-14

    same_len = sk_n.size == sk_L.size

    grids_coincide = same_len and bool(np.allclose(sk_L, sk_n, rtol=0.0, atol=atol_sig))

    meta["sigma_grids_coincide"] = bool(grids_coincide)

    meta["max_abs_sigma_n_minus_L"] = float(np.max(np.abs(sk_L - sk_n))) if same_len else float("nan")

    meta["mean_abs_sigma_n_minus_L"] = float(np.mean(np.abs(sk_L - sk_n))) if same_len else float("nan")

    meta["sigma_atol_nm_inv"] = float(atol_sig)

    remeshed = False

    if not grids_coincide:

        try:

            n_nodes = np.interp(sk_L, sk_n, n_nodes)

            remeshed = True

        except (TypeError, ValueError) as exc:

            raise ValueError(

                "profile corridors: n_nodes remesh failed (sigma_knots_n -> sigma_knots_L)."

            ) from exc

    meta["remeshed_n_sigma_n_to_sigma_L"] = bool(remeshed)

    if n_nodes.size != sk_L.size:

        raise ValueError(

            f"profile corridors: after alignment, len(n_nodes)={n_nodes.size} != K={sk_L.size}."

        )

    return sk_L, n_nodes, L_nodes, d_nm, meta


def _spectral_rmse_at_packed_nodes(

    cfg: SplineOptConfig,

    base_result: dict,

    sk: np.ndarray,

    d_nm: float,

    x_nodes: np.ndarray,


) -> tuple[float, float]:

    """Masked spectral RMSE for fixed [d, x_nodes] (no refit)."""

    lam_full = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    x_full = np.concatenate((np.asarray([float(d_nm)], dtype=np.float64), np.asarray(x_nodes, dtype=np.float64).ravel()))

    n_lam, k_lam = nk_from_x_pwlnk(

        x_full,

        lam_full,

        np.asarray(sk, dtype=np.float64).ravel(),

        cfg.k_clip_lo,

        cfg.k_clip_hi,

        sig_pre=None,

        n_mono_band_nm=cfg.n_mono_band_nm,

        profile_interp=str(cfg.nk_profile_interp or "smooth"),

    )

    mse, rmse = spectral_mse_rmse_masked_from_nk(

        cfg,

        base_result,

        lam_full,

        np.asarray(n_lam, dtype=np.float64).ravel(),

        np.asarray(k_lam, dtype=np.float64).ravel(),

        float(d_nm),

    )

    return float(mse), float(rmse)


def _x_nodes0_from_mesh_x_if_consistent(
    base_eff: dict[str, Any],
    *,
    sk: np.ndarray,
    d0_nm: float,
) -> np.ndarray | None:
    """Extract ``x_nodes`` (size 2K) from ``x_seg_spline_sigma`` or ``x`` when dimensions and *d* match.

    Same layout as the spline worker / mesh polish: ``x = [d, n_slice…, L_slice…]``. Using this path

    avoids ``n_phys → ξ → n_phys`` round-trip drift so packed-node spectral RMSE matches

    ``spectral_rmse_best_value`` at ``d0`` to near machine precision (when the dict *x* is authoritative).

    """

    sk_a = np.asarray(sk, dtype=np.float64).ravel()

    k = int(sk_a.size)

    if k < 2:

        return None

    want = 1 + 2 * k

    d_ref = float(d0_nm)

    if not np.isfinite(d_ref):

        return None

    # ULP-tight *d* check: same float pipeline should reproduce bit-identical *d* in x[0] and d_nm.

    d_atol = float(np.finfo(np.float64).eps) * max(64.0, abs(d_ref), 1.0)

    for key in ("x_seg_spline_sigma", "x"):

        raw = base_eff.get(key)

        if raw is None:

            continue

        xa = np.asarray(raw, dtype=np.float64).ravel()

        if xa.size != want:

            continue

        if not np.isfinite(float(xa[0])):

            continue

        if not np.isclose(float(xa[0]), d_ref, rtol=0.0, atol=d_atol):

            continue

        return np.concatenate(

            (xa[1 : 1 + k].astype(np.float64, copy=True), xa[1 + k : 1 + 2 * k].astype(np.float64, copy=True))

        )

    return None


def _log_corridor_base_geometry(

    *,

    sk: np.ndarray,

    n_phys: np.ndarray,

    L_nodes: np.ndarray,

    d0: float,

    sk_n_stored: np.ndarray | None,

    diag: dict[str, Any],

    rmse_ref_pipeline: float,

    rmse_seed_no_refit: float,

    mse_seed_no_refit: float,

    use_abs_delta: bool = False,


) -> None:

    """INFO logs: sigma grids, knots, seed RMSE vs pipeline ref."""

    k = int(sk.size)

    log.info(

        "%s ━━━ Profiling base (geometry) ━━━ x_encoding=%s | sigma_knots_n key=%s | "

        "n remesh (sigma_n->sigma_L)=%s | max|sigma_n-sigma_L|=%s mean|…|=%s (atol=%.3e) | sigma_grids_match=%s",

        _LOG_PREFIX,

        str(diag.get("x_encoding", "-")),

        "yes" if diag.get("sigma_knots_n_key_present") else "no",

        "yes" if diag.get("remeshed_n_sigma_n_to_sigma_L") else "no",

        f"{float(diag.get('max_abs_sigma_n_minus_L', float('nan'))):.6e}"

        if np.isfinite(float(diag.get("max_abs_sigma_n_minus_L", float("nan"))))

        else "n/a",

        f"{float(diag.get('mean_abs_sigma_n_minus_L', float('nan'))):.6e}"

        if np.isfinite(float(diag.get("mean_abs_sigma_n_minus_L", float("nan"))))

        else "n/a",

        float(diag.get("sigma_atol_nm_inv", 0.0)),

        str(diag.get("sigma_grids_coincide", False)),

    )

    log.info(

        "%s sigma_L (nm⁻¹) K=%d : %s",

        _LOG_PREFIX,

        k,

        np.array2string(np.asarray(sk, dtype=np.float64), precision=6, max_line_width=200),

    )

    if sk_n_stored is not None and sk_n_stored.size:

        log.info(

            "%s sigma_n (nm⁻¹) K=%d : %s",

            _LOG_PREFIX,

            int(sk_n_stored.size),

            np.array2string(np.asarray(sk_n_stored, dtype=np.float64), precision=6, max_line_width=200),

        )

    for i in range(k):

        sig = float(sk[i])

        lam_nm = 1.0 / max(sig, 1e-30)

        log.info(

            "%s   knot %2d/%d  sigma=%.6e nm⁻¹  lambda~%.2f nm  n=%.6f  ln_k=%.7f  k=%.6e",

            _LOG_PREFIX,

            i + 1,

            k,

            sig,

            lam_nm,

            float(n_phys[i]),

            float(L_nodes[i]),

            float(np.exp(np.clip(L_nodes[i], -80.0, 80.0))),

        )

    if use_abs_delta:

        log.info(

            "%s Spectral RMSE **sans refit** (graine bornée, d=d_opt) = %.8f (MSE=%.6e) | "

            "RMSE **seuil** (courbes nominale n_lam/k_lam base, même masque) = %.8f | écart graine-seuil=%+.6e. "

            "Si the seed ≫ seuil, vérifier sigma_n/sigma_L, mono ξ, clips. Les marches +/-d ré-optimisent n,L à d fixé.",

            _LOG_PREFIX,

            float(rmse_seed_no_refit),

            float(mse_seed_no_refit),

            float(rmse_ref_pipeline),

            float(rmse_seed_no_refit - rmse_ref_pipeline),

        )

    else:

        log.info(

            "%s Spectral RMSE **without refit** (clipped seed, d=d_opt) = %.8f (MSE=%.6e) | pipeline RMSE_ref=%.8f "

            "(spectral_rmse_segments / dict). seed-ref gap=%+.6e - if seed ≫ ref, check sigma_n/sigma_L grids, mono ξ, bound clips. "

            "Following corridor **refits** re-optimize n,L: their RMSE can be **> RMSE_ref** (expected).",

            _LOG_PREFIX,

            float(rmse_seed_no_refit),

            float(mse_seed_no_refit),

            float(rmse_ref_pipeline),

            float(rmse_seed_no_refit - rmse_ref_pipeline),

        )


def _bounds_for_nodes_only(cfg: SplineOptConfig, k: int) -> tuple[np.ndarray, np.ndarray]:

    """Bounds (n-slice, L-slice) and default x0 (size 2K)."""

    k = int(k)

    # n bounds: physical or ξ (then N_MONO_XI_BOUNDS apply via cfg/objective);

    # for profiling we reuse the same layout as the optimizer: optimize components

    # exactly as in x = [d, n_slice..., L_slice...].

    if cfg.n_mono_band_nm is None:

        n_lo, n_hi = float(N_MIN_LIMIT), float(N_MAX_LIMIT)

    else:

        # ξ bounds identical to pipeline (see certus_index_spline_core.N_MONO_XI_BOUNDS).

        # Local import to avoid a heavy cyclic import.

        from certus_index_spline_core import N_MONO_XI_BOUNDS

        n_lo, n_hi = map(float, N_MONO_XI_BOUNDS)

    lo_k = float(max(cfg.k_clip_lo, 1e-30))

    hi_k = float(max(cfg.k_clip_hi, lo_k * 1.0001))

    L_lo = float(np.log(lo_k))

    L_hi = float(np.log(hi_k))

    bounds = np.zeros((2 * k, 2), dtype=np.float64)

    bounds[:k] = [n_lo, n_hi]

    bounds[k:] = [L_lo, L_hi]

    x0 = 0.5 * (bounds[:, 0] + bounds[:, 1])

    # Neutral values (consistent with make_bounds_and_x0):

    if cfg.n_mono_band_nm is None:

        x0[:k] = np.clip(1.65, n_lo, n_hi)

    else:

        x0[:k] = np.clip(0.0, n_lo, n_hi)

    x0[k:] = np.clip(np.log(1e-3), L_lo, L_hi)

    return bounds, x0


def _fit_nodes_at_fixed_d(

    cfg: SplineOptConfig,

    sigma_knots: np.ndarray,

    d_target_nm: float,

    x_nodes_init: np.ndarray,

    bounds_nodes: np.ndarray,

    *,

    maxfun: int = 4000,
    keep_nominal_seed_if_refit_worse: bool = True,
    seed_keep_tol_rel: float = 0.0,
    seed_keep_tol_abs: float = 1e-5,
    pure_spectral: bool = False,


) -> dict[str, Any] | None:

    """Optimize (n_slice, L_slice) at fixed d and return a minimal snapshot.

    Returns None if the masked objective is empty or dimensions are inconsistent.

    """

    sk = np.asarray(sigma_knots, dtype=np.float64).ravel()

    k = int(sk.size)

    if k < 2:

        return None

    b = np.asarray(bounds_nodes, dtype=np.float64)

    x0 = np.asarray(x_nodes_init, dtype=np.float64).ravel().copy()

    if b.shape != (2 * k, 2) or x0.size != 2 * k:

        return None

    # Pure-spectral mode: disable n_lambda_rising penalty and lnk regularization so L-BFGS-B
    # genuinely minimises spectral RMSE at the target d instead of fleeing the nominal solution.
    if pure_spectral:
        cfg = replace(cfg, spline_pure_spectral_objective=True)
        log.debug(
            "%s _fit_nodes_at_fixed_d d=%.5g nm: pure_spectral=True -> n_lambda_rising penalty disabled for refit.",
            _LOG_PREFIX,
            float(d_target_nm),
        )

    # Masked objective (same weights / RMSE window); if empty, cannot profile.

    obj_stage = SplinePWLObjective(cfg, sk)

    # Build full x vector for obj(x): x = [d, n_slice..., L_slice...]

    def _pack(x_nodes: np.ndarray) -> np.ndarray:

        return np.concatenate((np.asarray([float(d_target_nm)], dtype=np.float64), x_nodes))

    x0 = clip_to_bounds(x0, b[:, 0], b[:, 1])

    def _mse_nodes(x_nodes: np.ndarray) -> float:

        x_full = _pack(x_nodes)

        mse0 = float(obj_stage(x_full))

        if not np.isfinite(mse0):

            return 1e30

        if bool(getattr(cfg, "spline_pure_spectral_objective", False)):

            return float(mse0)

        # ln(k) regularization consistent with SOL3b (spline_workers._run_free_knot_stage):

        # penalty on discrete curvature d2(L) at sigma knots.

        reg_w = float(max(getattr(cfg, "lnk_spline_reg_weight", 0.0) or 0.0, 0.0))

        if reg_w > 0.0:

            k_loc = int(sk.size)

            L_nodes = np.asarray(x_nodes, dtype=np.float64).ravel()[k_loc:]

            if L_nodes.size >= 3:

                d2 = np.diff(L_nodes, n=2)

                mse0 = float(mse0) + reg_w * float(np.mean(d2 * d2))

        return float(mse0)

    m0 = _mse_nodes(x0)

    if not np.isfinite(m0) or m0 >= 1e29:

        return None

    lam_full = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    x_full_seed = np.concatenate((np.asarray([float(d_target_nm)], dtype=np.float64), x0))

    n_lam_seed, k_lam_seed = nk_from_x_pwlnk(

        x_full_seed,

        lam_full,

        sk,

        cfg.k_clip_lo,

        cfg.k_clip_hi,

        sig_pre=None,

        n_mono_band_nm=cfg.n_mono_band_nm,

        profile_interp=str(cfg.nk_profile_interp or "smooth"),

    )

    mse_seed_spec, rmse_seed_spec = spectral_mse_rmse_masked_from_nk(

        cfg,

        {},

        lam_full,

        np.asarray(n_lam_seed, dtype=np.float64).ravel(),

        np.asarray(k_lam_seed, dtype=np.float64).ravel(),

        float(d_target_nm),

    )

    rmse_seed = float(rmse_seed_spec) if np.isfinite(float(rmse_seed_spec)) else float("nan")

    from spline_objective import spline_pwl_analytic_grad_supported

    def _jac_nodes(x_nodes: np.ndarray) -> np.ndarray | None:

        if not spline_pwl_analytic_grad_supported(cfg):

            return None

        x_full = _pack(x_nodes)

        g_full = obj_stage.analytic_gradient(x_full)

        if g_full is None:

            return None

        gn = np.asarray(g_full[1:], dtype=np.float64).ravel().copy()

        if bool(getattr(cfg, "spline_pure_spectral_objective", False)):

            return gn

        reg_w = float(max(getattr(cfg, "lnk_spline_reg_weight", 0.0) or 0.0, 0.0))

        if reg_w > 0.0:

            k_loc = int(sk.size)

            L_nodes = np.asarray(x_nodes, dtype=np.float64).ravel()[k_loc:]

            if L_nodes.size >= 3:

                d2 = np.diff(L_nodes, n=2)

                m = int(d2.size)

                if m > 0:

                    for j in range(m):

                        fac = reg_w * (2.0 * float(d2[j])) / float(m)

                        gn[k_loc + j] += fac * 1.0

                        gn[k_loc + j + 1] += fac * (-2.0)

                        gn[k_loc + j + 2] += fac * 1.0

        return gn

    _jac_n = None

    if spline_pwl_analytic_grad_supported(cfg):

        _tj = _jac_nodes(x0)

        if _tj is not None and np.all(np.isfinite(_tj)):

            _jac_n = _jac_nodes

    bds = [(float(b[i, 0]), float(b[i, 1])) for i in range(int(b.shape[0]))]

    res = minimize(

        _mse_nodes,

        x0,

        method="L-BFGS-B",

        jac=_jac_n,

        bounds=bds,

        options={"maxfun": int(max(300, maxfun)), "ftol": 1e-11, "gtol": 1e-8},

    )

    x_best = np.asarray(getattr(res, "x", x0), dtype=np.float64).ravel()

    x_best = clip_to_bounds(x_best, b[:, 0], b[:, 1])

    m_best_obj = float(_mse_nodes(x_best))

    if not np.isfinite(m_best_obj) or m_best_obj >= 1e29:

        return None

    # Rebuild n(lambda), k(lambda) on masked / full grid via nk_from_x_pwlnk.

    x_full_best = np.concatenate((np.asarray([float(d_target_nm)], dtype=np.float64), x_best))

    n_lam, k_lam = nk_from_x_pwlnk(

        x_full_best,

        lam_full,

        sk,

        cfg.k_clip_lo,

        cfg.k_clip_hi,

        sig_pre=None,

        n_mono_band_nm=cfg.n_mono_band_nm,

        profile_interp=str(cfg.nk_profile_interp or "smooth"),

    )

    mse_spec, rmse_spec = spectral_mse_rmse_masked_from_nk(

        cfg,

        {},

        lam_full,

        np.asarray(n_lam, dtype=np.float64).ravel(),

        np.asarray(k_lam, dtype=np.float64).ravel(),

        float(d_target_nm),

    )

    rmse = float(rmse_spec) if np.isfinite(float(rmse_spec)) else float("nan")

    rmse_refit_attempt = float(rmse)

    keep_seed_tol = max(float(seed_keep_tol_abs), float(seed_keep_tol_rel) * max(rmse_seed, 1e-12))

    keep_seed = bool(
        bool(keep_nominal_seed_if_refit_worse)
        and np.isfinite(rmse_seed)

        and np.isfinite(rmse)

        and rmse > rmse_seed + keep_seed_tol

    )

    if keep_seed:

        log.info(

            "Corridor fixed-d refit d=%.5g nm: kept nominal seed "
            "(spectral RMSE refit=%.5g > seed=%.5g + tol=%.5g).",

            float(d_target_nm),

            float(rmse_refit_attempt),

            float(rmse_seed),

            float(keep_seed_tol),

        )

        x_best = x0.copy()

        m_best_obj = float(m0)

        n_lam = np.asarray(n_lam_seed, dtype=np.float64).ravel()

        k_lam = np.asarray(k_lam_seed, dtype=np.float64).ravel()

        mse_spec = float(mse_seed_spec) if np.isfinite(float(mse_seed_spec)) else float("nan")

        rmse = rmse_seed

    # For reporting: physical n_nodes (useful for export) when n_mono is active.

    n_slice = x_best[:k]

    n_nodes_phys = (

        np.asarray(n_slice, dtype=np.float64).copy()

        if cfg.n_mono_band_nm is None

        else x_slice_n_to_physical_nodes(n_slice, sk, cfg.n_mono_band_nm)

    )

    L_nodes = x_best[k:].copy()

    _msg = str(getattr(res, "message", ""))[:160]

    if keep_seed:

        _msg = f"{_msg} | seed_kept_over_refit(spectral_RMSE)"[:160]

    return {

        "success": bool(getattr(res, "success", False)),

        "message": _msg,

        "nit": int(getattr(res, "nit", 0) or 0),

        "nfev": int(getattr(res, "nfev", 0) or 0),

        "d_nm": float(d_target_nm),

        "mse": float(mse_spec) if np.isfinite(float(mse_spec)) else float("nan"),

        "mse_objective": float(m_best_obj),

        "rmse": rmse,

        "rmse_seed_before_refit": float(rmse_seed) if np.isfinite(rmse_seed) else None,

        "seed_kept_over_refit": bool(keep_seed),

        "seed_keep_tolerance": float(keep_seed_tol),

        "seed_keep_rule": (
            "spectral_rmse_refit_gt_seed_plus_tol"
            if bool(keep_nominal_seed_if_refit_worse)
            else "disabled"
        ),

        "rmse_refit_attempted": float(rmse_refit_attempt) if np.isfinite(rmse_refit_attempt) else None,

        "chi2": None,

        # Exact warm start (same x space as optimizer: n_slice (physical or ξ) + L_nodes)

        "x_nodes_best": x_best,

        "sigma_knots": sk,

        "n_nodes_physical": n_nodes_phys,

        "L_nodes": L_nodes,

        "n_lam": np.asarray(n_lam, dtype=np.float64).ravel(),

        "k_lam": np.asarray(k_lam, dtype=np.float64).ravel(),

    }


def _hetero_sigma_masked_from_base(

    cfg: SplineOptConfig,

    base_result: dict,

    *,

    scale: float,

    floor_abs: float,


) -> tuple[np.ndarray | None, np.ndarray | None]:

    """sigma_T(lambda), sigma_R(lambda) on masked grid: max(floor, scale×|y_exp-y_th|) like base_result model."""

    from spline_objective import build_spline_objective_masked_grid

    from certus_physics import calculate_reflection_array, calculate_transmission_array

    from certus_index_utils import _ratio_theoretical_from_nk, _reflectance_ratio_theoretical_from_nk

    from certus_index_spline_core import _reflectance_absolute_backside_from_nk

    mg = build_spline_objective_masked_grid(cfg)

    if mg is None:

        return None, None

    lam_f, _sig_f, n_sub_f, _w, _inv_npix, t_exp_f, r_exp_f = mg

    lam_full = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    n_lam = np.asarray(base_result.get("n_lam", []), dtype=np.float64).ravel()

    k_lam = np.asarray(base_result.get("k_lam", []), dtype=np.float64).ravel()

    d_nm = float(base_result.get("d_nm", float("nan")))

    if n_lam.size != lam_full.size or k_lam.size != lam_full.size or not np.isfinite(d_nm):

        return None, None

    n_f = np.interp(lam_f, lam_full, n_lam)

    k_f = np.interp(lam_f, lam_full, k_lam)

    fl = float(max(floor_abs, 1e-12))

    sc = float(max(scale, 0.0))

    sigma_t_f: np.ndarray | None = None

    sigma_r_f: np.ndarray | None = None

    if cfg.data_type in (DataType.TRANSMISSION, DataType.BOTH) and t_exp_f is not None and float(cfg.weight_t) > 0:

        if cfg.t_is_ratio:

            t_th = _ratio_theoretical_from_nk(lam_f, n_f, k_f, n_sub_f, float(d_nm))

        else:

            t_th = calculate_transmission_array(lam_f, n_f, k_f, float(d_nm), n_sub_f)

        e = np.asarray(t_exp_f, dtype=np.float64) - np.asarray(t_th, dtype=np.float64)

        sigma_t_f = np.maximum(fl, sc * np.abs(e))

    if cfg.data_type in (DataType.REFLECTION, DataType.BOTH) and r_exp_f is not None and float(cfg.weight_r) > 0:

        if cfg.t_is_ratio:

            r_th = _reflectance_ratio_theoretical_from_nk(lam_f, n_f, k_f, n_sub_f, float(d_nm))

        else:

            r_th = _reflectance_absolute_backside_from_nk(lam_f, n_f, k_f, float(d_nm), n_sub_f)

        e = np.asarray(r_exp_f, dtype=np.float64) - np.asarray(r_th, dtype=np.float64)

        sigma_r_f = np.maximum(fl, sc * np.abs(e))

    return sigma_t_f, sigma_r_f


def _chi2_masked_constant_sigma(

    cfg: SplineOptConfig,

    *,

    sigma_t: float,

    sigma_r: float,

    n_lam_full: np.ndarray,

    k_lam_full: np.ndarray,

    d_nm: float,

    sigma_t_f: np.ndarray | None = None,

    sigma_r_f: np.ndarray | None = None,


) -> float:

    """χ² on objective mask: constant sigma or sigma_i (vectors aligned with lam_f / exp_f)."""

    from spline_objective import build_spline_objective_masked_grid

    from certus_physics import calculate_reflection_array, calculate_transmission_array

    from certus_index_utils import _ratio_theoretical_from_nk, _reflectance_ratio_theoretical_from_nk

    from certus_index_spline_core import _reflectance_absolute_backside_from_nk

    mg = build_spline_objective_masked_grid(cfg)

    if mg is None:

        return float("nan")

    lam_f, _sig_f, n_sub_f, w, _inv_npix, t_exp_f, r_exp_f = mg

    lam_full = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    n_full = np.asarray(n_lam_full, dtype=np.float64).ravel()

    k_full = np.asarray(k_lam_full, dtype=np.float64).ravel()

    if n_full.size != lam_full.size or k_full.size != lam_full.size:

        return float("nan")

    n_f = np.interp(lam_f, lam_full, n_full)

    k_f = np.interp(lam_f, lam_full, k_full)

    chi = 0.0

    if cfg.data_type in (DataType.TRANSMISSION, DataType.BOTH) and t_exp_f is not None and float(cfg.weight_t) > 0:

        if cfg.t_is_ratio:

            t_th = _ratio_theoretical_from_nk(lam_f, n_f, k_f, n_sub_f, float(d_nm))

        else:

            t_th = calculate_transmission_array(lam_f, n_f, k_f, float(d_nm), n_sub_f)

        e = (np.asarray(t_exp_f, dtype=np.float64) - t_th)

        if sigma_t_f is not None:

            st = np.asarray(sigma_t_f, dtype=np.float64).ravel()

            if st.size != e.size:

                return float("nan")

            chi += float(cfg.weight_t) * float(np.dot(w, (e / st) ** 2))

        else:

            chi += float(cfg.weight_t) * float(np.dot(w, (e / float(sigma_t)) ** 2))

    if cfg.data_type in (DataType.REFLECTION, DataType.BOTH) and r_exp_f is not None and float(cfg.weight_r) > 0:

        if cfg.t_is_ratio:

            r_th = _reflectance_ratio_theoretical_from_nk(lam_f, n_f, k_f, n_sub_f, float(d_nm))

        else:

            # Consistent with objective: absolute backside reflection in non-ratio mode.

            r_th = _reflectance_absolute_backside_from_nk(lam_f, n_f, k_f, float(d_nm), n_sub_f)

        e = (np.asarray(r_exp_f, dtype=np.float64) - r_th)

        if sigma_r_f is not None:

            sr = np.asarray(sigma_r_f, dtype=np.float64).ravel()

            if sr.size != e.size:

                return float("nan")

            chi += float(cfg.weight_r) * float(np.dot(w, (e / sr) ** 2))

        else:

            chi += float(cfg.weight_r) * float(np.dot(w, (e / float(sigma_r)) ** 2))

    return float(chi) if np.isfinite(chi) else float("nan")


def _best_fit_at_d(

    cfg: SplineOptConfig,

    *,

    sk: np.ndarray,

    d_nm: float,

    x_seed_primary: np.ndarray,

    x_seed_secondary: np.ndarray | None,

    x_seed_default: np.ndarray,

    bounds_nodes: np.ndarray,

    maxfun: int,

    use_lr: bool,

    sig_t: float,

    sig_r: float,

    chi2_min_ref: float | None,

    delta_chi2: float,

    pconf: ProfileCorridorConfig,

    stage_label: str,

    sigma_t_f: np.ndarray | None = None,

    sigma_r_f: np.ndarray | None = None,


) -> tuple[dict[str, Any] | None, bool, float]:

    """Multi-start: returns (best_fit, ok, metric_value_for_ok_check).

    - ok follows the mode rule (alpha or lr) using chi2_min_ref on the LR side.

    - metric_value_for_ok_check = RMSE (alpha) or χ² (lr) of best_fit.

    """

    rng = np.random.default_rng(int(pconf.rng_seed))

    seeds: list[np.ndarray] = []

    x1 = np.asarray(x_seed_primary, dtype=np.float64).ravel()

    seeds.append(x1)

    if x_seed_secondary is not None:

        seeds.append(np.asarray(x_seed_secondary, dtype=np.float64).ravel())

    seeds.append(np.asarray(x_seed_default, dtype=np.float64).ravel())

    # Jitter on seeds (when n_starts > len(seeds))

    k = int(np.asarray(sk).size)

    n_dim = 2 * k

    n_starts_eff = int(max(1, pconf.n_starts))

    while len(seeds) < n_starts_eff:

        base = seeds[0].copy() if seeds else x1.copy()

        j = base.copy()

        j[:k] += rng.normal(0.0, float(pconf.jitter_n), size=k)

        j[k:] += rng.normal(0.0, float(pconf.jitter_L), size=k)

        seeds.append(j)

    best_fit: dict[str, Any] | None = None

    best_metric = float("inf")

    best_rmse = float("inf")

    best_chi2 = float("nan")

    n_try = 0

    n_ok_fit = 0

    n_failed_fit = 0

    n_exceeds_limit = 0

    n_starts_cap = int(max(n_starts_eff, pconf.fit_max_n_starts))

    idx = 0

    while idx < int(min(len(seeds), n_starts_cap)):

        x0 = seeds[idx]

        n_try += 1

        fit = _fit_nodes_at_fixed_d(
            cfg,
            sk,
            float(d_nm),
            x0,
            bounds_nodes,
            maxfun=int(maxfun),
            keep_nominal_seed_if_refit_worse=bool(
                getattr(pconf, "seed_gate_keep_nominal_if_refit_worse", True)
            ),
            seed_keep_tol_rel=float(getattr(pconf, "seed_gate_tol_rel", 0.0)),
            seed_keep_tol_abs=float(getattr(pconf, "seed_gate_tol_abs", 1e-5)),
            pure_spectral=bool(getattr(pconf, "refit_pure_spectral", True)),
        )

        if fit is not None and str(fit.get("message", "")).upper().find("EXCEEDS LIMIT") >= 0:

            n_exceeds_limit += 1

            scale = float(max(getattr(pconf, "fit_retry_maxfun_scale", 1.0) or 1.0, 1.0))

            if scale > 1.0:

                fit_retry = _fit_nodes_at_fixed_d(

                    cfg,

                    sk,

                    float(d_nm),

                    np.asarray(fit.get("x_nodes_best", x0), dtype=np.float64).ravel(),

                    bounds_nodes,

                    maxfun=int(max(maxfun + 1, round(float(maxfun) * scale))),
                    keep_nominal_seed_if_refit_worse=bool(
                        getattr(pconf, "seed_gate_keep_nominal_if_refit_worse", True)
                    ),
                    seed_keep_tol_rel=float(getattr(pconf, "seed_gate_tol_rel", 0.0)),
                    seed_keep_tol_abs=float(getattr(pconf, "seed_gate_tol_abs", 1e-5)),
                    pure_spectral=bool(getattr(pconf, "refit_pure_spectral", True)),

                )

                if fit_retry is not None and np.isfinite(float(fit_retry.get("rmse", float("nan")))):

                    fit = fit_retry

        if fit is None or not np.isfinite(float(fit.get("rmse", float("nan")))):

            n_failed_fit += 1

            if bool(getattr(pconf, "fit_auto_n_starts", True)) and len(seeds) < n_starts_cap:

                jb = x1.copy()

                jb[:k] += rng.normal(0.0, float(pconf.jitter_n), size=k)

                jb[k:] += rng.normal(0.0, float(pconf.jitter_L), size=k)

                seeds.append(jb)

            idx += 1

            continue

        n_ok_fit += 1

        rm = float(fit["rmse"])

        if use_lr:

            chi = _chi2_masked_constant_sigma(

                cfg,

                sigma_t=sig_t,

                sigma_r=sig_r,

                n_lam_full=fit["n_lam"],

                k_lam_full=fit["k_lam"],

                d_nm=float(d_nm),

                sigma_t_f=sigma_t_f,

                sigma_r_f=sigma_r_f,

            )

            metric = float(chi) if np.isfinite(chi) else float("inf")

        else:

            metric = rm

        if metric < best_metric:

            best_metric = metric

            best_fit = fit

            best_rmse = rm

            best_chi2 = float(best_metric) if use_lr else float("nan")

        idx += 1

    if best_fit is None:

        log.info("%s %s: multi-start failed | d=%.6f nm | n_starts=%d", _LOG_PREFIX, stage_label, float(d_nm), int(pconf.n_starts))

        return None, False, float("nan")

    best_fit["n_try"] = int(n_try)

    best_fit["n_ok_fit"] = int(n_ok_fit)

    best_fit["n_failed_fit"] = int(n_failed_fit)

    best_fit["n_exceeds_limit"] = int(n_exceeds_limit)

    # Determine ok from mode.

    if use_lr:

        if chi2_min_ref is None or not np.isfinite(float(chi2_min_ref)):

            ok = False

        else:

            ok = np.isfinite(best_chi2) and best_chi2 <= float(chi2_min_ref) + float(delta_chi2)

        log.info(

            "%s %s: best-of-%d | d=%.6f nm | chi2=%.8f | rmse=%.8f (refit n,L at fixed d) | ok=%s | okfits=%d",

            _LOG_PREFIX,

            stage_label,

            int(n_try),

            float(d_nm),

            float(best_chi2),

            float(best_rmse),

            bool(ok),

            int(n_ok_fit),

        )

        return best_fit, ok, float(best_chi2)

    else:

        ok = True  # alpha filtering is done by caller with rmse_thresh

        log.info(

            "%s %s: best-of-%d | d=%.6f nm | rmse=%.8f (refit n,L at fixed d; not solver \"segments\" RMSE) | okfits=%d",

            _LOG_PREFIX,

            stage_label,

            int(n_try),

            float(d_nm),

            float(best_rmse),

            int(n_ok_fit),

        )

        return best_fit, ok, float(best_rmse)


def _corridor_profile_walk_side(

    walk_sign: float,

    pconf: ProfileCorridorConfig,

    cfg: SplineOptConfig,

    sk: np.ndarray,

    d0: float,

    x_nodes_center: np.ndarray,

    x0_default: np.ndarray,

    bounds_nodes: np.ndarray,

    maxfun_prof: int,

    use_lr: bool,

    chi2_min: float,

    delta_chi2: float,

    rmse_thresh_active: float,

    rmse_opt: float,

    sig_t: float,

    sig_r: float,

    sigma_t_f_hetero: np.ndarray | None,

    sigma_r_f_hetero: np.ndarray | None,


) -> dict[str, Any]:

    """One-sided d continuation (+d or -d). Use distinct ``pconf.rng_seed`` per thread when running in parallel."""

    d_vals: list[float] = []

    n_curves: list[np.ndarray] = []

    k_curves: list[np.ndarray] = []

    rmse_vals: list[float] = []

    chi2_vals: list[float] = []

    fit_nfev_values: list[float] = []

    fit_nit_values: list[float] = []

    fit_try_values: list[float] = []

    fit_fail_values: list[float] = []

    n_refines = 0
    seed_gate_eval_count = 0
    seed_gate_kept_count = 0
    seed_gate_delta_refit_minus_seed: list[float] = []

    def _refine_bracket(
        a_d: float,
        a_x: np.ndarray,
        y_a: float,
        b_d: float,
        b_x: np.ndarray,
        y_b: float,
        br_sign: float,
    ) -> tuple[float, np.ndarray] | None:

        if not bool(pconf.refine_boundary):

            return None

        

        if not (a_d < b_d if br_sign > 0 else a_d > b_d):

            return None

        for _ in range(int(max(1, pconf.refine_max_iter))):

            if abs(b_d - a_d) <= float(max(1e-6, pconf.refine_tol_nm)):

                break

            if y_b - y_a > 1e-12:
                frac = -y_a / (y_b - y_a)
                frac = min(max(frac, 0.2), 0.8)
            else:
                frac = 0.5
            m_d = a_d + frac * (b_d - a_d)
            m_x0 = a_x

            fitm, _okm, metricm = _best_fit_at_d(

                cfg,

                sk=sk,

                d_nm=float(m_d),

                x_seed_primary=m_x0,

                x_seed_secondary=x_nodes_center,

                x_seed_default=x0_default,

                bounds_nodes=bounds_nodes,

                maxfun=maxfun_prof,

                use_lr=use_lr,

                sig_t=sig_t,

                sig_r=sig_r,

                chi2_min_ref=float(chi2_min) if (use_lr and np.isfinite(chi2_min)) else None,

                delta_chi2=float(delta_chi2) if np.isfinite(delta_chi2) else 0.0,

                pconf=pconf,

                stage_label="Refine",

                sigma_t_f=sigma_t_f_hetero,

                sigma_r_f=sigma_r_f_hetero,

            )

            if fitm is None or not np.isfinite(float(fitm.get("rmse", float("nan")))):

                b_d = float(m_d)

                b_x = np.asarray(m_x0, dtype=np.float64).copy()

                continue

            rm = float(fitm["rmse"])

            ok_rb = False
            chi_m = float("nan")
            y_m = 0.0
            if use_lr:
                chi_m = float(metricm)
                ok_rb = np.isfinite(chi_m) and np.isfinite(chi2_min) and (chi_m <= chi2_min + delta_chi2)
                y_m = chi_m - (chi2_min + delta_chi2)
            else:
                ok_rb = rm <= rmse_thresh_active
                y_m = rm - rmse_thresh_active

            if ok_rb:
                a_d = float(m_d)
                y_a = y_m

                a_x = np.asarray(fitm["x_nodes_best"], dtype=np.float64).ravel().copy()

                d_vals.append(float(m_d))

                n_curves.append(np.asarray(fitm["n_lam"], dtype=np.float64))

                k_curves.append(np.asarray(fitm["k_lam"], dtype=np.float64))

                rmse_vals.append(rm)

                chi2_vals.append(float(chi_m) if np.isfinite(chi_m) else float("nan"))

            else:
                b_d = float(m_d)
                y_b = y_m
                b_x = np.asarray(fitm["x_nodes_best"], dtype=np.float64).ravel().copy()

        return float(a_d), np.asarray(a_x, dtype=np.float64).ravel().copy()

    d_prev = float(d0)

    x_prev = np.asarray(x_nodes_center, dtype=np.float64).copy()

    step = float(max(1e-6, getattr(pconf, "step_nm_initial", pconf.step_nm)))

    step_growth = float(max(getattr(pconf, "step_growth", 1.4) or 1.4, 1.0))

    step_cap = float(max(getattr(pconf, "step_nm_max", pconf.step_nm), 1e-6))

    span = 0.0

    nsteps = 0

    effective_max_span = float(pconf.max_span_nm)
    _d_range = float(abs(max(cfg.d_lo, cfg.d_hi) - min(cfg.d_lo, cfg.d_hi)))
    _max_span_safety = min(_d_range * 0.45, max(effective_max_span * 30.0, 300.0))
    _span_hist: list[float] = [0.0]
    _rmse_hist: list[float] = [float(rmse_opt) if np.isfinite(float(rmse_opt)) else float("nan")]

    last_good_d: float | None = None

    last_good_x: np.ndarray | None = None
    last_good_y: float = -delta_chi2 if use_lr else (rmse_opt - rmse_thresh_active)
    last_good_y: float = -delta_chi2 if use_lr else (rmse_opt - rmse_thresh_active)

    dir_lbl = "+d" if float(walk_sign) > 0 else "-d"

    n_valid_side = 0

    log.info(

        "%s Walk %s: start d=%.6f nm | step=%.4g nm | span_max=%.4g nm",

        _LOG_PREFIX,

        dir_lbl,

        d_prev,

        step,

        float(pconf.max_span_nm),

    )

    while nsteps < int(pconf.max_steps_each_side) and span < effective_max_span:

        d_try = float(d_prev + walk_sign * step)

        if d_try < float(min(cfg.d_lo, cfg.d_hi)) - 1e-9 or d_try > float(max(cfg.d_lo, cfg.d_hi)) + 1e-9:

            log.info(

                "%s Walk %s: stop (d bound) | d_try=%.6f nm not in [%.6f, %.6f]",

                _LOG_PREFIX,

                dir_lbl,

                d_try,

                float(min(cfg.d_lo, cfg.d_hi)),

                float(max(cfg.d_lo, cfg.d_hi)),

            )

            break

        # --- Iso-Phase Warm-Start (Physics scaling) ---
        # Optical path invariant: n * d ~ constant. To stay exactly in the interference 
        # valley without forcing the optimizer to slide down the gradient, we project n.
        x_smart_seed = x_prev.copy()
        try:
            from spline_objective import x_slice_n_to_physical_nodes
            from certus_index_spline_core import physical_nodes_to_x_slice_n
            from certus_core import N_MIN_LIMIT, N_MAX_LIMIT
            ratio_d = float(d_prev / d_try) if d_try > 0 else 1.0
            _n_old = x_slice_n_to_physical_nodes(x_prev[1:1+k], sk, cfg.n_mono_band_nm)
            _n_new = np.clip(_n_old * ratio_d, N_MIN_LIMIT, N_MAX_LIMIT)
            x_smart_seed[1:1+k] = physical_nodes_to_x_slice_n(_n_new, sk, cfg.n_mono_band_nm)
        except Exception as e:
            log.debug("Iso-Phase warm-start projection failed: %s", e)
            x_smart_seed = x_prev
            
        fit, _okfit, metricv = _best_fit_at_d(

            cfg,

            sk=sk,

            d_nm=float(d_try),

            x_seed_primary=x_smart_seed,

            x_seed_secondary=x_nodes_center,

            x_seed_default=x0_default,

            bounds_nodes=bounds_nodes,

            maxfun=maxfun_prof,

            use_lr=use_lr,

            sig_t=sig_t,

            sig_r=sig_r,

            chi2_min_ref=float(chi2_min) if (use_lr and np.isfinite(chi2_min)) else None,

            delta_chi2=float(delta_chi2) if np.isfinite(delta_chi2) else 0.0,

            pconf=pconf,

            stage_label=f"Step {dir_lbl}",

            sigma_t_f=sigma_t_f_hetero,

            sigma_r_f=sigma_r_f_hetero,

        )

        if fit is None or not np.isfinite(float(fit.get("rmse", float("nan")))):

            log.info("%s Walk %s: stop (fit invalid) | d=%.6f nm", _LOG_PREFIX, dir_lbl, d_try)

            break

        rm = float(fit["rmse"])
        rmse_seed_fit = fit.get("rmse_seed_before_refit")
        rmse_refit_fit = fit.get("rmse_refit_attempted")
        has_seed_gate = (
            rmse_seed_fit is not None
            and rmse_refit_fit is not None
            and np.isfinite(float(rmse_seed_fit))
            and np.isfinite(float(rmse_refit_fit))
        )
        if has_seed_gate:
            seed_gate_eval_count += 1
            if bool(fit.get("seed_kept_over_refit", False)):
                seed_gate_kept_count += 1
            seed_gate_delta_refit_minus_seed.append(float(rmse_refit_fit) - float(rmse_seed_fit))

        ok = False
        chi_v = float("nan")
        y_v = float("nan")
        
        if use_lr:
            chi_v = float(metricv)
            y_v = chi_v - (chi2_min + delta_chi2)
            ok = np.isfinite(chi_v) and np.isfinite(chi2_min) and (y_v <= 0)
        else:
            y_v = rm - rmse_thresh_active
            ok = rm <= rmse_thresh_active

        if use_lr:

            log.info(

                "%s Walk %s: d=%.6f nm | chi2=%.8f | Deltachi2=%+.8f | RMSE=%.8f | nit=%d nfev=%d",

                _LOG_PREFIX,

                dir_lbl,

                float(d_try),

                float(chi_v),

                float(chi_v - chi2_min) if np.isfinite(chi2_min) else float("nan"),

                float(rm),

                int(fit.get("nit", 0) or 0),

                int(fit.get("nfev", 0) or 0),

            )

        else:

            log.info(

                "%s Walk %s: d=%.6f nm | RMSE=%.8f | DeltaRMSE(vs_ref_best)=%+.8f | seed_gate=%s | DeltaRMSE(refit-seed)=%s | nit=%d nfev=%d",

                _LOG_PREFIX,

                dir_lbl,

                float(d_try),

                float(rm),

                float(rm - rmse_opt) if np.isfinite(rmse_opt) else float("nan"),

                "kept" if bool(fit.get("seed_kept_over_refit", False)) else "refit",

                (
                    f"{(float(rmse_refit_fit) - float(rmse_seed_fit)):+.8f}"
                    if has_seed_gate
                    else "n/a"
                ),

                int(fit.get("nit", 0) or 0),

                int(fit.get("nfev", 0) or 0),

            )

        if ok:

            d_vals.append(float(d_try))

            n_curves.append(np.asarray(fit["n_lam"], dtype=np.float64))

            k_curves.append(np.asarray(fit["k_lam"], dtype=np.float64))

            rmse_vals.append(rm)

            chi2_vals.append(float(chi_v) if np.isfinite(chi_v) else float("nan"))

            fit_nfev_values.append(float(fit.get("nfev", float("nan"))))

            fit_nit_values.append(float(fit.get("nit", float("nan"))))

            fit_try_values.append(float(fit.get("n_try", float("nan"))))

            fit_fail_values.append(float(fit.get("n_failed_fit", float("nan"))))

            x_prev = np.asarray(fit["x_nodes_best"], dtype=np.float64).ravel().copy()

            d_prev = float(d_try)

            span = abs(d_prev - float(d0))
            last_good_d = float(d_prev)
            last_good_x = x_prev.copy()
            last_good_y = float(y_v)

            nsteps += 1

            step = min(step_cap, step * step_growth)

            _span_hist.append(span)
            _rmse_hist.append(rm)
            if (
                len(_span_hist) >= 3
                and rm < rmse_thresh_active
                and np.isfinite(rmse_thresh_active)
                and np.isfinite(rm)
            ):
                _ds = _span_hist[-1] - _span_hist[-2]
                _dr = _rmse_hist[-1] - _rmse_hist[-2]
                if _ds > 1e-10 and _dr > 1e-15:
                    _slope = _dr / _ds
                    _nm_remaining = (rmse_thresh_active - rm) / _slope
                    secant_step = float(np.clip(_nm_remaining, step, step_cap * 2.0))
                    if secant_step > step:
                        step = secant_step

                    _projected = span + _nm_remaining * 1.15
                    if _projected > effective_max_span:
                        _new_span = min(_projected, _max_span_safety)
                        if _new_span > effective_max_span + 0.5:
                            log.info(
                                "%s Walk %s: auto-extend span %.1f -> %.1f nm "
                                "(RMSE slope=%.3g /nm, ~%.1f nm to threshold)",
                                _LOG_PREFIX,
                                dir_lbl,
                                effective_max_span,
                                _new_span,
                                _slope,
                                _nm_remaining,
                            )
                        effective_max_span = max(effective_max_span, _new_span)

            n_valid_side += 1

            continue

        if last_good_d is not None and last_good_x is not None:

            rb = _refine_bracket(last_good_d, last_good_x, last_good_y, float(d_try), x_prev, y_v, walk_sign)

            if rb is not None:

                n_refines += 1

                log.info(

                    "%s Walk %s: refined boundary on admissible side ~ d=%.6f nm",

                    _LOG_PREFIX,

                    dir_lbl,

                    float(rb[0]),

                )

        if use_lr and np.isfinite(chi_v) and np.isfinite(chi2_min):

            chi_lim = float(chi2_min) + float(delta_chi2)

            log.info(

                "%s Walk %s: stop (LR Deltaχ²) | d=%.6f nm χ²=%.6g > χ²_center+Delta=%.6g "

                "(Deltaχ² vs center=%+.6g | RMSE=%.8f; the alpha×RMSE threshold logged above does not apply to the LR criterion)",

                _LOG_PREFIX,

                dir_lbl,

                float(d_try),

                float(chi_v),

                float(chi_lim),

                float(chi_v - chi2_min),

                float(rm),

            )

        else:

            log.info(

                "%s Walk %s: stop (RMSE > seuil) | d=%.6f nm RMSE=%.8f > %.8f",

                _LOG_PREFIX,

                dir_lbl,

                float(d_try),

                float(rm),

                float(rmse_thresh_active),

            )

        break

    return {

        "d_vals": d_vals,

        "n_curves": n_curves,

        "k_curves": k_curves,

        "rmse_vals": rmse_vals,

        "chi2_vals": chi2_vals,

        "fit_nfev_values": fit_nfev_values,

        "fit_nit_values": fit_nit_values,

        "fit_try_values": fit_try_values,

        "fit_fail_values": fit_fail_values,

        "n_valid_side": int(n_valid_side),

        "n_refines": int(n_refines),
        "seed_gate_eval_count": int(seed_gate_eval_count),
        "seed_gate_kept_count": int(seed_gate_kept_count),
        "seed_gate_delta_refit_minus_seed": seed_gate_delta_refit_minus_seed,

    }


def compute_reg_sensitivity_scan(

    cfg: SplineOptConfig,

    base_result: dict,

    *,

    pconf: ProfileCorridorConfig,

    weights: np.ndarray,


) -> dict[str, Any]:

    """V2.3: sensitivity scan by varying ``cfg.lnk_spline_reg_weight``.

    For each weight, rerun ``compute_profiled_corridors_by_d`` (same thresholds / mode),

    then summarize:

      - admissible d interval

      - mean corridor width for n and k on lambda grid (mean(corridor_hi - corridor_lo)).

    Parallelism: ``cfg.corridor_reg_sensitivity_n_workers`` (default 1). Use ``<= 0`` for

    ``min(8, max(1, os.cpu_count()))`` when more than one weight.

    """

    w_arr = np.asarray(weights, dtype=np.float64).ravel()

    w_arr = w_arr[np.isfinite(w_arr) & (w_arr >= 0.0)]

    if w_arr.size == 0:

        return {}

    lam_full = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    out_weights: list[float] = []

    out_d_lo: list[float] = []

    out_d_hi: list[float] = []

    out_nw: list[float] = []

    out_kw: list[float] = []

    out_n_valid: list[int] = []

    def _reg_sens_one(iw: tuple[int, float]) -> tuple[int, float, float, float, float, float, int]:

        _idx, w = iw

        cfg_w = replace(cfg, lnk_spline_reg_weight=float(w))

        try:

            extra = compute_profiled_corridors_by_d(

                cfg_w, base_result, pconf=pconf, log_coaching=False

            )

        except Exception:

            log.exception("%s [REG-SENS] failed | reg_w=%.6g", _LOG_PREFIX, float(w))

            extra = {}

        d_int = extra.get("profile_d_interval_nm", None)

        dlo = float("nan")

        dhi = float("nan")

        if isinstance(d_int, (tuple, list)) and len(d_int) == 2:

            try:

                dlo = float(d_int[0])

                dhi = float(d_int[1])

            except (TypeError, ValueError):

                dlo, dhi = float("nan"), float("nan")

        n_lo = np.asarray(extra.get("corridor_n_lo", []), dtype=np.float64).ravel()

        n_hi = np.asarray(extra.get("corridor_n_hi", []), dtype=np.float64).ravel()

        k_lo = np.asarray(extra.get("corridor_k_lo", []), dtype=np.float64).ravel()

        k_hi = np.asarray(extra.get("corridor_k_hi", []), dtype=np.float64).ravel()

        n_width = float("nan")

        k_width = float("nan")

        if n_lo.size == lam_full.size and n_hi.size == lam_full.size:

            dn = n_hi - n_lo

            dn = dn[np.isfinite(dn)]

            if dn.size:

                n_width = float(np.mean(dn))

        if k_lo.size == lam_full.size and k_hi.size == lam_full.size:

            dk = k_hi - k_lo

            dk = dk[np.isfinite(dk)]

            if dk.size:

                k_width = float(np.mean(dk))

        n_valid = int(np.asarray(extra.get("profile_d_values_nm", [])).size)

        return (_idx, float(w), float(dlo), float(dhi), float(n_width), float(k_width), int(n_valid))

    nw = int(getattr(cfg, "corridor_reg_sensitivity_n_workers", 1) or 1)

    if nw <= 0:

        nw = min(8, max(1, os.cpu_count() or 1))

    w_jobs = list(enumerate(w_arr.tolist()))

    log.info(

        "%s [REG-SENS] Scan start | n=%d | workers=%d | weights=%s",

        _LOG_PREFIX,

        int(w_arr.size),

        int(min(nw, len(w_jobs)) if len(w_jobs) > 1 else 1),

        np.array2string(w_arr, precision=3),

    )

    if nw > 1 and len(w_jobs) > 1:

        try:

            with ThreadPoolExecutor(max_workers=min(nw, len(w_jobs))) as _pool:

                rows = list(_pool.map(_reg_sens_one, w_jobs))

        except Exception:

            log.exception("%s [REG-SENS] parallel failed - sequential fallback.", _LOG_PREFIX)

            rows = [_reg_sens_one(j) for j in w_jobs]

    else:

        rows = [_reg_sens_one(j) for j in w_jobs]

    rows.sort(key=lambda r: int(r[0]))

    for _idx, w, dlo, dhi, n_width, k_width, n_valid in rows:

        out_weights.append(float(w))

        out_d_lo.append(float(dlo))

        out_d_hi.append(float(dhi))

        out_nw.append(float(n_width))

        out_kw.append(float(k_width))

        out_n_valid.append(int(n_valid))

        log.info(

            "%s [REG-SENS] reg_w=%.6g | d=[%s,%s] nm | mean_width_n=%s mean_width_k=%s | n_valid=%d",

            _LOG_PREFIX,

            float(w),

            f"{dlo:.4f}" if np.isfinite(dlo) else "n/a",

            f"{dhi:.4f}" if np.isfinite(dhi) else "n/a",

            f"{n_width:.6g}" if np.isfinite(n_width) else "n/a",

            f"{k_width:.6g}" if np.isfinite(k_width) else "n/a",

            int(n_valid),

        )

    _log_coaching_reg_sensitivity_outcome(

        weights=np.asarray(out_weights, dtype=np.float64),

        d_lo=np.asarray(out_d_lo, dtype=np.float64),

        d_hi=np.asarray(out_d_hi, dtype=np.float64),

        nw=np.asarray(out_nw, dtype=np.float64),

        kw=np.asarray(out_kw, dtype=np.float64),

    )

    return {

        "reg_sens_enabled": True,

        "reg_sens_weights": np.asarray(out_weights, dtype=np.float64),

        "reg_sens_d_lo_nm": np.asarray(out_d_lo, dtype=np.float64),

        "reg_sens_d_hi_nm": np.asarray(out_d_hi, dtype=np.float64),

        "reg_sens_mean_width_n": np.asarray(out_nw, dtype=np.float64),

        "reg_sens_mean_width_k": np.asarray(out_kw, dtype=np.float64),

        "reg_sens_n_valid": np.asarray(out_n_valid, dtype=np.int64),

    }


def quick_pwlnk_refit_result_dict(

    cfg: SplineOptConfig,

    base_result: dict,

    *,

    maxfun: int,


) -> dict[str, Any] | None:

    """V2.5: one L-BFGS-B polish on (d, nodes) - same objective as ``SplinePWLObjective`` (canonical mesh).

    Warm-start from ``base_result`` (segment vector) without rerunning Smart Init / Swanepoel.

    """

    from certus_index_spline_core import _bounds_x0_for_sigma_knots, make_bounds_and_x0

    from spline_objective import (

        SplinePWLObjective,

        build_segment_optimizer_x_vector,

        nk_from_x_pwlnk,

        spectral_mse_rmse_masked_from_nk,

    )

    try:

        pair = build_segment_optimizer_x_vector(base_result, cfg)

        if pair is not None:

            x0_vec, sk = pair

            sk = np.asarray(sk, dtype=np.float64).ravel()

            bounds, _x0_def, _L_lo, _L_hi = _bounds_x0_for_sigma_knots(cfg, sk)

            x0 = clip_to_bounds(

                np.asarray(x0_vec, dtype=np.float64).ravel(), bounds[:, 0], bounds[:, 1]

            )

        else:

            bounds, x0_def, sk = make_bounds_and_x0(cfg, skip_smart_init=True)

            sk = np.asarray(sk, dtype=np.float64).ravel()

            x0 = clip_to_bounds(

                np.asarray(x0_def, dtype=np.float64).ravel(), bounds[:, 0], bounds[:, 1]

            )

    except Exception:

        log.debug("Bounds/x0 construction failed in quick_pwlnk_refit_result_dict", exc_info=True)

        return None

    from spline_objective import spline_pwl_analytic_grad_supported

    obj = SplinePWLObjective(cfg, sk)

    bds = [(float(bounds[i, 0]), float(bounds[i, 1])) for i in range(int(bounds.shape[0]))]

    _jac = None

    if spline_pwl_analytic_grad_supported(cfg):

        _gt = obj.analytic_gradient(x0)

        if _gt is not None and np.all(np.isfinite(_gt)):

            _jac = obj.analytic_gradient

    res = minimize(

        obj,

        x0,

        method="L-BFGS-B",

        jac=_jac,

        bounds=bds,

        options={"maxfun": int(max(300, maxfun)), "ftol": 1e-10, "gtol": 1e-7},

    )

    xb = clip_to_bounds(np.asarray(res.x, dtype=np.float64).ravel(), bounds[:, 0], bounds[:, 1])

    lam_full = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    d_nm = float(xb[0])

    n_lam, k_lam = nk_from_x_pwlnk(

        xb,

        lam_full,

        sk,

        cfg.k_clip_lo,

        cfg.k_clip_hi,

        sig_pre=None,

        n_mono_band_nm=cfg.n_mono_band_nm,

        profile_interp=str(cfg.nk_profile_interp or "smooth"),

    )

    _mse, rmse = spectral_mse_rmse_masked_from_nk(cfg, {}, lam_full, n_lam, k_lam, d_nm)

    k_nodes = int(sk.size)

    n_slice = xb[1 : 1 + k_nodes]

    L_nodes = xb[1 + k_nodes : 1 + 2 * k_nodes]

    n_nodes_phys = (

        np.asarray(n_slice, dtype=np.float64).copy()

        if cfg.n_mono_band_nm is None

        else x_slice_n_to_physical_nodes(n_slice, sk, cfg.n_mono_band_nm)

    )

    out = dict(base_result)

    out.update(

        {

            "sigma_knots": sk,

            "sigma_knots_L": sk,

            "d_nm": d_nm,

            "n_lam": np.asarray(n_lam, dtype=np.float64).ravel(),

            "k_lam": np.asarray(k_lam, dtype=np.float64).ravel(),

            "n_nodes_physical": n_nodes_phys,

            "L_nodes": np.asarray(L_nodes, dtype=np.float64).ravel(),

            "rmse": float(rmse) if np.isfinite(rmse) else float(out.get("rmse", float("nan"))),

            "x": xb,

        }

    )

    return out


def _bootstrap_single_replicate(

    cfg_b: SplineOptConfig,

    base_result: dict,

    *,

    pconf: ProfileCorridorConfig,

    qref: int,

    lam: np.ndarray,

    log_run_1based: int | None = None,


) -> dict[str, Any]:

    """Run profiling + quick refit for one draw (cfg_b); structured return for aggregation."""

    lam = np.asarray(lam, dtype=np.float64).ravel()

    out: dict[str, Any] = {"status": "exception", "nvalid": 0}

    try:

        base_for: dict[str, Any] = base_result

        if int(qref) > 0:

            br = quick_pwlnk_refit_result_dict(cfg_b, base_result, maxfun=int(qref))

            if br is not None:

                base_for = br

        extra = compute_profiled_corridors_by_d(

            cfg_b, base_for, pconf=pconf, log_coaching=False

        )

    except Exception:

        if log_run_1based is not None:

            log.exception("%s [BOOT] failed run=%d", _LOG_PREFIX, int(log_run_1based))

        else:

            log.exception("%s [BOOT] internal run failed", _LOG_PREFIX)

        return out

    d_int = extra.get("profile_d_interval_nm", None)

    out["nvalid"] = int(np.asarray(extra.get("profile_d_values_nm", [])).size)

    if not (isinstance(d_int, (tuple, list)) and len(d_int) == 2):

        out["status"] = "bad_interval"

        return out

    try:

        dlo = float(d_int[0])

        dhi = float(d_int[1])

    except (TypeError, ValueError):

        out["status"] = "bad_interval"

        return out

    n_lo = np.asarray(extra.get("corridor_n_lo", []), dtype=np.float64).ravel()

    n_hi = np.asarray(extra.get("corridor_n_hi", []), dtype=np.float64).ravel()

    k_lo = np.asarray(extra.get("corridor_k_lo", []), dtype=np.float64).ravel()

    k_hi = np.asarray(extra.get("corridor_k_hi", []), dtype=np.float64).ravel()

    if not (n_lo.size == lam.size == n_hi.size == k_lo.size == k_hi.size):

        out["status"] = "shape"

        return out

    out["status"] = "ok"

    out["dlo"] = dlo

    out["dhi"] = dhi

    out["n_lo"] = n_lo

    out["n_hi"] = n_hi

    out["k_lo"] = k_lo

    out["k_hi"] = k_hi

    return out


def _bootstrap_pool_entry(payload: tuple[Any, ...]) -> tuple[int, dict[str, Any]]:

    """ProcessPoolExecutor entry point (picklable, module level)."""

    b, cfg_b, base_result, pconf, qref, lam = payload

    r = _bootstrap_single_replicate(

        cfg_b,

        base_result,

        pconf=pconf,

        qref=int(qref),

        lam=lam,

        log_run_1based=int(b) + 1,

    )

    return int(b), r


def compute_bootstrap_corridors_by_d(

    cfg: SplineOptConfig,

    base_result: dict,

    *,

    pconf: ProfileCorridorConfig,

    n_boot: int,

    percentile: float = 0.95,

    seed: int = 0,

    sigma_t: float | None = None,

    sigma_r: float | None = None,

    mode: str = "parametric",

    block_len: int = 1,

    quick_refit_maxfun: int | None = None,

    n_workers: int | None = None,


) -> dict[str, Any]:

    """V2.4: parametric bootstrap for bands.

    Builds B synthetic (T/R) datasets by adding Gaussian noise (sigma_T, sigma_R),

    reruns ``compute_profiled_corridors_by_d`` and aggregates:

      - n/k corridor percentiles (on lambda)

      - distribution of d_interval bounds.

    ``n_workers``: parallelism for replicates (``ProcessPoolExecutor``). Default: ``cfg.corridor_bootstrap_n_workers`` or 1.

    On failure (pickle, worker), falls back to sequential.

    """

    def _theoretical_TR_from_base() -> tuple[np.ndarray | None, np.ndarray | None]:

        from certus_physics import calculate_reflection_array, calculate_transmission_array

        from certus_index_utils import _ratio_theoretical_from_nk, _reflectance_ratio_theoretical_from_nk

        from certus_index_spline_core import _reflectance_absolute_backside_from_nk

        lam = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

        n_lam = np.asarray(base_result.get("n_lam", []), dtype=np.float64).ravel()

        k_lam = np.asarray(base_result.get("k_lam", []), dtype=np.float64).ravel()

        d_nm = float(base_result.get("d_nm", float("nan")))

        if lam.size == 0 or n_lam.size != lam.size or k_lam.size != lam.size or not np.isfinite(d_nm):

            return None, None

        n_sub = np.asarray(cfg.n_sub, dtype=np.float64).ravel()

        if n_sub.size != lam.size:

            return None, None

        t_th: np.ndarray | None = None

        r_th: np.ndarray | None = None

        if cfg.data_type in (DataType.TRANSMISSION, DataType.BOTH) and cfg.t_exp is not None and float(cfg.weight_t) > 0:

            if cfg.t_is_ratio:

                t_th = _ratio_theoretical_from_nk(lam, n_lam, k_lam, n_sub, float(d_nm))

            else:

                t_th = calculate_transmission_array(lam, n_lam, k_lam, float(d_nm), n_sub)

        if cfg.data_type in (DataType.REFLECTION, DataType.BOTH) and cfg.r_exp is not None and float(cfg.weight_r) > 0:

            if cfg.t_is_ratio:

                r_th = _reflectance_ratio_theoretical_from_nk(lam, n_lam, k_lam, n_sub, float(d_nm))

            else:

                r_th = _reflectance_absolute_backside_from_nk(lam, n_lam, k_lam, float(d_nm), n_sub)

        return t_th, r_th

    def _resample_residuals(e: np.ndarray, L: int, rng: np.random.Generator) -> np.ndarray:

        ee = np.asarray(e, dtype=np.float64).ravel()

        n = int(ee.size)

        if n == 0:

            return ee.copy()

        L = int(max(1, min(L, n)))

        if L == 1:

            idx = rng.integers(0, n, size=n, endpoint=False)

            return ee[idx]

        # Moving block bootstrap (wrap-around).

        n_blocks = int(np.ceil(n / L))

        starts = rng.integers(0, n, size=n_blocks, endpoint=False)

        out = np.empty(n_blocks * L, dtype=np.float64)

        pos = 0

        for s in starts:

            j = (s + np.arange(L)) % n

            out[pos : pos + L] = ee[j]

            pos += L

        return out[:n]

    B = int(max(0, n_boot))

    if B <= 0:

        return {}

    p = float(np.clip(percentile, 0.5, 0.999999))

    q_lo = 0.5 * (1.0 - p)

    q_hi = 1.0 - q_lo

    rng = np.random.default_rng(int(seed))

    mode = str(mode or "parametric").strip().lower()

    if mode not in ("parametric", "residual"):

        mode = "parametric"

    blk = int(max(1, block_len))

    # Default sigma: consistent with LR mode (auto := RMSE_opt) when not provided.

    rmse_opt = float(base_result.get("rmse", float("nan")))

    sigma_auto = float(rmse_opt) if np.isfinite(rmse_opt) and rmse_opt > 0 else 1.0

    sig_t = float(sigma_t) if (sigma_t is not None and float(sigma_t) > 0) else sigma_auto

    sig_r = float(sigma_r) if (sigma_r is not None and float(sigma_r) > 0) else sigma_auto

    lam = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    t0 = np.asarray(cfg.t_exp, dtype=np.float64).ravel() if cfg.t_exp is not None else None

    r0 = np.asarray(cfg.r_exp, dtype=np.float64).ravel() if cfg.r_exp is not None else None

    if lam.size < 3:

        return {}

    if t0 is not None and t0.size != lam.size:

        return {}

    if r0 is not None and r0.size != lam.size:

        return {}

    from spline_objective import _spline_objective_lam_mask

    qref = int(quick_refit_maxfun) if quick_refit_maxfun is not None else 0

    hetero_param = bool(getattr(cfg, "corridor_profile_d_sigma_hetero", False)) and mode == "parametric"

    floor_abs = max(1e-8, 0.01 * float(sigma_auto))

    h_scale = float(max(getattr(cfg, "corridor_profile_d_sigma_hetero_scale", 1.0) or 0.0, 0.0))

    st_hetero, sr_hetero = _hetero_sigma_masked_from_base(

        cfg, base_result, scale=h_scale, floor_abs=floor_abs

    )

    sigma_T_full: np.ndarray | None = None

    sigma_R_full: np.ndarray | None = None

    if hetero_param:

        mobj = _spline_objective_lam_mask(cfg)

        if st_hetero is not None and t0 is not None:

            sigma_T_full = np.full(lam.size, float(sig_t), dtype=np.float64)

            sigma_T_full[mobj] = np.asarray(st_hetero, dtype=np.float64).ravel()

        if sr_hetero is not None and r0 is not None:

            sigma_R_full = np.full(lam.size, float(sig_r), dtype=np.float64)

            sigma_R_full[mobj] = np.asarray(sr_hetero, dtype=np.float64).ravel()

    d_lo_list: list[float] = []

    d_hi_list: list[float] = []

    n_lo_list: list[np.ndarray] = []

    n_hi_list: list[np.ndarray] = []

    k_lo_list: list[np.ndarray] = []

    k_hi_list: list[np.ndarray] = []

    n_ok = 0

    run_b: list[int] = []

    run_ok: list[int] = []

    run_dlo: list[float] = []

    run_dhi: list[float] = []

    run_nvalid: list[int] = []

    log.info(

        "%s [BOOT] start | mode=%s blk=%d | B=%d | p=%.3f (q=[%.4f,%.4f]) | sigma_T=%.6g sigma_R=%.6g | prof_mode=%s | "

        "hetero_sigma(lambda)=%s | quick_refit_maxfun=%d",

        _LOG_PREFIX,

        str(mode),

        int(blk),

        int(B),

        float(p),

        float(q_lo),

        float(q_hi),

        float(sig_t),

        float(sig_r),

        str(getattr(pconf, "mode", "alpha")),

        bool(hetero_param),

        int(qref),

    )

    # Residual mode: needs T_th / R_th and residuals.

    t_th0, r_th0 = (None, None)

    eT0, eR0 = (None, None)

    if mode == "residual":

        t_th0, r_th0 = _theoretical_TR_from_base()

        if t0 is not None and t_th0 is not None:

            m = np.isfinite(t0) & np.isfinite(t_th0)

            eT0 = (t0[m] - t_th0[m]).astype(np.float64, copy=False)

        if r0 is not None and r_th0 is not None:

            m = np.isfinite(r0) & np.isfinite(r_th0)

            eR0 = (r0[m] - r_th0[m]).astype(np.float64, copy=False)

        if (t0 is not None and (t_th0 is None or eT0 is None or eT0.size < 3)) and (r0 is None):

            return {}

        if (r0 is not None and (r_th0 is None or eR0 is None or eR0.size < 3)) and (t0 is None):

            return {}

    nw = int(n_workers) if n_workers is not None else int(getattr(cfg, "corridor_bootstrap_n_workers", 1) or 1)

    nw = max(1, nw)

    boot_workers_effective = 1

    replicates: list[tuple[int, SplineOptConfig]] = []

    for b in range(B):

        t_b = None

        r_b = None

        if mode == "parametric":

            mobj = _spline_objective_lam_mask(cfg)

            if t0 is not None:

                t_b = t0.copy()

                if hetero_param:

                    idx_t = mobj & np.isfinite(t_b)

                    if np.any(idx_t):

                        stv = (

                            sigma_T_full[idx_t]

                            if sigma_T_full is not None

                            else np.full(int(np.count_nonzero(idx_t)), float(sig_t), dtype=np.float64)

                        )

                        t_b[idx_t] = t_b[idx_t] + rng.normal(0.0, stv, size=int(np.count_nonzero(idx_t)))

                else:

                    m = np.isfinite(t_b)

                    if np.any(m):

                        t_b[m] = t_b[m] + rng.normal(0.0, float(sig_t), size=int(np.count_nonzero(m)))

            if r0 is not None:

                r_b = r0.copy()

                if hetero_param:

                    idx_r = mobj & np.isfinite(r_b)

                    if np.any(idx_r):

                        srv = (

                            sigma_R_full[idx_r]

                            if sigma_R_full is not None

                            else np.full(int(np.count_nonzero(idx_r)), float(sig_r), dtype=np.float64)

                        )

                        r_b[idx_r] = r_b[idx_r] + rng.normal(0.0, srv, size=int(np.count_nonzero(idx_r)))

                else:

                    m = np.isfinite(r_b)

                    if np.any(m):

                        r_b[m] = r_b[m] + rng.normal(0.0, float(sig_r), size=int(np.count_nonzero(m)))

        else:

            if t0 is not None and t_th0 is not None and eT0 is not None:

                t_b = t0.copy()

                m = np.isfinite(t_b) & np.isfinite(t_th0)

                e_star = _resample_residuals(eT0, blk, rng)

                t_b[m] = t_th0[m] + e_star[: int(np.count_nonzero(m))]

            if r0 is not None and r_th0 is not None and eR0 is not None:

                r_b = r0.copy()

                m = np.isfinite(r_b) & np.isfinite(r_th0)

                e_star = _resample_residuals(eR0, blk, rng)

                r_b[m] = r_th0[m] + e_star[: int(np.count_nonzero(m))]

        cfg_b = replace(cfg, t_exp=t_b, r_exp=r_b)

        replicates.append((int(b), cfg_b))

    batch_out: dict[int, dict[str, Any]] = {}

    use_parallel = nw > 1 and B > 1

    if use_parallel:

        try:

            from concurrent.futures import ProcessPoolExecutor, as_completed

            n_proc = int(min(nw, B))

            with ProcessPoolExecutor(max_workers=n_proc) as ex:

                futs = {

                    ex.submit(

                        _bootstrap_pool_entry,

                        (b, cfg_b, base_result, pconf, qref, lam),

                    ): int(b)

                    for b, cfg_b in replicates

                }

                for fut in as_completed(futs):

                    bi, r = fut.result()

                    batch_out[int(bi)] = r

            boot_workers_effective = n_proc

        except Exception:

            log.exception("%s [BOOT] parallel failed - sequential fallback.", _LOG_PREFIX)

            use_parallel = False

            batch_out.clear()

    if not use_parallel or len(batch_out) != B:

        if use_parallel and len(batch_out) != B:

            log.warning("%s [BOOT] incomplete parallel results - sequential fallback.", _LOG_PREFIX)

        batch_out = {}

        for b, cfg_b in replicates:

            batch_out[int(b)] = _bootstrap_single_replicate(

                cfg_b,

                base_result,

                pconf=pconf,

                qref=qref,

                lam=lam,

                log_run_1based=int(b) + 1,

            )

        boot_workers_effective = 1

    for b in range(B):

        r = batch_out[int(b)]

        st = str(r.get("status", "exception"))

        run_b.append(int(b))

        if st == "ok":

            d_lo_list.append(float(r["dlo"]))

            d_hi_list.append(float(r["dhi"]))

            n_lo_list.append(np.asarray(r["n_lo"], dtype=np.float64).ravel())

            n_hi_list.append(np.asarray(r["n_hi"], dtype=np.float64).ravel())

            k_lo_list.append(np.asarray(r["k_lo"], dtype=np.float64).ravel())

            k_hi_list.append(np.asarray(r["k_hi"], dtype=np.float64).ravel())

            n_ok += 1

            run_ok.append(1)

            run_dlo.append(float(r["dlo"]))

            run_dhi.append(float(r["dhi"]))

            run_nvalid.append(int(r.get("nvalid", 0)))

        else:

            run_ok.append(0)

            run_dlo.append(float("nan"))

            run_dhi.append(float("nan"))

            run_nvalid.append(int(r.get("nvalid", 0)))

        if (b + 1) % max(1, B // 10) == 0:

            log.info("%s [BOOT] progress %d/%d | ok=%d", _LOG_PREFIX, int(b + 1), int(B), int(n_ok))

    if n_ok < 3:

        log.warning("%s [BOOT] stop | too few valid runs (%d/%d)", _LOG_PREFIX, int(n_ok), int(B))

        _log_coaching_bootstrap_outcome(

            B=int(B),

            n_ok=int(n_ok),

            p=float(p),

            mode=str(mode),

            qref=int(qref),

            d_lo_q=float("nan"),

            d_hi_q=float("nan"),

        )

        return {}

    d_lo_a = np.asarray(d_lo_list, dtype=np.float64)

    d_hi_a = np.asarray(d_hi_list, dtype=np.float64)

    n_lo_a = np.stack(n_lo_list, axis=0)

    n_hi_a = np.stack(n_hi_list, axis=0)

    k_lo_a = np.stack(k_lo_list, axis=0)

    k_hi_a = np.stack(k_hi_list, axis=0)

    # Percentile bands: aggregate envelopes (lo/hi) from each run.

    # For k, compute quantiles in ln(k) space (better numerical stability), then map back to k.

    boot_n_lo = np.nanquantile(n_lo_a, q_lo, axis=0)

    boot_n_hi = np.nanquantile(n_hi_a, q_hi, axis=0)

    L_lo_a = np.log(np.maximum(k_lo_a, 1e-300))

    L_hi_a = np.log(np.maximum(k_hi_a, 1e-300))

    boot_L_lo = np.nanquantile(L_lo_a, q_lo, axis=0)

    boot_L_hi = np.nanquantile(L_hi_a, q_hi, axis=0)

    boot_k_lo = np.exp(boot_L_lo)

    boot_k_hi = np.exp(boot_L_hi)

    out = {

        "boot_enabled": True,

        "boot_n": int(B),

        "boot_n_ok": int(n_ok),

        "boot_seed": int(seed),

        "boot_percentile": float(p),

        "boot_q_lo": float(q_lo),

        "boot_q_hi": float(q_hi),

        "boot_sigma_t": float(sig_t),

        "boot_sigma_r": float(sig_r),

        "boot_mode": str(mode),

        "boot_block_len": int(blk),

        "boot_runs_b": np.asarray(run_b, dtype=np.int64),

        "boot_runs_ok": np.asarray(run_ok, dtype=np.int64),

        "boot_runs_d_lo_nm": np.asarray(run_dlo, dtype=np.float64),

        "boot_runs_d_hi_nm": np.asarray(run_dhi, dtype=np.float64),

        "boot_runs_n_valid": np.asarray(run_nvalid, dtype=np.int64),

        "boot_d_lo_samples_nm": d_lo_a,

        "boot_d_hi_samples_nm": d_hi_a,

        "boot_d_lo_q_nm": float(np.nanquantile(d_lo_a, q_lo)),

        "boot_d_hi_q_nm": float(np.nanquantile(d_hi_a, q_hi)),

        "boot_corridor_n_lo": np.asarray(boot_n_lo, dtype=np.float64),

        "boot_corridor_n_hi": np.asarray(boot_n_hi, dtype=np.float64),

        "boot_corridor_k_lo": np.asarray(boot_k_lo, dtype=np.float64),

        "boot_corridor_k_hi": np.asarray(boot_k_hi, dtype=np.float64),

        "boot_corridor_L_lo": np.asarray(boot_L_lo, dtype=np.float64),

        "boot_corridor_L_hi": np.asarray(boot_L_hi, dtype=np.float64),

        "boot_quick_refit_maxfun": int(qref),

        "boot_sigma_hetero_parametric": bool(hetero_param),

        "boot_n_workers_effective": int(boot_workers_effective),

    }

    log.info(

        "%s [BOOT] done | ok=%d/%d | d_lo q=%.4f | d_hi q=%.4f",

        _LOG_PREFIX,

        int(n_ok),

        int(B),

        float(out["boot_d_lo_q_nm"]),

        float(out["boot_d_hi_q_nm"]),

    )

    _log_coaching_bootstrap_outcome(

        B=int(B),

        n_ok=int(n_ok),

        p=float(p),

        mode=str(mode),

        qref=int(qref),

        d_lo_q=float(out["boot_d_lo_q_nm"]),

        d_hi_q=float(out["boot_d_hi_q_nm"]),

    )

    return out


def compute_profiled_corridors_by_d(

    cfg: SplineOptConfig,

    base_result: dict,

    *,

    pconf: ProfileCorridorConfig | None = None,

    log_coaching: bool = True,

    profile_polish_maxfun: int | None = None,


) -> dict[str, Any]:

    """Compute d interval and n/k corridors by profiling (refit nodes at fixed d).

    Returns a dict of fields to merge into the pipeline result (or empty dict if disabled / impossible).

    """

    pconf = pconf or ProfileCorridorConfig()

    if not bool(pconf.enabled):

        return {}

    # Strict limitation (pseudo-relaxation) to prevent robust 'n' destruction 
    # that compensates 'd' and causes massive corridor times + width.
    maxfun_prof = min(int(corridor_profile_refit_maxfun(cfg, profile_polish_maxfun)), 15)

    t0 = time.perf_counter()

    lam_full = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

    use_lr = str(getattr(pconf, "mode", "alpha")).strip().lower() == "lr"

    rmse_thr_sub = str(getattr(pconf, "rmse_threshold_mode", "alpha") or "alpha").strip().lower()

    use_abs_delta = (not use_lr) and rmse_thr_sub in ("abs_delta", "alpha_plus_delta")
    use_alpha_factor = rmse_thr_sub == "alpha_plus_delta"

    tol_abs = float(max(float(getattr(pconf, "rmse_abs_tolerance", 0.001) or 0.0), 0.0))

    scientific_nominal = (

        bool(getattr(pconf, "scientific_nominal_corridor", True))

        and use_abs_delta

        and (not use_lr)

    )

    nom_pack: dict[str, Any] | None = None

    if scientific_nominal:

        from spline_finalize import extract_nominal_best_polished_corridor_reference

        nom_pack = extract_nominal_best_polished_corridor_reference(base_result)

        if nom_pack is None:

            scientific_nominal = False

            log.info(

                "%s Corridor scientifique (best RMSE) indisponible - repli sur les courbes « base » du dict.",

                _LOG_PREFIX,

            )

        elif nom_pack is not None:

            sk_chk = np.asarray(nom_pack["sigma_knots"], dtype=np.float64).ravel()

            xsg_chk = nom_pack.get("x_seg_spline_sigma")

            xa_chk = (

                np.asarray(xsg_chk, dtype=np.float64).ravel()

                if xsg_chk is not None

                else np.zeros(0, dtype=np.float64)

            )

            if sk_chk.size < 2 or xa_chk.size != 1 + 2 * int(sk_chk.size):

                log.info(

                    "%s Corridor scientifique: ``x_seg_spline_sigma`` absent ou K incohérent - repli.",

                    _LOG_PREFIX,

                )

                scientific_nominal = False

                nom_pack = None

    base_eff: dict[str, Any] = dict(base_result)

    if scientific_nominal and nom_pack is not None:

        skn = np.asarray(nom_pack["sigma_knots"], dtype=np.float64).ravel()

        xsg = nom_pack.get("x_seg_spline_sigma")

        base_eff["n_lam"] = np.asarray(nom_pack["n_lam"], dtype=np.float64).copy()

        base_eff["k_lam"] = np.asarray(nom_pack["k_lam"], dtype=np.float64).copy()

        base_eff["d_nm"] = float(nom_pack["d_nm"])

        if skn.size >= 2:

            base_eff["sigma_knots"] = skn.copy()

        if xsg is not None:

            xa = np.asarray(xsg, dtype=np.float64).ravel()

            k_sig = int(skn.size)

            if xa.size == 1 + 2 * k_sig:

                base_eff["x_seg_spline_sigma"] = xa.copy()

                base_eff["x"] = xa.copy()

                n_slice_x = xa[1 : 1 + k_sig]

                L_slice_x = xa[1 + k_sig : 1 + 2 * k_sig]

                base_eff["L_nodes"] = np.asarray(L_slice_x, dtype=np.float64).copy()

                if cfg.n_mono_band_nm is None:

                    base_eff["n_nodes_physical"] = np.asarray(n_slice_x, dtype=np.float64).copy()

                else:

                    base_eff["n_nodes_physical"] = x_slice_n_to_physical_nodes(

                        n_slice_x, skn, cfg.n_mono_band_nm

                    )

    sk, n_nodes_phys0, L_nodes0, d0, _prof_geom = _extract_knots_and_nodes_from_result(base_eff)

    k = int(sk.size)

    sk_n_log = (

        np.asarray(base_eff.get("sigma_knots_n"), dtype=np.float64).ravel()

        if base_eff.get("sigma_knots_n") is not None

        else None

    )

    # Initial x_nodes: prefer canonical worker/polish vector x = [d, n_slice…, L…] (no n_phys↔ξ drift).

    x_nodes0 = _x_nodes0_from_mesh_x_if_consistent(base_eff, sk=sk, d0_nm=float(d0))

    corridor_seed_x_source = "mesh_x"

    if x_nodes0 is None:

        corridor_seed_x_source = "roundtrip_n_phys"

        # Build initial x_nodes consistent with pipeline parameterization:

        # n_slice = physical if no monotonicity; else encode n_phys as ξ.

        if cfg.n_mono_band_nm is None:

            n_slice0 = np.asarray(n_nodes_phys0, dtype=np.float64).copy()

        else:

            from spline_objective import physical_nodes_to_x_slice_n

            n_slice0 = physical_nodes_to_x_slice_n(n_nodes_phys0, sk, cfg.n_mono_band_nm)

        x_nodes0 = np.concatenate((n_slice0, np.asarray(L_nodes0, dtype=np.float64).copy()))

    bounds_nodes, x0_default = _bounds_for_nodes_only(cfg, k)

    x_nodes0_pre_clip = np.asarray(x_nodes0, dtype=np.float64).ravel().copy()

    x_nodes0 = clip_to_bounds(x_nodes0, bounds_nodes[:, 0], bounds_nodes[:, 1])

    if corridor_seed_x_source == "mesh_x" and not np.array_equal(x_nodes0, x_nodes0_pre_clip):

        log.warning(

            "%s Corridor seed: mesh *x* nodes were clipped to bounds (max |Delta|=%.3e) — "

            "spectral RMSE at seed may diverge from spectral_rmse_best_value.",

            _LOG_PREFIX,

            float(np.max(np.abs(x_nodes0 - x_nodes0_pre_clip))),

        )

    mse_seed0, rmse_seed0 = _spectral_rmse_at_packed_nodes(cfg, base_eff, sk, float(d0), x_nodes0)

    n_b = np.asarray(base_eff.get("n_lam"), dtype=np.float64).ravel()

    k_b = np.asarray(base_eff.get("k_lam"), dtype=np.float64).ravel()

    rmse_spectral_curves = float("nan")

    if n_b.size == lam_full.size and k_b.size == lam_full.size:

        _mse_bc, rmse_sc = spectral_mse_rmse_masked_from_nk(

            cfg, base_eff, lam_full, n_b, k_b, float(d0)

        )

        rmse_spectral_curves = float(rmse_sc) if np.isfinite(float(rmse_sc)) else float("nan")

    if scientific_nominal and nom_pack is not None:

        rmse_opt = float(nom_pack["rmse_best"])

        rmse_ref_tag = "spectral_rmse_best_value"

    else:

        rmse_opt, rmse_ref_tag = _pick_rmse_reference_for_profile(cfg, base_eff, sk, float(d0), x_nodes0)

        if use_abs_delta and np.isfinite(rmse_spectral_curves):

            rmse_opt = float(rmse_spectral_curves)

            rmse_ref_tag = "spectral_rmse_base_nk"

    # Calculate N_eff for intelligent statistical threshold
    _N_data = 0
    if cfg.t_exp is not None: _N_data += len(cfg.t_exp)
    if cfg.r_exp is not None: _N_data += len(cfg.r_exp)
    _N_params = 1 + 2 * sk.size if sk is not None else 25 # roughly
    _N_eff = max(5, _N_data - _N_params)
    
    alpha_intelligent = float(np.sqrt(1.0 + 3.84 / _N_eff))

    use_auto = (str(getattr(pconf, "rmse_threshold_mode", "")).strip().lower() == "auto" or str(getattr(pconf, "mode", "")).strip().lower() == "auto")
    
    # Force use_auto dynamically if they used the naive 1.05 and auto is technically better
    if abs(float(pconf.rmse_alpha) - 1.05) < 1e-3 and not use_abs_delta and not use_lr:
        use_auto = True

    # OVERRIDE: Force strict auto-statistical boundary to drastically tighten corridors 
    # instead of permissive manual GUI factors (e.g. 1.25) which widen 'n'.
    if not use_lr and not use_abs_delta:
        use_auto = True

    if use_auto and np.isfinite(rmse_opt):
        rmse_thresh = alpha_intelligent * float(rmse_opt)
        log.info("%s Corridor auto-threshold using N_eff=%d (N_obs=%d, N_params=%d) => intelligent_alpha=%.5f", _LOG_PREFIX, _N_eff, _N_data, _N_params, alpha_intelligent)
        
    elif use_lr:
        rmse_thresh = float(pconf.rmse_alpha) * float(rmse_opt) if np.isfinite(rmse_opt) else float("nan")

    elif use_abs_delta and np.isfinite(rmse_opt):
        _alpha_f = float(pconf.rmse_alpha) if use_alpha_factor else 1.0
        rmse_thresh = _alpha_f * float(rmse_opt) + tol_abs

    else:
        rmse_thresh = float(pconf.rmse_alpha) * float(rmse_opt) if np.isfinite(rmse_opt) else float("nan")

    if (

        scientific_nominal

        and nom_pack is not None

        and np.isfinite(float(rmse_opt))

        and np.isfinite(float(rmse_seed0))

    ):

        dv = abs(float(rmse_seed0) - float(rmse_opt))

        tol_rm = max(1e-12, abs(float(rmse_opt)) * 1e-9, float(np.sqrt(np.finfo(np.float64).eps)) * abs(float(rmse_opt)))

        log.info(

            "%s Corridor seed alignment | x_source=%s | RMSE(packed nodes @ d_opt)=%.12g | RMSE_ref(%s)=%.12g | |Delta|=%.3e (warn if >%.3e)",

            _LOG_PREFIX,

            str(corridor_seed_x_source),

            float(rmse_seed0),

            str(rmse_ref_tag),

            float(rmse_opt),

            float(dv),

            float(tol_rm),

        )

        if float(dv) > float(tol_rm):

            log.warning(

                "%s Corridor seed vs spectral_rmse_best_value: |Delta|=%.3e exceeds tol=%.3e — "

                "check x_seg_spline_sigma vs n_lam_seg_spline_sigma export, bounds clip, or mono ξ round-trip.",

                _LOG_PREFIX,

                float(dv),

                float(tol_rm),

            )

    rmse_thresh_active = float(rmse_thresh)

    auto_relaxed_alpha = False

    threshold_fallback_reason = ""

    threshold_basis_eff = str(getattr(pconf, "threshold_basis", "max") or "max").strip().lower()

    delta_chi2 = float(_chi2.ppf(float(np.clip(pconf.lr_conf_level, 1e-6, 0.999999)), 1)) if use_lr else float("nan")

    sigma_auto = float(rmse_opt) if np.isfinite(rmse_opt) and rmse_opt > 0 else 1.0

    sig_t = float(pconf.sigma_t) if (pconf.sigma_t is not None and float(pconf.sigma_t) > 0) else sigma_auto

    sig_r = float(pconf.sigma_r) if (pconf.sigma_r is not None and float(pconf.sigma_r) > 0) else sigma_auto

    if lam_full.size < 3 or not np.isfinite(rmse_opt) or not np.isfinite(rmse_thresh):

        if log_coaching:

            _log_coaching_corridor_failure(

                reason="rmse_meta_invalid",

                pconf=pconf,

                use_lr=use_lr,

                rmse_opt=rmse_opt,

                rmse_thresh=rmse_thresh,

                d0=float(d0),

            )

        return {"profile_d_status": "failed"}

    sigma_t_f_hetero: np.ndarray | None = None

    sigma_r_f_hetero: np.ndarray | None = None

    if use_lr and bool(getattr(pconf, "sigma_hetero_residual", False)):

        floor_abs = max(1e-8, 0.01 * float(rmse_opt)) if np.isfinite(rmse_opt) else 1e-6

        sigma_t_f_hetero, sigma_r_f_hetero = _hetero_sigma_masked_from_base(

            cfg,

            base_eff,

            scale=float(max(getattr(pconf, "sigma_hetero_scale", 1.0) or 0.0, 0.0)),

            floor_abs=floor_abs,

        )

        if sigma_t_f_hetero is not None or sigma_r_f_hetero is not None:

            log.info(

                "%s LR hetero sigma (max(floor, scale×|residual|)) | scale=%.4g floor_abs=%.4g",

                _LOG_PREFIX,

                float(pconf.sigma_hetero_scale),

                float(floor_abs),

            )

    _log_corridor_base_geometry(

        sk=np.asarray(sk, dtype=np.float64),

        n_phys=np.asarray(n_nodes_phys0, dtype=np.float64),

        L_nodes=np.asarray(L_nodes0, dtype=np.float64),

        d0=float(d0),

        sk_n_stored=sk_n_log,

        diag=_prof_geom,

        rmse_ref_pipeline=float(rmse_opt),

        rmse_seed_no_refit=float(rmse_seed0),

        mse_seed_no_refit=float(mse_seed0),

        use_abs_delta=bool(use_abs_delta),

    )

    if use_abs_delta:

        log.info(

            "%s Start | K_sigma=%d | d_opt=%.6f nm | RMSE_ref=%.8f (%s) | seuil absolu RMSE <= %.8f + Delta=%.6f -> %.8f | "

            "step=%.4g nm | span=%.4g nm | max_steps/side=%d | refine=%s tol=%.4g nm it=%d | nk_profile=%s | mono=%s | "

            "wT=%.4g wR=%.4g | rmse_fit_lambda_nm=%s",

            _LOG_PREFIX,

            int(k),

            float(d0),

            float(rmse_opt),

            str(rmse_ref_tag),

            float(rmse_opt),

            float(tol_abs),

            float(rmse_thresh),

            float(pconf.step_nm),

            float(pconf.max_span_nm),

            int(pconf.max_steps_each_side),

            bool(pconf.refine_boundary),

            float(pconf.refine_tol_nm),

            int(pconf.refine_max_iter),

            str(getattr(cfg, "nk_profile_interp", "smooth")),

            str(getattr(cfg, "n_mono_band_nm", None)),

            float(getattr(cfg, "weight_t", 0.0)),

            float(getattr(cfg, "weight_r", 0.0)),

            str(getattr(cfg, "rmse_fit_lambda_nm", None)),

        )

    else:

        log.info(

            "%s Start | K_sigma=%d | d_opt=%.6f nm | RMSE_ref=%.8f (%s) | alpha×RMSE threshold (alpha=%.3f -> %.8f) | "

            "step=%.4g nm | span=%.4g nm | max_steps/side=%d | refine=%s tol=%.4g nm it=%d | nk_profile=%s | mono=%s | "

            "wT=%.4g wR=%.4g | rmse_fit_lambda_nm=%s",

            _LOG_PREFIX,

            int(k),

            float(d0),

            float(rmse_opt),

            str(rmse_ref_tag),

            float(pconf.rmse_alpha),

            float(rmse_thresh),

            float(pconf.step_nm),

            float(pconf.max_span_nm),

            int(pconf.max_steps_each_side),

            bool(pconf.refine_boundary),

            float(pconf.refine_tol_nm),

            int(pconf.refine_max_iter),

            str(getattr(cfg, "nk_profile_interp", "smooth")),

            str(getattr(cfg, "n_mono_band_nm", None)),

            float(getattr(cfg, "weight_t", 0.0)),

            float(getattr(cfg, "weight_r", 0.0)),

            str(getattr(cfg, "rmse_fit_lambda_nm", None)),

        )

    log.info("%s L-BFGS-B budget per refit (profiling): maxfun=%d (main run polish=%d)", _LOG_PREFIX, maxfun_prof, int(cfg.polish_maxfun))

    if use_abs_delta:

        if scientific_nominal:

            log.info(

                "%s Reminder (corridor scientifique): RMSE_ref = **spectral_rmse_best_value** (meilleur modèle poli) ; "

                "seuil = RMSE_ref + Delta ; courbe nominale incluse **sans** élargissement d’enveloppe correctif.",

                _LOG_PREFIX,

            )

        else:

            log.info(

                "%s Reminder (seuil absolu): RMSE_ref = spectre masqué pour les courbes **n_lam/k_lam** de la base corridor. "

                "Pas de relèvement automatique du seuil ; les refits doivent rester <= RMSE_ref + Delta.",

                _LOG_PREFIX,

            )

    else:

        log.info(

            "%s Reminder: RMSE_ref (above) = spectrum for the **solver** solution (fixed nodes). "

            "Each « best-of » / refit RMSE = **n,L** re-optimization at fixed d (budget/jitter) -> can be **> RMSE_ref**; "

            "the alpha×RMSE threshold may then track the **center refit** (center_refit fallback or automatic lift).",

            _LOG_PREFIX,

        )

    if use_lr:

        log.info(

            "%s Mode LR | conf=%.4f -> Deltaχ²=%.6f | sigma_T=%.6g sigma_R=%.6g %s",

            _LOG_PREFIX,

            float(pconf.lr_conf_level),

            float(delta_chi2),

            float(sig_t),

            float(sig_r),

            "(+sigma_i residual)" if (sigma_t_f_hetero is not None or sigma_r_f_hetero is not None) else "(constants)",

        )

    # Store valid solutions.

    d_vals: list[float] = []

    n_curves: list[np.ndarray] = []

    k_curves: list[np.ndarray] = []

    rmse_vals: list[float] = []

    chi2_vals: list[float] = []

    fit_nfev_values: list[float] = []

    fit_nit_values: list[float] = []

    fit_try_values: list[float] = []

    fit_fail_values: list[float] = []

    boundary_refine_calls = 0

    pos_valid = 0

    neg_valid = 0

    # n(lambda),k(lambda) from center d_opt refit (same family as envelope) - for UI when the dict

    # shows another snapshot (e.g. best live SOL3 vs profiling PWL refits).

    corridor_ref_n_lam: np.ndarray | None = None

    corridor_ref_k_lam: np.ndarray | None = None

    # abs_delta: include the nominal n(lambda),k(lambda) as first family member (before center refit).

    if use_abs_delta and n_b.size == lam_full.size and k_b.size == lam_full.size:

        if scientific_nominal and nom_pack is not None:

            first_rmse = float(rmse_opt)

        elif np.isfinite(rmse_spectral_curves):

            first_rmse = float(rmse_spectral_curves)

        else:

            first_rmse = None

        if first_rmse is not None:

            d_vals.append(float(d0))

            n_curves.append(n_b.copy())

            k_curves.append(k_b.copy())

            rmse_vals.append(first_rmse)

            chi2_vals.append(float("nan"))

            fit_nfev_values.append(float("nan"))

            fit_nit_values.append(float("nan"))

            fit_try_values.append(float("nan"))

            fit_fail_values.append(float("nan"))

            if scientific_nominal and nom_pack is not None:

                corridor_ref_n_lam = np.asarray(nom_pack["n_lam"], dtype=np.float64).ravel().copy()

                corridor_ref_k_lam = np.asarray(nom_pack["k_lam"], dtype=np.float64).ravel().copy()

            else:

                corridor_ref_n_lam = n_b.copy()

                corridor_ref_k_lam = k_b.copy()

    # Center: best-of-N. In LR mode this sets chi2_min (reference) when successful.

    fit0, _ok0, metric0 = _best_fit_at_d(

        cfg,

        sk=sk,

        d_nm=float(d0),

        x_seed_primary=x_nodes0,

        x_seed_secondary=None,

        x_seed_default=x0_default,

        bounds_nodes=bounds_nodes,

        maxfun=maxfun_prof,

        use_lr=use_lr,

        sig_t=sig_t,

        sig_r=sig_r,

        chi2_min_ref=None,

        delta_chi2=float(delta_chi2) if np.isfinite(delta_chi2) else 0.0,

        pconf=pconf,

        stage_label="Center",

        sigma_t_f=sigma_t_f_hetero,

        sigma_r_f=sigma_r_f_hetero,

    )

    chi2_min = float(metric0) if (use_lr and np.isfinite(metric0)) else float("nan")

    rm_c = (

        float(fit0["rmse"])

        if fit0 is not None and np.isfinite(float(fit0.get("rmse", float("nan"))))

        else float("nan")

    )

    if (

        (not use_lr)

        and (not use_abs_delta)

        and bool(getattr(pconf, "auto_relax_threshold_to_include_center", True))

        and np.isfinite(rm_c)

        and np.isfinite(rmse_thresh_active)

    ):

        basis = threshold_basis_eff

        rmse_nom = float(rmse_opt)

        rmse_ctr = float(rm_c)

        rmse_basis = rmse_nom

        if basis == "center_refit":

            rmse_basis = rmse_ctr

        elif basis == "max":

            rmse_basis = max(rmse_nom, rmse_ctr)

        else:

            basis = "nominal"

            rmse_basis = rmse_nom

        rmse_thresh_active = float(pconf.rmse_alpha) * float(rmse_basis)

        ratio_guard = float(max(getattr(pconf, "threshold_ratio_guard", 1.25) or 1.25, 1.0))

        ratio_ctr = float(rmse_ctr / max(rmse_nom, 1e-30)) if np.isfinite(rmse_nom) and rmse_nom > 0 else float("inf")

        if ratio_ctr > ratio_guard and basis != "center_refit":

            rmse_thresh_active = float(pconf.rmse_alpha) * float(rmse_ctr)

            threshold_fallback_reason = f"center_refit_ratio_guard({ratio_ctr:.3f}>{ratio_guard:.3f})"

            basis = "center_refit"

            log.info(

                "%s RMSE threshold fallback: RMSE_refit_center/RMSE_ref=%.3f > guard=%.3f -> basis forced to **center_refit**: "

                "alpha×RMSE is now based on the **center refit** (not RMSE_ref alone), so d_opt is not rejected when only the "

                "nodes-only subproblem is worse than the full solver run.",

                _LOG_PREFIX,

                ratio_ctr,

                ratio_guard,

            )

        threshold_basis_eff = basis

        eps_ar = float(max(getattr(pconf, "auto_relax_epsilon", 0.002) or 0.0, 1e-12))

        relax_fac_cap = float(max(getattr(pconf, "auto_relax_max_factor", 1.5) or 1.5, 1.0))

        need = float(rm_c) * (1.0 + eps_ar)

        max_allowed = float(rmse_thresh) * relax_fac_cap if np.isfinite(rmse_thresh) else need

        if need > rmse_thresh_active:

            rmse_thresh_active = min(need, max_allowed)

            auto_relaxed_alpha = True

            log.info(

                "%s RMSE threshold auto-lifted: nominal alpha×RMSE_ref=%.8f -> effective=%.8f "

                "(center refit RMSE=%.8f; n,L refit at fixed d ≠ solver \"segments\" RMSE; threshold adjusted to include center).",

                _LOG_PREFIX,

                float(rmse_thresh),

                float(rmse_thresh_active),

                float(rm_c),

            )

    fit0_ok = False

    if fit0 is not None and np.isfinite(float(fit0.get("rmse", float("nan")))):

        fit0_ok = np.isfinite(chi2_min) if use_lr else (float(fit0["rmse"]) <= rmse_thresh_active)

    center_seed_kept = False
    center_seed_gate_eval_count = 0
    center_seed_gate_kept_count = 0
    center_seed_gate_delta_refit_minus_seed = float("nan")
    if fit0 is not None:
        _rm_seed0 = fit0.get("rmse_seed_before_refit")
        _rm_refit0 = fit0.get("rmse_refit_attempted")
        if (
            _rm_seed0 is not None
            and _rm_refit0 is not None
            and np.isfinite(float(_rm_seed0))
            and np.isfinite(float(_rm_refit0))
        ):
            center_seed_gate_eval_count = 1
            center_seed_gate_kept_count = int(bool(fit0.get("seed_kept_over_refit", False)))
            center_seed_gate_delta_refit_minus_seed = float(_rm_refit0) - float(_rm_seed0)
    if (not fit0_ok) and (not use_lr) and np.isfinite(float(rmse_seed0)) and np.isfinite(float(rmse_thresh_active)):
        if float(rmse_seed0) <= float(rmse_thresh_active):
            fit0_ok = True
            center_seed_kept = True

    if fit0 is not None and fit0_ok:

        if center_seed_kept:

            d_vals.append(float(d0))

            n_curves.append(np.asarray(base_eff.get("n_lam"), dtype=np.float64).ravel().copy())

            k_curves.append(np.asarray(base_eff.get("k_lam"), dtype=np.float64).ravel().copy())

            rmse_vals.append(float(rmse_seed0))

            chi2_vals.append(float("nan"))

            fit_nfev_values.append(float("nan"))

            fit_nit_values.append(float("nan"))

            fit_try_values.append(0.0)

            fit_fail_values.append(
                float(fit0.get("n_failed_fit", float("nan"))) if fit0 is not None else float("nan")
            )

            if scientific_nominal and nom_pack is not None:

                corridor_ref_n_lam = np.asarray(nom_pack["n_lam"], dtype=np.float64).ravel().copy()

                corridor_ref_k_lam = np.asarray(nom_pack["k_lam"], dtype=np.float64).ravel().copy()

            else:

                corridor_ref_n_lam = np.asarray(base_eff.get("n_lam"), dtype=np.float64).ravel().copy()

                corridor_ref_k_lam = np.asarray(base_eff.get("k_lam"), dtype=np.float64).ravel().copy()

            x_nodes_center = x_nodes0.copy()

            log.info(

                "%s Center: keeping nominal seed at d=d_opt | spectral RMSE **without refit**=%.8f <= active threshold %.8f "

                "(center refit RMSE=%s). Refitted center kept only as diagnostic; corridor remains anchored on the nominal model.",

                _LOG_PREFIX,

                float(rmse_seed0),

                float(rmse_thresh_active),

                (
                    f"{float(fit0['rmse']):.8f}"
                    if fit0 is not None and np.isfinite(float(fit0.get('rmse', float('nan'))))
                    else "n/a"
                ),

            )

        else:

            d_vals.append(float(fit0["d_nm"]))

            n_curves.append(np.asarray(fit0["n_lam"], dtype=np.float64))

            k_curves.append(np.asarray(fit0["k_lam"], dtype=np.float64))

            rmse_vals.append(float(fit0["rmse"]))

            chi2_vals.append(float(chi2_min) if use_lr else float("nan"))

            fit_nfev_values.append(float(fit0.get("nfev", float("nan"))))

            fit_nit_values.append(float(fit0.get("nit", float("nan"))))

            fit_try_values.append(float(fit0.get("n_try", float("nan"))))

            fit_fail_values.append(float(fit0.get("n_failed_fit", float("nan"))))

            if not scientific_nominal:

                corridor_ref_n_lam = np.asarray(fit0["n_lam"], dtype=np.float64).ravel().copy()

                corridor_ref_k_lam = np.asarray(fit0["k_lam"], dtype=np.float64).ravel().copy()

            x_nodes_center = np.asarray(fit0["x_nodes_best"], dtype=np.float64).ravel().copy()

            mo0 = fit0.get("mse_objective")

            mo0s = f"{float(mo0):.6e}" if mo0 is not None and np.isfinite(float(mo0)) else "n/a"

            if use_abs_delta:

                log.info(

                    "%s Center: OK | RMSE spectral **après refit** (d=d_opt)=%.8f | RMSE_ref seuil=%.8f (%s) | "

                    "refit_objective_MSE=%s | chi2_min=%s | Delta(refit - graine sans refit)=%+.6e | "

                    "Delta(refit - RMSE_ref nominale)=%+.6e",

                    _LOG_PREFIX,

                    float(fit0["rmse"]),

                    float(rmse_opt),

                    str(rmse_ref_tag),

                    mo0s,

                    f"{chi2_min:.8f}" if use_lr and np.isfinite(chi2_min) else "n/a",

                    float(fit0["rmse"]) - float(rmse_seed0),

                    float(fit0["rmse"]) - float(rmse_opt),

                )

            else:

                log.info(

                    "%s Center: OK | spectral RMSE **after refit** (d=d_opt)=%.8f | pipeline RMSE_ref=%.8f (%s) | "

                    "refit_objective_MSE=%s | chi2_min=%s | Delta(refit RMSE - no-refit seed)=%+.6e | "

                    "Delta(refit RMSE - RMSE_ref)=%+.6e",

                    _LOG_PREFIX,

                    float(fit0["rmse"]),

                    float(rmse_opt),

                    str(rmse_ref_tag),

                    mo0s,

                    f"{chi2_min:.8f}" if use_lr and np.isfinite(chi2_min) else "n/a",

                    float(fit0["rmse"]) - float(rmse_seed0),

                    float(fit0["rmse"]) - float(rmse_opt),

                )

    else:

        if not bool(pconf.include_center_even_if_nan):

            if log_coaching:

                _log_coaching_corridor_failure(

                    reason="centre_fail",

                    pconf=pconf,

                    use_lr=use_lr,

                    rmse_opt=rmse_opt,

                    rmse_thresh=rmse_thresh_active,

                    d0=float(d0),

                )

            return {"profile_d_status": "failed"}

        x_nodes_center = x_nodes0.copy()

        log.warning("%s Center: failed | continue=%s", _LOG_PREFIX, bool(pconf.include_center_even_if_nan))

    def _run_walks_sequential() -> tuple[dict[str, Any], dict[str, Any]]:

        return (

            _corridor_profile_walk_side(

                1.0,

                pconf,

                cfg,

                sk,

                d0,

                x_nodes_center,

                x0_default,

                bounds_nodes,

                maxfun_prof,

                use_lr,

                chi2_min,

                delta_chi2,

                rmse_thresh_active,

                rmse_opt,

                sig_t,

                sig_r,

                sigma_t_f_hetero,

                sigma_r_f_hetero,

            ),

            _corridor_profile_walk_side(

                -1.0,

                pconf,

                cfg,

                sk,

                d0,

                x_nodes_center,

                x0_default,

                bounds_nodes,

                maxfun_prof,

                use_lr,

                chi2_min,

                delta_chi2,

                rmse_thresh_active,

                rmse_opt,

                sig_t,

                sig_r,

                sigma_t_f_hetero,

                sigma_r_f_hetero,

            ),

        )

    parallel_walks = bool(getattr(cfg, "corridor_profile_d_parallel_walks", True))

    if parallel_walks:

        try:

            p_plus = replace(pconf, rng_seed=int(pconf.rng_seed) + 1_000_003)

            p_minus = replace(pconf, rng_seed=int(pconf.rng_seed) + 1_000_019)

            with ThreadPoolExecutor(max_workers=2) as _pool:

                _f_plus = _pool.submit(

                    _corridor_profile_walk_side,

                    1.0,

                    p_plus,

                    cfg,

                    sk,

                    d0,

                    x_nodes_center,

                    x0_default,

                    bounds_nodes,

                    maxfun_prof,

                    use_lr,

                    chi2_min,

                    delta_chi2,

                    rmse_thresh_active,

                    rmse_opt,

                    sig_t,

                    sig_r,

                    sigma_t_f_hetero,

                    sigma_r_f_hetero,

                )

                _f_minus = _pool.submit(

                    _corridor_profile_walk_side,

                    -1.0,

                    p_minus,

                    cfg,

                    sk,

                    d0,

                    x_nodes_center,

                    x0_default,

                    bounds_nodes,

                    maxfun_prof,

                    use_lr,

                    chi2_min,

                    delta_chi2,

                    rmse_thresh_active,

                    rmse_opt,

                    sig_t,

                    sig_r,

                    sigma_t_f_hetero,

                    sigma_r_f_hetero,

                )

                r_plus = _f_plus.result()

                r_minus = _f_minus.result()

        except Exception:

            log.exception("%s parallel +/-d walks failed - sequential fallback.", _LOG_PREFIX)

            r_plus, r_minus = _run_walks_sequential()

    else:

        r_plus, r_minus = _run_walks_sequential()

    d_vals.extend(r_plus["d_vals"])

    d_vals.extend(r_minus["d_vals"])

    n_curves.extend(r_plus["n_curves"])

    n_curves.extend(r_minus["n_curves"])

    k_curves.extend(r_plus["k_curves"])

    k_curves.extend(r_minus["k_curves"])

    rmse_vals.extend(r_plus["rmse_vals"])

    rmse_vals.extend(r_minus["rmse_vals"])

    chi2_vals.extend(r_plus["chi2_vals"])

    chi2_vals.extend(r_minus["chi2_vals"])

    fit_nfev_values.extend(r_plus["fit_nfev_values"])

    fit_nfev_values.extend(r_minus["fit_nfev_values"])

    fit_nit_values.extend(r_plus["fit_nit_values"])

    fit_nit_values.extend(r_minus["fit_nit_values"])

    fit_try_values.extend(r_plus["fit_try_values"])

    fit_try_values.extend(r_minus["fit_try_values"])

    fit_fail_values.extend(r_plus["fit_fail_values"])

    fit_fail_values.extend(r_minus["fit_fail_values"])

    pos_valid = int(r_plus["n_valid_side"])

    neg_valid = int(r_minus["n_valid_side"])

    boundary_refine_calls = int(r_plus["n_refines"]) + int(r_minus["n_refines"])
    seed_gate_eval_count = (
        int(r_plus.get("seed_gate_eval_count", 0))
        + int(r_minus.get("seed_gate_eval_count", 0))
        + int(center_seed_gate_eval_count)
    )
    seed_gate_kept_count = (
        int(r_plus.get("seed_gate_kept_count", 0))
        + int(r_minus.get("seed_gate_kept_count", 0))
        + int(center_seed_gate_kept_count)
    )
    seed_gate_deltas = np.asarray(
        list(np.asarray(r_plus.get("seed_gate_delta_refit_minus_seed", []), dtype=np.float64).ravel())
        + list(np.asarray(r_minus.get("seed_gate_delta_refit_minus_seed", []), dtype=np.float64).ravel())
        + (
            [float(center_seed_gate_delta_refit_minus_seed)]
            if np.isfinite(float(center_seed_gate_delta_refit_minus_seed))
            else []
        ),
        dtype=np.float64,
    )

    min_req = int(max(1, pconf.min_valid_points))

    if scientific_nominal:

        min_req = 1

    min_side = int(max(0, getattr(pconf, "min_valid_each_side", 1) or 1))

    if len(d_vals) < min_req:

        if len(d_vals) >= 1 and auto_relaxed_alpha:

            log.warning(

                "%s Continuing with %d valid point(s) (< min_valid_points=%d) after automatic RMSE threshold lift "

                "(envelope may be narrow or degenerate).",

                _LOG_PREFIX,

                int(len(d_vals)),

                min_req,

            )

        else:

            log.warning(

                "%s Abort: too few valid solutions (%d < %d).",

                _LOG_PREFIX,

                int(len(d_vals)),

                min_req,

            )

            if log_coaching:

                _log_coaching_corridor_failure(

                    reason="too_few_valid",

                    pconf=pconf,

                    use_lr=use_lr,

                    rmse_opt=rmse_opt,

                    rmse_thresh=rmse_thresh_active,

                    d0=float(d0),

                )

            return {"profile_d_status": "failed"}

    if min_side > 0 and (pos_valid < min_side or neg_valid < min_side):

        log.warning(

            "%s Degenerate corridor: valid_side(+d=%d, -d=%d) < min_valid_each_side=%d.",

            _LOG_PREFIX,

            int(pos_valid),

            int(neg_valid),

            int(min_side),

        )

    # Envelopes (corridors): min/max over all valid curves (linear n and linear k).

    # UI log₁₀(k): center refit lies in [k_lo, k_hi] in k but need not bisect [log10(k_lo), log10(k_hi)].

    n_stack = np.vstack([c.reshape(1, -1) for c in n_curves])

    k_stack = np.vstack([c.reshape(1, -1) for c in k_curves])

    n_lo = np.nanmin(n_stack, axis=0)

    n_hi = np.nanmax(n_stack, axis=0)

    k_lo = np.nanmin(k_stack, axis=0)

    k_hi = np.nanmax(k_stack, axis=0)

    if scientific_nominal and corridor_ref_n_lam is not None and corridor_ref_k_lam is not None:

        # Crucial Physical Coherence: The base nominal Polish *is* the anchor of the corridor. 

        # If the local searches shifted, this anchor MUST remain within its own bounded envelope.

        n_lo, n_hi, k_lo, k_hi = _expand_corridor_envelope_with_reported_nk(

            n_lo, n_hi, k_lo, k_hi, corridor_ref_n_lam, corridor_ref_k_lam

        )

        log.debug(

            "%s Scientific Nominal: Base reference natively injected into min/max bounds to ensure 100%% consistency.",

            _LOG_PREFIX,

        )

    elif not scientific_nominal:

        n_nom_r = np.asarray(base_result.get("n_lam"), dtype=np.float64).ravel()

        k_nom_r = np.asarray(base_result.get("k_lam"), dtype=np.float64).ravel()

        if int(n_nom_r.size) >= int(n_lo.size) and int(k_nom_r.size) >= int(k_lo.size):

            n_lo, n_hi, k_lo, k_hi = _expand_corridor_envelope_with_reported_nk(

                n_lo, n_hi, k_lo, k_hi, n_nom_r, k_nom_r

            )

            log.debug(

                "%s Envelope widened to include reported n_lam/k_lam (bold UI curve inside lo/hi).",

                _LOG_PREFIX,

            )

    # Legacy: align corridor_reference_* on stack[0]. Scientifique: référence = nominale best (déjà fixée).

    if (

        (not scientific_nominal)

        and corridor_ref_n_lam is not None

        and corridor_ref_k_lam is not None

        and int(n_stack.shape[0]) > 0

    ):

        corridor_ref_n_lam = np.asarray(n_stack[0], dtype=np.float64).ravel().copy()

        corridor_ref_k_lam = np.asarray(k_stack[0], dtype=np.float64).ravel().copy()

    if corridor_ref_n_lam is not None and corridor_ref_k_lam is not None and int(n_stack.shape[0]) > 0:

        m_chk = (

            np.isfinite(k_lo)

            & np.isfinite(k_hi)

            & np.isfinite(corridor_ref_k_lam)

            & (corridor_ref_k_lam > 0.0)

        )

        if np.any(m_chk):

            k_r = corridor_ref_k_lam[m_chk]

            lo_m = k_lo[m_chk]

            hi_m = k_hi[m_chk]

            bad_m = (k_r < lo_m - 1e-12) | (k_r > hi_m + 1e-12)

            if bool(np.any(bad_m)):

                idx_sub = int(np.where(bad_m)[0][0])

                idx_full = int(np.flatnonzero(m_chk)[idx_sub])

                log.warning(

                    "%s k corridor health: k_ref outside [k_lo,k_hi] at lambda[%d] (min/max inconsistency) - "

                    "k_ref=%.6e k_lo=%.6e k_hi=%.6e",

                    _LOG_PREFIX,

                    idx_full,

                    float(corridor_ref_k_lam[idx_full]),

                    float(k_lo[idx_full]),

                    float(k_hi[idx_full]),

                )

    d_arr = np.asarray(d_vals, dtype=np.float64)

    rm_arr = np.asarray(rmse_vals, dtype=np.float64)

    c_arr = np.asarray(chi2_vals, dtype=np.float64)

    n_stack = np.asarray(n_stack, dtype=np.float64)

    k_stack = np.asarray(k_stack, dtype=np.float64)

    order = np.argsort(d_arr)

    d_arr = d_arr[order]

    rm_arr = rm_arr[order]

    c_arr = c_arr[order] if c_arr.size == d_arr.size else np.full(d_arr.shape, np.nan, dtype=np.float64)

    if n_stack.ndim == 2 and n_stack.shape[0] == order.size:

        n_stack = n_stack[order, :]

    if k_stack.ndim == 2 and k_stack.shape[0] == order.size:

        k_stack = k_stack[order, :]

    dt_ms = 1000.0 * (time.perf_counter() - t0)

    log.info(

        "%s Done | n_valid=%d | d_interval=[%.6f, %.6f] nm (Delta=%.6f) | RMSE_min(valid)=%.8f RMSE_max(valid)=%.8f | elapsed=%.1f ms",

        _LOG_PREFIX,

        int(d_arr.size),

        float(np.min(d_arr)),

        float(np.max(d_arr)),

        float(np.max(d_arr) - np.min(d_arr)),

        float(np.nanmin(rm_arr)) if rm_arr.size else float("nan"),

        float(np.nanmax(rm_arr)) if rm_arr.size else float("nan"),

        float(dt_ms),

    )

    log.info(
        "%s Seed gate summary | kept=%d/%d (%.1f%%) | mean DeltaRMSE(refit-seed)=%s | "
        "(compte = pas de marche d profilés avec métadonnées seed/refit, pas chaque essai multi-start)",
        _LOG_PREFIX,
        int(seed_gate_kept_count),
        int(seed_gate_eval_count),
        100.0 * float(seed_gate_kept_count) / max(1.0, float(seed_gate_eval_count)),
        (
            f"{float(np.nanmean(seed_gate_deltas)):+.6e}"
            if seed_gate_deltas.size
            else "n/a"
        ),
    )

    if log_coaching:

        _log_coaching_corridor_outcome(

            pconf=pconf,

            use_lr=use_lr,

            use_abs_delta=bool(use_abs_delta),

            d0=float(d0),

            d_arr=d_arr,

            rm_arr=rm_arr,

            rmse_opt=rmse_opt,

            rmse_thresh=rmse_thresh_active,

            polish_maxfun=int(maxfun_prof),

            base_result=base_result,

        )

    out_prof: dict[str, Any] = {

        "profile_d_polish_maxfun_effective": int(maxfun_prof),

        "profile_d_enabled": True,

        "profile_d_sigma_hetero": bool(

            use_lr

            and bool(getattr(pconf, "sigma_hetero_residual", False))

            and (sigma_t_f_hetero is not None or sigma_r_f_hetero is not None)

        ),

        "profile_d_sigma_hetero_scale": float(getattr(pconf, "sigma_hetero_scale", 1.0) or 1.0)

        if use_lr

        else None,

        "profile_d_mode": str(pconf.mode),

        "profile_d_acceptance_mode": (

            "lr"

            if use_lr

            else (

                "delta_rmse_abs_best_polished"

                if (use_abs_delta and scientific_nominal)

                else ("delta_rmse_abs" if use_abs_delta else "alpha_heuristic")

            )

        ),

        "profile_d_scientific_nominal": bool(scientific_nominal),

        "profile_rmse_best_ref": float(rmse_opt) if scientific_nominal else None,

        "profile_delta_rmse_abs": float(tol_abs) if use_abs_delta else None,

        "profile_corridor_nominal_label": (

            str(nom_pack.get("label", "")) if scientific_nominal and nom_pack is not None else None

        ),

        "profile_d_rmse_threshold_mode": str(rmse_thr_sub) if not use_lr else "alpha",

        "profile_d_rmse_abs_tolerance": float(tol_abs) if use_abs_delta else None,

        "profile_d_rmse_alpha": float(pconf.rmse_alpha),

        "profile_d_rmse_opt": float(rmse_opt),

        "profile_d_rmse_ref_source": str(rmse_ref_tag),

        "profile_d_rmse_thresh": float(rmse_thresh_active),

        "profile_d_rmse_accept_max": float(rmse_thresh_active) if not use_lr else None,

        "profile_d_rmse_thresh_nominal": float(rmse_thresh) if (not use_lr) else None,

        "profile_d_threshold_basis_effective": str(threshold_basis_eff),

        "profile_d_threshold_fallback_reason": str(threshold_fallback_reason),

        "profile_d_auto_relaxed_threshold": bool(auto_relaxed_alpha),

        "profile_d_lr_conf": float(pconf.lr_conf_level) if use_lr else None,

        "profile_d_lr_delta_chi2": float(delta_chi2) if use_lr else None,

        "profile_d_sigma_t": float(sig_t) if use_lr else None,

        "profile_d_sigma_r": float(sig_r) if use_lr else None,

        "profile_d_values_nm": d_arr,

        "profile_d_rmse_values": rm_arr,

        "profile_d_chi2_values": c_arr,

        "profile_d_n_curves": np.asarray(n_stack, dtype=np.float64),

        "profile_d_k_curves": np.asarray(k_stack, dtype=np.float64),

        "profile_d_interval_nm": (float(np.min(d_arr)), float(np.max(d_arr))),

        "profile_d_status": (

            "degenerate"

            if (int(pos_valid) < min_side or int(neg_valid) < min_side)

            else "ok"

        ),

        "profile_d_valid_side_pos": int(pos_valid),

        "profile_d_valid_side_neg": int(neg_valid),

        "profile_d_boundary_refine_calls": int(boundary_refine_calls),

        "profile_d_fit_fail_rate": float(

            np.nanmean(

                np.asarray(fit_fail_values, dtype=np.float64)

                / np.maximum(np.asarray(fit_try_values, dtype=np.float64), 1.0)

            )

        )

        if fit_try_values

        else float("nan"),
        "profile_d_seed_gate_eval_count": int(seed_gate_eval_count),
        "profile_d_seed_gate_kept_count": int(seed_gate_kept_count),
        "profile_d_seed_gate_kept_rate": (
            float(seed_gate_kept_count) / float(seed_gate_eval_count)
            if int(seed_gate_eval_count) > 0
            else float("nan")
        ),
        "profile_d_seed_gate_center_kept": bool(center_seed_kept),
        "profile_d_seed_gate_mean_delta_refit_minus_seed": (
            float(np.nanmean(seed_gate_deltas)) if seed_gate_deltas.size else float("nan")
        ),
        "profile_d_seed_gate_min_delta_refit_minus_seed": (
            float(np.nanmin(seed_gate_deltas)) if seed_gate_deltas.size else float("nan")
        ),
        "profile_d_seed_gate_max_delta_refit_minus_seed": (
            float(np.nanmax(seed_gate_deltas)) if seed_gate_deltas.size else float("nan")
        ),
        "profile_d_seed_gate_std_delta_refit_minus_seed": (
            float(np.nanstd(seed_gate_deltas)) if seed_gate_deltas.size else float("nan")
        ),

        "profile_d_mean_nfev": float(np.nanmean(np.asarray(fit_nfev_values, dtype=np.float64)))

        if fit_nfev_values

        else float("nan"),

        "profile_d_mean_nit": float(np.nanmean(np.asarray(fit_nit_values, dtype=np.float64)))

        if fit_nit_values

        else float("nan"),

        "corridor_n_lo": np.asarray(n_lo, dtype=np.float64),

        "corridor_n_hi": np.asarray(n_hi, dtype=np.float64),

        "corridor_k_lo": np.asarray(k_lo, dtype=np.float64),

        "corridor_k_hi": np.asarray(k_hi, dtype=np.float64),

    }

    if corridor_ref_n_lam is not None and corridor_ref_k_lam is not None:

        out_prof["corridor_reference_n_lam"] = corridor_ref_n_lam

        out_prof["corridor_reference_k_lam"] = corridor_ref_k_lam

    return out_prof
