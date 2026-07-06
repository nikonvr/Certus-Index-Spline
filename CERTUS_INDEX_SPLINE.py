#!/usr/bin/env python3


# -*- coding: utf-8 -*-


"""


CERTUS-INDEX-SPLINE  Global fit of n(lambda), k(lambda) as piecewise-linear in sigma=1/lambda (ln k at nodes).


Standalone: no imports from CERTUS_INDEX nor certus_swanepool. Local optimization only.


"""


from __future__ import annotations


import argparse


import logging


import multiprocessing


import os


import sys


import time


from dataclasses import dataclass, replace


from typing import Any, Callable


from enum import Enum, auto


from threading import Event


import numpy as np


import pandas as pd


from PyQt6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QMetaObject,
    QObject,
    QPropertyAnimation,
    QSettings,
    QThread,
    Qt,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)


from PyQt6.QtGui import QAction


from PyQt6.QtWidgets import (

    QApplication,

    QCheckBox,

    QComboBox,

    QDialog,

    QDialogButtonBox,

    QDoubleSpinBox,

    QFileDialog,

    QGridLayout,

    QGroupBox,

    QHBoxLayout,

    QLabel,

    QMessageBox,

    QPlainTextEdit,

    QProgressBar,

    QPushButton,

    QScrollArea,

    QSlider,

    QSpinBox,

    QSplitter,

    QStackedWidget,

    QTabWidget,

    QTableWidget,

    QTableWidgetItem,

    QAbstractItemView,

    QVBoxLayout,

    QWidget,


)


import pyqtgraph as pg


from certus_core import (

    K_MAX_LIMIT,

    N_MAX_LIMIT,

    N_MIN_LIMIT,

    SUBSTRATES,

    SUBSTRATE_LIST,

    create_module_environment,

    setup_logging,


)


from certus_data import read_data_file_robust


from certus_physics import (
    calculate_T_substrate_array,

    clip_to_bounds,

    get_n_substrate_array_by_id,


)


from certus_index_utils import (

    _ratio_theoretical_from_nk,

    _reflectance_ratio_theoretical_from_nk,

    log_structured_json_event,


)


from certus_ui import (

    CertusBaseApp,

    CertusLogPanel,

    CertusScientificPlot,

    CertusTheme,

    ExcelTableWidget,

    FlashyCard,

    GenericWorker,

    apply_certus_theme,

    attach_excel_clipboard_context_menu,

    create_header_logo_widget,

    create_styled_button,

    get_certus_last_dir,

    init_certus_app,

    plot_widget_plot_finite,

    sanitize_xy_for_plot,

    set_certus_last_dir,

    setup_pyqtgraph_defaults,

    wrap_scientific_plot_with_toolbar,


)


from certus_reset_framework import create_reset_button


from certus_load_summary import build_summary_plain_text, show_load_summary_dialog


from certus_smart_init_curve_editor import SmartInitNKCurveEditorDialog


# Bootstrap


_env = create_module_environment(__file__, "CERTUS_INDEX_SPLINE")


_SCRIPT_DIR = _env["script_dir"]


logger = logging.getLogger("CERTUS_INDEX_SPLINE")


_QS_SPLINE_ORG = "CERTUS"


_QS_SPLINE_APP = "INDEX_SPLINE"


_QS_LAST_SPECTRUM = "last_spectrum_path"


_QS_SPECTRUM_FIT_T = "spectrum_fit_t"


_QS_SPECTRUM_FIT_TREL = "spectrum_fit_trel"


_QS_SPECTRUM_FIT_R = "spectrum_fit_r"


_QS_SPECTRUM_WT = "spectrum_weight_t"


_QS_SPECTRUM_WR = "spectrum_weight_r"


_QS_NK_PROFILE_INTERP = "nk_profile_interp"


_QS_SPLINE_SIMPLE_AUTO_UNCERTAINTY = "spline_simple_auto_uncertainty"


_QS_SPLINE_UNCERTAINTY_DEFAULTS_REV = "spline_uncertainty_defaults_rev"


_QS_NL_ALPHA_BUDGET = "nl_alpha_budget_mode"


_QS_NL_ALPHA_SECOND_PASS = "nl_alpha_second_pass_enabled"


_QS_NL_ALPHA_ADAPTIVE = "nl_alpha_adaptive_early_stop"


_QS_SMART_INIT_DEEP = "smart_init_deep_pglobal_after_manual"


_QS_SMART_INIT_TWO_PHASE = "smart_init_deep_two_phase_enabled"


_QS_SOL3_PHASE1_MAXFUN = "sol3_phase1_maxfun"


# Increment to reapply corridor / bootstrap / SiO₂ / NL alpha defaults on existing workstations once.


_UNCERTAINTY_DEFAULTS_REV: int = 10


# INDEX-SPLINE defaults for thin SiO₂ layers (~1.6-1.8 µm) on sapphire, TSIO2-type spectra (UV-IR, T/Tsub).


# Aligned on a validated session (Fast profile, mesh K=14 if lambda_max > 4000 nm, RMSE window 250-5000 nm).


SIO2_DEFAULT_D_LO_NM: float = 1600.0


SIO2_DEFAULT_D_HI_NM: float = 1800.0


SIO2_DEFAULT_NK_PROFILE_INTERP: str = "smooth"


SIO2_DEFAULT_RMSE_FIT_LAMBDA_LO_NM: float = 250.0


SIO2_DEFAULT_RMSE_FIT_LAMBDA_HI_NM: float = 5000.0


SIO2_DEFAULT_RMSE_FIT_LAMBDA_ENABLED: bool = True


# Auto-Best uses manual mode with a uniform sigma mesh and 12 anchor nodes


# (N_seg=11 -> K=12), quality preset, without adaptive/auto-K stages.


AUTO_BEST_MANUAL_N_SEG: int = 11  # K = N_seg + 1 = 12 wavelengths via sigma=1/lambda


# Adaptive mesh worker defaults (outside Auto-Best), aligned with core behavior.


from certus_index_spline_core import (

    SPLINE_PWL_K_NODES,

    SPLINE_MIN_RMSE_FIT_OBJECTIVE_POINTS,

    SPLINE_PERF_PRESETS,

    DataType,

    SplineOptConfig,

    default_n_mono_band_nm_from_spectrum,

    sigma_segment_indices,

    gui_perf_preset_only,

    export_spline_result_jsonable,

    build_sigma_knots,

    warm_start_interpolate_nodes,

    _to_fraction_T,

    ensure_lam_nm_array,

    prepare_exp_TR_for_fit,

    _find_lambda_column,

    _find_transmission_column,

    normalize_spectrum_dataframe,

    substrate_id_from_name,

    allowed_substrate_names,

    make_bounds_and_x0,

    warm_start_sigma_regrid,

    merge_spline_preset,

    reset_smart_init_preview_guard,

    rmse_at_spline_stage_x0_init,

    sol3_phase1_maxfun_effective,

    _canonical_knots_min_lambda_kw,

    canonical_spline_sigma_knots,

    bridge_sigma_knots_preserve_manual,

    log_rmse_mesh_bridge_diagnosis,

    _log_index_spline_best_config,


)


from spline_smart_init import (

    build_smart_manual_sigma_knots_from_preview_grid,

    interp_n_L_pwlnk_to_sigmas,

    pick_best_manual_material_preset,

    recalc_smart_init_spectral_preview,

    smart_init_sweep_node_thickness_rmse,


)


from spline_objective import (

    _spline_objective_lam_mask,

    build_spline_objective_masked_grid,

    nk_from_x_pwlnk,

    objective_lam_mask_on_target_grid,

    spline_objective_mse_on_masked_grid,

    spectral_rmse_weights,


)


from spline_pipeline import (
    enforce_local_optimization_policy,
    worker_run_corridor_profile_after_nl_choice,
    worker_spline_optimization,
)


from spline_workers import _run_single_spline_stage


from spline_profile_corridors import _expand_corridor_envelope_with_reported_nk


from spline_presets import _project_nb2o5_preset_to_sigma_knots, project_manual_material_preset


from spline_visual_utils import (

    live_monitor_nk_clipboard_tsv_2nm as _live_monitor_nk_clipboard_tsv_2nm,

    snap_spline_visual_dict as _snap_spline_visual_dict,


)


from spline_workers import worker_auto_best_split_knot_refinement


def _plot_spectrum_raw_scatter(

    plot_w: pg.PlotWidget,

    x: np.ndarray,

    y: np.ndarray,

    *,

    color: str,

    name: str,

    symbol_size: int = 5,


) -> None:

    """Raw spectral data: always in points (no line), CERTUS convention."""

    xf, yf = sanitize_xy_for_plot(x, y)

    if xf.size == 0:

        return

    plot_w.plot(

        xf,

        yf,

        pen=None,

        symbol="o",

        symbolSize=int(symbol_size),

        symbolBrush=pg.mkBrush(color),

        symbolPen=pg.mkPen(color, width=0.6),

        name=name,

    )


def _spectral_display_align(

    lam_nm: np.ndarray, *series: np.ndarray


) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:

    """

    Truncates all series to the same length as lam_nm, then sorts by increasing lambda.

    Without this, a non-monotonic lambda file results in PyQtGraph lines that "smear"

    the spectrum (phantom oscillations) even if the experimental points remain correct in the scatter plot.

    Also returns ``order`` (indices) to reorder other arrays of the same pre-truncation.

    """

    lam = np.asarray(lam_nm, dtype=np.float64).ravel()

    if lam.size == 0:

        z = np.array([], dtype=np.int64)

        return lam, [np.asarray(s, dtype=np.float64).ravel()[:0] for s in series], z

    n_use = lam.size

    arrs: list[np.ndarray] = []

    for s in series:

        a = np.asarray(s, dtype=np.float64).ravel()

        n_use = min(n_use, a.size)

        arrs.append(a)

    if n_use <= 0:

        zf = np.array([], dtype=np.float64)

        zi = np.array([], dtype=np.int64)

        return zf, [zf.copy() for _ in series], zi

    if n_use != lam.size:

        logger.warning(

            "Spectral display: inconsistent lengths (lambda=%d, truncation to %d).",

            lam.size,

            n_use,

        )

    lam_u = lam[:n_use]

    trimmed = [a[:n_use] for a in arrs]

    order = np.argsort(lam_u, kind="mergesort")

    lam_s = lam_u[order]

    out = [np.asarray(t)[order] for t in trimmed]

    return lam_s, out, order


def _mergesort_order_lambda(lam_nm: np.ndarray) -> np.ndarray:

    """Indices to permute spectral columns by increasing lambda (stable sort, same logic as UI)."""

    lam = np.asarray(lam_nm, dtype=np.float64).ravel()

    return np.argsort(lam, kind="mergesort") if lam.size else np.arange(0, dtype=np.intp)


def _smart_init_pw_nk_clipboard_df(curve_n: Any, curve_pk: Any) -> pd.DataFrame | None:

    """Build a DataFrame for Excel export from n(lambda) and ln k(lambda) plot items (k = exp(ln k), capped)."""

    xn, yn = curve_n.getData()

    xk, yk_ln = curve_pk.getData()

    xn = np.asarray(xn if xn is not None else [], dtype=float).ravel()

    yn = np.asarray(yn if yn is not None else [], dtype=float).ravel()

    xk = np.asarray(xk if xk is not None else [], dtype=float).ravel()

    yk_ln = np.asarray(yk_ln if yk_ln is not None else [], dtype=float).ravel()

    yk_k = np.full(yk_ln.shape, np.nan, dtype=float)

    m_ln = np.isfinite(yk_ln)

    yk_k[m_ln] = np.exp(np.minimum(yk_ln[m_ln], 700.0))

    n = int(max(xn.size, yn.size, xk.size, yk_k.size))

    if n == 0:

        return None

    def _pad(a: np.ndarray) -> np.ndarray:

        a = np.asarray(a, dtype=float).ravel()

        if a.size >= n:

            return a[:n].copy()

        return np.pad(a, (0, n - a.size), constant_values=np.nan)

    if xn.size == xk.size and xn.size > 0 and np.allclose(xn, xk, equal_nan=True):

        return pd.DataFrame({"lambda_nm": _pad(xn), "n": _pad(yn), "k": _pad(yk_k)})

    return pd.DataFrame(

        {

            "lambda_nm_n": _pad(xn),

            "n": _pad(yn),

            "lambda_nm_k": _pad(xk),

            "k": _pad(yk_k),

        }

    )


# --- GUI --------------------------------------------------------------------------


@dataclass


class SplineState:

    result: dict | None

    d_lo: float

    d_hi: float

    wt: float

    wr: float


class _CentiPercentProgressBar(QProgressBar):

    """Internal gauge 0..10000 (centi-percents) with label ``xx.xx %``."""

    def text(self) -> str:

        r = int(self.maximum() - self.minimum())

        if r <= 0:

            return super().text()

        pct = 100.0 * float(self.value() - self.minimum()) / float(r)

        return f"{pct:.2f}%"


class CorridorRMSEProfileWindow(QDialog):
    """Window displaying the RMSE = f(thickness) curve from corridor profiling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Corridor RMSE Profile  |  RMSE = f(thickness)")
        self.resize(600, 450)

        layout = QVBoxLayout(self)

        # Info label
        self.lbl_info = QLabel("No corridor data available. Run optimization with corridors enabled.")
        self.lbl_info.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 11px;")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        # Plot widget
        self.plot_rmse = CertusScientificPlot(title="RMSE vs Thickness d")
        self.plot_rmse.setLabel('bottom', 'd (nm)')
        self.plot_rmse.setLabel('left', 'RMSE')
        layout.addWidget(self.plot_rmse)

        # Button bar
        btn_layout = QHBoxLayout()

        self.btn_copy = create_styled_button("Copy data (TSV)", "secondary", parent=self)
        self.btn_copy.setToolTip("Copy d (nm) and RMSE values to clipboard (tab-separated)")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_layout.addWidget(self.btn_copy)

        btn_layout.addStretch()

        self.btn_close = create_styled_button("Close", "secondary", parent=self)
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

        apply_certus_theme(self)

        self._d_data: np.ndarray | None = None
        self._rmse_data: np.ndarray | None = None
        self._rmse_thresh: float | None = None

    def update_profile(self, d_nm: np.ndarray, rmse: np.ndarray, rmse_thresh: float | None = None):
        """Updates the plot with profiling data.

        Args:
            d_nm: Array des épaisseurs (nm)
            rmse: Array des valeurs RMSE correspondantes
            rmse_thresh: RMSE threshold used for acceptance (optional)
        """
        d_arr = np.asarray(d_nm, dtype=np.float64).ravel()
        r_arr = np.asarray(rmse, dtype=np.float64).ravel()

        if d_arr.size == 0 or r_arr.size == 0 or d_arr.size != r_arr.size:
            self.lbl_info.setText("No valid corridor profiling data.")
            self._d_data = None
            self._rmse_data = None
            return

        self._d_data = d_arr
        self._rmse_data = r_arr
        self._rmse_thresh = rmse_thresh

        # Trier par épaisseur
        order = np.argsort(d_arr)
        d_sorted = d_arr[order]
        r_sorted = r_arr[order]

        # Mettre à jour le graphique
        self.plot_rmse.clear()

        # Courbe RMSE(d)
        self.plot_rmse.add_curve(d_sorted, r_sorted, "RMSE(d)", color=CertusTheme.PRIMARY, width=2)

        # Ligne de seuil si disponible
        if rmse_thresh is not None and np.isfinite(rmse_thresh):
            d_span = float(d_sorted[-1] - d_sorted[0]) if d_sorted.size > 1 else 100.0
            d_lo = float(d_sorted[0]) - 0.1 * d_span
            d_hi = float(d_sorted[-1]) + 0.1 * d_span
            self.plot_rmse.add_curve(
                np.array([d_lo, d_hi]),
                np.array([rmse_thresh, rmse_thresh]),
                f"Threshold = {rmse_thresh:.6f}",
                color=CertusTheme.DANGER,
                width=1,
                style=Qt.PenStyle.DashLine
            )

        # Info
        n_points = d_arr.size
        d_min, d_max = float(d_sorted[0]), float(d_sorted[-1])
        r_min, r_max = float(np.min(r_sorted)), float(np.max(r_sorted))
        d_opt = float(d_sorted[np.argmin(r_sorted)])

        info_txt = (
            f"Points: {n_points} | "
            f"d interval: [{d_min:.2f}, {d_max:.2f}] nm | "
            f"d(opt) ≈ {d_opt:.2f} nm | "
            f"RMSE range: [{r_min:.6f}, {r_max:.6f}]"
        )
        if rmse_thresh is not None and np.isfinite(rmse_thresh):
            info_txt += f" | Threshold: {rmse_thresh:.6f}"

        self.lbl_info.setText(info_txt)
        self.plot_rmse.autoRange()

    def _copy_to_clipboard(self):
        """Copies (d, RMSE) data to clipboard."""
        if self._d_data is None or self._rmse_data is None:
            QMessageBox.information(self, "Clipboard", "No data to copy.")
            return

        lines = ["d_nm\tRMSE"]
        for d, r in zip(self._d_data, self._rmse_data):
            lines.append(f"{d:.6f}\t{r:.8f}")

        txt = "\n".join(lines)
        cb = QApplication.clipboard()
        if cb is None:
            QMessageBox.warning(self, "Clipboard", "Clipboard unavailable.")
            return

        cb.setText(txt)
        prev = self.btn_copy.text()
        self.btn_copy.setText("Copied!")
        QTimer.singleShot(1500, lambda t=prev: self.btn_copy.setText(t))


class CertusIndexSplineApp(CertusBaseApp):

    APP_NAME = "CERTUS-INDEX-SPLINE"

    APP_TITLE = "Indices PWL (sigma) - Local optimization"

    DEFAULT_WIDTH = 1280

    DEFAULT_HEIGHT = 720

    MIN_WIDTH = 960

    MIN_HEIGHT = 560

    _LIVE_LOG_REMINDER_S = 12.0

    smart_preview_requested = pyqtSignal(object)

    @staticmethod

    def _control_group_box_style() -> str:

        return (

            f"QGroupBox {{ font-weight: bold; border: 1px solid {CertusTheme.BORDER}; "

            f"border-radius: 6px; margin-top: 12px; padding-top: 10px; }}"

            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; "

            f"color: {CertusTheme.TEXT_MAIN}; }}"

        )

    def _update_persistent_nk_monitor(

        self, lam_arr: np.ndarray, n_arr: np.ndarray, k_arr: np.ndarray, d_nm: float | None = None

    ) -> None:

        mon = getattr(self, "_live_nk_monitor", None)

        if mon is None:

            return

        try:

            if hasattr(mon, "update_indices"):

                mon.update_indices(lam_arr, n_arr, k_arr, d_nm)

        except Exception:

            logger.debug("_update_persistent_nk_monitor failed", exc_info=True)

    def __init__(self) -> None:

        super().__init__()

        self._setup_logger(self.APP_NAME)

        self._log_prog_last: int = -1

        self._prog_ui_last: int = 0

        self._last_live_log_mono: float = 0.0

        self.df: pd.DataFrame | None = None

        self._worker: GenericWorker | None = None

        self._stop_event = Event()

        self._last_result: dict | None = None

        self._last_worker_result: dict | None = None

        self._last_run_cfg: SplineOptConfig | None = None

        self._worker_role: str = "idle"

        self._best_live_rmse: float = float("inf")

        self._best_live_result: dict | None = None

        self._corridor_rmse_manual_active: bool = False

        self._corridor_rmse_manual_lo: float = float("nan")

        self._corridor_rmse_manual_hi: float = float("nan")

        self._corridor_rmse_manual_slider_scale: int = 100

        self._last_spectrum_path: str = ""

        self._pending_auto_best_adaptive: dict[str, Any] | None = None

        self._auto_best_local_warm: dict[str, Any] | None = None

        self._auto_best_force_smart_init: bool = False

        self._auto_best_two_stage_refine: bool = False

        self._auto_best_second_stage_pending: dict[str, Any] | None = None

        self._preview_wait_event: Event | None = None

        self._rmse_fit_lambda_enabled: bool = SIO2_DEFAULT_RMSE_FIT_LAMBDA_ENABLED

        self._rmse_fit_lambda_lo: float = SIO2_DEFAULT_RMSE_FIT_LAMBDA_LO_NM

        self._rmse_fit_lambda_hi: float = SIO2_DEFAULT_RMSE_FIT_LAMBDA_HI_NM

        self._rmse_fit_lambda_lo_default: float = SIO2_DEFAULT_RMSE_FIT_LAMBDA_LO_NM

        self._rmse_fit_lambda_hi_default: float = SIO2_DEFAULT_RMSE_FIT_LAMBDA_HI_NM

        self._rmse_fit_overlay_items: list[Any] = []

        self._simple_auto_uncertainty: bool = True

        self._build_ui()

        self._restore_simple_auto_uncertainty_pref()

        self._restore_spectrum_fit_settings()

        self._restore_nl_alpha_budget_pref()

        self._restore_nl_alpha_second_pass_pref()

        self._restore_nl_alpha_adaptive_pref()

        self._restore_sol3_phase1_maxfun_pref()

        self._maybe_apply_uncertainty_defaults_migrated()

        self._refresh_corridors_gui_state_labels()

        self._update_epured_visibility()

        self._wire_spectrum_fit_settings_persistence()

        self._wire_sol3_phase1_maxfun_persistence()

        self._persist_spectrum_fit_settings()

        self.smart_preview_requested.connect(self._on_smart_preview_requested)

        apply_certus_theme(

            self,

            plots=[

                self.plot_T,

                self.plot_n,

                self.plot_lgk,

                self.plot_n_corridor,

                self.plot_lgk_corridor,

                self.plot_corridor_rmse_d,

            ],

        )

        self._finalize_init()

    def _load_defaults(self) -> None:

        self.df = None

        self._last_result = None

        self._best_live_rmse = float("inf")

        self._best_live_result = None

        if hasattr(self, "lbl_file"):

            self.lbl_file.setText("(no file)")

        if hasattr(self, "lbl_status"):

            self.lbl_status.setText("Ready")

        if hasattr(self, "table_nk"):

            self._refresh_data_table()

        self._rmse_fit_lambda_enabled = bool(SIO2_DEFAULT_RMSE_FIT_LAMBDA_ENABLED)

        self._rmse_fit_lambda_lo = float(SIO2_DEFAULT_RMSE_FIT_LAMBDA_LO_NM)

        self._rmse_fit_lambda_hi = float(SIO2_DEFAULT_RMSE_FIT_LAMBDA_HI_NM)

        self._rmse_fit_lambda_lo_default = float(SIO2_DEFAULT_RMSE_FIT_LAMBDA_LO_NM)

        self._rmse_fit_lambda_hi_default = float(SIO2_DEFAULT_RMSE_FIT_LAMBDA_HI_NM)

        self._remove_rmse_fit_region_overlay()

    def _restore_spectrum_fit_settings(self) -> None:

        """Reads step 3 from QSettings (T, T/Tsub ratio, R, wT, wR)."""

        if not hasattr(self, "chk_t"):

            return

        s = QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP)

        widgets = (

            self.chk_t,

            self.chk_trel,

            self.chk_r,

            self.w_t,

            self.w_r,

        )

        for w in widgets:

            w.blockSignals(True)

        try:

            vt = s.value(_QS_SPECTRUM_FIT_T)

            if vt is not None:

                self.chk_t.setChecked(bool(vt))

            vrel = s.value(_QS_SPECTRUM_FIT_TREL)

            if vrel is not None:

                self.chk_trel.setChecked(bool(vrel))

            vr = s.value(_QS_SPECTRUM_FIT_R)

            if vr is not None:

                self.chk_r.setChecked(bool(vr))

            wtv = s.value(_QS_SPECTRUM_WT)

            if wtv is not None:

                self.w_t.setValue(float(wtv))

            wrv = s.value(_QS_SPECTRUM_WR)

            if wrv is not None:

                self.w_r.setValue(float(wrv))

        finally:

            for w in widgets:

                w.blockSignals(False)

    def _persist_spectrum_fit_settings(self) -> None:

        """Saves step 3 to QSettings (read at next launch)."""

        if not hasattr(self, "chk_t"):

            return

        s = QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP)

        s.setValue(_QS_SPECTRUM_FIT_T, self.chk_t.isChecked())

        s.setValue(_QS_SPECTRUM_FIT_TREL, self.chk_trel.isChecked())

        s.setValue(_QS_SPECTRUM_FIT_R, self.chk_r.isChecked())

        s.setValue(_QS_SPECTRUM_WT, float(self.w_t.value()))

        s.setValue(_QS_SPECTRUM_WR, float(self.w_r.value()))

        s.setValue(_QS_NK_PROFILE_INTERP, "smooth")

    def _wire_spectrum_fit_settings_persistence(self) -> None:

        self.chk_t.toggled.connect(self._persist_spectrum_fit_settings)

        self.chk_trel.toggled.connect(self._persist_spectrum_fit_settings)

        self.chk_r.toggled.connect(self._persist_spectrum_fit_settings)

        self.w_t.valueChanged.connect(self._persist_spectrum_fit_settings)

        self.w_r.valueChanged.connect(self._persist_spectrum_fit_settings)

    def _save_undo_state(self) -> None:

        """Store current state before computation in undo stack (Ctrl+Z via CertusBaseApp)."""

        if not hasattr(self, "undo_stack"): return

        state = SplineState(

            result=dict(self._last_result) if self._last_result is not None else None,

            d_lo=self.d_lo.value(),

            d_hi=self.d_hi.value(),

            wt=self.w_t.value(),

            wr=self.w_r.value(),

        )

        self.undo_stack.append(state)

        if hasattr(self, "undo_btn"):

            self.undo_btn.setEnabled(True)

    def _setup_logger(self, name: str):

        """Route core logs (CERTUS, CERTUS_INDEX_SPLINE) to the same GUI queue."""

        from certus_core import QueueHandler

        super()._setup_logger(name)

        qh = next(

            (h for h in (self.logger.handlers or []) if isinstance(h, QueueHandler)),

            None,

        )

        if qh is None:

            return

        for ln in ("CERTUS", "CERTUS_INDEX_SPLINE"):

            lg = logging.getLogger(ln)

            if any(

                isinstance(h, QueueHandler)

                and getattr(h, "log_queue", None) is self.log_queue

                for h in lg.handlers

            ):

                continue

            lg.setLevel(logging.INFO)

            lg.addHandler(qh)

    def _log_optimization_header(self, cfg: SplineOptConfig) -> None:

        """Startup INFO block (CERTUS_INDEX+ detail: context + displayed RMSE reminder)."""

        if not self.logger:

            return

        lam = np.asarray(cfg.lam_nm, dtype=np.float64)

        npt = int(lam.size)

        if npt:

            l0, l1 = float(np.nanmin(lam)), float(np.nanmax(lam))

        else:

            l0 = l1 = float("nan")

        path_hint = (

            getattr(self, "_last_spectrum_path", "").strip()

            or (str(self.lbl_file.text()).strip() if hasattr(self, "lbl_file") else "")

        )

        dt_name = cfg.data_type.name if hasattr(cfg.data_type, "name") else str(cfg.data_type)

        self.logger.info(" INDEX-SPLINE  optimization start ")

        self.logger.info("Spectrum: %s", path_hint or "(unknown path)")

        self.logger.info(

            "substrate: %s | lambda [%g, %g] nm | %d points | substrate-normalized T: %s",

            cfg.substrate_name,

            l0,

            l1,

            npt,

            cfg.t_is_ratio,

        )

        self.logger.info(

            "Target: %s | weights wT=%.4g wR=%.4g | spectral quadrature: ln lambda (trapezoids, no cap)",

            dt_name,

            cfg.weight_t,

            cfg.weight_r,

        )

        self.logger.info(

            "d  [%.2f, %.2f] nm | segments sur maillage sigma (n_seg)=%d",

            cfg.d_lo,

            cfg.d_hi,

            cfg.n_seg,

        )

        if cfg.n_mono_band_nm is not None:

            a, b = float(cfg.n_mono_band_nm[0]), float(cfg.n_mono_band_nm[1])

            self.logger.info(

            "n(sigma) monotonicity on segments intersecting lambda[%.0f, %.0f] nm | continuous-law penalty w=%.4g",

                min(a, b),

                max(a, b),

                float(cfg.n_mono_continuous_penalty),

            )

        else:

            self.logger.info("n(sigma) monotonicity on fixed band: disabled")

        w_nlam = float(getattr(cfg, "n_lambda_rising_penalty_weight", 0.0) or 0.0)

        band_nlam = getattr(cfg, "n_lambda_rising_penalty_band_nm", None)

        if w_nlam > 0.0 and band_nlam is not None:

            b0, b1 = float(band_nlam[0]), float(band_nlam[1])

            self.logger.info(

                "n increasing with lambda forbidden (segments sigma ∩ lambda[%.0f, %.0f] nm) | penalty w=%.4g",

                min(b0, b1),

                max(b0, b1),

                w_nlam,

            )

        else:

            self.logger.info("Penalty for increasing n(lambda): disabled (w=0 or band None)")

        n_fit = int(np.count_nonzero(_spline_objective_lam_mask(cfg)))

        if cfg.rmse_fit_lambda_nm is not None:

            rl0, rl1 = float(cfg.rmse_fit_lambda_nm[0]), float(cfg.rmse_fit_lambda_nm[1])

            self.logger.info(

                "RMSE fit lambda: [%.4g, %.4g] nm (%d points)",

                min(rl0, rl1),

                max(rl0, rl1),

                n_fit,

            )

        else:

            self.logger.info("RMSE fit lambda: full spectrum (%d objective points)", n_fit)

        self.logger.info("Local optimizer: polish maxfun=%d", cfg.polish_maxfun)

        self.logger.info(

            "SOL3: L-BFGS-B phase 1 maxfun=%d (effective; UI sol3_phase1_maxfun=%s)",

            int(sol3_phase1_maxfun_effective(cfg)),

            getattr(cfg, "sol3_phase1_maxfun", None),

        )

        self.logger.info("substrate: no Deltan_sub refinement (substrat nominal).")

        prof = str(self.cb_profilee.currentData() or "fast") if hasattr(self, "cb_profilee") else "fast"

        self.logger.info("GUI performance profile: %s", prof)

        self.logger.info("Auto-Ksigma: disabled (K fixed to n_seg+1 initial knots)")

        self.logger.info(

            "[Reminder] During optimization, live snapshots follow the best RMSE seen at that moment. "

            "At the end, rmse/mse in the dict may reflect final indices (polish spectral spline cubique sigma) "

            "- compare to pipeline_best_rmse_watermark if needed. Curves on screen = n_lam/k_lam from final dict."

        )

    def _build_ui(self) -> None:

        setup_pyqtgraph_defaults()

        central = QWidget()

        self.setCentralWidget(central)

        outer = QVBoxLayout(central)

        outer.setContentsMargins(0, 0, 0, 0)

        outer.setSpacing(0)

        outer.addWidget(

            create_header_logo_widget(

                title_text="CERTUS-INDEX-SPLINE",

                subtitle_text=self.APP_TITLE,

                module_name="CERTUS_INDEX_SPLINE",

            )

        )

        root_layout = QHBoxLayout()

        root_layout.setContentsMargins(6, 6, 6, 6)

        outer.addLayout(root_layout, 1)

        left_scroll = QScrollArea()

        left_scroll.setWidgetResizable(True)

        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        left_scroll.setMinimumWidth(300)

        left_scroll.setMaximumWidth(580)

        left_inner = QWidget()

        left_inner.setMinimumWidth(460)

        left_lay = QVBoxLayout(left_inner)

        left_lay.setContentsMargins(0, 0, 4, 0)

        left_gb = QGroupBox("Recommended workflow  INDEX-SPLINE")

        left_gb.setToolTip(

            "General order: 1 file -> 24 Basic tab -> (57 Advanced if needed) -> 8 Run. "

            "Tooltips describe each control in detail."

        )

        gb_lay = QVBoxLayout(left_gb)

        st1 = QLabel(

            "<b>Step 1</b>  Load the spectrum (lambda + T required, R optional). "

            "Check rendering in the <i>T / R Spectrum</i> plot tab."

        )

        st1.setWordWrap(True)

        st1.setStyleSheet(f"color: {CertusTheme.PRIMARY}; font-size: 11px;")

        st1.setToolTip(

            "First action: without a valid file, run is not possible. "

            "CSV / Excel formats; columns normalized to lambda (nm) and T."

        )

        gb_lay.addWidget(st1)

        self.lbl_file = QLabel("(no file)")

        self.lbl_file.setWordWrap(True)

        self.lbl_file.setToolTip("Path of the last loaded file (preview).")

        btn_load = create_styled_button("1  Load spectrum...", "secondary")

        btn_load.setToolTip(

            "Step 1: open a file containing at least lambda and transmission T. "

            "In Basic, enable T/Tsub if T already is T_film/T_bare_sub ratio (often in %)."

        )

        btn_load.clicked.connect(self._on_load)

        gb_lay.addWidget(self.lbl_file)

        gb_lay.addWidget(btn_load)

        st_tabs = QLabel(

            "<b>Steps 2 to 4</b>  <i>Basic</i> tab."

        )

        st_tabs.setWordWrap(True)

        st_tabs.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 10px;")

        st_tabs.setToolTip(

            "Basic: substrate, thickness, fitted targets (T/R), then sigma mesh and local optimization."

        )

        gb_lay.addWidget(st_tabs)

        self.ctrl_tabs = QTabWidget()

        self.ctrl_tabs.setToolTip(

            "Basic = steps 24 in displayed order."

        )

        self.ctrl_tabs.addTab(self._build_controls_basic_panel(), "Basic (2 -> 4)")

        gb_lay.addWidget(self.ctrl_tabs, 1)

        st8 = QLabel(

            "<b>Step 5</b>  <b>Run Optimization</b>."

        )

        st8.setWordWrap(True)

        st8.setStyleSheet(f"color: {CertusTheme.PRIMARY}; font-size: 10px;")

        st8.setToolTip(

            "Automatically sets sigma-knot count and runs the best possible optimization (Smart Init)."

        )

        gb_lay.addWidget(st8)

        self.btn_run = create_styled_button(" Run Optimization", "accent")

        self.btn_run.setToolTip(

            "Start global optimization."

        )

        self.btn_run.clicked.connect(self._on_run)

        self.btn_stop = create_styled_button("Stop", "secondary")

        self.btn_stop.setToolTip(

            "Stop current optimization and keep the best configuration found so far."

        )

        self.btn_stop.setEnabled(False)

        self.btn_stop.clicked.connect(self._on_stop)

        row_btn = QHBoxLayout()

        row_btn.addWidget(self.btn_run)

        row_btn.addWidget(self.btn_stop)

        gb_lay.addLayout(row_btn)

        row_quick = QHBoxLayout()

        self.btn_corridor_toggle = QPushButton("Corridors")

        self.btn_corridor_toggle.setCheckable(True)

        self.btn_corridor_toggle.setChecked(True)

        self.btn_corridor_toggle.setToolTip(

            "Enables or disables n/k corridor calculation (thickness d profiling) at the end of optimization.\n"

            "Default mode (RMSE_ref+Delta + best RMSE): acceptance envelope around the best polished spectral model.\n"

            "Equivalent to 'Corridors n/k -> Enable' in advanced settings."

        )

        self.btn_corridor_toggle.toggled.connect(self._on_btn_corridor_toggled)

        row_quick.addWidget(self.btn_corridor_toggle)

        self.lbl_corridors_run_state = QLabel()

        self.lbl_corridors_run_state.setTextFormat(Qt.TextFormat.RichText)

        self.lbl_corridors_run_state.setToolTip(

            "Indicates if n/k corridor profiling will be executed at the end of the run (equivalent to 'Enable' in Basic -> Advanced)."

        )

        row_quick.addWidget(self.lbl_corridors_run_state)

        self.btn_nl_toggle = QPushButton("Non-lin. alpha")

        self.btn_nl_toggle.setCheckable(True)

        self.btn_nl_toggle.setChecked(True)

        self.btn_nl_toggle.setToolTip(

            "Enabled by default - uncheck to skip. After fit: alpha sweep from 0.995 to 1.005 (0.0005 step); "

            "at each alpha, L-BFGS-B on thickness + nodes (MSE mask with alpha×T/R). 'Slow' budget by default, then "

            "2nd pass on alpha_opt (adjustable in NL alpha tab). Best (alpha, mesh) pair retained."

        )

        row_quick.addWidget(self.btn_nl_toggle)

        row_quick.addStretch(1)

        gb_lay.addLayout(row_quick)

        if hasattr(self, "chk_corridor_d"):

            self.btn_corridor_toggle.setChecked(self.chk_corridor_d.isChecked())

            self.chk_corridor_d.stateChanged.connect(lambda _s: self._sync_corridor_btn_from_chk())

        self.prog = _CentiPercentProgressBar()

        self.prog.setRange(0, 10000)

        self.prog.setToolTip(

            "INDEX_SPLINE worker progression: 0–100 % monotonic (échelle interne 1/100 %, "

            "SOL2 → SOL3 → fixed-mesh finalization → k floor, polish spectral spline cubique sigma, NL alpha). "

            "Long phases (notamment SOL3 / deep polish) are weighted by their optimization budgets and "

            "emit keep-alive updates about every 5 s so the run does not appear frozen."

        )

        gb_lay.addWidget(self.prog)

        self._prog_anim = QPropertyAnimation(self.prog, b"value", self)

        self._prog_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.lbl_status = QLabel("Ready")

        self.lbl_status.setWordWrap(True)

        self.lbl_status.setToolTip("Last action or error; details also in the Journal tab.")

        gb_lay.addWidget(self.lbl_status)

        reset_btn = create_reset_button(self, use_app_reset=True)

        reset_btn.setToolTip(

            "Clear everything and return to first-launch state: no loaded spectrum, "

            "no result, default controls, empty plots/logs, detached windows closed."

        )

        gb_lay.addWidget(reset_btn)

        gb_lay.addStretch(1)

        left_lay.addWidget(left_gb, 1)

        left_scroll.setWidget(left_inner)

        right_panel = self._build_plot_tabs_panel()

        inner_split = QSplitter(Qt.Orientation.Horizontal)

        inner_split.addWidget(left_scroll)

        inner_split.addWidget(right_panel)

        inner_split.setStretchFactor(1, 1)

        self.log_panel = CertusLogPanel(title="OPTIMIZATION LOG  LIVE INFO")

        self.log_panel.setMinimumHeight(120)

        self.log_text = self.log_panel.log_text

        self.widgets["log_text"] = self.log_text

        self.main_split = QSplitter(Qt.Orientation.Vertical)

        self.main_split.addWidget(inner_split)

        self.main_split.addWidget(self.log_panel)

        self.main_split.setStretchFactor(0, 4)

        self.main_split.setStretchFactor(1, 1)

        root_layout.addWidget(self.main_split)

        self._refresh_corridors_gui_state_labels()

        menu_bar = self.menuBar()

        # Menu View
        menu_view = menu_bar.addMenu("View")

        act_rmse_profile = QAction("Corridor RMSE Profile...", self)
        act_rmse_profile.setShortcut("Ctrl+Shift+R")
        act_rmse_profile.triggered.connect(self._show_corridor_rmse_profile_window)
        menu_view.addAction(act_rmse_profile)

        menu_view.addSeparator()

        menu_param = menu_bar.addMenu("Settings")

        act_adv = QAction("Advanced settings...", self)

        act_adv.triggered.connect(self._open_advanced_settings_dialog)

        menu_param.addAction(act_adv)

    def _get_log_widget(self) -> Any | None:

        return getattr(self.log_panel, "log_text", None) if hasattr(self, "log_panel") else None

    def _restore_simple_auto_uncertainty_pref(self) -> None:

        s = QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP)

        v = s.value(_QS_SPLINE_SIMPLE_AUTO_UNCERTAINTY)

        if v is None:

            self._simple_auto_uncertainty = True

        else:

            self._simple_auto_uncertainty = bool(v)

    def _persist_simple_auto_uncertainty_pref(self) -> None:

        QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP).setValue(

            _QS_SPLINE_SIMPLE_AUTO_UNCERTAINTY, bool(getattr(self, "_simple_auto_uncertainty", True))

        )

    def _restore_nl_alpha_budget_pref(self) -> None:

        """NL alpha: L-BFGS-B budget per step (slow by default)."""

        if not hasattr(self, "cb_nl_alpha_budget"):

            return

        s = QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP)

        v = s.value(_QS_NL_ALPHA_BUDGET, "slow")

        mode = str(v or "slow").strip().lower()

        if mode not in ("slow", "fast"):

            mode = "slow"

        self.cb_nl_alpha_budget.blockSignals(True)

        try:

            iq = self.cb_nl_alpha_budget.findData(mode)

            if iq >= 0:

                self.cb_nl_alpha_budget.setCurrentIndex(int(iq))

        finally:

            self.cb_nl_alpha_budget.blockSignals(False)

    def _persist_nl_alpha_budget_pref(self) -> None:

        if not hasattr(self, "cb_nl_alpha_budget"):

            return

        mode = str(self.cb_nl_alpha_budget.currentData() or "slow")

        QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP).setValue(_QS_NL_ALPHA_BUDGET, mode)

    def _restore_nl_alpha_second_pass_pref(self) -> None:

        if not hasattr(self, "chk_nl_second_pass"):

            return

        s = QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP)

        v = s.value(_QS_NL_ALPHA_SECOND_PASS)

        on = True if v is None else bool(v)

        self.chk_nl_second_pass.blockSignals(True)

        try:

            self.chk_nl_second_pass.setChecked(on)

        finally:

            self.chk_nl_second_pass.blockSignals(False)

    def _persist_nl_alpha_second_pass_pref(self) -> None:

        if not hasattr(self, "chk_nl_second_pass"):

            return

        QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP).setValue(

            _QS_NL_ALPHA_SECOND_PASS, bool(self.chk_nl_second_pass.isChecked())

        )

    def _restore_nl_alpha_adaptive_pref(self) -> None:

        if not hasattr(self, "chk_nl_adaptive_scan"):

            return

        s = QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP)

        v = s.value(_QS_NL_ALPHA_ADAPTIVE)

        on = True if v is None else bool(v)

        self.chk_nl_adaptive_scan.blockSignals(True)

        try:

            self.chk_nl_adaptive_scan.setChecked(on)

        finally:

            self.chk_nl_adaptive_scan.blockSignals(False)

    def _persist_nl_alpha_adaptive_pref(self) -> None:

        if not hasattr(self, "chk_nl_adaptive_scan"):

            return

        QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP).setValue(

            _QS_NL_ALPHA_ADAPTIVE, bool(self.chk_nl_adaptive_scan.isChecked())

        )

    def _restore_sol3_phase1_maxfun_pref(self) -> None:

        if not hasattr(self, "sp_sol3_p1_maxfun"):

            return

        s = QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP)

        v = s.value(_QS_SOL3_PHASE1_MAXFUN, 10000)

        try:

            vi = int(v)

        except (TypeError, ValueError):

            vi = 10000

        vi = max(500, min(500000, vi))

        self.sp_sol3_p1_maxfun.blockSignals(True)

        try:

            self.sp_sol3_p1_maxfun.setValue(vi)

        finally:

            self.sp_sol3_p1_maxfun.blockSignals(False)

    def _persist_sol3_phase1_maxfun_pref(self) -> None:

        if not hasattr(self, "sp_sol3_p1_maxfun"):

            return

        QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP).setValue(

            _QS_SOL3_PHASE1_MAXFUN, int(self.sp_sol3_p1_maxfun.value())

        )

    def _wire_sol3_phase1_maxfun_persistence(self) -> None:

        if not hasattr(self, "sp_sol3_p1_maxfun"):

            return

        self.sp_sol3_p1_maxfun.valueChanged.connect(self._persist_sol3_phase1_maxfun_pref)

    def _apply_sio2_default_fit_parameters(self) -> None:

        """Thickness bounds, RMSE lambda window and nk interpolation for thin SiO₂ layers (cf. TSIO2 / sapphire logs)."""

        if hasattr(self, "d_lo"):

            self.d_lo.setValue(float(SIO2_DEFAULT_D_LO_NM))

        if hasattr(self, "d_hi"):

            self.d_hi.setValue(float(SIO2_DEFAULT_D_HI_NM))

        self._rmse_fit_lambda_enabled = bool(SIO2_DEFAULT_RMSE_FIT_LAMBDA_ENABLED)

        self._rmse_fit_lambda_lo = float(SIO2_DEFAULT_RMSE_FIT_LAMBDA_LO_NM)

        self._rmse_fit_lambda_hi = float(SIO2_DEFAULT_RMSE_FIT_LAMBDA_HI_NM)

        self._rmse_fit_lambda_lo_default = float(SIO2_DEFAULT_RMSE_FIT_LAMBDA_LO_NM)

        self._rmse_fit_lambda_hi_default = float(SIO2_DEFAULT_RMSE_FIT_LAMBDA_HI_NM)

    def _maybe_apply_uncertainty_defaults_migrated(self) -> None:

        """Applies automatic uncertainty defaults once (migration / new install)."""

        s = QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP)

        try:

            rev = int(s.value(_QS_SPLINE_UNCERTAINTY_DEFAULTS_REV, 0) or 0)

        except (TypeError, ValueError):

            rev = 0

        if rev < _UNCERTAINTY_DEFAULTS_REV:

            if rev < 5:

                self._apply_recommended_uncertainty_and_perf_defaults()

                self._apply_sio2_default_fit_parameters()

                if hasattr(self, "chk_trel"):

                    self.chk_trel.setChecked(True)

            if rev < 6 and hasattr(self, "btn_nl_toggle"):

                self.btn_nl_toggle.setChecked(True)

            if rev < 7 and hasattr(self, "cb_nl_alpha_budget"):

                self.cb_nl_alpha_budget.blockSignals(True)

                try:

                    iq = self.cb_nl_alpha_budget.findData("slow")

                    if iq >= 0:

                        self.cb_nl_alpha_budget.setCurrentIndex(int(iq))

                finally:

                    self.cb_nl_alpha_budget.blockSignals(False)

                self._persist_nl_alpha_budget_pref()

            if rev < 8 and hasattr(self, "chk_nl_second_pass"):

                self.chk_nl_second_pass.setChecked(True)

                self._persist_nl_alpha_second_pass_pref()

            if rev < 10 and hasattr(self, "chk_nl_adaptive_scan"):

                self.chk_nl_adaptive_scan.setChecked(True)

                self._persist_nl_alpha_adaptive_pref()

            if rev < 9 and hasattr(self, "cb_corr_mode"):

                self.cb_corr_mode.blockSignals(True)

                try:

                    iq = self.cb_corr_mode.findData("abs_delta")

                    if iq >= 0:

                        self.cb_corr_mode.setCurrentIndex(int(iq))

                finally:

                    self.cb_corr_mode.blockSignals(False)

                if hasattr(self, "sp_corr_rmse_delta"):

                    self.sp_corr_rmse_delta.setValue(0.001)

                self._on_corr_mode_changed()

            s.setValue(_QS_SPLINE_UNCERTAINTY_DEFAULTS_REV, int(_UNCERTAINTY_DEFAULTS_REV))

    def _apply_recommended_uncertainty_and_perf_defaults(self) -> None:

        """Speed-oriented defaults: Fast profile, n/k corridors enabled (accelerated refits); optional bootstrap / reg. scan."""

        if not hasattr(self, "cb_profilee"):

            return

        self.cb_profilee.blockSignals(True)

        try:

            iq = self.cb_profilee.findData("fast")

            if iq >= 0:

                self.cb_profilee.setCurrentIndex(int(iq))

        finally:

            self.cb_profilee.blockSignals(False)

        self._on_profilee_changed()

        if hasattr(self, "chk_corridor_d"):

            self.chk_corridor_d.setChecked(True)

        if hasattr(self, "cb_corr_mode"):

            ia = self.cb_corr_mode.findData("alpha")

            if ia >= 0:

                self.cb_corr_mode.setCurrentIndex(int(ia))

        if hasattr(self, "chk_corr_scientific_nominal"):

            self.chk_corr_scientific_nominal.setChecked(True)

        if hasattr(self, "chk_corr_sigma_hetero"):

            self.chk_corr_sigma_hetero.setChecked(False)

        if hasattr(self, "sp_corr_hetero_scale"):

            self.sp_corr_hetero_scale.setValue(1.0)

        if hasattr(self, "sp_corr_sigma"):

            self.sp_corr_sigma.setValue(0.0)

        if hasattr(self, "sp_corr_starts"):

            self.sp_corr_starts.setValue(1)

        if hasattr(self, "chk_corr_reg_sens"):

            self.chk_corr_reg_sens.setChecked(False)

        if hasattr(self, "chk_corr_boot"):

            self.chk_corr_boot.setChecked(False)

        if hasattr(self, "sp_corr_boot_n"):

            self.sp_corr_boot_n.setValue(40)

        self._refresh_corridors_gui_state_labels()

        if hasattr(self, "sp_corr_boot_p"):

            self.sp_corr_boot_p.setValue(0.95)

        if hasattr(self, "chk_corr_boot_refit"):

            self.chk_corr_boot_refit.setChecked(False)

        if hasattr(self, "sp_corr_boot_maxfun"):

            self.sp_corr_boot_maxfun.setValue(4000)

        if hasattr(self, "sp_corr_boot_workers"):

            self.sp_corr_boot_workers.setValue(1)

        if hasattr(self, "sp_corr_span"):

            self.sp_corr_span.setValue(15.0)

        if hasattr(self, "sp_corr_prof_maxfun"):

            self.sp_corr_prof_maxfun.setValue(2500)

        if hasattr(self, "cb_corr_boot_mode"):

            ip = self.cb_corr_boot_mode.findData("parametric")

            if ip >= 0:

                self.cb_corr_boot_mode.setCurrentIndex(int(ip))

    def _update_epured_visibility(self) -> None:

        if not hasattr(self, "_stack_box4_adv"):

            return

        epure = bool(getattr(self, "_simple_auto_uncertainty", True))

        self._stack_box4_adv.setCurrentIndex(0 if epure else 1)

    def _open_advanced_settings_dialog(self) -> None:

        lay_page = getattr(self, "_box4_full_adv_layout", None)

        if not hasattr(self, "_w_full_adv") or lay_page is None:

            return

        dlg = QDialog(self)

        dlg.setWindowTitle("Advanced settings - INDEX-SPLINE")

        dlg.resize(560, 620)

        outer = QVBoxLayout(dlg)

        chk = QCheckBox(

            "Simplified Basic panel (recommended): hide advanced budgets and uncertainty details"

        )

        chk.setChecked(bool(getattr(self, "_simple_auto_uncertainty", True)))

        chk.setToolTip(

            "Unchecked: after OK, controls stay visible in step 4. "

            "Checked: summary only in the panel; settings remain available here."

        )

        outer.addWidget(chk)

        scroll = QScrollArea()

        scroll.setWidgetResizable(True)

        host = QWidget()

        host_lay = QVBoxLayout(host)

        host_lay.setContentsMargins(0, 0, 0, 0)

        lay_page.removeWidget(self._w_full_adv)

        host_lay.addWidget(self._w_full_adv)

        scroll.setWidget(host)

        outer.addWidget(scroll, 1)

        bb = QDialogButtonBox(

            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        )

        bb.accepted.connect(dlg.accept)

        bb.rejected.connect(dlg.reject)

        outer.addWidget(bb)

        code = dlg.exec()

        host_lay.removeWidget(self._w_full_adv)

        lay_page.addWidget(self._w_full_adv)

        self._w_full_adv.show()

        if code == QDialog.DialogCode.Accepted:

            self._simple_auto_uncertainty = chk.isChecked()

            self._persist_simple_auto_uncertainty_pref()

        self._update_epured_visibility()

    def _show_corridor_rmse_profile_window(self) -> None:
        """Displays the RMSE = f(thickness) window with corridor profiling data."""
        win = getattr(self, "_corridor_rmse_profile_win", None)
        if win is None or not win.isVisible():
            win = CorridorRMSEProfileWindow(self)
            self._corridor_rmse_profile_win = win

        # Mettre à jour avec les données disponibles
        if self._last_result is not None:
            d_prof = np.asarray(self._last_result.get("profile_d_values_nm", []), dtype=np.float64)
            r_prof = np.asarray(self._last_result.get("profile_d_rmse_values", []), dtype=np.float64)
            rmse_thresh = self._last_result.get("profile_d_rmse_thresh")
            if d_prof.size > 0 and r_prof.size == d_prof.size:
                win.update_profile(d_prof, r_prof, rmse_thresh)

        win.show()
        win.raise_()
        win.activateWindow()

    def _build_controls_basic_panel(self) -> QWidget:

        """Steps 2 to 4: substrate / thickness, spectral targets, mesh and optimizer."""

        w = QWidget()

        v = QVBoxLayout(w)

        v.setContentsMargins(4, 4, 4, 4)

        gst = self._control_group_box_style()

        hint = QLabel(

            "Follow this order: <b>2</b> then <b>3</b> then <b>4</b>, then go to step 8 (Run) "

            "or open the Advanced tab if you use auto-K / continuous laws."

        )

        hint.setWordWrap(True)

        hint.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 10px;")

        hint.setToolTip("The model needs correct substrate and thickness bounds before any optimization.")

        v.addWidget(hint)

        box2 = QGroupBox("2  Substrate & thickness d (nm)")

        box2.setStyleSheet(gst)

        box2.setToolTip(

            "Step 2: set substrate optical index n_sub(lambda) and single-layer thickness bounds. "

            "This must be physically consistent before running the fit."

        )

        g2 = QGridLayout(box2)

        r = 0

        lb_d0 = QLabel("d_min (nm) :")

        lb_d0.setToolTip("Lower search bound for physical layer thickness (nm).")

        g2.addWidget(lb_d0, r, 0)

        self.d_lo = QDoubleSpinBox()

        self.d_lo.setRange(1.0, 50000.0)

        self.d_lo.setValue(float(SIO2_DEFAULT_D_LO_NM))

        self.d_lo.setToolTip(

            "Lower bound for d (nm). Default thin SiO₂ ~1.7 µm: 1600 nm - widen if needed."

        )

        g2.addWidget(self.d_lo, r, 1)

        r += 1

        lb_d1 = QLabel("d_max (nm) :")

        lb_d1.setToolTip("Upper search bound for thickness (nm).")

        g2.addWidget(lb_d1, r, 0)

        self.d_hi = QDoubleSpinBox()

        self.d_hi.setRange(1.0, 50000.0)

        self.d_hi.setValue(float(SIO2_DEFAULT_D_HI_NM))

        self.d_hi.setToolTip(

            "Upper bound for d (nm). Default thin SiO₂: 1800 nm - must be > d_min."

        )

        g2.addWidget(self.d_hi, r, 1)

        r += 1

        lb_sub = QLabel("Substrate :")

        lb_sub.setToolTip(

            "Bare substrate material used to compute T_sub and the multilayer model (CERTUS list)."

        )

        g2.addWidget(lb_sub, r, 0)

        self.cb_sub = QComboBox()

        for name in allowed_substrate_names():

            self.cb_sub.addItem(name, name)

        # Default: Sapphire (Al2O3)

        idx_sapphire = self.cb_sub.findText("Sapphire (Al2O3)", Qt.MatchFlag.MatchContains)

        if idx_sapphire >= 0:

            self.cb_sub.setCurrentIndex(idx_sapphire)

        self.cb_sub.setToolTip("Select the same substrate used for measurement (internal tabulated dispersion).")

        g2.addWidget(self.cb_sub, r, 1)

        g2.setColumnStretch(1, 1)

        v.addWidget(box2)

        box3 = QGroupBox("3  What to fit on the spectrum (T, T/Tsub, R)")

        box3.setStyleSheet(gst)

        box3.setToolTip(

            "Step 3: define what the T column represents. Checked = T_film/T_bare_sub ratio "

            "(and R/T_bare_sub if R), often in % (100 = ratio 1). Unchecked = absolute T and R. "

            "wT / wR weight RMSE when both channels are active."

        )

        g3 = QGridLayout(box3)

        r3 = 0

        self.chk_t = QCheckBox("Fit transmission T")

        self.chk_t.setChecked(True)

        self.chk_t.setToolTip("Include file T column in objective. Disable only when fitting R only.")

        g3.addWidget(self.chk_t, r3, 0, 1, 2)

        r3 += 1

        self.chk_trel = QCheckBox("T = T_film / T_substrate (ratio, e.g. % -> fraction)")

        self.chk_trel.setChecked(True)

        self.chk_trel.setToolTip(

            "Checked (usual case): T column is T_film / bare-substrate T ratio (backside included), same for R. "

            "Often provided in percent (100 = ratio 1). Fit compares against ratio model without dividing by T_sub again. "

            "Unchecked: columns are absolute transmission/reflection (or %). "

            "RMSE objective uses ln lambda weighting and optional mixed T/R loss."

        )

        self.chk_trel.toggled.connect(self._on_trel_plot_refresh)

        g3.addWidget(self.chk_trel, r3, 0, 1, 2)

        r3 += 1

        self.chk_r = QCheckBox("Fit reflection R (if R column exists)")

        self.chk_r.setToolTip("Requires an R column; combines T and R if both are enabled and wR > 0.")

        g3.addWidget(self.chk_r, r3, 0, 1, 2)

        r3 += 1

        lb_w = QLabel("Weights in MSE:")

        lb_w.setToolTip(

            "wT and wR weight T and R errors respectively (mixed mode). Use wR = 0 for T-only fitting."

        )

        g3.addWidget(lb_w, r3, 0)

        self.w_t = QDoubleSpinBox()

        self.w_t.setRange(0.0, 100.0)

        self.w_t.setValue(1.0)

        self.w_t.setToolTip("Relative weight of T error in global RMSE.")

        self.w_r = QDoubleSpinBox()

        self.w_r.setRange(0.0, 100.0)

        self.w_r.setValue(1.0)

        self.w_r.setToolTip("Relative weight of R error. Set to 0 to ignore R in fitting.")

        h_w = QHBoxLayout()

        h_w.addWidget(QLabel("wT"))

        h_w.addWidget(self.w_t)

        h_w.addWidget(QLabel("wR"))

        h_w.addWidget(self.w_r)

        hw = QWidget()

        hw.setLayout(h_w)

        g3.addWidget(hw, r3, 1)

        v.addWidget(box3)

        btn_rmse_win = create_styled_button("Spectral RMSE window (lambda)...", "secondary")

        btn_rmse_win.setToolTip(

            "Limits the wavelengths used in the optimization MSE/RMSE. "

            "The displayed spectrum remains complete; only points in the band count for the adjustment."

        )

        btn_rmse_win.clicked.connect(self._on_rmse_fit_window_dialog)

        v.addWidget(btn_rmse_win)

        box4 = QGroupBox("4  Sigma mesh, spectral weights and local budget")

        box4.setStyleSheet(gst)

        box4.setToolTip(

            "Step 4 (manual mode without auto-K): segment count in sigma=1/lambda, spectral RMSE setup, "

            "and global optimization budgets. If auto-K is enabled (Advanced), K min/max control knot growth."

        )

        g4 = QGridLayout(box4)

        r4 = 0

        # First row of group 4: control visible immediately (scroll / legacy screenshots).

        lb_prof = QLabel("n, ln k between sigma nodes:")

        lb_prof.setToolTip(

            "Fixed interpolation: cubic spline in sigma between nodes (not-a-knot if K>=4, otherwise internal linear fallback). "

            "The solver optimizes values at the nodes; the forward model is this spline. "

            "The final spectral polish recalculates d, n, L on the same mesh with the same masked objective."

        )

        g4.addWidget(lb_prof, r4, 0)

        self.lbl_nk_profile_fixed = QLabel("Spline cubique en sigma (unique)")

        self.lbl_nk_profile_fixed.setMinimumWidth(200)

        self.lbl_nk_profile_fixed.setToolTip(lb_prof.toolTip())

        g4.addWidget(self.lbl_nk_profile_fixed, r4, 1)

        r4 += 1

        lbl_mod = QLabel(

            "Model: n and L = ln k at sigma = 1/lambda nodes (k = e^L). Between nodes: cubic spline in sigma only."

        )

        lbl_mod.setWordWrap(True)

        lbl_mod.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 10px;")

        lbl_mod.setToolTip(

            "The optimizer adjusts values at the nodes; the theoretical spectrum uses the cubic sigma-spline.\n"

            "Logs / Excel: sigma-mesh polish RMSE (sigma-spline) vs solver reference before polish."

        )

        g4.addWidget(lbl_mod, r4, 0, 1, 2)

        r4 += 1

        # --- Local budget + uncertainty: full panel (hidden in simplified interface) ---

        self._w_full_adv = QWidget()

        v_adv = QVBoxLayout(self._w_full_adv)

        v_adv.setContentsMargins(0, 0, 0, 0)

        v_adv.setSpacing(6)

        lb_pg = QLabel("Local iterations hint (max):")

        lb_pg.setToolTip("Legacy control kept for compatibility; inactive in local-only INDEX-SPLINE mode.")

        self.sp_pg_iter = QSpinBox()

        self.sp_pg_iter.setRange(5, 120)

        self.sp_pg_iter.setValue(
            int(SPLINE_PERF_PRESETS.get("fast", {}).get("pglobal_max_iter", 35) or 35)
        )

        self.sp_pg_iter.setToolTip(

            "Inactive in local-only mode; kept only for preset/config compatibility."

        )

        self.sp_pg_iter.setEnabled(False)

        row_pg = QHBoxLayout()

        row_pg.addWidget(lb_pg)

        row_pg.addWidget(self.sp_pg_iter, 1)

        v_adv.addLayout(row_pg)

        lb_s3p1 = QLabel("SOL3 — L-BFGS-B phase 1 maxfun:")

        lb_s3p1.setToolTip(

            "Evaluation budget (maxfun) for the primary L-BFGS-B descent with free sigma knots "

            "(SOL3 stage). Increase if logs indicate 'STOP: TOTAL NO. OF F.G EVALUATIONS EXCEEDS LIMIT' "

            "in phase 1. Stored in Qt preferences. Effective solver floor: 300; legacy default 10000."

        )

        self.sp_sol3_p1_maxfun = QSpinBox()

        self.sp_sol3_p1_maxfun.setRange(500, 500000)

        self.sp_sol3_p1_maxfun.setSingleStep(500)

        self.sp_sol3_p1_maxfun.setValue(10000)

        self.sp_sol3_p1_maxfun.setToolTip(lb_s3p1.toolTip())

        row_s3p1 = QHBoxLayout()

        row_s3p1.addWidget(lb_s3p1)

        row_s3p1.addWidget(self.sp_sol3_p1_maxfun, 1)

        v_adv.addLayout(row_s3p1)

        lb_pr = QLabel("Performance profile:")

        lb_pr.setToolTip(

            "Budget preset: polish budget and local searches. "

            "\"Maximal\" gives best quality at the expense of runtime."

        )

        self.cb_profilee = QComboBox()

        for lab, key in [

            ("Fast", "fast"),

            ("Standard", "standard"),

            ("Quality", "quality"),

            ("Maximal", "max"),

        ]:

            self.cb_profilee.addItem(lab, key)

        self.cb_profilee.setCurrentIndex(0)

        self.cb_profilee.setToolTip(

            "When the profile changes, a recommended local budget may be applied automatically to the spin."

        )

        self.cb_profilee.currentIndexChanged.connect(self._on_profilee_changed)

        row_pf = QHBoxLayout()

        row_pf.addWidget(lb_pr)

        row_pf.addWidget(self.cb_profilee, 1)

        v_adv.addLayout(row_pf)

        lb_mesh_dlam = QLabel("Pas min. Deltalambda/lambdā (maillage sigma) :")

        lb_mesh_dlam.setToolTip(

            "Contrainte sur le maillage canonique : min(Deltalambda entre nœuds) / lambdā >= cette valeur, "

            "with lambdā = (lambda_min + lambda_max) / 2 from file. Number of segments is reduced if needed; "

            "IR extension (+2 knots) is omitted if it violates threshold.\n"

            "0 = disabled (nominal behavior without this constraint)."

        )

        self.sp_mesh_min_dlam = QDoubleSpinBox()

        self.sp_mesh_min_dlam.setRange(0.0, 0.5)

        self.sp_mesh_min_dlam.setDecimals(4)

        self.sp_mesh_min_dlam.setSingleStep(0.0025)

        self.sp_mesh_min_dlam.setValue(0.02)

        self.sp_mesh_min_dlam.setSpecialValueText("disabled")

        self.sp_mesh_min_dlam.setToolTip(lb_mesh_dlam.toolTip())

        row_mesh_dlam = QHBoxLayout()

        row_mesh_dlam.addWidget(lb_mesh_dlam)

        row_mesh_dlam.addWidget(self.sp_mesh_min_dlam, 1)

        v_adv.addLayout(row_mesh_dlam)

        lb_cor = QLabel("Corridors n/k (d profiling):")

        lb_cor.setToolTip(

            "Calculates a plausible thickness interval and n(lambda), k(lambda) corridors by fixing d, then re-optimizing\n"

            "the n and ln k nodes (same penalties and masked RMSE as the fit).\n\n"

            "RMSE_ref+Delta mode (default): acceptance RMSE <= best polished spectral RMSE + Delta; the nominal 'best' curve\n"

            "is a native member of the envelope (not a pseudo-CI centered on a heuristic refit).\n"

            "Mode alpha : RMSE(d) <= alpha × RMSE_opt (heuristique).\n\n"

            "Enabled by default at end of optimization; results in 'Corridors n/k' tab."

        )

        self.chk_corridor_d = QCheckBox("Enable")

        self.chk_corridor_d.setChecked(True)

        self.chk_corridor_d.setToolTip(lb_cor.toolTip())

        row_cd = QHBoxLayout()

        row_cd.addWidget(lb_cor)

        row_cd.addWidget(self.chk_corridor_d)

        self.lbl_corridors_state_adv = QLabel()

        self.lbl_corridors_state_adv.setTextFormat(Qt.TextFormat.RichText)

        self.lbl_corridors_state_adv.setToolTip(

            "Read-only: same state as the 'Corridors' button under Run (Yes = computation at end of optimization)."

        )

        row_cd.addWidget(self.lbl_corridors_state_adv)

        row_cd.addStretch(1)

        v_adv.addLayout(row_cd)

        row_cor = QHBoxLayout()

        self.cb_corr_mode = QComboBox()

        self.cb_corr_mode.addItem("Heuristic (alpha×RMSE_opt)", "alpha")

        self.cb_corr_mode.addItem("RMSE_ref + Delta (absolute)", "abs_delta")

        self.cb_corr_mode.addItem("Likelihood ratio (Deltaχ²) - sigma constant or residual", "lr")

        _iad = self.cb_corr_mode.findData("abs_delta")

        self.cb_corr_mode.setCurrentIndex(int(_iad) if _iad >= 0 else 0)

        self.cb_corr_mode.setToolTip(

            "alpha : RMSE(d) <= alpha×RMSE_opt (heuristic).\n"

            "RMSE_ref+Delta: RMSE(d) <= RMSE_ref + Delta (same spectral mask). With 'best RMSE' checked, RMSE_ref = "

            "spectral_rmse_best_value (best polish) ; otherwise base curves from dict.\n"

            "LR: Deltaχ² <= χ²(1,conf); constant sigma or sigma_i(lambda) ∝ |residual| if 'sigma(lambda) residual'."

        )

        row_cor.addWidget(QLabel("mode"))

        row_cor.addWidget(self.cb_corr_mode)

        self.cb_corr_mode.currentIndexChanged.connect(self._on_corr_mode_changed)

        self.sp_corr_alpha = QDoubleSpinBox()

        self.sp_corr_alpha.setDecimals(3)

        self.sp_corr_alpha.setRange(1.000, 2.000)

        self.sp_corr_alpha.setSingleStep(0.005)

        self.sp_corr_alpha.setValue(1.05)

        self.sp_corr_alpha.setToolTip("alpha: RMSE threshold = alpha × RMSE_opt (e.g. 1.05 = +5%).")

        self.lbl_corr_alpha = QLabel("alpha")

        row_cor.addWidget(self.lbl_corr_alpha)

        row_cor.addWidget(self.sp_corr_alpha)

        self.lbl_corr_rmse_delta = QLabel("Delta RMSE abs.")

        self.sp_corr_rmse_delta = QDoubleSpinBox()

        self.sp_corr_rmse_delta.setDecimals(4)

        self.sp_corr_rmse_delta.setRange(0.0001, 0.05)

        self.sp_corr_rmse_delta.setSingleStep(0.0005)

        self.sp_corr_rmse_delta.setValue(0.001)

        self.sp_corr_rmse_delta.setToolTip(

            "Absolute margin on masked spectral RMSE: a refit at fixed d is accepted if "

            "RMSE <= RMSE_ref + Delta (default 1e-3 = best RMSE + 0.001). "

            "With best polished RMSE, RMSE_ref is the one of the exported model."

        )

        row_cor.addWidget(self.lbl_corr_rmse_delta)

        row_cor.addWidget(self.sp_corr_rmse_delta)

        self.chk_corr_scientific_nominal = QCheckBox("best RMSE")

        self.chk_corr_scientific_nominal.setChecked(True)

        self.chk_corr_scientific_nominal.setToolTip(

            "Mode corridor scientifique (uniquement si mode = RMSE_ref+Delta) : RMSE_ref = spectral_rmse_best_value ; "

            "nominal curves and nodes aligned on best polished model; no envelope widening toward "

            "main solver curve. Uncheck for legacy abs_delta behavior on 'base' curves only."

        )

        row_cor.addWidget(self.chk_corr_scientific_nominal)

        self.sp_corr_conf = QDoubleSpinBox()

        self.sp_corr_conf.setDecimals(3)

        self.sp_corr_conf.setRange(0.50, 0.999)

        self.sp_corr_conf.setSingleStep(0.01)

        self.sp_corr_conf.setValue(0.95)

        self.sp_corr_conf.setToolTip("LR confidence level (df=1): e.g. 0.95 -> Deltaχ²~3.84.")

        row_cor.addSpacing(8)

        row_cor.addWidget(QLabel("conf"))

        row_cor.addWidget(self.sp_corr_conf)

        self.sp_corr_sigma = QDoubleSpinBox()

        self.sp_corr_sigma.setDecimals(6)

        self.sp_corr_sigma.setRange(0.0, 1.0)

        self.sp_corr_sigma.setSingleStep(0.001)

        self.sp_corr_sigma.setValue(0.0)

        self.sp_corr_sigma.setToolTip(

            "Constant sigma (T and R) in fraction units (not %). 0 = auto (sigma := RMSE_opt)."

        )

        row_cor.addSpacing(8)

        row_cor.addWidget(QLabel("sigma"))

        row_cor.addWidget(self.sp_corr_sigma)

        self.sp_corr_step = QDoubleSpinBox()

        self.sp_corr_step.setDecimals(2)

        self.sp_corr_step.setRange(0.1, 50.0)

        self.sp_corr_step.setSingleStep(0.5)

        self.sp_corr_step.setValue(1.0)

        self.sp_corr_step.setToolTip("d continuation step size (nm).")

        row_cor.addSpacing(8)

        row_cor.addWidget(QLabel("step (nm)"))

        row_cor.addWidget(self.sp_corr_step)

        self.sp_corr_span = QDoubleSpinBox()

        self.sp_corr_span.setDecimals(1)

        self.sp_corr_span.setRange(1.0, 2000.0)

        self.sp_corr_span.setSingleStep(1.0)

        self.sp_corr_span.setValue(15.0)

        self.sp_corr_span.setToolTip(

            "Maximum offset |d - d_opt| explored in each direction (+d and -d), in nm (not the sum). "

            "Default 15 nm ~ local neighborhood around the optimal thickness."

        )

        row_cor.addSpacing(8)

        row_cor.addWidget(QLabel("span (nm)"))

        row_cor.addWidget(self.sp_corr_span)

        # Multi-start (V2.2): robustness to local minima during fixed-d refit.

        self.sp_corr_starts = QSpinBox()

        self.sp_corr_starts.setRange(1, 25)

        self.sp_corr_starts.setValue(1)

        self.sp_corr_starts.setToolTip(

            "Number of initializations (multi-start) per d value. 1 = continuation only (fast). "

            ">1 increases robustness (best solution kept), at the cost of computation time."

        )

        row_cor.addSpacing(10)

        row_cor.addWidget(QLabel("starts"))

        row_cor.addWidget(self.sp_corr_starts)

        self.sp_corr_jn = QDoubleSpinBox()

        self.sp_corr_jn.setDecimals(3)

        self.sp_corr_jn.setRange(0.0, 1.0)

        self.sp_corr_jn.setSingleStep(0.01)

        self.sp_corr_jn.setValue(0.02)

        self.sp_corr_jn.setToolTip("Gaussian jitter sigma on n (or ξ if monotonicity active) for additional starts.")

        row_cor.addSpacing(6)

        row_cor.addWidget(QLabel("j_n"))

        row_cor.addWidget(self.sp_corr_jn)

        self.sp_corr_jL = QDoubleSpinBox()

        self.sp_corr_jL.setDecimals(3)

        self.sp_corr_jL.setRange(0.0, 5.0)

        self.sp_corr_jL.setSingleStep(0.05)

        self.sp_corr_jL.setValue(0.15)

        self.sp_corr_jL.setToolTip("Gaussian jitter sigma on L=ln k for additional starts.")

        row_cor.addSpacing(6)

        row_cor.addWidget(QLabel("j_L"))

        row_cor.addWidget(self.sp_corr_jL)

        self.sp_corr_seed = QSpinBox()

        self.sp_corr_seed.setRange(-2**31, 2**31 - 1)

        self.sp_corr_seed.setValue(0)

        self.sp_corr_seed.setToolTip("RNG seed for multi-start reproducibility (jitter).")

        row_cor.addSpacing(6)

        row_cor.addWidget(QLabel("seed"))

        row_cor.addWidget(self.sp_corr_seed)

        # V2.3: ln(k) regularization sensitivity scan (d2(L)^2 weight).

        self.chk_corr_reg_sens = QCheckBox("scan reg")

        self.chk_corr_reg_sens.setChecked(False)

        self.chk_corr_reg_sens.setToolTip(

            "Runs a scan (log grid) of the ln(k) regularization weight and re-launches d profiling for each value.\n"

            "Goal: verify the robustness of the d interval and n/k corridors to regularization choices."

        )

        row_cor.addSpacing(10)

        row_cor.addWidget(self.chk_corr_reg_sens)

        self.sp_corr_reg_pts = QSpinBox()

        self.sp_corr_reg_pts.setRange(2, 15)

        self.sp_corr_reg_pts.setValue(5)

        self.sp_corr_reg_pts.setToolTip("Number of points in the regularization scan log grid.")

        row_cor.addSpacing(6)

        row_cor.addWidget(QLabel("pts"))

        row_cor.addWidget(self.sp_corr_reg_pts)

        self.sp_corr_reg_dec = QSpinBox()

        self.sp_corr_reg_dec.setRange(0, 6)

        self.sp_corr_reg_dec.setValue(2)

        self.sp_corr_reg_dec.setToolTip("Number of decades on each side of the base weight (lnk_spline_reg_weight).")

        row_cor.addSpacing(6)

        row_cor.addWidget(QLabel("dec"))

        row_cor.addWidget(self.sp_corr_reg_dec)

        # V2.4: parametric bootstrap for publication-level bands

        self.chk_corr_boot = QCheckBox("bootstrap")

        self.chk_corr_boot.setChecked(False)

        self.chk_corr_boot.setToolTip(

            "Parametric bootstrap: generates B T/R datasets by adding Gaussian noise (sigma_T, sigma_R),\n"

            "re-launches d profiling for each replication, then computes percentile bands on n(lambda), k(lambda)\n"

            "and a distribution of the d interval."

        )

        row_cor.addSpacing(10)

        row_cor.addWidget(self.chk_corr_boot)

        self.sp_corr_boot_n = QSpinBox()

        self.sp_corr_boot_n.setRange(5, 500)

        self.sp_corr_boot_n.setValue(40)

        self.sp_corr_boot_n.setToolTip("Number of bootstrap replications (B). Larger = more robust, but slower.")

        row_cor.addSpacing(6)

        row_cor.addWidget(QLabel("B"))

        row_cor.addWidget(self.sp_corr_boot_n)

        self.sp_corr_boot_p = QDoubleSpinBox()

        self.sp_corr_boot_p.setDecimals(3)

        self.sp_corr_boot_p.setRange(0.50, 0.999)

        self.sp_corr_boot_p.setSingleStep(0.01)

        self.sp_corr_boot_p.setValue(0.95)

        self.sp_corr_boot_p.setToolTip("Central percentile (e.g. 0.95 => bounds 2.5% / 97.5%).")

        row_cor.addSpacing(6)

        row_cor.addWidget(QLabel("p"))

        row_cor.addWidget(self.sp_corr_boot_p)

        self.sp_corr_boot_seed = QSpinBox()

        self.sp_corr_boot_seed.setRange(-2**31, 2**31 - 1)

        self.sp_corr_boot_seed.setValue(0)

        self.sp_corr_boot_seed.setToolTip("Seed RNG bootstrap.")

        row_cor.addSpacing(6)

        row_cor.addWidget(QLabel("seedB"))

        row_cor.addWidget(self.sp_corr_boot_seed)

        self.cb_corr_boot_mode = QComboBox()

        self.cb_corr_boot_mode.addItem("parametric (T/R + N(0,sigma))", "parametric")

        self.cb_corr_boot_mode.addItem("residual (T_th + residuals*)", "residual")

        self.cb_corr_boot_mode.setCurrentIndex(0)

        self.cb_corr_boot_mode.setToolTip(

            "parametric: adds Gaussian noise to measurements.\n"

            "residual: non-parametric bootstrap on residuals (more realistic if noise is non-Gaussian / correlated)."

        )

        row_cor.addSpacing(8)

        row_cor.addWidget(self.cb_corr_boot_mode)

        self.sp_corr_boot_block = QSpinBox()

        self.sp_corr_boot_block.setRange(1, 5000)

        self.sp_corr_boot_block.setValue(1)

        self.sp_corr_boot_block.setToolTip("Block length (in lambda points) for residual bootstrap. 1 = iid.")

        row_cor.addSpacing(6)

        row_cor.addWidget(QLabel("blk"))

        row_cor.addWidget(self.sp_corr_boot_block)

        self._on_corr_mode_changed()

        w_cor = QWidget()

        w_cor.setLayout(row_cor)

        v_adv.addWidget(w_cor)

        row_cor_prof = QHBoxLayout()

        self.sp_corr_prof_maxfun = QSpinBox()

        self.sp_corr_prof_maxfun.setRange(0, 200000)

        self.sp_corr_prof_maxfun.setSingleStep(500)

        self.sp_corr_prof_maxfun.setValue(2500)

        self.sp_corr_prof_maxfun.setToolTip(

            "L-BFGS-B budget (maxfun) for each refit of n, ln k nodes at fixed d during corridor profiling.\n"

            "0 = reuse the main run polish_maxfun (often 8000+, very slow per step).\n"

            "Typ. 1500-4000 for a local scan; increase if 'EXCEEDS LIMIT' messages or poor refits."

        )

        row_cor_prof.addWidget(QLabel("d profiling: maxfun / refit"))

        row_cor_prof.addWidget(self.sp_corr_prof_maxfun)

        row_cor_prof.addStretch(1)

        w_cor_prof = QWidget()

        w_cor_prof.setLayout(row_cor_prof)

        v_adv.addWidget(w_cor_prof)

        row_cor_v25 = QHBoxLayout()

        self.chk_corr_sigma_hetero = QCheckBox("sigma(lambda) residual (LR + param. boot.)")

        self.chk_corr_sigma_hetero.setChecked(False)

        self.chk_corr_sigma_hetero.setToolTip(

            "In LR mode: χ² with sigma_i = max(floor, scale×|y_exp-y_th|) on the objective grid.\n"

            "Parametric bootstrap: same sigma_i for Gaussian noise on T/R (objective points only)."

        )

        row_cor_v25.addWidget(self.chk_corr_sigma_hetero)

        self.sp_corr_hetero_scale = QDoubleSpinBox()

        self.sp_corr_hetero_scale.setDecimals(3)

        self.sp_corr_hetero_scale.setRange(0.0, 20.0)

        self.sp_corr_hetero_scale.setSingleStep(0.05)

        self.sp_corr_hetero_scale.setValue(1.0)

        self.sp_corr_hetero_scale.setToolTip("Scale factor on |residual| for sigma_i(lambda) (0 = floor only).")

        row_cor_v25.addSpacing(6)

        row_cor_v25.addWidget(QLabel("scale sigma(lambda)"))

        row_cor_v25.addWidget(self.sp_corr_hetero_scale)

        self.chk_corr_boot_refit = QCheckBox("refit rapide par tirage (bootstrap)")

        self.chk_corr_boot_refit.setChecked(False)

        self.chk_corr_boot_refit.setToolTip(

            "After each bootstrap trial: a short L-BFGS-B on (d + nodes) using noisy T/R, "

            "same spectral objective as main run (n,L interp. in sigma = cubic spline), "

            "then profiling in d from this refit (often more consistent than freezing initial mesh)."

        )

        row_cor_v25.addSpacing(12)

        row_cor_v25.addWidget(self.chk_corr_boot_refit)

        self.sp_corr_boot_maxfun = QSpinBox()

        self.sp_corr_boot_maxfun.setRange(0, 200000)

        self.sp_corr_boot_maxfun.setValue(4000)

        self.sp_corr_boot_maxfun.setToolTip("L-BFGS-B maxfun budget per bootstrap refit (0 = disabled even if box is checked).")

        row_cor_v25.addSpacing(6)

        row_cor_v25.addWidget(QLabel("maxfun"))

        row_cor_v25.addWidget(self.sp_corr_boot_maxfun)

        self.sp_corr_boot_workers = QSpinBox()

        self.sp_corr_boot_workers.setRange(1, 64)

        self.sp_corr_boot_workers.setValue(1)

        self.sp_corr_boot_workers.setToolTip(

            "Number of parallel processes for bootstrap replications (1 = sequential). "

            f"Typ. 2-{max(2, min(8, (multiprocessing.cpu_count() or 4)))} on this machine "

            f"({multiprocessing.cpu_count() or '?'} cores). "

            "Pickle or worker failure -> automatic fallback to sequential."

        )

        row_cor_v25.addSpacing(10)

        row_cor_v25.addWidget(QLabel("proc."))

        row_cor_v25.addWidget(self.sp_corr_boot_workers)

        row_cor_v25.addStretch(1)

        w_cor2 = QWidget()

        w_cor2.setLayout(row_cor_v25)

        v_adv.addWidget(w_cor2)

        page_full_adv = QWidget()

        self._box4_full_adv_layout = QVBoxLayout(page_full_adv)

        self._box4_full_adv_layout.setContentsMargins(0, 0, 0, 0)

        self._box4_full_adv_layout.addWidget(self._w_full_adv)

        page_epure = QWidget()

        lay_ep = QVBoxLayout(page_epure)

        lay_ep.setContentsMargins(0, 4, 0, 0)

        lbl_ep = QLabel(

            "<b>Robustness</b> - Default: 'Fast' profile and <b>n/k corridors</b> (d profiling) "

            "<b>enabled</b> after the fit. <b>Bootstrap</b> and regularization scan remain optional "

            "(advanced settings).<br><br>"

            "<span style='color:#888;font-size:10px;'>Local budget, profiling / bootstrap details: advanced settings.</span>"

        )

        lbl_ep.setWordWrap(True)

        lbl_ep.setToolTip(

            "Corridors: n/k envelopes and d interval after optimization (enabled by default). "

            "Bootstrap and heavy options: advanced settings."

        )

        lay_ep.addWidget(lbl_ep)

        btn_open_adv = create_styled_button("Advanced settings...", "secondary")

        btn_open_adv.setToolTip("Optimization budgets and detailed uncertainty / corridor options.")

        btn_open_adv.clicked.connect(self._open_advanced_settings_dialog)

        lay_ep.addWidget(btn_open_adv)

        lay_ep.addStretch(1)

        self._stack_box4_adv = QStackedWidget()

        self._stack_box4_adv.addWidget(page_epure)

        self._stack_box4_adv.addWidget(page_full_adv)

        g4.addWidget(self._stack_box4_adv, r4, 0, 1, 2)

        r4 += 1

        g4.setColumnStretch(1, 1)

        v.addWidget(box4)

        v.addStretch(1)

        return w

    def _build_plot_tabs_panel(self) -> QWidget:

        """Panneau droit type Swanepoel : barre detachage + onglets graphiques."""

        self.tabs_main = QTabWidget()

        self.tabs_main.addTab(self._build_tab_spectrum(), "Spectrum T / R")

        self.tabs_main.addTab(self._build_tab_indices(), "n & log₁₀ k")

        self._tab_corridor_panel = self._build_tab_corridor()

        self.tabs_main.addTab(self._tab_corridor_panel, "Corridors n/k")

        self._idx_tab_corridor = self.tabs_main.indexOf(self._tab_corridor_panel)

        self._tab_corridor_rmse_panel = self._build_tab_corridor_rmse()

        self.tabs_main.addTab(self._tab_corridor_rmse_panel, "Corridor RMSE(d)")

        self._idx_tab_corridor_rmse = self.tabs_main.indexOf(self._tab_corridor_rmse_panel)

        self._tab_nl_panel = self._build_tab_nl()

        self.tabs_main.addTab(self._tab_nl_panel, "NL alpha")

        self._idx_tab_nl = self.tabs_main.indexOf(self._tab_nl_panel)

        self.tabs_main.addTab(self._build_tab_data(), "Data")

        self.tabs_main.addTab(self._build_tab_log(), "Log")

        self.tabs_main.addTab(self._build_tab_why(), "CERTUS")

        hdr = QWidget()

        hl = QHBoxLayout(hdr)

        hl.setContentsMargins(4, 2, 4, 2)

        detach_btn = QPushButton(" Detach plot")

        detach_btn.setFixedHeight(26)

        detach_btn.setToolTip(

            "Clone le first plot de longlet actif (tab a 2 graphes : aussi Ctrl+Shift+D sur le plot)."

        )

        detach_btn.clicked.connect(self._detach_current_plot)

        hl.addWidget(detach_btn)

        hl.addStretch(1)

        out = QWidget()

        vl = QVBoxLayout(out)

        vl.setContentsMargins(0, 0, 0, 0)

        vl.setSpacing(0)

        vl.addWidget(hdr)

        vl.addWidget(self.tabs_main, 1)

        return out

    def _build_tab_spectrum(self) -> QWidget:

        panel = QWidget()

        lay = QVBoxLayout(panel)

        lay.setContentsMargins(0, 0, 0, 0)

        row_axis = QHBoxLayout()

        row_axis.addWidget(QLabel("X Axis:"))

        self.cb_spectrum_xmode = QComboBox()

        self.cb_spectrum_xmode.addItem("Lambda (nm)", "lambda")

        self.cb_spectrum_xmode.addItem("Sigma (nm⁻1)", "sigma")

        self.cb_spectrum_xmode.addItem("Sigma2 (nm⁻2)", "sigma2")

        self.cb_spectrum_xmode.currentIndexChanged.connect(self._on_spectrum_x_mode_changed)

        row_axis.addWidget(self.cb_spectrum_xmode)

        row_axis.addStretch(1)

        lay.addLayout(row_axis)

        self.plot_T = CertusScientificPlot(

            title="Spectrum", y_label="T, R or T/T_sub", x_label="lambda (nm)"

        )

        self.plot_T.showGrid(x=True, y=True, alpha=0.25)

        lay.addWidget(wrap_scientific_plot_with_toolbar(self, self.plot_T), 1)

        return panel

    def _build_tab_indices(self) -> QWidget:

        panel = QWidget()

        lay = QVBoxLayout(panel)

        lay.setContentsMargins(0, 0, 0, 0)

        self.plot_n = CertusScientificPlot(title="n(lambda)", y_label="n", x_label="lambda (nm)")

        self.plot_n.showGrid(x=True, y=True, alpha=0.25)

        self.plot_lgk = CertusScientificPlot(

            title="log₁₀ k(lambda)", y_label="log₁₀ k", x_label="lambda (nm)"

        )

        self.plot_lgk.showGrid(x=True, y=True, alpha=0.25)

        spl = QSplitter(Qt.Orientation.Vertical)

        spl.addWidget(wrap_scientific_plot_with_toolbar(self, self.plot_n))

        spl.addWidget(wrap_scientific_plot_with_toolbar(self, self.plot_lgk))

        spl.setStretchFactor(0, 1)

        spl.setStretchFactor(1, 1)

        lay.addWidget(spl, 1)

        return panel

    def _build_tab_corridor_rmse(self) -> QWidget:

        """Tab for corridor profile RMSE as a function of thickness d."""

        panel = QWidget()

        lay = QVBoxLayout(panel)

        lay.setContentsMargins(6, 6, 6, 6)

        hint = QLabel(

            "<b>RMSE(d) corridor profile</b> — points are evaluated during corridor profiling. "

            "The <b>green marker</b> and dashed vertical line indicate the best computed thickness "

            "(<i>d*</i>, minimum RMSE among sampled points)."

        )

        hint.setWordWrap(True)

        hint.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 11px;")

        lay.addWidget(hint)

        self.lbl_corridor_rmse_summary = QLabel("No corridor RMSE profile available yet.")

        self.lbl_corridor_rmse_summary.setStyleSheet(f"color: {CertusTheme.TEXT_MAIN}; font-size: 11px;")

        lay.addWidget(self.lbl_corridor_rmse_summary)

        row_rob = QHBoxLayout()

        row_rob.addWidget(QLabel("Robust ΔRMSE:"))

        self.sp_corridor_rmse_delta = QDoubleSpinBox()

        self.sp_corridor_rmse_delta.setDecimals(6)

        self.sp_corridor_rmse_delta.setRange(1e-6, 0.01)

        self.sp_corridor_rmse_delta.setSingleStep(1e-5)

        self.sp_corridor_rmse_delta.setValue(2e-4)

        self.sp_corridor_rmse_delta.setToolTip(

            "Target increment above RMSE(d*) for robust interval proposal via local quadratic fit."

        )

        row_rob.addWidget(self.sp_corridor_rmse_delta)

        row_rob.addWidget(QLabel("Local half-window (points):"))

        self.sp_corridor_rmse_win = QSpinBox()

        self.sp_corridor_rmse_win.setRange(2, 8)

        self.sp_corridor_rmse_win.setValue(3)

        self.sp_corridor_rmse_win.setToolTip(

            "Number of sampled points on each side of d* used in local quadratic robust fit."

        )

        row_rob.addWidget(self.sp_corridor_rmse_win)

        row_rob.addStretch(1)

        lay.addLayout(row_rob)

        row_manual = QHBoxLayout()

        row_manual.addWidget(QLabel("Manual centered corridor (± nm):"))

        self.sl_corridor_manual_half = QSlider(Qt.Orientation.Horizontal)

        self.sl_corridor_manual_half.setRange(0, 1)

        self.sl_corridor_manual_half.setValue(0)

        self.sl_corridor_manual_half.setSingleStep(1)

        self.sl_corridor_manual_half.setPageStep(5)

        self.sl_corridor_manual_half.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.sl_corridor_manual_half.setEnabled(False)

        self.sl_corridor_manual_half.setToolTip(

            "Centered half-width around d* used to define a manual corridor interval before regeneration."

        )

        self.sl_corridor_manual_half.setStyleSheet(

            "QSlider::groove:horizontal { height: 8px; background: #d7deea; border-radius: 4px; }"

            "QSlider::sub-page:horizontal { background: #7a3cff; border-radius: 4px; }"

            "QSlider::add-page:horizontal { background: #eef2f8; border-radius: 4px; }"

            "QSlider::handle:horizontal { width: 14px; margin: -4px 0; border-radius: 7px; background: #ff4d4f; }"

        )

        self.sl_corridor_manual_half.valueChanged.connect(self._on_corridor_manual_slider_changed)

        row_manual.addWidget(self.sl_corridor_manual_half, 1)

        self.lbl_corridor_manual_half = QLabel("±0.00 nm")

        self.lbl_corridor_manual_half.setMinimumWidth(90)

        row_manual.addWidget(self.lbl_corridor_manual_half)

        self.btn_corridor_manual_robust = create_styled_button("Use robust interval", "secondary", parent=self)

        self.btn_corridor_manual_robust.setEnabled(False)

        self.btn_corridor_manual_robust.setToolTip(

            "Sets the manual centered corridor width from the current robust interval proposal."

        )

        self.btn_corridor_manual_robust.clicked.connect(self._use_robust_corridor_interval)

        row_manual.addWidget(self.btn_corridor_manual_robust)

        self.btn_generate_manual_corridor = create_styled_button("Generate corridor", "primary", parent=self)

        self.btn_generate_manual_corridor.setEnabled(False)

        self.btn_generate_manual_corridor.setToolTip(

            "Regenerates the visual n/k corridor and the data table from the selected centered interval."

        )

        self.btn_generate_manual_corridor.clicked.connect(self._apply_manual_corridor_selection)

        row_manual.addWidget(self.btn_generate_manual_corridor)

        lay.addLayout(row_manual)

        row_manual_meta = QHBoxLayout()

        self.lbl_corridor_manual_dmin = QLabel("d_min: -")

        self.lbl_corridor_manual_dcenter = QLabel("d*: -")

        self.lbl_corridor_manual_dmax = QLabel("d_max: -")

        self.lbl_corridor_manual_interval = QLabel("Manual interval: -")

        for _lab in (

            self.lbl_corridor_manual_dmin,

            self.lbl_corridor_manual_dcenter,

            self.lbl_corridor_manual_dmax,

            self.lbl_corridor_manual_interval,

        ):

            _lab.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 11px;")

        row_manual_meta.addWidget(self.lbl_corridor_manual_dmin)

        row_manual_meta.addSpacing(8)

        row_manual_meta.addWidget(self.lbl_corridor_manual_dcenter)

        row_manual_meta.addSpacing(8)

        row_manual_meta.addWidget(self.lbl_corridor_manual_dmax)

        row_manual_meta.addStretch(1)

        row_manual_meta.addWidget(self.lbl_corridor_manual_interval)

        lay.addLayout(row_manual_meta)

        self.plot_corridor_rmse_d = CertusScientificPlot(

            title="Corridor profile: RMSE(d)", y_label="RMSE", x_label="d (nm)"

        )

        self.plot_corridor_rmse_d.showGrid(x=True, y=True, alpha=0.25)

        self._corridor_rmse_d_vals = np.array([], dtype=np.float64)

        self._corridor_rmse_vals = np.array([], dtype=np.float64)

        self._corridor_rmse_best_idx = -1

        self._corridor_rmse_robust_lo = float("nan")

        self._corridor_rmse_robust_hi = float("nan")

        self._corridor_rmse_robust_ok = False

        self.sp_corridor_rmse_delta.valueChanged.connect(self._refresh_corridor_rmse_robust_view)

        self.sp_corridor_rmse_win.valueChanged.connect(self._refresh_corridor_rmse_robust_view)

        _scene = self.plot_corridor_rmse_d.plotItem.scene()

        if _scene is not None:

            _scene.sigMouseClicked.connect(self._on_corridor_rmse_plot_clicked)

        lay.addWidget(wrap_scientific_plot_with_toolbar(self, self.plot_corridor_rmse_d), 1)

        return panel

    def _refresh_corridor_rmse_robust_view(self) -> None:

        src = self._corridor_profile_source_result()

        if src is None:

            return

        try:

            self._plot_corridor_rmse_tab(src)

        except Exception:

            logger.debug("Corridor RMSE robust refresh failed", exc_info=True)

    def _corridor_profile_source_result(self) -> dict[str, Any] | None:

        for cand in (self._last_result, self._last_worker_result):

            if not isinstance(cand, dict):

                continue

            d_prof = np.asarray(cand.get("profile_d_values_nm", []), dtype=np.float64).ravel()

            r_prof = np.asarray(cand.get("profile_d_rmse_values", []), dtype=np.float64).ravel()

            if d_prof.size > 0 and r_prof.size == d_prof.size:

                return cand

        return self._last_result

    def _corridor_manual_half_width_nm(self) -> float:

        if not hasattr(self, "sl_corridor_manual_half"):

            return 0.0

        scale = max(1, int(getattr(self, "_corridor_rmse_manual_slider_scale", 100) or 100))

        return float(self.sl_corridor_manual_half.value()) / float(scale)

    def _set_corridor_manual_interval_preview(self, d_best: float, half_width_nm: float) -> None:

        hw = float(max(0.0, half_width_nm))

        self._corridor_rmse_manual_lo = float(d_best - hw)

        self._corridor_rmse_manual_hi = float(d_best + hw)

        if hasattr(self, "lbl_corridor_manual_half"):

            self.lbl_corridor_manual_half.setText(f"±{hw:.2f} nm")

        if hasattr(self, "lbl_corridor_manual_interval"):

            self.lbl_corridor_manual_interval.setText(

                f"Manual interval: [{self._corridor_rmse_manual_lo:.2f}, {self._corridor_rmse_manual_hi:.2f}] nm"

            )

    def _set_corridor_manual_bounds_labels(self, d_min: float, d_best: float, d_max: float) -> None:

        if hasattr(self, "lbl_corridor_manual_dmin"):

            self.lbl_corridor_manual_dmin.setText(

                f"d_min: {d_min:.2f} nm" if np.isfinite(d_min) else "d_min: -"

            )

        if hasattr(self, "lbl_corridor_manual_dcenter"):

            self.lbl_corridor_manual_dcenter.setText(

                f"d*: {d_best:.2f} nm" if np.isfinite(d_best) else "d*: -"

            )

        if hasattr(self, "lbl_corridor_manual_dmax"):

            self.lbl_corridor_manual_dmax.setText(

                f"d_max: {d_max:.2f} nm" if np.isfinite(d_max) else "d_max: -"

            )

    def _reset_corridor_manual_controls(self) -> None:

        if hasattr(self, "sl_corridor_manual_half"):

            self.sl_corridor_manual_half.blockSignals(True)

            self.sl_corridor_manual_half.setRange(0, 1)

            self.sl_corridor_manual_half.setValue(0)

            self.sl_corridor_manual_half.setEnabled(False)

            self.sl_corridor_manual_half.blockSignals(False)

        if hasattr(self, "btn_generate_manual_corridor"):

            self.btn_generate_manual_corridor.setEnabled(False)

        if hasattr(self, "btn_corridor_manual_robust"):

            self.btn_corridor_manual_robust.setEnabled(False)

        if hasattr(self, "lbl_corridor_manual_half"):

            self.lbl_corridor_manual_half.setText("±0.00 nm")

        if hasattr(self, "lbl_corridor_manual_interval"):

            self.lbl_corridor_manual_interval.setText("Manual interval: -")

        self._set_corridor_manual_bounds_labels(float("nan"), float("nan"), float("nan"))

    def _sync_corridor_manual_controls(self, d_s: np.ndarray, i_best: int) -> None:

        if (

            not hasattr(self, "sl_corridor_manual_half")

            or not hasattr(self, "btn_generate_manual_corridor")

            or d_s.size == 0

            or i_best < 0

            or i_best >= int(d_s.size)

        ):

            self._reset_corridor_manual_controls()

            return

        d_best = float(d_s[i_best])

        self._set_corridor_manual_bounds_labels(float(d_s[0]), d_best, float(d_s[-1]))

        max_half = float(max(d_best - float(d_s[0]), float(d_s[-1]) - d_best, 0.0))

        scale = max(1, int(getattr(self, "_corridor_rmse_manual_slider_scale", 100) or 100))

        max_steps = max(1, int(round(max_half * scale)))

        cur_half = self._corridor_manual_half_width_nm()

        if not np.isfinite(cur_half) or cur_half <= 0.0:

            if np.isfinite(self._corridor_rmse_robust_lo) and np.isfinite(self._corridor_rmse_robust_hi):

                cur_half = 0.5 * max(0.0, float(self._corridor_rmse_robust_hi - self._corridor_rmse_robust_lo))

            elif d_s.size >= 2:

                cur_half = max(float(np.nanmedian(np.abs(np.diff(d_s)))), 0.0)

            else:

                cur_half = 0.0

        cur_half = float(min(max(cur_half, 0.0), max_half))

        self.sl_corridor_manual_half.blockSignals(True)

        self.sl_corridor_manual_half.setRange(0, max_steps)

        self.sl_corridor_manual_half.setTickInterval(max(1, max_steps // 8))

        self.sl_corridor_manual_half.setValue(int(round(cur_half * scale)))

        self.sl_corridor_manual_half.setEnabled(max_steps > 0)

        self.sl_corridor_manual_half.blockSignals(False)

        self.btn_generate_manual_corridor.setEnabled(True)

        if hasattr(self, "btn_corridor_manual_robust"):

            self.btn_corridor_manual_robust.setEnabled(bool(self._corridor_rmse_robust_ok))

        self._set_corridor_manual_interval_preview(d_best, cur_half)

    def _on_corridor_manual_slider_changed(self, _value: int) -> None:

        d_s = np.asarray(getattr(self, "_corridor_rmse_d_vals", []), dtype=np.float64).ravel()

        i_best = int(getattr(self, "_corridor_rmse_best_idx", -1))

        if d_s.size == 0 or i_best < 0 or i_best >= int(d_s.size):

            return

        self._set_corridor_manual_interval_preview(float(d_s[i_best]), self._corridor_manual_half_width_nm())

        src = self._corridor_profile_source_result()

        if src is not None:

            try:

                self._plot_corridor_rmse_tab(src)

            except Exception:

                logger.debug("Manual corridor slider refresh failed", exc_info=True)

    def _use_robust_corridor_interval(self) -> None:

        if not bool(getattr(self, "_corridor_rmse_robust_ok", False)):

            return

        d_s = np.asarray(getattr(self, "_corridor_rmse_d_vals", []), dtype=np.float64).ravel()

        i_best = int(getattr(self, "_corridor_rmse_best_idx", -1))

        if d_s.size == 0 or i_best < 0 or i_best >= int(d_s.size):

            return

        d_best = float(d_s[i_best])

        d_lo_rb = float(getattr(self, "_corridor_rmse_robust_lo", float("nan")))

        d_hi_rb = float(getattr(self, "_corridor_rmse_robust_hi", float("nan")))

        if not (np.isfinite(d_lo_rb) and np.isfinite(d_hi_rb) and d_hi_rb >= d_lo_rb):

            return

        half = 0.5 * float(max(0.0, d_hi_rb - d_lo_rb))

        scale = max(1, int(getattr(self, "_corridor_rmse_manual_slider_scale", 100) or 100))

        max_half = float(max(d_best - float(d_s[0]), float(d_s[-1]) - d_best, 0.0))

        half = float(min(max(0.0, half), max_half))

        if hasattr(self, "sl_corridor_manual_half"):

            self.sl_corridor_manual_half.setValue(int(round(half * scale)))

    def _build_manual_corridor_payload(

        self,

        source: dict[str, Any],

        display: dict[str, Any],

        d_lo_nm: float,

        d_hi_nm: float,

    ) -> dict[str, Any] | None:

        d_vals = np.asarray(source.get("profile_d_values_nm", []), dtype=np.float64).ravel()

        n_curves = np.asarray(source.get("profile_d_n_curves", []), dtype=np.float64)

        k_curves = np.asarray(source.get("profile_d_k_curves", []), dtype=np.float64)

        if d_vals.size == 0 or n_curves.ndim != 2 or k_curves.ndim != 2:

            return None

        if n_curves.shape[0] != d_vals.size or k_curves.shape[0] != d_vals.size:

            return None

        if n_curves.shape[1] == 0 or k_curves.shape[1] != n_curves.shape[1]:

            return None

        d_lo = float(min(d_lo_nm, d_hi_nm))

        d_hi = float(max(d_lo_nm, d_hi_nm))

        sel = np.isfinite(d_vals) & (d_vals >= d_lo - 1e-12) & (d_vals <= d_hi + 1e-12)

        if not np.any(sel):

            i_near = int(np.argmin(np.abs(d_vals - 0.5 * (d_lo + d_hi))))

            sel = np.zeros_like(d_vals, dtype=bool)

            sel[i_near] = True

        n_pick = np.asarray(n_curves[sel, :], dtype=np.float64)

        k_pick = np.asarray(k_curves[sel, :], dtype=np.float64)

        if n_pick.ndim != 2 or k_pick.ndim != 2 or n_pick.shape[0] == 0:

            return None

        n_lo = np.nanmin(n_pick, axis=0)

        n_hi = np.nanmax(n_pick, axis=0)

        k_lo = np.nanmin(k_pick, axis=0)

        k_hi = np.nanmax(k_pick, axis=0)

        ref_n = np.asarray(source.get("corridor_reference_n_lam", display.get("n_lam", [])), dtype=np.float64).ravel()

        ref_k = np.asarray(source.get("corridor_reference_k_lam", display.get("k_lam", [])), dtype=np.float64).ravel()

        if ref_n.size == n_lo.size and ref_k.size == k_lo.size:

            n_lo, n_hi, k_lo, k_hi = _expand_corridor_envelope_with_reported_nk(

                n_lo,

                n_hi,

                k_lo,

                k_hi,

                ref_n,

                ref_k,

            )

        d_sel = np.asarray(d_vals[sel], dtype=np.float64)

        return {

            "profile_d_enabled": True,

            "corridor_n_lo": np.asarray(n_lo, dtype=np.float64),

            "corridor_n_hi": np.asarray(n_hi, dtype=np.float64),

            "corridor_k_lo": np.asarray(k_lo, dtype=np.float64),

            "corridor_k_hi": np.asarray(k_hi, dtype=np.float64),

            "corridor_reference_n_lam": np.asarray(ref_n, dtype=np.float64),

            "corridor_reference_k_lam": np.asarray(ref_k, dtype=np.float64),

            "manual_corridor_active": True,

            "manual_corridor_interval_nm": (float(np.nanmin(d_sel)), float(np.nanmax(d_sel))),

            "manual_corridor_selected_count": int(d_sel.size),

        }

    def _apply_manual_corridor_selection(self) -> None:

        source = self._corridor_profile_source_result()

        display = self._last_result

        if not isinstance(source, dict) or not isinstance(display, dict):

            QMessageBox.information(self, "Generate corridor", "No corridor profile is available yet.")

            return

        d_s = np.asarray(source.get("profile_d_values_nm", []), dtype=np.float64).ravel()

        i_best = int(getattr(self, "_corridor_rmse_best_idx", -1))

        if d_s.size == 0 or i_best < 0 or i_best >= int(d_s.size):

            QMessageBox.information(self, "Generate corridor", "No valid RMSE(d) profile is available.")

            return

        d_best = float(d_s[i_best])

        half = self._corridor_manual_half_width_nm()

        d_lo = float(d_best - half)

        d_hi = float(d_best + half)

        payload = self._build_manual_corridor_payload(source, display, d_lo, d_hi)

        if payload is None:

            QMessageBox.warning(self, "Generate corridor", "Unable to rebuild a manual corridor from this interval.")

            return

        updated = dict(display)

        for k, v in source.items():

            if k.startswith("profile_d_") and k not in updated:

                updated[k] = v

        updated.update(payload)

        self._corridor_rmse_manual_active = True

        self._corridor_rmse_manual_lo = float(payload["manual_corridor_interval_nm"][0])

        self._corridor_rmse_manual_hi = float(payload["manual_corridor_interval_nm"][1])

        self._last_result = updated

        self._plot_result(updated)

        self._refresh_data_table()

        self.lbl_status.setText(

            f"Manual corridor regenerated on [{self._corridor_rmse_manual_lo:.2f}, {self._corridor_rmse_manual_hi:.2f}] nm"

        )

        if self.logger:

            self.logger.info(

                "GUI manual corridor | interval=[%.6f, %.6f] nm | selected_points=%d",

                float(self._corridor_rmse_manual_lo),

                float(self._corridor_rmse_manual_hi),

                int(payload.get("manual_corridor_selected_count", 0)),

            )

    @staticmethod

    def _robust_interval_from_local_quadratic(

        d_s: np.ndarray,

        r_s: np.ndarray,

        i_best: int,

        delta_rmse: float,

        half_window_pts: int,

    ) -> tuple[bool, float, float, float, float]:

        if d_s.size < 5 or r_s.size != d_s.size:

            return False, float("nan"), float("nan"), float("nan"), float("nan")

        i0 = int(max(0, i_best - max(1, half_window_pts)))

        i1 = int(min(d_s.size, i_best + max(1, half_window_pts) + 1))

        if i1 - i0 < 5:

            return False, float("nan"), float("nan"), float("nan"), float("nan")

        d_loc = np.asarray(d_s[i0:i1], dtype=np.float64)

        r_loc = np.asarray(r_s[i0:i1], dtype=np.float64)

        d_best = float(d_s[i_best])

        x = d_loc - d_best

        try:

            c2, c1, _c0 = np.polyfit(x, r_loc, 2)

        except Exception:

            return False, float("nan"), float("nan"), float("nan"), float("nan")

        if (not np.isfinite(c2)) or (not np.isfinite(c1)) or c2 <= 0.0:

            return False, float("nan"), float("nan"), float("nan"), float("nan")

        dd = float(np.sqrt(max(delta_rmse / c2, 0.0)))

        d_lo = d_best - dd

        d_hi = d_best + dd

        slope_best = float(c1)

        return True, d_lo, d_hi, slope_best, float(c2)

    def _on_corridor_rmse_plot_clicked(self, ev) -> None:

        if not hasattr(self, "plot_corridor_rmse_d"):

            return

        d_s = np.asarray(getattr(self, "_corridor_rmse_d_vals", []), dtype=np.float64).ravel()

        r_s = np.asarray(getattr(self, "_corridor_rmse_vals", []), dtype=np.float64).ravel()

        if d_s.size == 0 or r_s.size != d_s.size:

            return

        try:

            vb = self.plot_corridor_rmse_d.plotItem.vb

            p = vb.mapSceneToView(ev.scenePos())

            x = float(p.x())

        except Exception:

            return

        if not np.isfinite(x):

            return

        i_sel = int(np.argmin(np.abs(d_s - x)))

        d_sel = float(d_s[i_sel])

        r_sel = float(r_s[i_sel])

        i_best = int(getattr(self, "_corridor_rmse_best_idx", -1))

        if i_best < 0 or i_best >= d_s.size:

            i_best = int(np.argmin(r_s))

        d_best = float(d_s[i_best])

        r_best = float(r_s[i_best])

        d_lo_rb = float(getattr(self, "_corridor_rmse_robust_lo", float("nan")))

        d_hi_rb = float(getattr(self, "_corridor_rmse_robust_hi", float("nan")))

        rb_ok = bool(getattr(self, "_corridor_rmse_robust_ok", False))

        if hasattr(self, "lbl_corridor_rmse_summary"):

            tail = (

                f" | robust interval ≈ [{d_lo_rb:.3f}, {d_hi_rb:.3f}] nm"

                if rb_ok and np.isfinite(d_lo_rb) and np.isfinite(d_hi_rb)

                else ""

            )

            self.lbl_corridor_rmse_summary.setText(

                f"Best computed thickness: d* = {d_best:.3f} nm | RMSE(d*) = {r_best:.6f} | "

                f"selected: d = {d_sel:.3f} nm, RMSE = {r_sel:.6f}, ΔRMSE = {r_sel - r_best:+.6e}{tail}"

            )

    def _build_tab_corridor(self) -> QWidget:

        """Tab for n, log₁₀ k + corridor / bootstrap envelopes (fills _plot_result)."""

        panel = QWidget()

        lay = QVBoxLayout(panel)

        lay.setContentsMargins(6, 6, 6, 6)

        hint = QLabel(

            "<b>Acceptance envelope (profiling in d)</b> - n(lambda) and k(lambda) bands after optimization. "

            "Calculation starts from <b>best polished spectral RMSE</b> ('best RMSE' + RMSE_ref+Delta default): "

            "displayed reference curve is the scientific nominal, and the envelope groups models whose "

            "masked RMSE remains <= RMSE<sub>best</sub> + Delta. This is not a Bayesian confidence interval."

            "<br><br>"

            "<b>Automatic</b> execution at end of run if 'Corridors n/k -> Enable' is checked. "

            "Onglet ouvert seul lorsque corridors ou bootstrap sont disponibles."

            "<br><br>"

            "<b>log₁₀ k:</b> the <b>bold</b> orange curve follows the main optimization result ('n &amp; log₁₀ k' tab). "

            "Shaded area = min/max of linear <i>k</i> of refits accepted at various <i>d</i> "

            "(without corrective widening in scientific mode). <b>Dashed</b> orange curve only if a central refit "

            "differs significantly from bold. <b>Crosshair:</b> value follows bold curve at cursor lambda."

        )

        hint.setWordWrap(True)

        hint.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 11px;")

        lay.addWidget(hint)

        self.lbl_corridors_tab_state = QLabel()

        self.lbl_corridors_tab_state.setTextFormat(Qt.TextFormat.RichText)

        self.lbl_corridors_tab_state.setStyleSheet(f"color: {CertusTheme.TEXT_MAIN}; font-size: 11px;")

        self.lbl_corridors_tab_state.setToolTip(

            "Corridor option state in the UI for the next optimization run."

        )

        lay.addWidget(self.lbl_corridors_tab_state)

        self.plot_n_corridor = CertusScientificPlot(

            title="n(lambda) + corridors", y_label="n", x_label="lambda (nm)"

        )

        self.plot_n_corridor.showGrid(x=True, y=True, alpha=0.25)

        self.plot_lgk_corridor = CertusScientificPlot(

            title="log₁₀ k(lambda) + corridors", y_label="log₁₀ k", x_label="lambda (nm)"

        )

        self.plot_lgk_corridor.showGrid(x=True, y=True, alpha=0.25)

        spl = QSplitter(Qt.Orientation.Vertical)

        spl.addWidget(wrap_scientific_plot_with_toolbar(self, self.plot_n_corridor))

        spl.addWidget(wrap_scientific_plot_with_toolbar(self, self.plot_lgk_corridor))

        spl.setStretchFactor(0, 1)

        spl.setStretchFactor(1, 1)

        lay.addWidget(spl, 1)

        return panel

    def _build_tab_nl(self) -> QWidget:

        """Measurement non-linearity: n, log k with / without alpha_NL; RMSE metrics."""

        panel = QWidget()

        lay = QVBoxLayout(panel)

        lay.setContentsMargins(6, 6, 6, 6)

        hint = QLabel(

            "<b>Non-linearity correction (alpha × measurements)</b> - By default, <b>Non-lin. alpha</b> is enabled: the same alpha "

            "from 0.995 to 1.005 (step 0.0005) are traversed starting from the closest to <b>1</b>, then by "

            "<b>adjacent</b> steps (1 -> 0.9995 -> 1.0005 -> …). Each L-BFGS-B restarts from the previous step solution. "

            "The best MSE criterion is kept. Masked T (and R) are multiplied by alpha. "

            "<b>Slow budget</b> ~ 1.35× the run polish (capped); <b>2nd pass</b> on alpha_opt with even "

            "larger maxfun. Curves <i>without NL</i> = main model; <i>with NL</i> = after this post-processing."

        )

        hint.setWordWrap(True)

        hint.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 11px;")

        lay.addWidget(hint)

        row_nl_b = QHBoxLayout()

        row_nl_b.addWidget(QLabel("L-BFGS-B balance (joint NL alpha polish):"))

        self.cb_nl_alpha_budget = QComboBox()

        self.cb_nl_alpha_budget.addItem(

            "Slow - maxfun ~ 1.35× run polish (default, 100k ceiling)", "slow"

        )

        self.cb_nl_alpha_budget.addItem(

            "Fast - shared budget 2×polish / n_steps (20k/step ceiling)", "fast"

        )

        self.cb_nl_alpha_budget.setToolTip(

            "Slow: larger maxfun budget for the joint local polish on alpha, thickness and spline nodes.\n"

            "Fast: reduced maxfun budget for a quicker, less exhaustive joint polish."

        )

        self.cb_nl_alpha_budget.currentIndexChanged.connect(lambda _i: self._persist_nl_alpha_budget_pref())

        row_nl_b.addWidget(self.cb_nl_alpha_budget, 1)

        lay.addLayout(row_nl_b)

        self.chk_nl_second_pass = QCheckBox(

            "Reinforced maxfun for joint NL alpha polish"

        )

        self.chk_nl_second_pass.setChecked(True)

        self.chk_nl_second_pass.setToolTip(

            "Controls the maxfun budget used by the final joint local polish on alpha, thickness and spline nodes "

            "with alpha strongly constrained around 1. Disable or reduce if you want faster runs."

        )

        self.chk_nl_second_pass.stateChanged.connect(lambda _s: self._persist_nl_alpha_second_pass_pref())

        lay.addWidget(self.chk_nl_second_pass)

        self.chk_nl_adaptive_scan = QCheckBox(

            "Legacy alpha sweep option (unused by joint polish)"

        )

        self.chk_nl_adaptive_scan.setChecked(True)

        self.chk_nl_adaptive_scan.setToolTip(

            "This control is kept only for settings compatibility. The current NL alpha engine no longer uses an alpha sweep "

            "and instead performs one constrained joint local polish with alpha≈1."

        )

        self.chk_nl_adaptive_scan.setEnabled(False)

        self.chk_nl_adaptive_scan.stateChanged.connect(lambda _s: self._persist_nl_alpha_adaptive_pref())

        lay.addWidget(self.chk_nl_adaptive_scan)

        self.lbl_nl_summary = QLabel("-")

        self.lbl_nl_summary.setWordWrap(True)

        self.lbl_nl_summary.setTextFormat(Qt.TextFormat.RichText)

        lay.addWidget(self.lbl_nl_summary)

        self.plot_n_nl = CertusScientificPlot(

            title="n(lambda) - no NL vs with alpha NL", y_label="n", x_label="lambda (nm)"

        )

        self.plot_n_nl.showGrid(x=True, y=True, alpha=0.25)

        self.plot_lgk_nl = CertusScientificPlot(

            title="log₁₀ k(lambda) - no NL vs with alpha NL", y_label="log₁₀ k", x_label="lambda (nm)"

        )

        self.plot_lgk_nl.showGrid(x=True, y=True, alpha=0.25)

        spl = QSplitter(Qt.Orientation.Vertical)

        spl.addWidget(wrap_scientific_plot_with_toolbar(self, self.plot_n_nl))

        spl.addWidget(wrap_scientific_plot_with_toolbar(self, self.plot_lgk_nl))

        spl.setStretchFactor(0, 1)

        spl.setStretchFactor(1, 1)

        lay.addWidget(spl, 1)

        return panel

    def _build_tab_data(self) -> QWidget:

        panel = QWidget()

        lay = QVBoxLayout(panel)

        lay.setContentsMargins(10, 10, 10, 10)

        tb = QHBoxLayout()

        self.btn_copy_nk = create_styled_button("Copier tout le tableau (TSV)", "secondary")

        self.btn_copy_nk.setEnabled(False)

        self.btn_copy_nk.setToolTip("Toutes les colonnes (lambda, n…, k…) - collage Excel")

        self.btn_copy_nk.clicked.connect(self._copy_nk_to_clipboard)

        tb.addWidget(self.btn_copy_nk)

        self.btn_export_nk = create_styled_button("Export CSV...", "primary")

        self.btn_export_nk.setEnabled(False)

        self.btn_export_nk.clicked.connect(self._export_nk_csv)

        tb.addWidget(self.btn_export_nk)

        tb.addStretch(1)

        lay.addLayout(tb)

        spl_prev = QSplitter(Qt.Orientation.Horizontal)

        self.plot_data_preview_n = CertusScientificPlot(

            title="n preview",

            y_label="n",

            x_label="lambda (nm)",

        )

        self.plot_data_preview_n.showGrid(x=True, y=True, alpha=0.25)

        self.plot_data_preview_k = CertusScientificPlot(

            title="k preview",

            y_label="k",

            x_label="lambda (nm)",

        )

        self.plot_data_preview_k.showGrid(x=True, y=True, alpha=0.25)

        self.plot_data_preview_k.setLogMode(False, True)

        spl_prev.addWidget(wrap_scientific_plot_with_toolbar(self, self.plot_data_preview_n))

        spl_prev.addWidget(wrap_scientific_plot_with_toolbar(self, self.plot_data_preview_k))

        spl_prev.setStretchFactor(0, 1)

        spl_prev.setStretchFactor(1, 1)

        lay.addWidget(spl_prev, 1)

        self.table_nk = ExcelTableWidget()

        self.table_nk.setColumnCount(9)

        self.table_nk.setHorizontalHeaderLabels(

            [

                "lambda (nm)",

                "n",

                "n_alpha",

                "n env min",

                "n env max",

                "k",

                "k_alpha",

                "k env min",

                "k env max",

            ]

        )

        self.table_nk.setEditTriggers(ExcelTableWidget.EditTrigger.NoEditTriggers)

        self.table_nk.horizontalHeader().setStretchLastSection(True)

        self.table_nk.setToolTip(

            "lambda grid by spectral region: 2 nm step (<=400 nm), 5 nm (400-1200 nm), "

            "10 nm beyond; n, k and envelopes interpolated from result mesh. "

            "n_alpha / k_alpha: nonlinear indices if available. "

            "Enveloppes : bornes du corridor (profilage d). "

            "Previews: all n (or k) curves, envelope band if corridor; synchronized lambda cursor. "

            "Ctrl+C: copy selection (TSV) -> Excel."

        )

        lay.addWidget(self.table_nk, 2)

        return panel

    def _build_tab_log(self) -> QWidget:

        w = QWidget()

        lay = QVBoxLayout(w)

        lay.setContentsMargins(12, 12, 12, 12)

        info = QLabel(

            "The detailed stream (local stages, K stages, polish, continuous laws) appears in the "

            "<b>OPTIMIZATION LOG</b> panel under the plots. "

            "Use <b>Copy Logs</b> on that panel to copy all text."

        )

        info.setWordWrap(True)

        info.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 11px;")

        lay.addWidget(info)

        lay.addStretch(1)

        return w

    def _build_tab_why(self) -> QWidget:

        panel = QWidget()

        grid = QGridLayout(panel)

        grid.setSpacing(16)

        grid.setContentsMargins(24, 24, 24, 24)

        intro = QLabel(

            "<b>CERTUS-INDEX-SPLINE.</b> Global fit of "

            "<i>n(lambda)</i>, <i>k(lambda)</i> as piecewise-linear in sigma=1/lambda (ln k at knots), "

            "with <b>local L-BFGS-B</b> polish. Advanced mode: catalog of continuous laws "

            "on normalized <i>u</i> and 19-D re-optimization if spectral RMSE improves."

        )

        intro.setWordWrap(True)

        intro.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 11px;")

        grid.addWidget(intro, 0, 0, 1, 2)

        cards = [

            ("Local L-BFGS-B", "Local optimization with tunable budgets.", ""),

            ("Spectral weights Deltaln lambda", "RMSE weighted trapezoidal rule on ln lambda grid (no cap).", ""),

            ("Auto-K and adaptive mesh", "K growth or SMART-style sigma insertions; warm start.", ""),

            ("Continuous laws (advanced)", "Rank n(u), ln k(u) families then optimize d + 18 parameters.", ""),

        ]

        for i, (title, desc, icon) in enumerate(cards):

            grid.addWidget(FlashyCard(title, desc, icon=icon), 1 + i // 2, i % 2)

        return panel

    def _detach_current_plot(self) -> None:

        w = self.tabs_main.currentWidget()

        if w is None:

            return

        plots = w.findChildren(CertusScientificPlot)

        if not plots:

            QMessageBox.information(self, "Detach", "No scientific plot in this tab.")

            return

        tab_name = self.tabs_main.tabText(self.tabs_main.currentIndex())

        self.open_detached_certus_plot(

            plots[0], title=f"{self.APP_NAME}  {tab_name}"

        )

    @staticmethod

    def _lam_uniform_grid_nm(lo_h: float, hi_h: float, step: float) -> np.ndarray:

        if not (np.isfinite(lo_h) and np.isfinite(hi_h) and hi_h > lo_h):

            return np.array([], dtype=np.float64)

        st = float(np.ceil(lo_h / step) * step)

        en = float(np.floor(hi_h / step) * step)

        if en < st - 1e-9:

            return np.array([0.5 * (lo_h + hi_h)], dtype=np.float64)

        if abs(en - st) < 1e-9:

            return np.array([st], dtype=np.float64)

        return np.arange(st, en + 1e-9, step, dtype=np.float64)

    @staticmethod

    def _lam_piecewise_report_grid_nm(lo: float, hi: float) -> np.ndarray:

        """2 nm step on [lambda_min, 400], 5 nm on ]400, 1200], 10 nm beyond (nm)."""

        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):

            return np.array([], dtype=np.float64)

        parts: list[np.ndarray] = []

        a, b = float(lo), float(min(hi, 400.0))

        if b >= a - 1e-9:

            parts.append(CertusIndexSplineApp._lam_uniform_grid_nm(a, b, 2.0))

        a, b = float(max(lo, 400.0)), float(min(hi, 1200.0))

        if b >= a - 1e-9:

            parts.append(CertusIndexSplineApp._lam_uniform_grid_nm(a, b, 5.0))

        a, b = float(max(lo, 1200.0)), float(hi)

        if b >= a - 1e-9:

            parts.append(CertusIndexSplineApp._lam_uniform_grid_nm(a, b, 10.0))

        if not parts:

            return np.array([0.5 * (lo + hi)], dtype=np.float64)

        return np.unique(np.concatenate(parts))

    @staticmethod

    def _fmt_n_data_tab(nv: float) -> str:

        if not np.isfinite(nv):

            return ""

        return f"{float(nv):.4f}"

    @staticmethod

    def _fmt_k_data_tab(kv: float) -> str:

        if not np.isfinite(kv) or kv < 0:

            return ""

        v = float(kv)

        if v == 0.0:

            return "0"

        if v > 0.001:

            return f"{v:.3g}"

        return f"{v:.1e}"

    def _prepare_nk_data_tab_series(

        self, r: dict[str, Any]

    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:

        """n/k series (and NL, envelopes) interpolated on piecewise lambda grid."""

        lam = np.asarray(r["lam_nm"], dtype=np.float64).ravel()

        n_ = np.asarray(r["n_lam"], dtype=np.float64).ravel()

        k_ = np.asarray(r["k_lam"], dtype=np.float64).ravel()

        m0 = int(min(lam.size, n_.size, k_.size))

        if m0 <= 0:

            return None

        order = np.argsort(lam[:m0], kind="mergesort")

        ls = lam[:m0][order]

        ns = n_[:m0][order]

        ks = k_[:m0][order]

        fg = np.isfinite(ls)

        if not np.any(fg):

            return None

        lo = float(np.nanmin(ls[fg]))

        hi = float(np.nanmax(ls[fg]))

        lam_g = self._lam_piecewise_report_grid_nm(lo, hi)

        if lam_g.size == 0:

            return None

        n_g = np.interp(lam_g, ls, ns, left=np.nan, right=np.nan)

        k_g = np.interp(lam_g, ls, ks, left=np.nan, right=np.nan)

        m = m0

        lam_full = lam[:m0]

        n_nl_g = np.full_like(lam_g, np.nan)

        k_nl_g = np.full_like(lam_g, np.nan)

        n_nl = r.get("n_lam_nl")

        k_nl = r.get("k_lam_nl")

        if n_nl is not None and k_nl is not None:

            nn = np.asarray(n_nl, dtype=np.float64).ravel()

            kn = np.asarray(k_nl, dtype=np.float64).ravel()

            if nn.size >= m and kn.size >= m:

                nn_s = nn[:m][order]

                kn_s = kn[:m][order]

                n_nl_g = np.interp(lam_g, ls, nn_s, left=np.nan, right=np.nan)

                k_nl_g = np.interp(lam_g, ls, kn_s, left=np.nan, right=np.nan)

        n_lo_g = np.full_like(lam_g, np.nan)

        n_hi_g = np.full_like(lam_g, np.nan)

        k_lo_g = np.full_like(lam_g, np.nan)

        k_hi_g = np.full_like(lam_g, np.nan)

        if bool(r.get("profile_d_enabled", False)) or bool(r.get("manual_corridor_active", False)):

            cn_lo = np.asarray(r.get("corridor_n_lo", []), dtype=np.float64).ravel()

            cn_hi = np.asarray(r.get("corridor_n_hi", []), dtype=np.float64).ravel()

            ck_lo = np.asarray(r.get("corridor_k_lo", []), dtype=np.float64).ravel()

            ck_hi = np.asarray(r.get("corridor_k_hi", []), dtype=np.float64).ravel()

            lsz = lam_full.size

            if (

                cn_lo.size == lsz

                and cn_hi.size == lsz

                and ck_lo.size == lsz

                and ck_hi.size == lsz

            ):

                n_lo_g = np.interp(lam_g, ls, cn_lo[:m][order], left=np.nan, right=np.nan)

                n_hi_g = np.interp(lam_g, ls, cn_hi[:m][order], left=np.nan, right=np.nan)

                k_lo_g = np.interp(lam_g, ls, ck_lo[:m][order], left=np.nan, right=np.nan)

                k_hi_g = np.interp(lam_g, ls, ck_hi[:m][order], left=np.nan, right=np.nan)

        return (lam_g, n_g, k_g, n_nl_g, k_nl_g, n_lo_g, n_hi_g, k_lo_g, k_hi_g)

    def _copy_nk_to_clipboard(self) -> None:

        if self._last_result is None:

            QMessageBox.information(self, "Clipboard", "Run an optimization first.")

            return

        r = self._last_result

        ser = self._prepare_nk_data_tab_series(r)

        if ser is None:

            QMessageBox.information(self, "Clipboard", "Grille vide.")

            return

        (

            lam_g,

            n_g,

            k_g,

            n_nl_g,

            k_nl_g,

            n_lo_g,

            n_hi_g,

            k_lo_g,

            k_hi_g,

        ) = ser

        hdr = (

            "lambda_nm\tn\tn_alpha\tn_envelope_min\tn_envelope_max\t"

            "k\tk_alpha\tk_envelope_min\tk_envelope_max"

        )

        lines = [hdr]

        m = int(lam_g.size)

        for i in range(m):

            row = f"{float(lam_g[i]):.4f}\t"

            row += self._fmt_n_data_tab(float(n_g[i])) + "\t"

            row += self._fmt_n_data_tab(float(n_nl_g[i])) + "\t"

            row += self._fmt_n_data_tab(float(n_lo_g[i])) + "\t"

            row += self._fmt_n_data_tab(float(n_hi_g[i])) + "\t"

            row += self._fmt_k_data_tab(float(k_g[i])) + "\t"

            row += self._fmt_k_data_tab(float(k_nl_g[i])) + "\t"

            row += self._fmt_k_data_tab(float(k_lo_g[i])) + "\t"

            row += self._fmt_k_data_tab(float(k_hi_g[i]))

            lines.append(row)

        cb = QApplication.clipboard()

        if cb is None:

            QMessageBox.warning(self, "Clipboard", "Clipboard unavailable.")

            return

        cb.setText("\n".join(lines))

        self.lbl_status.setText("Data table copied (TSV).")

        self.btn_copy_nk.setText(" Copied!")

        QTimer.singleShot(

            1800, lambda: self.btn_copy_nk.setText("Copier tout le tableau (TSV)")

        )

    def _export_nk_csv(self) -> None:

        if self._last_result is None:

            QMessageBox.warning(self, "Export", "No result to export.")

            return

        start_dir = get_certus_last_dir()

        if not start_dir or not os.path.isdir(start_dir):

            start_dir = str(_SCRIPT_DIR)

        suggested = os.path.join(start_dir, "certus_index_spline_nk.csv")

        path, _ = QFileDialog.getSaveFileName(

            self,

            "Export indices",

            suggested,

            "CSV (*.csv);;All (*.*)",

        )

        if not path:

            return

        ser = self._prepare_nk_data_tab_series(self._last_result)

        if ser is None:

            QMessageBox.warning(self, "Export", "Empty grid.")

            return

        (

            lam_g,

            n_g,

            k_g,

            n_nl_g,

            k_nl_g,

            n_lo_g,

            n_hi_g,

            k_lo_g,

            k_hi_g,

        ) = ser

        m = int(lam_g.size)

        try:

            with open(path, "w", encoding="utf-8") as fh:

                fh.write(

                    "lambda_nm,n,n_alpha,n_envelope_min,n_envelope_max,"

                    "k,k_alpha,k_envelope_min,k_envelope_max\n"

                )

                for i in range(m):

                    line = f"{float(lam_g[i]):.4f},"

                    line += self._fmt_n_data_tab(float(n_g[i])) + ","

                    line += self._fmt_n_data_tab(float(n_nl_g[i])) + ","

                    line += self._fmt_n_data_tab(float(n_lo_g[i])) + ","

                    line += self._fmt_n_data_tab(float(n_hi_g[i])) + ","

                    line += self._fmt_k_data_tab(float(k_g[i])) + ","

                    line += self._fmt_k_data_tab(float(k_nl_g[i])) + ","

                    line += self._fmt_k_data_tab(float(k_lo_g[i])) + ","

                    line += self._fmt_k_data_tab(float(k_hi_g[i])) + "\n"

                    fh.write(line)

            set_certus_last_dir(path)

            self.lbl_status.setText(f"CSV saved: {path}")

        except OSError as e:

            QMessageBox.critical(self, "Export", str(e))

    @staticmethod

    def _interp_preview_axis(xs: np.ndarray, ys: np.ndarray, xq: float) -> float:

        xs = np.asarray(xs, dtype=np.float64).ravel()

        ys = np.asarray(ys, dtype=np.float64).ravel()

        m = np.isfinite(xs) & np.isfinite(ys)

        if int(np.count_nonzero(m)) < 2:

            return float("nan")

        xv, yv = xs[m], ys[m]

        o = np.argsort(xv, kind="mergesort")

        xv, yv = xv[o], yv[o]

        xf = float(xq)

        if xf < float(xv[0]) or xf > float(xv[-1]):

            return float("nan")

        return float(np.interp(xf, xv, yv))

    @staticmethod

    def _vb_mid_y_plot(w: CertusScientificPlot) -> float:

        try:

            y0, y1 = w.plotItem.vb.viewRange()[1]

            return 0.5 * (float(y0) + float(y1))

        except Exception:

            return 0.0

    def _on_data_preview_plot_mouse_moved(

        self,

        src: CertusScientificPlot,

        pos: Any,

        x: float,

        y: float,

        y_show: Any,

    ) -> None:

        """Synchronizes both previews (lambda) and tooltip with all interpolated n and k."""

        del pos, y_show

        s = getattr(self, "_data_preview_series", None)

        if not isinstance(s, dict):

            return

        pn = self.plot_data_preview_n

        pk = self.plot_data_preview_k

        lam = s.get("lam")

        if lam is None:

            return

        lam_a = np.asarray(lam, dtype=np.float64).ravel()

        def _fmt_nq(v: float) -> str:

            return self._fmt_n_data_tab(v) if np.isfinite(v) else "-"

        def _fmt_kq(v: float) -> str:

            if not np.isfinite(v) or v < 0:

                return "-"

            return self._fmt_k_data_tab(float(v))

        n_at = self._interp_preview_axis(lam_a, s["n"], x)

        nnl_at = self._interp_preview_axis(lam_a, s["n_nl"], x)

        nlo_at = self._interp_preview_axis(lam_a, s["n_lo"], x)

        nhi_at = self._interp_preview_axis(lam_a, s["n_hi"], x)

        k_at = self._interp_preview_axis(lam_a, s["k"], x)

        knl_at = self._interp_preview_axis(lam_a, s["k_nl"], x)

        klo_at = self._interp_preview_axis(lam_a, s["k_lo"], x)

        khi_at = self._interp_preview_axis(lam_a, s["k_hi"], x)

        lam_txt = float(x)

        txt = (

            f"lambda = {lam_txt:.2f} nm\n"

            f"n={_fmt_nq(n_at)}  n_alpha={_fmt_nq(nnl_at)}  n_min={_fmt_nq(nlo_at)}  n_max={_fmt_nq(nhi_at)}\n"

            f"k={_fmt_kq(k_at)}  k_alpha={_fmt_kq(knl_at)}  k_min={_fmt_kq(klo_at)}  k_max={_fmt_kq(khi_at)}"

        )

        k_floor = float(s.get("k_floor", 1e-30))

        if not (np.isfinite(k_floor) and k_floor > 0.0):

            k_floor = 1e-30

        pn.vLine.setPos(x)

        pk.vLine.setPos(x)

        pn.hLine.setVisible(False)

        pk.hLine.setVisible(False)

        for w in (pn, pk):

            try:

                xr = w.plotItem.vb.viewRange()[0]

                x_lo, x_hi = float(xr[0]), float(xr[1])

                span = x_hi - x_lo

                if span > 0 and x > x_lo + 0.78 * span:

                    w.info_label.setAnchor((1, 1))

                else:

                    w.info_label.setAnchor((0, 1))

            except Exception:

                w.info_label.setAnchor((0, 1))

        pn.info_label.setText(txt)

        pk.info_label.setText(txt)

        if src is pn:

            pn_y = float(y)

            if np.isfinite(k_at) and float(k_at) > 0.0:

                pk_y = float(k_at)

            elif np.isfinite(k_at) and float(k_at) == 0.0:

                pk_y = k_floor

            else:

                pk_y = self._vb_mid_y_plot(pk)

        else:

            pk_y = float(y)

            pn_y = float(n_at) if np.isfinite(n_at) else self._vb_mid_y_plot(pn)

        pn.info_label.setPos(x, pn_y)

        pk.info_label.setPos(x, pk_y)

    def _refresh_data_preview_plots(

        self,

        ser: tuple[np.ndarray, ...] | None = None,

    ) -> None:

        """Data mini-graphs: all n / all k, table grid, synchronized lambda."""

        if not hasattr(self, "plot_data_preview_n") or not hasattr(self, "plot_data_preview_k"):

            return

        pn = self.plot_data_preview_n

        pk = self.plot_data_preview_k

        pn.clear()

        pk.clear()

        pn._certus_crosshair_label_fn = None

        pk._certus_crosshair_label_fn = None

        pn._certus_crosshair_vertical_only = False

        pk._certus_crosshair_vertical_only = False

        self._data_preview_series = None

        if ser is None:

            pn._apply_sensible_empty_range()

            pk._apply_sensible_empty_range()

            return

        (

            lam_g,

            n_g,

            k_g,

            n_nl_g,

            k_nl_g,

            n_lo_g,

            n_hi_g,

            k_lo_g,

            k_hi_g,

        ) = ser

        lam = np.asarray(lam_g, dtype=np.float64).ravel()

        nv = np.asarray(n_g, dtype=np.float64).ravel()

        kv = np.asarray(k_g, dtype=np.float64).ravel()

        n_nl_v = np.asarray(n_nl_g, dtype=np.float64).ravel()

        k_nl_v = np.asarray(k_nl_g, dtype=np.float64).ravel()

        n_lo_v = np.asarray(n_lo_g, dtype=np.float64).ravel()

        n_hi_v = np.asarray(n_hi_g, dtype=np.float64).ravel()

        k_lo_v = np.asarray(k_lo_g, dtype=np.float64).ravel()

        k_hi_v = np.asarray(k_hi_g, dtype=np.float64).ravel()

        mk = np.isfinite(lam) & np.isfinite(kv) & (kv > 0.0)

        kk = kv[mk]

        k_pos = kk[kk > 0.0]

        k_floor = float(np.nanmin(k_pos)) if k_pos.size > 0 else 1e-30

        self._data_preview_series = {

            "lam": lam.copy(),

            "n": nv.copy(),

            "n_nl": n_nl_v.copy(),

            "n_lo": n_lo_v.copy(),

            "n_hi": n_hi_v.copy(),

            "k": kv.copy(),

            "k_nl": k_nl_v.copy(),

            "k_lo": k_lo_v.copy(),

            "k_hi": k_hi_v.copy(),

            "k_floor": k_floor,

        }

        def _add_legend(plot: CertusScientificPlot) -> None:

            try:

                plot.addLegend(offset=(8, 8))

            except Exception:

                pass

        # --- n preview : enveloppe puis courbes ---

        m_n_env = (

            np.isfinite(n_lo_v)

            & np.isfinite(n_hi_v)

            & (n_hi_v >= n_lo_v)

        )

        if np.any(m_n_env):

            le = lam[m_n_env]

            ylo = n_lo_v[m_n_env]

            yhi = n_hi_v[m_n_env]

            o = np.argsort(le, kind="mergesort")

            le, ylo, yhi = le[o], ylo[o], yhi[o]

            if le.size >= 2:

                cl = pg.PlotCurveItem(le, ylo, pen=pg.mkPen((0, 87, 255, 80), width=1))

                cu = pg.PlotCurveItem(le, yhi, pen=pg.mkPen((0, 87, 255, 80), width=1))

                pn.addItem(cl)

                pn.addItem(cu)

                pn.addItem(pg.FillBetweenItem(cl, cu, brush=pg.mkBrush(0, 87, 255, 40)))

        if np.any(np.isfinite(n_nl_v)):

            xnl, ynl = sanitize_xy_for_plot(lam, n_nl_v)

            if xnl.size >= 2:

                plot_widget_plot_finite(

                    pn,

                    xnl,

                    ynl,

                    pen=pg.mkPen("#0a8f5a", width=1.6),

                    name="n_alpha",

                )

        xn, yn = sanitize_xy_for_plot(lam, nv)

        if xn.size >= 2:

            c_n = plot_widget_plot_finite(

                pn, xn, yn, pen=pg.mkPen("#0057ff", width=2.4), name="n"

            )

            if c_n is not None:

                setattr(c_n, "_certus_crosshair_primary", True)

        # --- k preview : enveloppe (k>0) puis courbes ---

        m_k_env = (

            np.isfinite(k_lo_v)

            & np.isfinite(k_hi_v)

            & (k_lo_v > 0.0)

            & (k_hi_v > 0.0)

            & (k_hi_v >= k_lo_v)

        )

        if np.any(m_k_env):

            lek = lam[m_k_env]

            ylok = k_lo_v[m_k_env]

            yhik = k_hi_v[m_k_env]

            ok = np.argsort(lek, kind="mergesort")

            lek, ylok, yhik = lek[ok], ylok[ok], yhik[ok]

            if lek.size >= 2:

                clk = pg.PlotCurveItem(lek, ylok, pen=pg.mkPen((245, 158, 11, 90), width=1))

                cuk = pg.PlotCurveItem(lek, yhik, pen=pg.mkPen((245, 158, 11, 90), width=1))

                pk.addItem(clk)

                pk.addItem(cuk)

                pk.addItem(pg.FillBetweenItem(clk, cuk, brush=pg.mkBrush(255, 160, 40, 35)))

        if np.any(np.isfinite(k_nl_v) & (k_nl_v > 0.0)):

            knlp = np.where(np.isfinite(k_nl_v) & (k_nl_v > 0.0), k_nl_v, np.nan)

            xknl, yknl = sanitize_xy_for_plot(lam, knlp)

            if xknl.size >= 2:

                plot_widget_plot_finite(

                    pk,

                    xknl,

                    yknl,

                    pen=pg.mkPen("#0a8f5a", width=1.6),

                    name="k_alpha",

                )

        kk_plot = np.where(np.isfinite(kv) & (kv > 0.0), kv, np.nan)

        xk, yk = sanitize_xy_for_plot(lam, kk_plot)

        if xk.size >= 2:

            c_k = plot_widget_plot_finite(

                pk, xk, yk, pen=pg.mkPen("#f59e0b", width=2.4), name="k"

            )

            if c_k is not None:

                setattr(c_k, "_certus_crosshair_primary", True)

        pn.setLogMode(False, False)

        pk.setLogMode(False, True)

        _add_legend(pn)

        _add_legend(pk)

        pn._certus_mouse_moved_hook = self._on_data_preview_plot_mouse_moved

        pk._certus_mouse_moved_hook = self._on_data_preview_plot_mouse_moved

        pn.autoRange()

        pk.autoRange()

    def _refresh_data_table(self, result_override: dict | None = None) -> None:

        if not hasattr(self, "table_nk"):

            return

        t = self.table_nk

        t.setRowCount(0)

        result_eff = result_override if isinstance(result_override, dict) else self._last_result

        if result_eff is None:

            self.btn_copy_nk.setEnabled(False)

            self.btn_export_nk.setEnabled(False)

            self._refresh_data_preview_plots(ser=None)

            return

        ser = self._prepare_nk_data_tab_series(result_eff)

        if ser is None:

            self.btn_copy_nk.setEnabled(False)

            self.btn_export_nk.setEnabled(False)

            self._refresh_data_preview_plots(ser=None)

            return

        (

            lam_g,

            n_g,

            k_g,

            n_nl_g,

            k_nl_g,

            n_lo_g,

            n_hi_g,

            k_lo_g,

            k_hi_g,

        ) = ser

        m = int(lam_g.size)

        t.setColumnCount(9)

        t.setHorizontalHeaderLabels(

            [

                "lambda (nm)",

                "n",

                "n_alpha",

                "n env min",

                "n env max",

                "k",

                "k_alpha",

                "k env min",

                "k env max",

            ]

        )

        t.setRowCount(m)

        n_valid = 0

        def _cell_n(x: float) -> QTableWidgetItem:

            if not np.isfinite(x):

                return QTableWidgetItem("-")

            return QTableWidgetItem(self._fmt_n_data_tab(float(x)))

        def _cell_k(x: float) -> QTableWidgetItem:

            if not np.isfinite(x) or x < 0:

                return QTableWidgetItem("-")

            return QTableWidgetItem(self._fmt_k_data_tab(float(x)))

        for i in range(m):

            t.setItem(i, 0, QTableWidgetItem(f"{float(lam_g[i]):.4f}"))

            if np.isfinite(n_g[i]) and np.isfinite(k_g[i]) and float(k_g[i]) >= 0.0:

                n_valid += 1

            t.setItem(i, 1, _cell_n(float(n_g[i])))

            t.setItem(i, 2, _cell_n(float(n_nl_g[i])))

            t.setItem(i, 3, _cell_n(float(n_lo_g[i])))

            t.setItem(i, 4, _cell_n(float(n_hi_g[i])))

            t.setItem(i, 5, _cell_k(float(k_g[i])))

            t.setItem(i, 6, _cell_k(float(k_nl_g[i])))

            t.setItem(i, 7, _cell_k(float(k_lo_g[i])))

            t.setItem(i, 8, _cell_k(float(k_hi_g[i])))

        self.btn_copy_nk.setEnabled(m > 0 and n_valid > 0)

        self.btn_export_nk.setEnabled(m > 0 and n_valid > 0)

        self._refresh_data_preview_plots(ser=ser)

    def reset_to_defaults(self) -> None:

        """Reinitialisation complete (bouton Clear / Reset CERTUS)."""

        self._stop_event.set()

        self._cleanup_thread()

        self._stop_event = Event()

        self._last_result = None

        self._last_worker_result = None

        self._corridor_rmse_manual_active = False

        self._corridor_rmse_manual_lo = float("nan")

        self._corridor_rmse_manual_hi = float("nan")

        for win in list(getattr(self, "detached_plot_windows", {}).values()):

            try:

                win.close()

            except Exception:

                logger.debug("Detached plot window close failed", exc_info=True)

        self.detached_plot_windows.clear()

        self._load_defaults()

        self.sp_pg_iter.setValue(
            int(SPLINE_PERF_PRESETS.get("fast", {}).get("pglobal_max_iter", 35) or 35)
        )

        self.cb_profilee.setCurrentIndex(0)

        self._on_profilee_changed()

        if hasattr(self, "sp_sol3_p1_maxfun"):

            self.sp_sol3_p1_maxfun.setValue(10000)

            self._persist_sol3_phase1_maxfun_pref()

        if hasattr(self, "sp_mesh_min_dlam"):

            self.sp_mesh_min_dlam.setValue(0.02)

        self._simple_auto_uncertainty = True

        self._persist_simple_auto_uncertainty_pref()

        QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP).setValue(

            _QS_SPLINE_UNCERTAINTY_DEFAULTS_REV, int(_UNCERTAINTY_DEFAULTS_REV)

        )

        self._apply_recommended_uncertainty_and_perf_defaults()

        self._apply_sio2_default_fit_parameters()

        self._update_epured_visibility()

        if hasattr(self, "btn_nl_toggle"):

            self.btn_nl_toggle.setChecked(True)

        self.chk_t.setChecked(True)

        self.chk_trel.setChecked(True)

        self.chk_r.setChecked(False)

        self.w_t.setValue(1.0)

        self.w_r.setValue(1.0)

        self.cb_weight.setCurrentIndex(0)

        self.cb_sub.setCurrentIndex(0)

        if hasattr(self, "ctrl_tabs"):

            self.ctrl_tabs.setCurrentIndex(0)

        if hasattr(self, "tabs_main"):

            self.tabs_main.setCurrentIndex(0)

        self._prog_reset_bar()

        self.btn_run.setEnabled(True)

        self.btn_stop.setEnabled(False)

        self.plot_T.clear()

        self.plot_n.clear()

        self.plot_lgk.clear()

        if hasattr(self, "plot_n_corridor"):

            self.plot_n_corridor.clear()

        if hasattr(self, "plot_lgk_corridor"):

            self.plot_lgk_corridor.clear()

        if hasattr(self, "plot_corridor_rmse_d"):

            self.plot_corridor_rmse_d.clear()

        self._corridor_rmse_d_vals = np.array([], dtype=np.float64)

        self._corridor_rmse_vals = np.array([], dtype=np.float64)

        self._corridor_rmse_best_idx = -1

        self._corridor_rmse_robust_lo = float("nan")

        self._corridor_rmse_robust_hi = float("nan")

        self._corridor_rmse_robust_ok = False

        if hasattr(self, "lbl_corridor_rmse_summary"):

            self.lbl_corridor_rmse_summary.setText("No corridor RMSE profile available yet.")

        if hasattr(self, "plot_n_nl"):

            self.plot_n_nl.clear()

        if hasattr(self, "plot_lgk_nl"):

            self.plot_lgk_nl.clear()

        if hasattr(self, "lbl_nl_summary"):

            self.lbl_nl_summary.setText("-")

        self._refresh_data_table()

        if hasattr(self, "log_panel"):

            self.log_panel.log_text.clear()

        self.lbl_status.setText("Reset")

        self._persist_spectrum_fit_settings()

        self._refresh_corridors_gui_state_labels()

    def _on_trel_plot_refresh(self) -> None:

        if self.df is None:

            return

        if hasattr(self, "tabs_main"):

            self.tabs_main.setCurrentIndex(0)

        self._plot_data_raw()

    def _on_profilee_changed(self) -> None:

        key = str(self.cb_profilee.currentData() or "fast")

        pr = SPLINE_PERF_PRESETS.get(key, {})

        v = pr.get("pglobal_max_iter")

        if v is not None:

            self.sp_pg_iter.blockSignals(True)

            self.sp_pg_iter.setValue(int(v))

            self.sp_pg_iter.blockSignals(False)

    def _spectrum_open_dialog_start_path(self) -> str:

        """Dernier file spectrum (pre-selection Qt) sinon last dossier suite, sinon script."""

        s = QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP)

        last_file = str(s.value(_QS_LAST_SPECTRUM, "") or "").strip()

        if last_file and os.path.isfile(last_file):

            return last_file

        d = get_certus_last_dir()

        if d and os.path.isdir(d):

            return d

        return str(_SCRIPT_DIR)

    def _persist_last_spectrum_path(self, path: str) -> None:

        ap = os.path.abspath(path)

        self._last_spectrum_path = ap

        set_certus_last_dir(ap)

        QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP).setValue(_QS_LAST_SPECTRUM, ap)

    def _on_load(self) -> None:

        path, _ = QFileDialog.getOpenFileName(

            self,

            "Spectrum",

            self._spectrum_open_dialog_start_path(),

            "Data (*.csv *.xlsx *.xls);;All (*.*)",

        )

        if not path:

            return

        try:

            raw = read_data_file_robust(path)

            self.df = normalize_spectrum_dataframe(raw)

            if self.df is None or "lambda" not in self.df.columns:

                raise ValueError("Invalid wavelength or spectrum column after normalization.")

            self._persist_last_spectrum_path(path)

            self.lbl_file.setText(path)

            if hasattr(self, "tabs_main"):

                self.tabs_main.setCurrentIndex(0)

            self._sync_rmse_lambda_bounds_from_file()

            self._plot_data_raw()

            self.lbl_status.setText(f"Loaded: {len(self.df)} points")

            if os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen":

                lam = np.asarray(self.df.get("lambda", []), dtype=np.float64).ravel()

                lam_f = lam[np.isfinite(lam)]

                lmin = float(np.min(lam_f)) if lam_f.size else float("nan")

                lmax = float(np.max(lam_f)) if lam_f.size else float("nan")

                cols = [str(c) for c in self.df.columns]

                summary = build_summary_plain_text(

                    "CERTUS INDEX SPLINE - Load Summary",

                    [

                        f"File: {os.path.abspath(path)}",

                        "",

                        "General",

                        (f"Rows: {int(len(self.df))}", int(len(self.df)) <= 0),

                        "",

                        "Data",

                        f"Columns: {', '.join(cols)}",

                        (f"Wavelength range: [{lmin:.1f}, {lmax:.1f}] nm", not (np.isfinite(lmin) and np.isfinite(lmax) and lmax > lmin)),

                        "",

                        "Compatibility checks",

                        (

                            f"Transmission column present: {'yes' if any(c.lower().startswith('t') for c in cols) else 'no'}",

                            not any(c.lower().startswith("t") for c in cols),

                        ),

                    ],

                )

                show_load_summary_dialog(self, "INDEX SPLINE Load Summary", summary)

        except Exception as e:

            QMessageBox.critical(self, "Loading", str(e))

            logger.exception("load")

    def _add_curve(

        self,

        widget,

        x: np.ndarray,

        y: np.ndarray,

        color: str,

        name: str,

        is_scatter: bool = False,

        *,

        crosshair_primary: bool = False,

        pen: Any | None = None,

    ) -> bool:

        xf, yf = sanitize_xy_for_plot(x, y)

        if xf.size == 0:

            return False

        if is_scatter:

            _plot_spectrum_raw_scatter(widget, xf, yf, color=color, name=name)

        else:

            p = pen if pen is not None else pg.mkPen(color, width=2)

            curve = plot_widget_plot_finite(widget, xf, yf, pen=p, name=name)

            if curve is not None and crosshair_primary:

                setattr(curve, "_certus_crosshair_primary", True)

        return True

    def _transform_spectrum_x(self, lam_nm: np.ndarray) -> tuple[np.ndarray, str]:

        mode = str(

            getattr(self, "cb_spectrum_xmode", None).currentData()

            if hasattr(self, "cb_spectrum_xmode")

            else "lambda"

        )

        lam = np.asarray(lam_nm, dtype=np.float64).ravel()

        if mode == "sigma":

            return 1.0 / np.maximum(lam, 1e-30), "sigma (nm⁻1)"

        if mode == "sigma2":

            s = 1.0 / np.maximum(lam, 1e-30)

            return s * s, "sigma2 (nm⁻2)"

        return lam, "lambda (nm)"

    def _apply_spectrum_x_axis_label(self, lbl: str) -> None:

        try:

            self.plot_T.setLabel("bottom", lbl)

        except (AttributeError, RuntimeError):

            try:

                self.plot_T.plotItem.setLabel("bottom", lbl)

            except (AttributeError, RuntimeError):

                logger.debug("_apply_spectrum_x_axis_label failed", exc_info=True)

    def _on_spectrum_x_mode_changed(self) -> None:

        if self._last_result is not None:

            self._plot_result(self._last_result)

        elif self.df is not None:

            self._plot_data_raw()

    def _format_spectrum_plot_title(self, r: dict) -> str:

        """Spectrum plot title: RMSE and thickness of the displayed snapshot + config summary."""

        rmse = float(r.get("rmse", float("nan")))

        d_nm = float(r.get("d_nm", float("nan")))

        rmse_s = f"{rmse:.6f}" if np.isfinite(rmse) else ""

        d_s = f"{d_nm:.2f} nm" if np.isfinite(d_nm) else ""

        rmse_lbl = "RMSE (bande lambda)" if r.get("rmse_fit_lambda_nm") is not None else "RMSE"

        bits: list[str] = []

        sk = r.get("sigma_knots")

        k_sig = int(np.asarray(sk, dtype=np.float64).size) if sk is not None else 0

        if k_sig > 0:

            bits.append(f"Ksigma={k_sig} ({k_sig - 1} seg.)")

        bits.append("interp sigma=cubic spline")

        if r.get("auto_knot_stages"):

            kb = r.get("auto_knots_K_best")

            if kb is not None:

                bits.append(f"auto-K (K*={int(kb)})")

            else:

                bits.append("auto-K")

        prof = str(self.cb_profilee.currentData() or "").strip() if hasattr(self, "cb_profilee") else ""

        if prof and prof != "fast":

            bits.append(f"profile={prof}")

        cfg_s = "  ".join(bits) if bits else ""

        return f"Spectrum  {rmse_lbl} {rmse_s}  d={d_s}  {cfg_s}"

    def _apply_spectrum_plot_title(self, r: dict | None) -> None:

        t = "Spectrum" if r is None else self._format_spectrum_plot_title(r)

        self.plot_T.plotItem.setTitle(t, color=CertusTheme.PRIMARY, size="11pt")

        self.plot_T._certus_init_title = t

    def _plot_data_raw(self) -> None:

        if self.df is None:

            return

        lam = ensure_lam_nm_array(self.df["lambda"].to_numpy(dtype=np.float64))

        x_plot, x_lbl = self._transform_spectrum_x(lam)

        self.plot_T.clear()

        any_curve = False

        if "T" in self.df.columns:

            y_raw = _to_fraction_T(self.df["T"].to_numpy(dtype=np.float64))

            y = y_raw

            tlabel = "T/Tsub exp" if self.chk_trel.isChecked() else "T exp"

            if self._add_curve(self.plot_T, x_plot, y, CertusTheme.PRIMARY, tlabel, True):

                any_curve = True

        if "R" in self.df.columns:

            y_raw = _to_fraction_T(self.df["R"].to_numpy(dtype=np.float64))

            y = y_raw

            rlabel = "R/Tsub exp" if self.chk_trel.isChecked() else "R exp"

            if self._add_curve(self.plot_T, x_plot, y, CertusTheme.SECONDARY, rlabel, True):

                any_curve = True

        if any_curve:

            self.plot_T.autoRange()

        self._apply_spectrum_x_axis_label(x_lbl)

        self._apply_spectrum_plot_title(None)

        self._update_rmse_fit_region_overlay()

    def _remove_rmse_fit_region_overlay(self) -> None:

        items = getattr(self, "_rmse_fit_overlay_items", None) or []

        for ri in items:

            if ri is None:

                continue

            try:

                self.plot_T.removeItem(ri)

            except (AttributeError, RuntimeError):

                logger.debug("_remove_rmse_fit_region_overlay: removeItem failed", exc_info=True)

        self._rmse_fit_overlay_items = []

    def _spectrum_plot_lambda_span_nm(self) -> tuple[float, float] | None:

        """lambda span of displayed spectrum (file first, else last result grid)."""

        if self.df is not None and "lambda" in self.df.columns:

            lam = ensure_lam_nm_array(self.df["lambda"].to_numpy(dtype=np.float64))

            lam = lam[np.isfinite(lam)]

            if lam.size:

                return float(np.min(lam)), float(np.max(lam))

        lr = getattr(self, "_last_result", None)

        if lr is not None and lr.get("lam_nm") is not None:

            lam = np.asarray(lr["lam_nm"], dtype=np.float64).ravel()

            lam = lam[np.isfinite(lam)]

            if lam.size:

                return float(np.min(lam)), float(np.max(lam))

        return None

    def _update_rmse_fit_region_overlay(self) -> None:

        self._remove_rmse_fit_region_overlay()

        if not getattr(self, "_rmse_fit_lambda_enabled", False):

            return

        span = self._spectrum_plot_lambda_span_nm()

        if span is None:

            return

        lam_file_lo, lam_file_hi = span

        if not (np.isfinite(lam_file_lo) and np.isfinite(lam_file_hi)):

            return

        if lam_file_lo > lam_file_hi:

            lam_file_lo, lam_file_hi = lam_file_hi, lam_file_lo

        wlo = float(self._rmse_fit_lambda_lo)

        whi = float(self._rmse_fit_lambda_hi)

        if not (np.isfinite(wlo) and np.isfinite(whi)):

            return

        lam_win_lo, lam_win_hi = min(wlo, whi), max(wlo, whi)

        def _x_span_lam(la: float, lb: float) -> tuple[float, float] | None:

            if not (np.isfinite(la) and np.isfinite(lb)):

                return None

            if la > lb:

                la, lb = lb, la

            if lb - la <= 0.0:

                return None

            try:

                xv, _ = self._transform_spectrum_x(np.array([la, lb], dtype=np.float64))

            except (TypeError, ValueError, RuntimeError):

                return None

            return float(np.min(xv)), float(np.max(xv))

        eps_lam = max(1e-6, 1e-9 * max(abs(lam_file_hi), 1.0))

        eps_x = max(1e-12, 1e-15 * max(abs(lam_file_lo), abs(lam_file_hi), 1.0))

        def _add_region(xa: float, xb: float, *, brush, z: float) -> None:

            lo_x, hi_x = (xa, xb) if xa <= xb else (xb, xa)

            if hi_x - lo_x <= eps_x:

                return

            reg = pg.LinearRegionItem(values=(lo_x, hi_x), movable=False, brush=brush)

            reg.setZValue(z)

            self.plot_T.addItem(reg)

            self._rmse_fit_overlay_items.append(reg)

        gray_brush = pg.mkBrush(120, 120, 120, 85)

        # Excludes lambda < window (within file envelope) - correct if sigma/sigma² (nonlinear in lambda on axis)

        if lam_win_lo > lam_file_lo + eps_lam:

            la, lb = lam_file_lo, min(lam_file_hi, lam_win_lo)

            if lb - la > eps_lam:

                xs = _x_span_lam(la, lb)

                if xs is not None:

                    _add_region(xs[0], xs[1], brush=gray_brush, z=-8.0)

        # Exclude lambda > window

        if lam_win_hi < lam_file_hi - eps_lam:

            la, lb = max(lam_file_lo, lam_win_hi), lam_file_hi

            if lb - la > eps_lam:

                xs = _x_span_lam(la, lb)

                if xs is not None:

                    _add_region(xs[0], xs[1], brush=gray_brush, z=-8.0)

        x_active = _x_span_lam(lam_win_lo, lam_win_hi)

        if x_active is not None:

            xa, xb = x_active

            if xb - xa > eps_x:

                reg_active = pg.LinearRegionItem(

                    values=(xa, xb),

                    movable=False,

                    brush=pg.mkBrush(0, 120, 215, 40),

                )

                reg_active.setZValue(-5.0)

                self.plot_T.addItem(reg_active)

                self._rmse_fit_overlay_items.append(reg_active)

    def _sync_rmse_lambda_bounds_from_file(self) -> None:

        if self.df is None or "lambda" not in self.df.columns:

            return

        lam = ensure_lam_nm_array(self.df["lambda"].to_numpy(dtype=np.float64))

        lam = lam[np.isfinite(lam)]

        if lam.size == 0:

            return

        lo, hi = float(np.min(lam)), float(np.max(lam))

        self._rmse_fit_lambda_lo_default = lo

        self._rmse_fit_lambda_hi_default = hi

        if not self._rmse_fit_lambda_enabled:

            self._rmse_fit_lambda_lo = lo

            self._rmse_fit_lambda_hi = hi

    def _on_rmse_fit_window_dialog(self) -> None:

        dlg = QDialog(self)

        dlg.setWindowTitle("Spectral RMSE Window")

        lay = QVBoxLayout(dlg)

        chk = QCheckBox("Limit optimization MSE/RMSE to a lambda band (nm)")

        chk.setChecked(self._rmse_fit_lambda_enabled)

        chk.setToolTip(

            "If checked: only experimental points in [lambda_min, lambda_max] enter the spectral loss. "

            "Plots always use the full loaded file."

        )

        lay.addWidget(chk)

        g = QGridLayout()

        lb_lo = QLabel("lambda_min (nm)")

        lb_hi = QLabel("lambda_max (nm)")

        sp_lo = QDoubleSpinBox()

        sp_hi = QDoubleSpinBox()

        for sp in (sp_lo, sp_hi):

            sp.setRange(200.0, 20000.0)

            sp.setDecimals(4)

            sp.setSingleStep(1.0)

        sp_lo.setValue(float(self._rmse_fit_lambda_lo))

        sp_hi.setValue(float(self._rmse_fit_lambda_hi))

        sp_lo.setToolTip("Lower bound (nm) of the RMSE band.")

        sp_hi.setToolTip("Upper bound (nm) of the RMSE band.")

        g.addWidget(lb_lo, 0, 0)

        g.addWidget(sp_lo, 0, 1)

        g.addWidget(lb_hi, 1, 0)

        g.addWidget(sp_hi, 1, 1)

        lay.addLayout(g)

        btn_reset = QPushButton("Reset")

        btn_reset.setToolTip(

            "Reset lambda_min / lambda_max to [min file, max file] of the loaded spectrum."

        )

        lay.addWidget(btn_reset)

        def _apply_enable(en: bool) -> None:

            sp_lo.setEnabled(en)

            sp_hi.setEnabled(en)

            btn_reset.setEnabled(en)

        def _do_reset() -> None:

            sp_lo.setValue(float(self._rmse_fit_lambda_lo_default))

            sp_hi.setValue(float(self._rmse_fit_lambda_hi_default))

        chk.toggled.connect(_apply_enable)

        _apply_enable(chk.isChecked())

        btn_reset.clicked.connect(_do_reset)

        bb = QDialogButtonBox(

            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        )

        bb.accepted.connect(dlg.accept)

        bb.rejected.connect(dlg.reject)

        lay.addWidget(bb)

        if dlg.exec() != QDialog.DialogCode.Accepted:

            return

        self._rmse_fit_lambda_enabled = chk.isChecked()

        self._rmse_fit_lambda_lo = float(sp_lo.value())

        self._rmse_fit_lambda_hi = float(sp_hi.value())

        if self._last_result is not None:

            self._plot_result(self._last_result)

        elif self.df is not None:

            self._plot_data_raw()

        else:

            self._update_rmse_fit_region_overlay()

    def _on_stop(self) -> None:

        self._stop_event.set()

        self.lbl_status.setText("Stop requested...")

    def _on_corr_mode_changed(self, _idx: int = 0) -> None:

        """Show alpha vs Delta RMSE spinboxes according to corridor mode (LR hides both thresholds)."""

        if not hasattr(self, "cb_corr_mode"):

            return

        m = str(self.cb_corr_mode.currentData() or "alpha")

        show_alpha = m == "alpha"

        show_delta = m == "abs_delta"

        if hasattr(self, "sp_corr_alpha"):

            self.sp_corr_alpha.setVisible(show_alpha)

        if hasattr(self, "lbl_corr_alpha"):

            self.lbl_corr_alpha.setVisible(show_alpha)

        if hasattr(self, "sp_corr_rmse_delta"):

            self.sp_corr_rmse_delta.setVisible(show_delta)

        if hasattr(self, "lbl_corr_rmse_delta"):

            self.lbl_corr_rmse_delta.setVisible(show_delta)

        if hasattr(self, "chk_corr_scientific_nominal"):

            self.chk_corr_scientific_nominal.setVisible(show_delta)

    def _refresh_corridors_gui_state_labels(self) -> None:

        """Show Yes / No according to corridor enable state (quick button = Basic checkbox)."""

        on = bool(

            getattr(self, "btn_corridor_toggle", None) is not None

            and self.btn_corridor_toggle.isChecked()

        )

        if on:

            rich = (

                f'Corridors (next run): <span style="color:{CertusTheme.SUCCESS};"><b>Yes</b></span>'

            )

        else:

            rich = (

                f'Corridors (next run): <span style="color:{CertusTheme.TEXT_SUB};"><b>No</b></span>'

            )

        if hasattr(self, "lbl_corridors_run_state"):

            self.lbl_corridors_run_state.setText(rich)

        if hasattr(self, "lbl_corridors_state_adv"):

            self.lbl_corridors_state_adv.setText(rich)

        if hasattr(self, "lbl_corridors_tab_state"):

            self.lbl_corridors_tab_state.setText(rich)

    def _on_btn_corridor_toggled(self, checked: bool) -> None:

        if not hasattr(self, "chk_corridor_d"):

            return

        self.chk_corridor_d.blockSignals(True)

        self.chk_corridor_d.setChecked(bool(checked))

        self.chk_corridor_d.blockSignals(False)

        self._refresh_corridors_gui_state_labels()

    def _sync_corridor_btn_from_chk(self) -> None:

        if not hasattr(self, "btn_corridor_toggle") or not hasattr(self, "chk_corridor_d"):

            return

        self.btn_corridor_toggle.blockSignals(True)

        self.btn_corridor_toggle.setChecked(self.chk_corridor_d.isChecked())

        self.btn_corridor_toggle.blockSignals(False)

        self._refresh_corridors_gui_state_labels()

    def _plot_nl_tab(

        self,

        r: dict,

        lam_m: np.ndarray,

        n_base: np.ndarray,

        k_base: np.ndarray,

    ) -> None:

        if not hasattr(self, "plot_n_nl") or not hasattr(self, "lbl_nl_summary"):

            return

        self.plot_n_nl.clear()

        self.plot_lgk_nl.clear()

        lam_m = np.asarray(lam_m, dtype=np.float64).ravel()

        n_base = np.asarray(n_base, dtype=np.float64).ravel()

        k_base = np.asarray(k_base, dtype=np.float64).ravel()

        bits: list[str] = []

        a = r.get("nl_alpha_opt")

        if a is not None:

            try:

                af = float(a)

            except (TypeError, ValueError):

                af = float("nan")

            if np.isfinite(af):

                bits.append(f"alpha<sub>NL</sub> = {af:.6f}")

            else:

                bits.append("alpha<sub>NL</sub> : -")

        else:

            bits.append("alpha<sub>NL</sub> : -")

        dnl = r.get("d_nm_nl")

        if dnl is not None:

            try:

                df = float(dnl)

                if np.isfinite(df):

                    bits.append(f"d (NL) = {df:.2f} nm")

            except (TypeError, ValueError):

                pass

        rref = r.get("nl_rmse_reference_best")

        rmo = r.get("nl_rmse_vs_meas_orig")

        rms = r.get("nl_rmse_vs_meas_scaled")

        if rref is not None:

            try:

                rf = float(rref)

                if np.isfinite(rf):

                    bits.append(f"Ref. RMSE (best spectral, no post NL) = {rf:.6f}")

            except (TypeError, ValueError):

                pass

        if rmo is not None:

            try:

                of = float(rmo)

                if np.isfinite(of):

                    bits.append(f"NL model RMSE vs raw measurements = {of:.6f}")

            except (TypeError, ValueError):

                pass

        if rms is not None:

            try:

                sf = float(rms)

                if np.isfinite(sf):

                    bits.append(f"NL model RMSE vs alpha×measurements (adjusted criterion) = {sf:.6f}")

            except (TypeError, ValueError):

                pass

        ok = r.get("nl_optim_ok")

        msg = str(r.get("nl_optim_message") or "")

        if ok is False and msg:

            bits.append(f"<span style='color:#c44'>{msg}</span>")

        gn = r.get("nl_alpha_grid_n")

        gs = r.get("nl_alpha_grid_step")

        sel_crit = str(r.get("nl_alpha_selection_criterion") or "")

        if sel_crit == "joint_objective_alpha_plus_x":

            bits.append(

                "<span style='color:#666'>joint local polish: alpha optimized together with d and spline nodes under a strong prior alpha≈1.</span>"

            )

        if gn is not None and gs is not None:

            try:

                bits.append(

                    "<span style='color:#666'>alpha sweep: {:d} values (step {:g}) - at each alpha, L-BFGS-B on "

                    "d and knots (masked MSE); best pair kept.</span>".format(int(gn), float(gs))

                )

            except (TypeError, ValueError):

                pass

        ne = r.get("nl_alpha_steps_evaluated")

        if ne is not None and gn is not None:

            try:

                if int(ne) != int(gn):

                    bits.append(

                        "<span style='color:#666'>steps evaluated: {:d} / {:d}".format(int(ne), int(gn))

                        + (" (early stop)</span>" if r.get("nl_alpha_scan_early_stopped") else "</span>")

                    )

            except (TypeError, ValueError):

                pass

        nid = r.get("nl_alpha_identifiability_note")

        if nid:

            bits.append(f"Identifiability: <span style='color:#666'>{nid}</span>")

        if r.get("nl_alpha_identifiable") is False:

            bits.append("<span style='color:#666'>No strong evidence for an α shift vs flat profile / α≈1.</span>")

        bh = r.get("nl_alpha_budget_maxfun_hits")

        if bh is not None:

            try:

                bits.append(f"L-BFGS-B budget_maxfun hits (grid): {int(bh)}")

            except (TypeError, ValueError):

                pass

        if (gn is None or gs is None) and r.get("nl_second_pass_applied"):

            bits.append("<span style='color:#666'>NL optimization in 2 L-BFGS-B passes (refinement).</span>")

        if r.get("n_lam_nl") is None and a is None:

            bits.append(

                "<span style='color:#888'>Run an optimization to compute alpha<sub>NL</sub> "

                "(“Non-lin. alpha” is checked by default; uncheck to disable).</span>"

            )

        self.lbl_nl_summary.setText("<br/>".join(bits))

        # lambda axis: keep strictly positive wavelengths (avoids sigma or abscissa artifacts).

        nlam = min(lam_m.size, n_base.size, k_base.size)

        if nlam < 2:

            return

        lam_u = lam_m[:nlam]

        nb_u = n_base[:nlam]

        kb_u = k_base[:nlam]

        mpos = np.isfinite(lam_u) & (lam_u > 1e-6) & np.isfinite(nb_u) & np.isfinite(kb_u)

        lam_p = lam_u[mpos]

        nb_p = nb_u[mpos]

        kb_p = kb_u[mpos]

        if lam_p.size < 2:

            return

        lk_b = np.full(kb_p.shape, np.nan, dtype=np.float64)

        mk = np.isfinite(kb_p) & (kb_p >= 0.0)

        lk_b[mk] = np.log10(np.maximum(kb_p[mk], 1e-30))

        self._add_curve(self.plot_n_nl, lam_p, nb_p, CertusTheme.TEXT_SUB, "n (no NL)")

        self._add_curve(self.plot_lgk_nl, lam_p, lk_b, CertusTheme.TEXT_SUB, "log₁₀ k (no NL)")

        n_nl = r.get("n_lam_nl")

        k_nl = r.get("k_lam_nl")

        lam_src = r.get("nl_lam_nm")

        if lam_src is None:

            lam_src = r.get("lam_nm")

        lam_src = (

            np.asarray(lam_src, dtype=np.float64).ravel()

            if lam_src is not None

            else np.array([], dtype=np.float64)

        )

        if n_nl is not None and k_nl is not None and lam_src.size >= 2:

            n_nl = np.asarray(n_nl, dtype=np.float64).ravel()

            k_nl = np.asarray(k_nl, dtype=np.float64).ravel()

            nn = int(min(lam_src.size, n_nl.size, k_nl.size))

            if nn >= 2:

                lam_src = lam_src[:nn]

                n_nl = n_nl[:nn]

                k_nl = k_nl[:nn]

                order = np.argsort(lam_src, kind="mergesort")

                lam0s = lam_src[order]

                n_nls = n_nl[order]

                k_nls = k_nl[order]

                msrc = np.isfinite(lam0s) & (lam0s > 1e-6)

                lam0s = lam0s[msrc]

                n_nls = n_nls[msrc]

                k_nls = k_nls[msrc]

                if lam0s.size >= 2:

                    n_nli = np.interp(lam_p, lam0s, n_nls, left=np.nan, right=np.nan)

                    k_nli = np.interp(lam_p, lam0s, k_nls, left=np.nan, right=np.nan)

                    self._add_curve(self.plot_n_nl, lam_p, n_nli, "#00aa44", "n (alpha NL)")

                    lk_nl = np.full(k_nli.shape, np.nan, dtype=np.float64)

                    mk2 = np.isfinite(k_nli) & (k_nli >= 0.0)

                    lk_nl[mk2] = np.log10(np.maximum(k_nli[mk2], 1e-30))

                    self._add_curve(self.plot_lgk_nl, lam_p, lk_nl, "#00aa44", "log₁₀ k (alpha NL)")

        self.plot_n_nl.autoRange()

        self.plot_lgk_nl.autoRange()

        span_lo = float(np.nanmin(lam_p))

        span_hi = float(np.nanmax(lam_p))

        if np.isfinite(span_lo) and np.isfinite(span_hi) and span_hi > span_lo:

            pad = 0.02 * (span_hi - span_lo)

            self.plot_n_nl.plotItem.setXRange(span_lo - pad, span_hi + pad, padding=0.0)

            self.plot_lgk_nl.plotItem.setXRange(span_lo - pad, span_hi + pad, padding=0.0)

    def _build_opt_config(self, *, notify: bool = True) -> SplineOptConfig | None:

        if self.df is None:

            if notify:

                QMessageBox.warning(self, "Data", "Load a file first.")

            return None

        sub_name = str(self.cb_sub.currentData() or self.cb_sub.currentText())

        sid = substrate_id_from_name(sub_name)

        lam = ensure_lam_nm_array(self.df["lambda"].to_numpy(dtype=np.float64))

        # Align ``SplineOptConfig.n_seg`` with the actual mesh (12 or 14 sigma knots) **before** any worker:

        # same rule as ``make_bounds_and_x0`` (IR extension if max(lambda) > 4000 nm), plus optional Deltalambda/lambdā min (Advanced).

        _mesh_mdl = (

            float(self.sp_mesh_min_dlam.value())

            if hasattr(self, "sp_mesh_min_dlam")

            else 0.02

        )

        _kmd_mesh: dict[str, float] = {}

        if _mesh_mdl > 0.0:

            _kmd_mesh["min_delta_lambda_over_lambda_mean"] = _mesh_mdl

        k_mesh_sigma = int(

            canonical_spline_sigma_knots(

                float(np.min(lam)), float(np.max(lam)), **_kmd_mesh

            ).size

        )

        n_seg_mesh = max(1, k_mesh_sigma - 1)

        try:

            n_sub = get_n_substrate_array_by_id(sid, lam)

        except Exception as e:

            if notify:

                QMessageBox.critical(self, "Substrate", str(e))

            return None

        has_t = "T" in self.df.columns

        has_r = "R" in self.df.columns

        use_t = self.chk_t.isChecked() and has_t

        use_r = self.chk_r.isChecked() and has_r

        if not use_t and not use_r:

            if notify:

                QMessageBox.warning(self, "Fit", "Enable at least T or R according to available columns.")

            return None

        if use_t and use_r:

            dt = DataType.BOTH

        elif use_r:

            dt = DataType.REFLECTION

        else:

            dt = DataType.TRANSMISSION

        t_raw = self.df["T"].to_numpy(dtype=np.float64) if has_t else None

        r_raw = self.df["R"].to_numpy(dtype=np.float64) if has_r else None

        t_is_ratio = bool(self.chk_trel.isChecked())

        t_exp, r_exp = prepare_exp_TR_for_fit(lam, n_sub, t_raw, r_raw, t_is_ratio=t_is_ratio)

        overlay = gui_perf_preset_only(str(self.cb_profilee.currentData() or "fast"))

        rmse_fit_lambda_nm: tuple[float, float] | None = None

        if getattr(self, "_rmse_fit_lambda_enabled", False):

            rl0 = float(self._rmse_fit_lambda_lo)

            rl1 = float(self._rmse_fit_lambda_hi)

            rmse_fit_lambda_nm = (min(rl0, rl1), max(rl0, rl1))

        _n_mono_band = default_n_mono_band_nm_from_spectrum(lam)

        cfg = SplineOptConfig(

            lam_nm=lam,

            t_exp=t_exp,

            r_exp=r_exp,

            n_sub=n_sub,

            data_type=dt,

            n_seg=int(n_seg_mesh),

            d_lo=float(self.d_lo.value()),

            d_hi=float(self.d_hi.value()),

            weight_t=float(self.w_t.value()) if use_t else 0.0,

            weight_r=float(self.w_r.value()) if use_r else 0.0,

            substrate_name=sub_name,

            t_is_ratio=t_is_ratio,

            pglobal_max_iter=0,

            polish_maxfun=int(overlay.get("polish_maxfun", 8000)),

            sol3_phase1_maxfun=(

                int(self.sp_sol3_p1_maxfun.value())

                if hasattr(self, "sp_sol3_p1_maxfun")

                else None

            ),

            pglobal_max_feval=None,

            pglobal_max_time=None,

            pglobal_local_search_budget=None,

            spline_local_only=True,

            n_mono_band_nm=_n_mono_band,

            n_mono_continuous_penalty=0.008,

            n_lambda_rising_penalty_band_nm=_n_mono_band,

            n_lambda_rising_penalty_weight=3000.0,

            rmse_fit_lambda_nm=rmse_fit_lambda_nm,

            nk_profile_interp="smooth",

            nonlinear_alpha_refinement_enabled=bool(

                getattr(self, "btn_nl_toggle", None) and self.btn_nl_toggle.isChecked()

            ),

            nonlinear_alpha_budget_mode=str(

                getattr(self, "cb_nl_alpha_budget", None).currentData() or "slow"

            )

            if hasattr(self, "cb_nl_alpha_budget")

            else "slow",

            nonlinear_alpha_second_pass_enabled=bool(

                getattr(self, "chk_nl_second_pass", None) and self.chk_nl_second_pass.isChecked()

            )

            if hasattr(self, "chk_nl_second_pass")

            else True,

            nl_alpha_adaptive_early_stop=bool(

                getattr(self, "chk_nl_adaptive_scan", None) and self.chk_nl_adaptive_scan.isChecked()

            )

            if hasattr(self, "chk_nl_adaptive_scan")

            else True,

            corridor_profile_d_enabled=bool(getattr(self, "chk_corridor_d", None) and self.chk_corridor_d.isChecked()),

            corridor_profile_d_mode=str(self.cb_corr_mode.currentData() or "abs_delta") if hasattr(self, "cb_corr_mode") else "abs_delta",

            corridor_profile_d_rmse_alpha=float(getattr(self, "sp_corr_alpha", None).value()) if hasattr(self, "sp_corr_alpha") else 1.05,

            corridor_profile_d_rmse_abs_tolerance=float(getattr(self, "sp_corr_rmse_delta", None).value())

            if hasattr(self, "sp_corr_rmse_delta")

            else 0.001,

            corridor_scientific_nominal_enabled=(

                not hasattr(self, "chk_corr_scientific_nominal")

                or bool(self.chk_corr_scientific_nominal.isChecked())

            ),

            corridor_profile_d_step_nm=float(getattr(self, "sp_corr_step", None).value()) if hasattr(self, "sp_corr_step") else 1.0,

            corridor_profile_d_max_span_nm=float(getattr(self, "sp_corr_span", None).value()) if hasattr(self, "sp_corr_span") else 15.0,

            corridor_profile_d_polish_maxfun=(

                None

                if (

                    hasattr(self, "sp_corr_prof_maxfun")

                    and int(self.sp_corr_prof_maxfun.value()) <= 0

                )

                else int(self.sp_corr_prof_maxfun.value())

            )

            if hasattr(self, "sp_corr_prof_maxfun")

            else None,

            corridor_profile_d_lr_conf_level=float(getattr(self, "sp_corr_conf", None).value()) if hasattr(self, "sp_corr_conf") else 0.95,

            corridor_profile_d_sigma_t=(

                None

                if (hasattr(self, "sp_corr_sigma") and float(self.sp_corr_sigma.value()) <= 0.0)

                else float(self.sp_corr_sigma.value())

            )

            if hasattr(self, "sp_corr_sigma")

            else None,

            corridor_profile_d_sigma_r=(

                None

                if (hasattr(self, "sp_corr_sigma") and float(self.sp_corr_sigma.value()) <= 0.0)

                else float(self.sp_corr_sigma.value())

            )

            if hasattr(self, "sp_corr_sigma")

            else None,

            corridor_profile_d_n_starts=int(getattr(self, "sp_corr_starts", None).value()) if hasattr(self, "sp_corr_starts") else 1,

            corridor_profile_d_jitter_n=float(getattr(self, "sp_corr_jn", None).value()) if hasattr(self, "sp_corr_jn") else 0.02,

            corridor_profile_d_jitter_L=float(getattr(self, "sp_corr_jL", None).value()) if hasattr(self, "sp_corr_jL") else 0.15,

            corridor_profile_d_rng_seed=int(getattr(self, "sp_corr_seed", None).value()) if hasattr(self, "sp_corr_seed") else 0,

            corridor_profile_d_sigma_hetero=bool(

                getattr(self, "chk_corr_sigma_hetero", None) and self.chk_corr_sigma_hetero.isChecked()

            )

            if hasattr(self, "chk_corr_sigma_hetero")

            else False,

            corridor_profile_d_sigma_hetero_scale=float(getattr(self, "sp_corr_hetero_scale", None).value())

            if hasattr(self, "sp_corr_hetero_scale")

            else 1.0,

            corridor_reg_sensitivity_enabled=bool(getattr(self, "chk_corr_reg_sens", None) and self.chk_corr_reg_sens.isChecked()),

            corridor_reg_sensitivity_points=int(getattr(self, "sp_corr_reg_pts", None).value()) if hasattr(self, "sp_corr_reg_pts") else 5,

            corridor_reg_sensitivity_decades=int(getattr(self, "sp_corr_reg_dec", None).value()) if hasattr(self, "sp_corr_reg_dec") else 2,

            corridor_bootstrap_enabled=bool(getattr(self, "chk_corr_boot", None) and self.chk_corr_boot.isChecked()),

            corridor_bootstrap_n=int(getattr(self, "sp_corr_boot_n", None).value()) if hasattr(self, "sp_corr_boot_n") else 40,

            corridor_bootstrap_seed=int(getattr(self, "sp_corr_boot_seed", None).value()) if hasattr(self, "sp_corr_boot_seed") else 0,

            corridor_bootstrap_percentile=float(getattr(self, "sp_corr_boot_p", None).value()) if hasattr(self, "sp_corr_boot_p") else 0.95,

            corridor_bootstrap_mode=str(getattr(self, "cb_corr_boot_mode", None).currentData() or "parametric") if hasattr(self, "cb_corr_boot_mode") else "parametric",

            corridor_bootstrap_block_len=int(getattr(self, "sp_corr_boot_block", None).value()) if hasattr(self, "sp_corr_boot_block") else 1,

            corridor_bootstrap_quick_refit=bool(

                getattr(self, "chk_corr_boot_refit", None) and self.chk_corr_boot_refit.isChecked()

            )

            if hasattr(self, "chk_corr_boot_refit")

            else False,

            corridor_bootstrap_quick_refit_maxfun=int(getattr(self, "sp_corr_boot_maxfun", None).value())

            if hasattr(self, "sp_corr_boot_maxfun")

            else 4000,

            corridor_bootstrap_n_workers=int(getattr(self, "sp_corr_boot_workers", None).value())

            if hasattr(self, "sp_corr_boot_workers")

            else 1,

            spline_min_delta_lambda_over_lambda_mean=float(_mesh_mdl),

        )

        if rmse_fit_lambda_nm is not None:

            n_ok = int(np.count_nonzero(_spline_objective_lam_mask(cfg)))

            if n_ok < SPLINE_MIN_RMSE_FIT_OBJECTIVE_POINTS:

                if notify:

                    QMessageBox.warning(

                        self,

                        "RMSE window",

                        f"Too few spectral points in the band ({n_ok} < {SPLINE_MIN_RMSE_FIT_OBJECTIVE_POINTS}). "

                        "Widen the window or disable the limit.",

                    )

                return None

        return cfg

    def _smart_mesh_objective_lam_mask_float(self, lam_r: np.ndarray) -> np.ndarray:

        """Same lambda mask as the spline objective on grid ``lam_r`` (result / SMART)."""

        lam_r = np.asarray(lam_r, dtype=np.float64).ravel()

        cfg_opt = self._build_opt_config(notify=False)

        if cfg_opt is not None:

            return objective_lam_mask_on_target_grid(cfg_opt, lam_r).astype(np.float64, copy=False)

        rw = self._rmse_fit_lambda_tuple_for_report()

        m = np.isfinite(lam_r).astype(np.float64, copy=False)

        if rw is not None:

            lo = float(min(rw[0], rw[1]))

            hi = float(max(rw[0], rw[1]))

            m *= ((lam_r >= lo) & (lam_r <= hi)).astype(np.float64)

        return m

    def _rmse_fit_lambda_tuple_for_report(self) -> tuple[float, float] | None:

        """lambda window for display/export (result first, else GUI)."""

        rw = None

        if self._last_result is not None:

            rw = self._last_result.get("rmse_fit_lambda_nm")

        if rw is None and getattr(self, "_rmse_fit_lambda_enabled", False):

            rl0 = float(self._rmse_fit_lambda_lo)

            rl1 = float(self._rmse_fit_lambda_hi)

            rw = (min(rl0, rl1), max(rl0, rl1))

        return rw

    def _smart_init_preview_hook(self, payload: dict) -> bool:

        """Called from the worker (QThread) after n_init/L_init logs; UI must run on the GUI thread."""

        app = QApplication.instance()

        if app is None:

            return True

        self._preview_ret = None

        if QThread.currentThread() == app.thread():

            return self._show_smart_init_preview_dialog(payload)

        # Thread worker -> GUI: demander explicitement la preview via signal Qt.

        self._preview_payload = payload

        self._preview_result = True

        self._preview_wait_event = Event()

        self.smart_preview_requested.emit(payload)

        # Augmentation du timeout a 10 minutes (600s) pour laisser le temps du tuning manual

        ok = self._preview_wait_event.wait(timeout=600.0)

        # Securite PyQt : rapatrier l'etat mute depuis le thread principal via variable d'instance.

        ret_tuple = getattr(self, "_preview_ret", None)

        if ret_tuple is not None:

            cfg = payload.get("cfg")

            if cfg is not None:

                sk, ne, Le, d_nm, rmse = ret_tuple

                cfg.smart_preview_exact_sigma_knots = sk

                cfg.smart_preview_exact_n_L = (ne, Le)

                cfg.smart_preview_d_nm_override = d_nm

                cfg.smart_preview_accepted_rmse = rmse

                # Signal au moteur de calcul qu'une injection manualle est disponible

                cfg.smart_init_manual_force_restart = True 

            self._preview_ret = None

        if not ok:

            logger.warning("Smart Init preview: GUI timeout (600s), continuing optimization.")

            return True

        return bool(getattr(self, "_preview_result", True))

    @pyqtSlot(object)

    def _on_smart_preview_requested(self, payload: object) -> None:

        try:

            if isinstance(payload, dict):

                self._preview_result = self._show_smart_init_preview_dialog(payload)

            else:

                self._preview_result = True

        except Exception:

            logger.exception("Smart Init preview: GUI error")

            self._preview_result = True

        finally:

            if self._preview_wait_event is not None:

                self._preview_wait_event.set()

    def _show_smart_init_preview_dialog(self, payload: dict) -> bool:

        """Manual Smart Init: PWL n and ln k on K sigma knots (K=12 or 14 from spectrum / RMSE window);

        if RMSE window is on: uniform sigma² mesh on the objective (often K=12) then bridge to worker K on Continue."""

        cfg = payload.get("cfg")

        sk = np.asarray(payload.get("sigma_knots"), dtype=np.float64).ravel()

        grids = payload.get("preview_grids")

        if cfg is None or grids is None or sk.size < 2:

            QMessageBox.warning(

                self,

                "Smart Init",

                "Incomplete preview data (cfg or grids). Continuing without adjustment.",

            )

            return True

        k_n = int(sk.size)

        n_phys = np.asarray(payload["n_nodes_physical"], dtype=np.float64).ravel().copy()

        L_nodes = np.asarray(payload["L_nodes"], dtype=np.float64).ravel().copy()

        if n_phys.size != k_n or L_nodes.size != k_n:

            QMessageBox.warning(self, "Smart Init", "n / L sizes are inconsistent with sigma knots.")

            return True

        rel_step = 0.005  # +/-0,5 % sur n et sur L = ln k

        L_lo_g = float(grids["L_lo"])

        L_hi_g = float(grids["L_hi"])

        # During this dialog: free physical n (non-monotone in sigma) if a mono band is active elsewhere;

        # ξ monotone reprojection applies on Continue (worker).

        _relax_si_mono = cfg.n_mono_band_nm is not None

        # Nb₂O₅ preset (ref. 12 abscissas in data): on open, Swanepoel is replaced by Nb₂O₅

        # only if K=12 (legacy). For K=14 (IR extension), keep Swanepoel n/L; user can still

        # apply Nb₂O₅ via “Apply preset” (interpolation on current sigma grid).

        if sk.size == SPLINE_PWL_K_NODES:

            sk, n_phys, L_nodes, d_total = _project_nb2o5_preset_to_sigma_knots(sk)

            payload["d_best_nm"] = d_total

        # Truncated RMSE lambda window: sigma mesh for +/- columns = uniform in sigma² on sig_f (objective), not full spectrum.

        if getattr(cfg, "rmse_fit_lambda_nm", None) is not None:

            sig_f_g = np.asarray(grids["sig_f"], dtype=np.float64).ravel()

            if int(sig_f_g.size) >= 2:

                sk_win = build_smart_manual_sigma_knots_from_preview_grid(

                    sig_f_g, n_uniform_in_sigma2=11

                )

                n_phys, L_nodes = interp_n_L_pwlnk_to_sigmas(sk, n_phys, L_nodes, sk_win)

                sk = sk_win

                k_n = int(sk.size)

        self.smart_preview_sk_arr = np.asarray(sk, dtype=np.float64).ravel().copy()

        # (sigma, n, L) triplets aligned on the instance: avoids drift vs local n_phys / L_nodes

        # after recomputation (curves, +/-) if other code reads smart_preview_sk_arr alone.

        self.smart_preview_n_phys = np.asarray(n_phys, dtype=np.float64).ravel().copy()

        self.smart_preview_L_nodes = np.asarray(L_nodes, dtype=np.float64).ravel().copy()

        self._si_mesh_sk_snap = self.smart_preview_sk_arr.copy()

        _d0 = payload.get("d_best_nm")

        if _d0 is None or not np.isfinite(float(_d0)):

            preview_d_nm = float(0.5 * (float(cfg.d_lo) + float(cfg.d_hi)))

        else:

            preview_d_nm = float(_d0)

        _, rm0 = rmse_at_spline_stage_x0_init(

            cfg,

            sk,

            n_phys,

            L_nodes,

            preview_d_nm,

            relax_n_mono=_relax_si_mono,

        )

        best_rmse = float(rm0)

        best_n = n_phys.copy()

        best_L = L_nodes.copy()

        current_rmse = float(rm0)

        dlg = QDialog(self)

        dlg.setWindowTitle(

            f"Smart Init  PWL n and ln k ({k_n} sigma knots · presets Nb₂O₅ · SiO₂ · Ta₂O₅)"

        )

        dlg.setMinimumWidth(1180)

        dlg.setMinimumHeight(620)

        # Auxiliary window for n(lambda) and log k(lambda)

        aux_dlg = QDialog(dlg)

        aux_dlg.setWindowTitle("Optical Profiles  n(lambda) and ln k(lambda)")

        aux_dlg.setMinimumWidth(500)

        aux_dlg.setMinimumHeight(500)

        aux_lay = QVBoxLayout(aux_dlg)

        pw_nk = CertusScientificPlot()

        pw_nk.showGrid(x=True, y=True, alpha=0.3)

        pw_nk.setLabel("bottom", "lambda (nm)")

        pw_nk.addLegend()

        attach_excel_clipboard_context_menu(pw_nk)

        aux_lay.addWidget(wrap_scientific_plot_with_toolbar(aux_dlg, pw_nk))

        # Curves for n and ln k

        curve_n = pw_nk.plot(pen=pg.mkPen(CertusTheme.PRIMARY, width=2), name="n(lambda)")

        # Axe Y secondaire pour ln k

        main_vb = pw_nk.plotItem.vb

        p_extra = pg.ViewBox()

        pw_nk.scene().addItem(p_extra)

        pw_nk.getAxis("right").linkToView(p_extra)

        p_extra.setXLink(main_vb)

        curve_pk = pg.PlotCurveItem(pen=pg.mkPen(CertusTheme.ACCENT, width=2, style=Qt.PenStyle.DashLine), name="ln k(lambda)")

        # Clipboard / CSV export: always k, never ln k (see certus_ui _export_y_values_for_item).

        curve_pk._certus_export_y_as_exp_k = True

        curve_pk._certus_export_name_override = "k"

        p_extra.addItem(curve_pk)

        pw_nk._certus_clipboard_df_provider = lambda: _smart_init_pw_nk_clipboard_df(

            curve_n, curve_pk

        )

        def update_aux_layout():

            p_extra.setGeometry(main_vb.sceneBoundingRect())

        main_vb.sigResized.connect(update_aux_layout)

        lay = QVBoxLayout(dlg)

        h_x_main = QHBoxLayout()

        h_x_main.addWidget(QLabel("X axis (spectrum):"))

        cb_x_main = QComboBox()

        cb_x_main.addItems(["Lambda (nm)", "Sigma (nm⁻1)", "Sigma2 (nm⁻2)"])

        h_x_main.addWidget(cb_x_main)

        h_x_main.addStretch()

        lay.addLayout(h_x_main)

        if _relax_si_mono:

            lbl_mono_relax = QLabel(

                "<b>Manual tuning</b>: <i>n</i> may be <b>non-monotone</b> in sigma between knots here "

                "(sliders / editor). <b>After Continue</b>: optimization uses the "

                "<b>ξ reparametrization</b> - <i>n</i> non-decreasing in sigma on the run’s lambda band "

                "(so in practice <i>n</i> <b>decreasing or quasi-flat</b> as lambda increases on these segments), "

                "plus a penalty (UV-VIS band) if <i>n</i> rises too much with lambda "

                "(small slack on this penalty is configurable)."

            )

            lbl_mono_relax.setWordWrap(True)

            lbl_mono_relax.setStyleSheet(

                f"color: {CertusTheme.WARNING}; font-size: 11px; padding: 2px 0;"

            )

            lay.addWidget(lbl_mono_relax)

        d_lo_nm = float(cfg.d_lo)

        d_hi_nm = float(cfg.d_hi)

        _D_SLIDER_STEPS = 5000

        def _d_from_slider_int(iv: int) -> float:

            if d_hi_nm <= d_lo_nm + 1e-30:

                return float(d_lo_nm)

            t = float(iv) / float(_D_SLIDER_STEPS)

            return float(d_lo_nm + t * (d_hi_nm - d_lo_nm))

        def _slider_int_from_d_nm(dv: float) -> int:

            if d_hi_nm <= d_lo_nm + 1e-30:

                return 0

            dv = float(np.clip(dv, d_lo_nm, d_hi_nm))

            t = (dv - d_lo_nm) / (d_hi_nm - d_lo_nm)

            return int(round(t * _D_SLIDER_STEPS))

        row_d = QHBoxLayout()

        row_d.addWidget(QLabel("Thickness d:"))

        slider_d = QSlider(Qt.Orientation.Horizontal)

        slider_d.setRange(0, _D_SLIDER_STEPS)

        slider_d.setToolTip(

            "Slider between fit d min and d max. The +/- buttons on n and ln k do not change d; "

            "move this slider to try different thickness."

        )

        lbl_d_slider = QLabel()

        lbl_d_slider.setMinimumWidth(220)

        row_d.addWidget(slider_d, 1)

        row_d.addWidget(lbl_d_slider)

        lay.addLayout(row_d)

        btn_show_nk = QPushButton("Display profiles n, ln k (lambda)")

        btn_show_nk.setFixedWidth(200)

        btn_show_nk.clicked.connect(aux_dlg.show)

        lay.addWidget(btn_show_nk)

        pw = CertusScientificPlot()

        pw.setMinimumHeight(300)

        pw.showGrid(x=True, y=True, alpha=0.35)

        lam_m = np.asarray(payload["lam_nm"], dtype=np.float64)

        y_exp = np.asarray(payload["t_exp"], dtype=np.float64)

        y_th0 = np.asarray(payload["t_theo"], dtype=np.float64)

        def _interp_t_at_lam_knots(lam_grid: np.ndarray, t_grid: np.ndarray, cur_sk: np.ndarray) -> np.ndarray:

            lam_g = np.asarray(lam_grid, dtype=np.float64).ravel()

            t_g = np.asarray(t_grid, dtype=np.float64).ravel()

            lam_k = 1.0 / np.maximum(np.asarray(cur_sk, dtype=np.float64).ravel(), 1e-30)

            o = np.argsort(lam_g)

            return np.interp(lam_k, lam_g[o], t_g[o], left=t_g[o[0]], right=t_g[o[-1]])

        sig_pts = (1.0 / np.maximum(lam_m, 1e-9))**2

        ord_sig = np.argsort(sig_pts)

        sig_s = sig_pts[ord_sig]

        y_exp_s = y_exp[ord_sig]

        y_th0_s = y_th0[ord_sig]

        y_lab = "T/T_sub" if payload.get("t_is_ratio") else "T"

        pw.setLabel("bottom", "sigma2 = 1/lambda2 (nm⁻2)")

        pw.setLabel("left", y_lab)

        pw.addLegend()

        curve_exp = pw.plot(

            [], [],

            pen=None,

            symbol="o",

            symbolSize=5,

            symbolBrush=pg.mkBrush(CertusTheme.ACCENT),

            name="Measurement",

        )

        curve_theo = pw.plot(

            [], [],

            pen=pg.mkPen(CertusTheme.PRIMARY, width=2.5),

            name="Theoretical (PWL n, ln k | d = slider)",

        )

        knot_markers = pw.plot(

            [], [],

            pen=None,

            symbol="s",

            symbolSize=9,

            symbolBrush=pg.mkBrush("#c97800"),

            name="T at knots",

        )

        def get_xv(sx, mode):

            if mode == "Sigma (nm⁻1)": return float(sx)

            elif mode == "Sigma2 (nm⁻2)": return float(sx)**2

            else: return 1.0/float(sx) if sx != 0 else 0

        knot_lines = []

        def redraw_knot_lines():

            for line in knot_lines:

                try:

                    pw.removeItem(line)

                except (AttributeError, RuntimeError):

                    pass

            knot_lines.clear()

            pen_k = pg.mkPen("#1a9f3c", width=1.8)

            mode = cb_x_main.currentText()

            sk_lines = np.asarray(

                getattr(self, "smart_preview_sk_arr", sk), dtype=np.float64

            ).ravel()

            for sx in sk_lines:

                il = pg.InfiniteLine(get_xv(sx, mode), angle=90, pen=pen_k)

                pw.addItem(il)

                knot_lines.append(il)

        redraw_knot_lines()

        cb_x_main.currentIndexChanged.connect(redraw_knot_lines)

        current_t_th = y_th0.copy()

        def _study_lambda_window_nm() -> tuple[float, float]:

            """Useful lambda band: RMSE window if set, else loaded spectrum (finite points)."""

            lam = np.asarray(lam_m, dtype=np.float64).ravel()

            ok = np.isfinite(lam) & (lam > 0)

            if not np.any(ok):

                return 400.0, 1200.0

            lo_d = float(np.min(lam[ok]))

            hi_d = float(np.max(lam[ok]))

            rw = getattr(cfg, "rmse_fit_lambda_nm", None)

            if rw is None:

                return lo_d, hi_d

            lo_w = float(min(rw[0], rw[1]))

            hi_w = float(max(rw[0], rw[1]))

            lo = max(lo_d, lo_w)

            hi = min(hi_d, hi_w)

            if hi <= lo:

                return lo_d, hi_d

            return lo, hi

        def _apply_manual_spectrum_plot_range() -> None:

            """Auto scales centered on the study region (lambda or sigma / sigma²), not the full axis span."""

            lo_s, hi_s = _study_lambda_window_nm()

            pad_l = max((hi_s - lo_s) * 0.02, 1e-6)

            mode = cb_x_main.currentIndex()

            if mode == 0:

                x_lo, x_hi = float(lo_s - pad_l), float(hi_s + pad_l)

            elif mode == 1:

                x_lo = 1.0 / float(hi_s + pad_l)

                x_hi = 1.0 / float(max(lo_s - pad_l, 1e-30))

            else:

                x_lo = (1.0 / float(hi_s + pad_l)) ** 2

                x_hi = (1.0 / float(max(lo_s - pad_l, 1e-30))) ** 2

            if x_hi < x_lo:

                x_lo, x_hi = x_hi, x_lo

            cur_sk = getattr(self, "smart_preview_sk_arr", sk_arr)

            k_vals = (

                1.0 / np.maximum(cur_sk, 1e-30),

                np.asarray(cur_sk, dtype=np.float64),

                np.asarray(cur_sk, dtype=np.float64) ** 2,

            )[mode]

            kv = np.asarray(k_vals, dtype=np.float64).ravel()

            kv = kv[np.isfinite(kv)]

            if kv.size:

                x_lo = min(float(x_lo), float(np.min(kv)))

                x_hi = max(float(x_hi), float(np.max(kv)))

            if x_hi < x_lo:

                x_lo, x_hi = x_hi, x_lo

            pad_x = max((x_hi - x_lo) * 0.015, 1e-24)

            lam = np.asarray(lam_m, dtype=np.float64).ravel()

            ye = np.asarray(y_exp, dtype=np.float64).ravel()

            yt = np.asarray(current_t_th, dtype=np.float64).ravel()

            n = int(min(lam.size, ye.size, yt.size))

            if n <= 0:

                return

            lam, ye, yt = lam[:n], ye[:n], yt[:n]

            m = np.isfinite(lam) & (lam >= lo_s) & (lam <= hi_s)

            if not np.any(m):

                m = np.isfinite(lam)

            yy = np.concatenate([ye[m], yt[m]])

            yy = yy[np.isfinite(yy)]

            cur_sk2 = getattr(self, "smart_preview_sk_arr", sk_arr)

            knot_t = _interp_t_at_lam_knots(lam_m, current_t_th, cur_sk2)

            kt = np.asarray(knot_t, dtype=np.float64).ravel()

            kt = kt[np.isfinite(kt)]

            if kt.size:

                yy = np.concatenate([yy, kt]) if yy.size else kt

            if yy.size == 0:

                yy = np.array([0.0, 1.0], dtype=np.float64)

            y_lo, y_hi = float(np.min(yy)), float(np.max(yy))

            if y_hi <= y_lo:

                y_hi = y_lo + 1e-6

            pad_y = max((y_hi - y_lo) * 0.08, 1e-5)

            pw.setXRange(float(x_lo - pad_x), float(x_hi + pad_x), padding=0)

            pw.setYRange(float(y_lo - pad_y), float(y_hi + pad_y), padding=0)

        def refresh_nk_plots_aux(lam_nk: np.ndarray, n_lam: np.ndarray, k_lam: np.ndarray) -> None:

            curve_n.setData(lam_nk, n_lam)

            ln_k = np.log(np.maximum(k_lam, 1e-12))

            curve_pk.setData(lam_nk, ln_k)

            lo_s, hi_s = _study_lambda_window_nm()

            pad_l = max((hi_s - lo_s) * 0.02, 1e-6)

            lam_a = np.asarray(lam_nk, dtype=np.float64).ravel()

            n_a = np.asarray(n_lam, dtype=np.float64).ravel()

            ln_a = np.asarray(ln_k, dtype=np.float64).ravel()

            n_pts = min(lam_a.size, n_a.size, ln_a.size)

            if n_pts <= 0:

                return

            lam_a = lam_a[:n_pts]

            n_a = n_a[:n_pts]

            ln_a = ln_a[:n_pts]

            m = np.isfinite(lam_a) & (lam_a >= lo_s) & (lam_a <= hi_s)

            if not np.any(m):

                m = np.isfinite(lam_a)

            main_vb.setXRange(float(lo_s - pad_l), float(hi_s + pad_l), padding=0)

            nn = n_a[m]

            nn = nn[np.isfinite(nn)]

            if nn.size > 0:

                n_lo, n_hi = float(np.min(nn)), float(np.max(nn))

                pr = max((n_hi - n_lo) * 0.06, 1e-6)

                main_vb.setYRange(n_lo - pr, n_hi + pr, padding=0)

            lk = ln_a[m]

            lk = lk[np.isfinite(lk)]

            if lk.size > 0:

                lk_lo, lk_hi = float(np.min(lk)), float(np.max(lk))

                pr = max((lk_hi - lk_lo) * 0.08, 1e-6)

                p_extra.setYRange(lk_lo - pr, lk_hi + pr, padding=0)

        # --- NEW : MONITORING LIVE DES INDICES ---

        class LiveIndexMonitor(QDialog):

            def __init__(self, parent=None):

                super().__init__(parent)

                self.setWindowTitle("Monitoring Indices (Live)")

                self.resize(550, 700)

                l = QVBoxLayout(self)

                h = QHBoxLayout()

                h.addWidget(QLabel("X Axis Unit:"))

                self.cb = QComboBox()

                self.cb.addItems(["Lambda (nm)", "Sigma (nm⁻1)", "Sigma2 (nm⁻2)"])

                def on_unit_change():

                    if hasattr(self, "_last_data"):

                        self.update_indices(*self._last_data)

                self.cb.currentIndexChanged.connect(on_unit_change)

                h.addWidget(self.cb)

                self._btn_copy_nk_2nm = create_styled_button(

                    "Copy lambda, n, k (2 nm step)", "secondary", parent=self

                )

                self._btn_copy_nk_2nm.setToolTip(

                    "Clipboard: lambda (integer nm), n, k sorted by increasing lambda, "

                    "interpolated on a 2 nm grid (TSV)."

                )

                self._btn_copy_nk_2nm.clicked.connect(self._copy_nk_clipboard_2nm)

                h.addWidget(self._btn_copy_nk_2nm)

                h.addStretch()

                l.addLayout(h)

                self.lbl_d = QLabel("d =  nm")

                self.lbl_d.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 11px;")

                l.addWidget(self.lbl_d)

                self.p_n = CertusScientificPlot(title="Index n")

                self.p_k = CertusScientificPlot(title="Index k  Log Scale")

                self.p_k.setLogMode(False, True) 

                l.addWidget(self.p_n)

                l.addWidget(self.p_k)

                apply_certus_theme(self)

            def _copy_nk_clipboard_2nm(self) -> None:

                if not hasattr(self, "_last_data") or self._last_data is None:

                    QMessageBox.information(

                        self,

                        "Clipboard",

                        "No n, k data (wait for live update).",

                    )

                    return

                lam_arr, n_arr, k_arr, _ = self._last_data

                txt = _live_monitor_nk_clipboard_tsv_2nm(lam_arr, n_arr, k_arr)

                if not txt:

                    QMessageBox.information(

                        self,

                        "Clipboard",

                        "No valid points for export.",

                    )

                    return

                cb = QApplication.clipboard()

                if cb is None:

                    QMessageBox.warning(self, "Clipboard", "Clipboard unavailable.")

                    return

                cb.setText(txt)

                prev = self._btn_copy_nk_2nm.text()

                self._btn_copy_nk_2nm.setText("Copied!")

                QTimer.singleShot(

                    1500,

                    lambda t=prev: self._btn_copy_nk_2nm.setText(t),

                )

            def update_indices(self, lam_arr, n_arr, k_arr, d_nm=None):

                self._last_data = (lam_arr, n_arr, k_arr, d_nm)

                mode = self.cb.currentIndex()

                if mode == 0:

                    x, lbl = lam_arr, "lambda (nm)"

                elif mode == 1:

                    x, lbl = 1.0 / lam_arr, "sigma (nm⁻1)"

                else:

                    x, lbl = (1.0 / lam_arr)**2, "sigma2 (nm⁻2)"

                if d_nm is not None and np.isfinite(float(d_nm)):

                    self.lbl_d.setText(f"d = {float(d_nm):.1f} nm")

                self.p_n.setLabel('bottom', lbl)

                self.p_k.setLabel('bottom', lbl)

                if "n" not in self.p_n._curves:

                    self.p_n.add_curve(x, n_arr, "n", color=CertusTheme.PRIMARY, width=2)

                else:

                    self.p_n.update_curve("n", x, n_arr)

                if "k" not in self.p_k._curves:

                    self.p_k.add_curve(x, k_arr, "k", color=CertusTheme.DANGER, width=2)

                else:

                    self.p_k.update_curve("k", x, k_arr)

                study_fn = getattr(self, "_study_lam_window_fn", None)

                if not callable(study_fn):

                    self.p_n.autoRange()

                    self.p_k.autoRange()

                    return

                try:

                    lo_s, hi_s = study_fn()

                except (TypeError, ValueError, RuntimeError):

                    self.p_n.autoRange()

                    self.p_k.autoRange()

                    return

                if not (hi_s > lo_s and np.isfinite(lo_s) and np.isfinite(hi_s)):

                    self.p_n.autoRange()

                    self.p_k.autoRange()

                    return

                pad_l = max((hi_s - lo_s) * 0.02, 1e-6)

                lam_f = np.asarray(lam_arr, dtype=np.float64).ravel()

                n_f = np.asarray(n_arr, dtype=np.float64).ravel()

                k_f = np.asarray(k_arr, dtype=np.float64).ravel()

                npt = min(lam_f.size, n_f.size, k_f.size)

                if npt <= 0:

                    self.p_n.autoRange()

                    self.p_k.autoRange()

                    return

                lam_f, n_f, k_f = lam_f[:npt], n_f[:npt], k_f[:npt]

                mwin = np.isfinite(lam_f) & (lam_f >= lo_s) & (lam_f <= hi_s)

                if not np.any(mwin):

                    mwin = np.isfinite(lam_f)

                if mode == 0:

                    x_lo, x_hi = float(lo_s - pad_l), float(hi_s + pad_l)

                elif mode == 1:

                    x_lo = 1.0 / float(hi_s + pad_l)

                    x_hi = 1.0 / float(max(lo_s - pad_l, 1e-30))

                else:

                    x_lo = (1.0 / float(hi_s + pad_l)) ** 2

                    x_hi = (1.0 / float(max(lo_s - pad_l, 1e-30))) ** 2

                if x_hi < x_lo:

                    x_lo, x_hi = x_hi, x_lo

                pad_x = max((x_hi - x_lo) * 0.02, 1e-24)

                x0, x1 = float(x_lo - pad_x), float(x_hi + pad_x)

                self.p_n.plotItem.setXRange(x0, x1, padding=0)

                self.p_k.plotItem.setXRange(x0, x1, padding=0)

                nn = n_f[mwin]

                nn = nn[np.isfinite(nn)]

                if nn.size > 0:

                    n_lo, n_hi = float(np.min(nn)), float(np.max(nn))

                    pr = max((n_hi - n_lo) * 0.07, 1e-6)

                    self.p_n.plotItem.setYRange(n_lo - pr, n_hi + pr, padding=0)

                kk = k_f[mwin]

                kk = kk[np.isfinite(kk) & (kk > 0)]

                if kk.size > 0:

                    k_lo = max(float(np.min(kk)), 1e-30)

                    k_hi = max(float(np.max(kk)), k_lo * 1.0001)

                    self.p_k.plotItem.setYRange(k_lo * 0.85, k_hi * 1.15, padding=0)

        mon = getattr(self, "_live_nk_monitor", None)

        if mon is None or not hasattr(mon, "update_indices"):

            mon = LiveIndexMonitor(self)

            self._live_nk_monitor = mon

        mon._study_lam_window_fn = _study_lambda_window_nm

        mon.show()

        # Positionner a droite du dialog de preview (si visible).

        try:

            mon.move(dlg.x() + dlg.width() + 10, dlg.y())

        except (AttributeError, RuntimeError):

            pass

        def refresh_nk_plots_mon(lam_u, n_lam_u, k_lam_u) -> None:

            mon.update_indices(lam_u, n_lam_u, k_lam_u, preview_d_nm)

        lbl_stats = QLabel()

        lbl_stats.setWordWrap(True)

        def refresh_stats(dv: float, rm: float) -> None:

            nonlocal current_rmse

            current_rmse = float(rm)

            rmse_lbl = "RMSE"

            if (

                cfg.data_type == DataType.BOTH

                and float(cfg.weight_t) > 0.0

                and float(cfg.weight_r) > 0.0

            ):

                rmse_lbl = "RMSE (√MSE objectif T+R, comme au 1er cout optimization)"

            lbl_stats.setText(

                f"Summary: dialogue mesh K={k_n} sigma knots (worker = canonical file K after 'Continue' if different). "

                f"Squares = theoretical T at knots. Continue -> local refinement on this mesh. "

                f"d {dv:.2f} nm | stage {rmse_lbl} start (x0 after clip, before L-BFGS-B) {rm:.6f} "

                f"| best reached {best_rmse:.6f}"

            )

        refresh_stats(preview_d_nm, rm0)

        # Colonnes alignees sous les sigma du plot (espacements  Deltasigma sur l'axe).

        sk_arr = np.asarray(sk, dtype=np.float64).ravel()

        sig2_arr = sk_arr**2

        sig_sort_idx = np.argsort(sk_arr)

        sk_sorted = sk_arr[sig_sort_idx]

        sig2_sorted = sig2_arr[sig_sort_idx]

        s2_lo_f = float(min(float(np.min(sig_pts)), float(np.min(sig2_arr))))

        s2_hi_f = float(max(float(np.max(sig_pts)), float(np.max(sig2_arr))))

        span_sig2 = max(s2_hi_f - s2_lo_f, 1e-30)

        def update_main_x_axes():

            mode = cb_x_main.currentIndex()

            x_L = lam_m

            x_s = 1.0 / np.maximum(lam_m, 1e-30)

            x_s2 = x_s**2

            cur_sk = getattr(self, 'smart_preview_sk_arr', sk_arr)

            k_L = 1.0 / np.maximum(cur_sk, 1e-30)

            k_s = cur_sk

            k_s2 = cur_sk**2

            x_vals = [x_L, x_s, x_s2][mode]

            k_vals = [k_L, k_s, k_s2][mode]

            lbl = ["lambda (nm)", "sigma (nm⁻1)", "sigma2 = 1/lambda2 (nm⁻2)"][mode]

            pw.setLabel("bottom", lbl)

            o = np.argsort(x_vals)

            curve_exp.setData(x_vals[o], y_exp[o])

            curve_theo.setData(x_vals[o], current_t_th[o])

            knot_t = _interp_t_at_lam_knots(lam_m, current_t_th, cur_sk)

            knot_markers.setData(k_vals, knot_t)

            for j, il in enumerate(knot_lines):

                if j < len(k_vals):

                    il.setPos(k_vals[j])

            _apply_manual_spectrum_plot_range()

        cb_x_main.currentIndexChanged.connect(update_main_x_axes)

        cb_x_main.setCurrentIndex(2)

        lbl_lam_cols: list[QLabel] = []

        lbl_sig_cols: list[QLabel] = []

        lbl_n_cols: list[QLabel] = []

        lbl_L_cols: list[QLabel] = []

        knot_bar = QWidget()

        knot_h = QHBoxLayout(knot_bar)

        knot_h.setContentsMargins(2, 4, 2, 2)

        knot_h.setSpacing(0)

        def _stretch_sig(delta: float) -> int:

            return max(1, int(max(0.0, float(delta)) / span_sig2 * 28000.0))

        n_btn_pairs: list[tuple[QPushButton, QPushButton]] = []

        L_btn_pairs: list[tuple[QPushButton, QPushButton]] = []

        n_auto_btns: list[QPushButton] = []

        L_auto_btns: list[QPushButton] = []

        curve_editor_holder: list[SmartInitNKCurveEditorDialog] = []

        def rebuild_knot_ui(new_kn):

            nonlocal k_n

            k_n = new_kn

            # Vidage du layout actuel

            while knot_h.count():

                item = knot_h.takeAt(0)

                if item.widget():

                    item.widget().deleteLater()

            lbl_lam_cols.clear()

            lbl_sig_cols.clear()

            lbl_n_cols.clear()

            lbl_L_cols.clear()

            n_btn_pairs.clear()

            L_btn_pairs.clear()

            n_auto_btns.clear()

            L_auto_btns.clear()

            # Reconstruction des colonnes

            current_sk = getattr(self, 'smart_preview_sk_arr', sk_arr)

            sig2_sorted_loc = np.sort(current_sk**2)

            # UPDATE DES MARQUEURS SUR LE GRAPHE (consolide)

            redraw_knot_lines()

            knot_h.addStretch(_stretch_sig(float(sig2_sorted_loc[0] - s2_lo_f)))

            for j in range(k_n):

                # ... (creation widgets)

                col_w = QWidget()

                cv = QVBoxLayout(col_w)

                cv.setContentsMargins(0, 0, 0, 0)

                cv.setSpacing(2)

                col_w.setFixedWidth(112)

                lam_l = QLabel()

                lam_l.setAlignment(Qt.AlignmentFlag.AlignHCenter)

                lam_l.setStyleSheet(f"font-size: 9px; color: {CertusTheme.TEXT_SUB};")

                sig_l = QLabel()

                sig_l.setAlignment(Qt.AlignmentFlag.AlignHCenter)

                sig_l.setStyleSheet(f"font-size: 9px; color: {CertusTheme.TEXT_SUB};")

                lbl_lam_cols.append(lam_l)

                lbl_sig_cols.append(sig_l)

                cap_n = QLabel("n")

                cap_n.setAlignment(Qt.AlignmentFlag.AlignHCenter)

                cap_n.setStyleSheet(f"font-size: 8px; color: {CertusTheme.TEXT_SUB};")

                row_n = QHBoxLayout()

                row_n.setSpacing(1)

                bm_n = QPushButton("")

                bm_n.setFixedWidth(22)

                val_n = QLabel()

                val_n.setAlignment(Qt.AlignmentFlag.AlignCenter)

                val_n.setMinimumWidth(36)

                val_n.setStyleSheet("font-size: 10px;")

                bp_n = QPushButton("+")

                bp_n.setFixedWidth(22)

                b_auto_n = QPushButton("auto")

                b_auto_n.setFixedWidth(34)

                b_auto_n.setStyleSheet("font-size: 7px; padding: 1px 2px;")

                lbl_n_cols.append(val_n)

                row_n.addWidget(bm_n)

                row_n.addWidget(val_n, 1)

                row_n.addWidget(bp_n)

                row_n.addWidget(b_auto_n)

                n_auto_btns.append(b_auto_n)

                cap_L = QLabel("ln k")

                cap_L.setAlignment(Qt.AlignmentFlag.AlignHCenter)

                cap_L.setStyleSheet(f"font-size: 8px; color: {CertusTheme.TEXT_SUB};")

                row_L = QHBoxLayout()

                row_L.setSpacing(1)

                bm_L = QPushButton("")

                bm_L.setFixedWidth(22)

                val_L = QLabel()

                val_L.setAlignment(Qt.AlignmentFlag.AlignCenter)

                val_L.setMinimumWidth(36)

                val_L.setStyleSheet("font-size: 10px;")

                bp_L = QPushButton("+")

                bp_L.setFixedWidth(22)

                b_auto_L = QPushButton("auto")

                b_auto_L.setFixedWidth(34)

                b_auto_L.setStyleSheet("font-size: 7px; padding: 1px 2px;")

                lbl_L_cols.append(val_L)

                row_L.addWidget(bm_L)

                row_L.addWidget(val_L, 1)

                row_L.addWidget(bp_L)

                row_L.addWidget(b_auto_L)

                L_auto_btns.append(b_auto_L)

                n_btn_pairs.append((bm_n, bp_n))

                L_btn_pairs.append((bm_L, bp_L))

                cv.addWidget(lam_l)

                cv.addWidget(sig_l)

                cv.addWidget(cap_n)

                cv.addLayout(row_n)

                cv.addWidget(cap_L)

                cv.addLayout(row_L)

                knot_h.addWidget(col_w, 0)

                if j + 1 < k_n:

                    knot_h.addStretch(_stretch_sig(float(sig2_sorted_loc[j + 1] - sig2_sorted_loc[j])))

            # Rewire +/- / auto buttons for the current k_n sigma knots

            current_sk = getattr(self, 'smart_preview_sk_arr', sk_arr)

            sig2_sorted_loc = np.sort(current_sk**2)

            sig_sort_idx_loc = np.argsort(current_sk)

            # Rewire events

            for j in range(k_n):

                oi = int(sig_sort_idx_loc[j])

                bm_n, bp_n = n_btn_pairs[j]

                bm_L, bp_L = L_btn_pairs[j]

                wire_hold_button(bm_n, oi, -1, is_ln_k=False)

                wire_hold_button(bp_n, oi, +1, is_ln_k=False)

                wire_hold_button(bm_L, oi, -1, is_ln_k=True)

                wire_hold_button(bp_L, oi, +1, is_ln_k=True)

                n_auto_btns[j].clicked.connect(lambda _=False, r=oi: run_auto(r, False))

                L_auto_btns[j].clicked.connect(lambda _=False, r=oi: run_auto(r, True))

            sync_knot_labels()

            for _ce in curve_editor_holder:

                try:

                    _ce.refresh_plots()

                except (AttributeError, RuntimeError):

                    pass

        knot_bar.setMinimumHeight(140)

        def sync_knot_labels() -> None:

            cur_sk = getattr(self, 'smart_preview_sk_arr', sk_arr)

            cur_sort_idx = np.argsort(cur_sk)

            cur_kn = int(cur_sk.size)

            for j in range(cur_kn):

                oi = int(cur_sort_idx[j])

                lam_v = 1.0 / max(float(cur_sk[oi]), 1e-30)

                lbl_lam_cols[j].setText(f"{lam_v:.1f} nm")

                lbl_sig_cols[j].setText(f"{float(cur_sk[oi]):.5f}")

                lbl_n_cols[j].setText(f"{float(n_phys[oi]):.4f}")

                lbl_L_cols[j].setText(f"{float(L_nodes[oi]):.4f}")

        def sync_d_slider_label() -> None:

            lbl_d_slider.setText(

                f"{preview_d_nm:.2f} nm   [d min={d_lo_nm:.1f}, d max={d_hi_nm:.1f}]"

            )

        def set_slider_from_preview_d() -> None:

            slider_d.blockSignals(True)

            slider_d.setValue(_slider_int_from_d_nm(preview_d_nm))

            slider_d.blockSignals(False)

            sync_d_slider_label()

        def _set_n_knot_curve(i: int, v: float) -> None:

            nonlocal n_phys

            nn = np.asarray(n_phys, dtype=np.float64).copy()

            nn[int(i)] = float(np.clip(v, N_MIN_LIMIT, N_MAX_LIMIT))

            n_phys = nn

        def _set_L_knot_curve(i: int, v: float) -> None:

            nonlocal L_nodes

            LL = np.asarray(L_nodes, dtype=np.float64).copy()

            LL[int(i)] = float(np.clip(v, L_lo_g, L_hi_g))

            L_nodes = LL

        def do_recalc() -> None:

            nonlocal n_phys, L_nodes, best_rmse, best_n, best_L, preview_d_nm

            cur_sk = getattr(self, 'smart_preview_sk_arr', sk_arr)

            out = recalc_smart_init_spectral_preview(

                cfg,

                cur_sk,

                n_phys,

                L_nodes,

                grids,

                d_nm_fixed=float(preview_d_nm),

                relax_n_mono=_relax_si_mono,

            )

            if out is None:

                return

            n_phys = np.asarray(out["n_nodes_physical"], dtype=np.float64).ravel().copy()

            L_nodes = np.asarray(out["L_nodes"], dtype=np.float64).ravel().copy()

            preview_d_nm = float(out["d_best_nm"])

            sk_out = np.asarray(out.get("sigma_knots", cur_sk), dtype=np.float64).ravel().copy()

            self.smart_preview_sk_arr = sk_out

            self.smart_preview_n_phys = n_phys.copy()

            self.smart_preview_L_nodes = L_nodes.copy()

            self._si_mesh_sk_snap = sk_out.copy()

            lam_u = np.asarray(out["lam_nm"], dtype=np.float64).ravel()

            ou = np.argsort((1.0 / np.maximum(lam_u, 1e-9))**2)

            nonlocal current_t_th

            current_t_th = np.asarray(out["t_theo"], dtype=np.float64).ravel()

            update_main_x_axes()

            lam_uu = lam_u

            sync_knot_labels()

            _, rm_depart = rmse_at_spline_stage_x0_init(

                cfg,

                sk_out,

                n_phys,

                L_nodes,

                preview_d_nm,

                relax_n_mono=_relax_si_mono,

            )

            if rm_depart < best_rmse:

                best_rmse = rm_depart

                best_n = n_phys.copy()

                best_L = L_nodes.copy()

            refresh_stats(preview_d_nm, rm_depart)

            # Mise a jour plot LIVE n, k (si dispo dans l'output de preview)

            if "n_lam" in out and "k_lam" in out:

                n_lam_u = np.asarray(out["n_lam"], dtype=np.float64).ravel()

                k_lam_u = np.asarray(out["k_lam"], dtype=np.float64).ravel()

                refresh_nk_plots_aux(lam_uu[ou], n_lam_u[ou], k_lam_u[ou])

                refresh_nk_plots_mon(lam_uu[ou], n_lam_u[ou], k_lam_u[ou])

            for _ce in curve_editor_holder:

                try:

                    _ce.refresh_plots()

                except (AttributeError, RuntimeError):

                    pass

        _nk_curve_editor = SmartInitNKCurveEditorDialog(

            dlg,

            n_lo=float(N_MIN_LIMIT),

            n_hi=float(N_MAX_LIMIT),

            L_lo=float(L_lo_g),

            L_hi=float(L_hi_g),

            k_clip_lo=float(getattr(cfg, "k_clip_lo", 1e-30) or 1e-30),

            get_sk=lambda: np.asarray(

                getattr(self, "smart_preview_sk_arr", sk_arr), dtype=np.float64

            ).ravel(),

            get_n_phys=lambda: n_phys,

            get_L_nodes=lambda: L_nodes,

            set_n_at=_set_n_knot_curve,

            set_L_at=_set_L_knot_curve,

            request_recalc=do_recalc,

            study_lambda_window=_study_lambda_window_nm,

        )

        curve_editor_holder.append(_nk_curve_editor)

        _nk_curve_editor.show()

        def _place_nk_editor() -> None:

            try:

                fr = dlg.frameGeometry()

                _nk_curve_editor.move(

                    max(24, fr.left() - _nk_curve_editor.width() - 20),

                    fr.top() + 32,

                )

            except (AttributeError, RuntimeError):

                pass

        QTimer.singleShot(0, _place_nk_editor)

        def run_auto(row: int, is_ln_k: bool) -> None:

            nonlocal n_phys, L_nodes, preview_d_nm

            cur_sk = np.asarray(

                getattr(self, "smart_preview_sk_arr", sk_arr), dtype=np.float64

            ).ravel()

            n_loc = np.asarray(n_phys, dtype=np.float64).ravel()

            L_loc = np.asarray(L_nodes, dtype=np.float64).ravel()

            if n_loc.size != cur_sk.size or L_loc.size != cur_sk.size:

                spn = getattr(self, "smart_preview_n_phys", None)

                spl = getattr(self, "smart_preview_L_nodes", None)

                if spn is not None and spl is not None:

                    spn_a = np.asarray(spn, dtype=np.float64).ravel()

                    spl_a = np.asarray(spl, dtype=np.float64).ravel()

                    if spn_a.size == spl_a.size == cur_sk.size:

                        n_phys = spn_a.copy()

                        L_nodes = spl_a.copy()

                        n_loc = spn_a

                        L_loc = spl_a

            if n_loc.size != cur_sk.size or L_loc.size != cur_sk.size:

                sk_snap = getattr(self, "_si_mesh_sk_snap", None)

                if sk_snap is not None:

                    sk_snap = np.asarray(sk_snap, dtype=np.float64).ravel()

                    if (

                        sk_snap.size >= 2

                        and sk_snap.size == n_loc.size == L_loc.size

                        and cur_sk.size >= 2

                    ):

                        n_phys, L_nodes = interp_n_L_pwlnk_to_sigmas(

                            sk_snap, n_loc, L_loc, cur_sk

                        )

                        n_loc = n_phys

                        L_loc = L_nodes

            if n_loc.size != cur_sk.size or L_loc.size != cur_sk.size:

                QMessageBox.warning(

                    dlg,

                    "Smart Init  auto",

                    f"Inconsistent sigma / n / ln k (K_sigma={cur_sk.size}, len(n)={n_loc.size}, len(L)={L_loc.size}). "

                    "Try again after a recalculation (slight movement of a knot or d slider).",

                )

                return

            try:

                out = smart_init_sweep_node_thickness_rmse(

                    cfg,

                    cur_sk,

                    n_phys,

                    L_nodes,

                    int(row),

                    is_ln_k=bool(is_ln_k),

                    d_lo=float(cfg.d_lo),

                    d_hi=float(cfg.d_hi),

                    L_lo=L_lo_g,

                    L_hi=L_hi_g,

                    time_budget_s=2.9,

                    d_nm_current=float(preview_d_nm),

                    relax_n_mono=_relax_si_mono,

                )

            except Exception as exc:

                QMessageBox.warning(

                    dlg,

                    "Smart Init  auto",

                    f"Balayage d + ({'ln k' if is_ln_k else 'n'}) : {exc}",

                )

                return

            n_phys = np.asarray(out["n_nodes_physical"], dtype=np.float64).ravel().copy()

            L_nodes = np.asarray(out["L_nodes"], dtype=np.float64).ravel().copy()

            preview_d_nm = float(out["d_nm"])

            set_slider_from_preview_d()

            do_recalc()

        def on_slider_d_changed(_iv: int) -> None:

            nonlocal preview_d_nm

            preview_d_nm = _d_from_slider_int(slider_d.value())

            sync_d_slider_label()

            do_recalc()

        slider_d.valueChanged.connect(on_slider_d_changed)

        set_slider_from_preview_d()

        def bump_n_scaled(row: int, direction: int, mult: float) -> None:

            step = rel_step * float(mult)

            f = 1.0 + float(direction) * step

            n_phys[row] = float(np.clip(n_phys[row] * f, N_MIN_LIMIT, N_MAX_LIMIT))

            do_recalc()

        def bump_L_scaled(row: int, direction: int, mult: float) -> None:

            step = rel_step * float(mult)

            f = 1.0 + float(direction) * step

            L_nodes[row] = float(np.clip(L_nodes[row] * f, L_lo_g, L_hi_g))

            do_recalc()

        def wire_hold_button(

            btn: QPushButton,

            row: int,

            direction: int,

            *,

            is_ln_k: bool,

        ) -> None:

            t = QTimer(dlg)

            t.setInterval(78)

            ntick: list[int] = [0]

            def on_tick() -> None:

                ntick[0] += 1

                mult = min(24.0, 1.0 + (ntick[0] - 1) * 0.85)

                if is_ln_k:

                    bump_L_scaled(row, direction, mult)

                else:

                    bump_n_scaled(row, direction, mult)

            t.timeout.connect(on_tick)

            def on_press() -> None:

                ntick[0] = 1

                if is_ln_k:

                    bump_L_scaled(row, direction, 1.0)

                else:

                    bump_n_scaled(row, direction, 1.0)

                t.stop()

                def maybe_start_repeat() -> None:

                    if btn.isDown():

                        t.start()

                QTimer.singleShot(400, maybe_start_repeat)

            def on_release() -> None:

                t.stop()

                ntick[0] = 0

            btn.pressed.connect(on_press)

            btn.released.connect(on_release)

        def recall_best() -> None:

            nonlocal n_phys, L_nodes

            if not np.isfinite(best_rmse):

                return

            cur_k = int(

                np.asarray(

                    getattr(self, "smart_preview_sk_arr", sk_arr), dtype=np.float64

                ).ravel().size

            )

            if best_n.size != cur_k or best_L.size != cur_k:

                QMessageBox.information(

                    dlg,

                    "Smart Init",

                    "The sigma mesh has changed since this 'best' (preset, Autofind, etc.): "

                    f"impossible to recall n and ln k (best K={best_n.size}, current K={cur_k}).",

                )

                return

            n_phys = best_n.copy()

            L_nodes = best_L.copy()

            do_recalc()

        rebuild_knot_ui(k_n)  # Appel initial  ici wire_hold_button est deja defini

        attach_excel_clipboard_context_menu(pw)

        lay.addWidget(wrap_scientific_plot_with_toolbar(dlg, pw), stretch=1)

        lbl_nodes = QLabel(

            f"<b>Knot adjustment (increasing sigma)</b> - <b>n &amp; k Editor</b> window on the left: drag points "

            f"(<i>k</i> in log); here: <b>- / +</b> +/-{100 * rel_step:.1f} % on <i>n</i> and <i>L</i> (= ln <i>k</i>), "

            f"<b>without</b> auto thickness recalculation (d slider above); "

            f"<b>hold down</b> to accelerate; <b>auto</b>: d + param sweep <=3 s."

        )

        lbl_nodes.setWordWrap(True)

        lbl_nodes.setStyleSheet(f"color: {CertusTheme.TEXT_SUB}; font-size: 11px;")

        lay.addWidget(lbl_nodes)

        lay.addWidget(knot_bar)

        lay.addWidget(lbl_stats)

        row_hint = QHBoxLayout()

        lbl_row_hint = QLabel()

        def update_hint_text():

            # Help text: same K as worker after Continue (avoids claiming “12 knots” for a 5 µm file).

            lam_h = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

            k_h = int(

                canonical_spline_sigma_knots(

                    float(np.nanmin(lam_h)),

                    float(np.nanmax(lam_h)),

                    **_canonical_knots_min_lambda_kw(cfg),

                ).size

            )

            n_h = max(1, k_h - 1)

            lbl_row_hint.setText(

                f" Continue: fixed mesh {k_h} sigma knots, {n_h} segments between knots "

                "(canonical grid [lambda_min, lambda_max]); local refinement; knots and RMSE logged in CERTUS."

            )

        update_hint_text()

        row_hint.addWidget(lbl_row_hint, stretch=1)

        btn_recall = QPushButton("Recall best")

        btn_recall.setToolTip("Restore n and ln k profiles with the lowest RMSE since dialog start.")

        btn_recall.clicked.connect(recall_best)

        btn_copy = QPushButton("Copy to clipboard")

        def on_copy() -> None:

            cur_sk = getattr(self, 'smart_preview_sk_arr', sk_arr)

            lines = [

                f"RMSE: {current_rmse:.8f}",

                f"d: {preview_d_nm:.6f} nm",

                "Nodes (sigma, n, ln k):"

            ]

            for idx in np.argsort(cur_sk):

                lines.append(f"  {cur_sk[idx]:.8e} | {n_phys[idx]:.6f} | {L_nodes[idx]:.6f}")

            QApplication.clipboard().setText("\n".join(lines))

            btn_copy.setText("Copied!")

            QTimer.singleShot(1500, lambda: btn_copy.setText("Copy to clipboard"))

        btn_copy.clicked.connect(on_copy)

        row_hint.addWidget(btn_copy)

        row_hint.addWidget(btn_recall)

        def apply_manual_preset_from_projector(

            projector: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, float]],

            feedback_btn: QPushButton | None,

            idle_label: str,

        ) -> None:

            nonlocal sk, n_phys, L_nodes, preview_d_nm

            target_sk = np.asarray(getattr(self, "smart_preview_sk_arr", sk_arr), dtype=np.float64).ravel()

            self.smart_preview_sk_arr, self.smart_preview_n_phys, self.smart_preview_L_nodes, preview_d_nm = (

                projector(target_sk)

            )

            sk = np.asarray(self.smart_preview_sk_arr, dtype=np.float64).ravel().copy()

            n_phys = self.smart_preview_n_phys.copy()

            L_nodes = self.smart_preview_L_nodes.copy()

            self.smart_preview_sig2 = self.smart_preview_sk_arr**2

            _skp = np.asarray(self.smart_preview_sk_arr, dtype=np.float64).ravel()

            if getattr(self, "sk_sorted", None) is not None:

                try:

                    _ss = np.asarray(self.sk_sorted, dtype=np.float64).ravel()

                    if _ss.size == _skp.size:

                        self.sk_sorted[:] = _skp

                except (TypeError, ValueError, IndexError):

                    pass

            from scipy.optimize import minimize_scalar

            def obj_d_only(dv: float) -> float:

                _, rm = rmse_at_spline_stage_x0_init(

                    cfg,

                    self.smart_preview_sk_arr,

                    self.smart_preview_n_phys,

                    self.smart_preview_L_nodes,

                    float(dv),

                    relax_n_mono=_relax_si_mono,

                )

                return float(rm)

            res_d = minimize_scalar(

                obj_d_only,

                bounds=(cfg.d_lo, cfg.d_hi),

                method="bounded",

                options={"xatol": 0.01},

            )

            if res_d.success:

                preview_d_nm = float(res_d.x)

                self.smart_preview_d_nm = preview_d_nm

            for line in knot_lines:

                try:

                    pw.removeItem(line)

                except (AttributeError, RuntimeError):

                    pass

            knot_lines.clear()

            pen_k = pg.mkPen("#1a9f3c", width=1.8)

            for sx in self.smart_preview_sk_arr:

                mode = self.cb_unit_x.currentText() if hasattr(self, "cb_unit_x") else "Sigma2"

                if mode == "Sigma":

                    xv = float(sx)

                elif mode == "Sigma2":

                    xv = float(sx) ** 2

                else:

                    xv = 1.0 / float(sx) if sx != 0 else 0.0

                il = pg.InfiniteLine(xv, angle=90, pen=pen_k)

                pw.addItem(il)

                knot_lines.append(il)

            rebuild_knot_ui(len(self.smart_preview_sk_arr))

            set_slider_from_preview_d()

            do_recalc()

            # Align cache used by RMSE / editor on state after preview (fix display Ta₂O₅, etc.).

            self.smart_preview_n_phys = np.asarray(n_phys, dtype=np.float64).copy()

            self.smart_preview_L_nodes = np.asarray(L_nodes, dtype=np.float64).copy()

            if feedback_btn is not None:

                feedback_btn.setText(f"OK - {len(self.smart_preview_sk_arr)} nodes")

                QTimer.singleShot(

                    1500,

                    lambda b=feedback_btn, t=idle_label: b.setText(t),

                )

        cb_material_preset = QComboBox()

        cb_material_preset.setMinimumWidth(168)

        cb_material_preset.setToolTip(

            "Choose a material: Nb₂O₅ (reference 12 sigma + d), SiO₂ or Ta₂O₅ (lambda tabulation), "

            "then 'Apply preset' - PWL interpolation on current sigma grid, d mini-optimization.\n"

            "When opening the dialog, the **three** presets are automatically tested; the best RMSE "

            "(same criteria as preview) is applied."

        )

        for _label, _pid in (

            ("Nb₂O₅ (ref.)", "nb2o5"),

            ("SiO₂", "sio2"),

            ("Ta₂O₅", "ta2o5"),

        ):

            cb_material_preset.addItem(_label, _pid)

        btn_apply_material = QPushButton("Apply preset")

        def on_apply_material_preset() -> None:

            pid = str(cb_material_preset.currentData() or "nb2o5")

            dh = float(preview_d_nm)

            def _run(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:

                return project_manual_material_preset(pid, ts, d_nm_hint=dh)

            apply_manual_preset_from_projector(_run, btn_apply_material, "Apply preset")

        btn_apply_material.clicked.connect(on_apply_material_preset)

        row_hint.addWidget(cb_material_preset)

        row_hint.addWidget(btn_apply_material)

        def _auto_try_three_material_presets() -> None:

            """Compares Nb₂O₅ / SiO₂ / Ta₂O₅ on the current sigma grid and applies the best one (mini-opt d)."""

            nonlocal preview_d_nm

            target_sk = np.asarray(

                getattr(self, "smart_preview_sk_arr", sk_arr), dtype=np.float64

            ).ravel()

            if int(target_sk.size) < 2:

                return

            try:

                picked = pick_best_manual_material_preset(

                    cfg,

                    target_sk,

                    d_nm_hint=float(preview_d_nm),

                    relax_n_mono=bool(_relax_si_mono),

                )

            except Exception as exc:

                if self.logger:

                    self.logger.warning(

                        "INDEX_SPLINE [Smart Init] Auto-selection of 3 material presets: %s",

                        exc,

                    )

                return

            if picked is None:

                if self.logger:

                    self.logger.info(

                        "INDEX_SPLINE [Smart Init] Material presets: no valid RMSE score - "

                        "keeping Swanepoel / current profile."

                    )

                return

            winner, rm_w, d_w, _nw, _Lw, score_rows = picked

            if self.logger:

                parts = [f"{pid}->RMSE={rm:.6f}" for pid, rm in score_rows]

                self.logger.info(

                    "INDEX_SPLINE [Smart Init] Material presets (d mini-opt for each): %s | "

                    "kept **%s** (RMSE=%.6f, d~%.2f nm)",

                    " ; ".join(parts),

                    winner,

                    rm_w,

                    d_w,

                )

            preview_d_nm = float(d_w)

            iw = cb_material_preset.findData(winner)

            if iw >= 0:

                cb_material_preset.blockSignals(True)

                try:

                    cb_material_preset.setCurrentIndex(int(iw))

                finally:

                    cb_material_preset.blockSignals(False)

            def _proj(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:

                return project_manual_material_preset(

                    winner, ts, d_nm_hint=float(preview_d_nm)

                )

            apply_manual_preset_from_projector(_proj, None, "")

        btn_autofind = QPushButton("Autofind")

        btn_autofind.setToolTip(

            "SOL2 only (~30 s): local L-BFGS-B polish on fixed sigma (canonical file mesh), "

            "without SOL3 / free nodes or pipeline suite. Seed = current profile re-interpolated in PWL."

        )

        autofind_prog = QProgressBar()

        autofind_prog.setRange(0, 100)

        autofind_prog.setValue(0)

        autofind_prog.setMinimumWidth(220)

        autofind_prog.setFormat("Autofind 0% (0.0/30.0s)")

        def on_autofind() -> None:

            nonlocal n_phys, L_nodes, preview_d_nm, current_rmse, best_rmse, best_n, best_L

            cur_sk = np.asarray(getattr(self, "smart_preview_sk_arr", sk_arr), dtype=np.float64).ravel()

            if cur_sk.size < 2:

                QMessageBox.warning(dlg, "Autofind", "Invalid knot grid.")

                return

            if n_phys.size != cur_sk.size or L_nodes.size != cur_sk.size:

                QMessageBox.warning(

                    dlg,

                    "Autofind",

                    "Current n / ln k vectors are inconsistent with knot count.",

                )

                return

            lam_nm_af = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

            lam_min_af = float(np.nanmin(lam_nm_af))

            lam_max_af = float(np.nanmax(lam_nm_af))

            # Realign towards a worker grid of same K as canonical, keeping manual

            # knots when possible (adding knots rather than full remesh).

            _mdl_br = float(getattr(cfg, "spline_min_delta_lambda_over_lambda_mean", 0.0) or 0.0)

            sk_canon = bridge_sigma_knots_preserve_manual(

                cur_sk,

                lam_min_af,

                lam_max_af,

                rmse_fit_lambda_nm=getattr(cfg, "rmse_fit_lambda_nm", None),

                min_delta_lambda_over_lambda_mean=_mdl_br if _mdl_br > 0.0 else None,

            )

            # Same PWL regridding as "Continue" (not np.interp with plateaus at edges).

            try:

                n_on, L_on = interp_n_L_pwlnk_to_sigmas(

                    np.asarray(cur_sk, dtype=np.float64).ravel(),

                    np.asarray(n_phys, dtype=np.float64).ravel(),

                    np.asarray(L_nodes, dtype=np.float64).ravel(),

                    np.asarray(sk_canon, dtype=np.float64).ravel(),

                )

            except (ValueError, TypeError) as exc:

                QMessageBox.warning(

                    dlg,

                    "Autofind",

                    f"PWL regridding to canonical mesh impossible: {exc}",

                )

                return

            k_loc = int(np.asarray(sk_canon, dtype=np.float64).size)

            if int(np.asarray(n_on).size) != k_loc or int(np.asarray(L_on).size) != k_loc:

                QMessageBox.warning(

                    dlg,

                    "Autofind",

                    f"Inconsistent sizes after regridding (K={k_loc}, len(n)={np.asarray(n_on).size}).",

                )

                return

            x0_loc = np.concatenate(

                (

                    np.asarray([float(preview_d_nm)], dtype=np.float64),

                    np.asarray(n_on, dtype=np.float64).ravel(),

                    np.asarray(L_on, dtype=np.float64).ravel(),

                )

            )

            _polish_af = min(int(getattr(cfg, "polish_maxfun", 8000) or 8000), 3200)

            _smlf_af = min(

                max(int(getattr(cfg, "stage_mandatory_local_maxfun", 0) or 0), 400),

                900,

            )

            enforce_local_optimization_policy(cfg)

            auto_cfg = replace(

                cfg,

                n_seg=int(max(1, k_loc - 1)),

                sigma_knots_override=None,

                smart_preview_exact_sigma_knots=np.asarray(sk_canon, dtype=np.float64).copy(),

                smart_preview_exact_n_L=(

                    np.asarray(n_on, dtype=np.float64).copy(),

                    np.asarray(L_on, dtype=np.float64).copy(),

                ),

                smart_preview_d_nm_override=float(preview_d_nm),

                x0_warm=x0_loc.copy(),

                smart_init_manual_force_restart=False,

                pglobal_trust_region_by_k=False,

                pglobal_max_time=None,

                pglobal_max_iter=0,

                pglobal_max_feval=None,

                pglobal_local_search_budget=None,

                spline_local_only=True,

                stage_mandatory_local_maxfun=int(_smlf_af),

                polish_maxfun=int(_polish_af),

                smart_init_preview_hook=None,

            )

            btn_autofind.setEnabled(False)

            autofind_prog.setValue(0)

            autofind_prog.setFormat("Autofind 0% (0.0/30.0s)")

            prev_txt = btn_autofind.text()

            btn_autofind.setText("Autofind...")

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            try:

                import threading

                timeout_s = 30.0

                stop_ev = Event()

                done_ev = threading.Event()

                result_box: dict[str, Any] = {"best": None, "error": None}

                best_live: dict[str, Any] | None = None

                def _live_capture(payload: dict[str, Any]) -> None:

                    nonlocal best_live

                    if not isinstance(payload, dict):

                        return

                    try:

                        rm = float(payload.get("rmse", float("inf")))

                    except (TypeError, ValueError):

                        rm = float("inf")

                    if not np.isfinite(rm):

                        return

                    if best_live is None or rm < float(best_live.get("rmse", float("inf"))):

                        best_live = dict(payload)

                def _run_autofind() -> None:

                    try:

                        # Single SOL2 stage (fixed sigma): not the complete pipeline (SOL3 / free nodes / 

                        # spectral polish), which could greatly exceed UI budget and change K.

                        res_sol2, _ = _run_single_spline_stage(

                            auto_cfg,

                            stop_ev,

                            lambda _pc, _msg: None,

                            live_cb=_live_capture,

                            pipeline_seq="AUTOFIND_SOL2",

                            fatal_finish=None,

                        )

                        result_box["best"] = res_sol2

                    except Exception as exc:

                        result_box["error"] = exc

                    finally:

                        done_ev.set()

                QApplication.processEvents()

                th = threading.Thread(target=_run_autofind, daemon=True)

                th.start()

                t0 = time.perf_counter()

                timeout_reached = False

                while True:

                    elapsed = max(0.0, time.perf_counter() - t0)

                    elapsed_clamped = min(elapsed, timeout_s)

                    pct = int(

                        min(

                            99,

                            max(

                                0,

                                round(

                                    100.0

                                    * elapsed_clamped

                                    / max(timeout_s, 1e-9)

                                ),

                            ),

                        )

                    )

                    autofind_prog.setValue(pct)

                    autofind_prog.setFormat(

                        f"Autofind {pct}% ({elapsed_clamped:.1f}/{timeout_s:.1f}s)"

                    )

                    QApplication.processEvents()

                    if done_ev.is_set():

                        break

                    if elapsed >= timeout_s:

                        timeout_reached = True

                        stop_ev.set()

                        break

                    time.sleep(0.05)

                if timeout_reached and not done_ev.wait(0.8):

                    # Coupure UI au timeout : appliquer le dernier snapshot live si le fil n’a pas fini.

                    result_box["best"] = best_live if isinstance(best_live, dict) else None

                if result_box.get("error") is not None and result_box.get("best") is None:

                    raise result_box["error"]

                best = result_box.get("best")

                if not isinstance(best, dict):

                    if isinstance(best_live, dict):

                        best = best_live

                    else:

                        raise RuntimeError(

                            "Autofind: stopped at timeout without any useful RMSE snapshot."

                        )

                autofind_prog.setValue(100)

                autofind_prog.setFormat(f"Autofind 100% ({timeout_s:.1f}/{timeout_s:.1f}s)")

            except Exception as exc:

                QMessageBox.warning(

                    dlg,

                    "Autofind",

                    f"SOL2 local search failed: {exc}",

                )

                return

            finally:

                QApplication.restoreOverrideCursor()

                btn_autofind.setEnabled(True)

                btn_autofind.setText(prev_txt)

                QApplication.processEvents()

            try:

                n_new = np.asarray(best.get("n_nodes_physical", n_phys), dtype=np.float64).ravel()

                L_new = np.asarray(best.get("L_nodes", L_nodes), dtype=np.float64).ravel()

                if n_new.size == k_loc and L_new.size == k_loc:

                    n_phys = n_new.copy()

                    L_nodes = L_new.copy()

                elif n_new.size != k_loc or L_new.size != k_loc:

                    QMessageBox.warning(

                        dlg,

                        "Autofind",

                        f"SOL2 result ignored for n/L: sizes {n_new.size} / {L_new.size} "

                        f"for K={k_loc} expected - keeping current profile, only *d* may be updated.",

                    )

                if n_phys.size == k_loc and L_nodes.size == k_loc:

                    self.smart_preview_sk_arr = np.asarray(sk_canon, dtype=np.float64).copy()

                    self.smart_preview_n_phys = np.asarray(n_phys, dtype=np.float64).ravel().copy()

                    self.smart_preview_L_nodes = np.asarray(L_nodes, dtype=np.float64).ravel().copy()

                    self._si_mesh_sk_snap = self.smart_preview_sk_arr.copy()

                    rebuild_knot_ui(k_loc)

                    update_hint_text()

                try:

                    _d_b = float(best.get("d_nm", preview_d_nm))

                except (TypeError, ValueError):

                    _d_b = float(preview_d_nm)

                if np.isfinite(_d_b):

                    preview_d_nm = _d_b

                set_slider_from_preview_d()

                do_recalc()

                try:

                    _rm_b = float(best.get("rmse", current_rmse))

                except (TypeError, ValueError):

                    _rm_b = float(current_rmse)

                if np.isfinite(_rm_b):

                    current_rmse = min(float(current_rmse), _rm_b)

                if current_rmse < best_rmse:

                    best_rmse = current_rmse

                    best_n = n_phys.copy()

                    best_L = L_nodes.copy()

                refresh_stats(preview_d_nm, current_rmse)

                btn_autofind.setText("Autofind completed")

                QTimer.singleShot(1500, lambda: btn_autofind.setText("Autofind"))

            except Exception as exc:

                QMessageBox.warning(

                    dlg,

                    "Autofind",

                    f"Error when applying Autofind result: {exc}",

                )

        btn_autofind.clicked.connect(on_autofind)

        row_hint.addWidget(btn_autofind)

        row_hint.addWidget(autofind_prog)

        lay.addLayout(row_hint)

        chk_si_deep = QCheckBox(

            "Deep SOL2 after Smart Init (legacy option inactive in local-only mode)"

        )

        chk_si_deep.setChecked(False)

        chk_si_deep.setToolTip(

            "Manual Smart Init is now always handed off to the worker in local L-BFGS-B mode. "

            "This legacy option is kept visible only for compatibility and has no effect."

        )

        chk_si_deep.setEnabled(False)

        lay.addWidget(chk_si_deep)

        chk_si_two_phase = QCheckBox(

            "Two-phase deep SOL2 (legacy option inactive in local-only mode)"

        )

        chk_si_two_phase.setChecked(False)

        chk_si_two_phase.setToolTip(

            "Legacy compatibility flag only; no second global phase exists anymore in local-only mode."

        )

        chk_si_two_phase.setEnabled(False)

        lay.addWidget(chk_si_two_phase)

        bb = QDialogButtonBox(

            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        )

        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Continue optimization")

        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("Stop")

        def on_keep() -> None:

            if cfg is not None:

                # Epaisseur reglee lors du dialog

                d_final = float(preview_d_nm)

                # Nuds physiques (n, ln k) regles lors du dialog

                n_phys_final = np.asarray(n_phys, dtype=np.float64).copy()

                L_nodes_final = np.asarray(L_nodes, dtype=np.float64).copy()

                sk_final = np.asarray(

                    getattr(self, "smart_preview_sk_arr", sk), dtype=np.float64

                ).ravel().copy()

                # RMSE displayed in the dialog = objective on current mesh (often K=12 if 

                # rmse_fit_lambda_nm was regridded in sigma²). The worker goes to a target K mesh (often 14 in IR),

                # while keeping manual knots as much as possible then adding knots: the SOL2 RMSE

                # must be compared on this worker mesh.

                rmse_preview_mesh = float(current_rmse)

                rmse_worker_mesh = rmse_preview_mesh

                sk_canon_keep: np.ndarray | None = None

                try:

                    lam_c = np.asarray(cfg.lam_nm, dtype=np.float64).ravel()

                    lam_min_c = float(np.min(lam_c))

                    lam_max_c = float(np.max(lam_c))

                    _mdl_ck = float(getattr(cfg, "spline_min_delta_lambda_over_lambda_mean", 0.0) or 0.0)

                    sk_canon = bridge_sigma_knots_preserve_manual(

                        sk_final,

                        lam_min_c,

                        lam_max_c,

                        rmse_fit_lambda_nm=getattr(cfg, "rmse_fit_lambda_nm", None),

                        min_delta_lambda_over_lambda_mean=_mdl_ck if _mdl_ck > 0.0 else None,

                    )

                    sk_canon_keep = sk_canon

                    n_on_canon, L_on_canon = interp_n_L_pwlnk_to_sigmas(

                        sk_final, n_phys_final, L_nodes_final, sk_canon

                    )

                    # False = same ξ encoding / bounds as ``make_bounds_and_x0`` + SOL2 (not relaxed preview).

                    _, rmse_worker_mesh = rmse_at_spline_stage_x0_init(

                        cfg,

                        sk_canon,

                        n_on_canon,

                        L_on_canon,

                        d_final,

                        relax_n_mono=False,

                    )

                    if self.logger:

                        log_rmse_mesh_bridge_diagnosis(

                            cfg,

                            sk_final,

                            n_phys_final,

                            L_nodes_final,

                            sk_canon,

                            n_on_canon,

                            L_on_canon,

                            d_final,

                            self.logger,

                            relax_preview_mono=bool(_relax_si_mono),

                        )

                except Exception as exc:

                    if self.logger:

                        self.logger.warning(

                            "GUI Smart Init [Keep] | could not recalculate RMSE on worker mesh: %s",

                            exc,

                        )

                    rmse_worker_mesh = rmse_preview_mesh

                # 1. Sauvegarde via signal multi-thread (securite PyQt)

                self._preview_ret = (

                    sk_final.copy(),

                    n_phys_final.copy(),

                    L_nodes_final.copy(),

                    d_final,

                    float(rmse_worker_mesh),

                )

                # 2. Injection directe dans l'objet config (pour le Worker)

                cfg.smart_preview_node_override = (n_phys_final.copy(), L_nodes_final.copy())

                cfg.smart_preview_exact_sigma_knots = sk_final.copy()

                cfg.smart_preview_exact_n_L = (n_phys_final.copy(), L_nodes_final.copy())

                cfg.smart_preview_d_nm_override = d_final

                cfg.smart_preview_accepted_rmse = float(rmse_worker_mesh)

                QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP).setValue(

                    _QS_SMART_INIT_DEEP, bool(chk_si_deep.isChecked())

                )

                QSettings(_QS_SPLINE_ORG, _QS_SPLINE_APP).setValue(

                    _QS_SMART_INIT_TWO_PHASE, bool(chk_si_two_phase.isChecked())

                )

                cfg.gui_run_pglobal_opt_in = False

                cfg.spline_local_only = True

                cfg.spline_smart_init_deep_two_phase = False

                if self.logger:

                    self.logger.info(

                        "GUI Smart Init [Keep] | retained preview transferred to worker in forced local-only mode.",

                    )

                    _k_gui = int(sk_final.size)

                    _k_wrk = (

                        int(sk_canon_keep.size)

                        if sk_canon_keep is not None

                        else _k_gui

                    )

                    self.logger.info(

                        "GUI Smart Init [Keep] | Fil conducteur: aperçu/fenêtre RMSE=%.6f (K=%d) → valeur retenue "

                        "pour le worker (SOL2 / départ INDEX_SPLINE) RMSE=%.6f (K=%d). "

                        "The second value is used for optimization.",

                        rmse_preview_mesh,

                        _k_gui,

                        rmse_worker_mesh,

                        _k_wrk,

                    )

                    self.logger.info(

                        "GUI Smart Init [Keep] | d_final=%.4f nm | si K diffère: détail noeuds / extrapolation dans "

                        "``INDEX_SPLINE_smart_init_Ksrc_to_worker_mesh`` puis FACTUAL SOL1/SOL2 ; écart preview vs "

                        "worker: lignes ``DIAG_RMSE_BRIDGE`` [A]/[B].",

                        d_final,

                    )

                    self.logger.info(

                        "GUI Smart Init [Keep] | preview relax_n_mono=%s — si True avec bande mono, l'aperçu n'applique "

                        "pas n_mono ; le worker garde l'objectif complet (mono + pénalités).",

                        bool(_relax_si_mono),

                    )

                    self.logger.info(

                        "GUI Smart Init [Keep] | même grille spectrale que le worker: rmse_fit_lambda_nm=%s | "

                        "wT=%.4g wR=%.4g | t_is_ratio=%s | data_type=%s | n_mono_band_nm=%s | nk_profile_interp=%s",

                        str(getattr(cfg, "rmse_fit_lambda_nm", None)),

                        float(getattr(cfg, "weight_t", 0.0)),

                        float(getattr(cfg, "weight_r", 0.0)),

                        bool(getattr(cfg, "t_is_ratio", False)),

                        str(getattr(cfg, "data_type", "")),

                        str(getattr(cfg, "n_mono_band_nm", None)),

                        str(getattr(cfg, "nk_profile_interp", "smooth")),

                    )

                    self.logger.info(

                        "GUI Smart Init [Keep] | sigma_knots (nm⁻¹) K=%d : %s",

                        int(sk_final.size),

                        np.array2string(np.asarray(sk_final, dtype=np.float64), precision=6, max_line_width=200),

                    )

                    ord_sig = np.argsort(sk_final)

                    for rank, idx in enumerate(ord_sig, start=1):

                        sigv = float(sk_final[idx])

                        lamv = 1.0 / max(sigv, 1e-30)

                        self.logger.info(

                            "GUI Smart Init [Node %2d/%2d] lambda=%10.4f nm  sigma=%.10e  n=%.6f  ln k=%.6f",

                            rank, int(sk_final.size), lamv, sigv, float(n_phys_final[idx]), float(L_nodes_final[idx])

                        )

                # 3. Preparation du vecteur x encode (pour packaging et worker)

                from spline_workers import _pack_spline_stage_result

                from spline_objective import physical_nodes_to_x_slice_n

                n_xi = physical_nodes_to_x_slice_n(n_phys_final, sk_final, cfg.n_mono_band_nm)

                x_final = np.concatenate(([d_final], n_xi, L_nodes_final))

                # --- UPDATE VISUELLE IMMEDIATE ---

                ui_snap = _pack_spline_stage_result(

                    cfg, sk_final, x_final,

                    float(rmse_worker_mesh**2),

                    0, 0 # nfev, nit fake

                )

                self._plot_result(ui_snap)

                mono_s = ""

                if cfg.n_mono_band_nm is not None:

                    mono_s = " | Optimization: monotonic n(sigma) on the lambda band + penalty if n rises too much with lambda."

                if abs(rmse_worker_mesh - rmse_preview_mesh) > 1e-8:

                    status_rmse = (

                        f"RMSE dialogue K={int(sk_final.size)} {rmse_preview_mesh:.6f} "

                        f"-> worker (canonique) {rmse_worker_mesh:.6f}"

                    )

                else:

                    status_rmse = f"RMSE {rmse_worker_mesh:.6f}"

                self.lbl_status.setText(

                    f"Smart Init validated: {status_rmse} | d={d_final:.2f} nm. "

                    f"Launching optimization…{mono_s}"

                )

            dlg.accept()

        # --- INITIALISATION IMMEDIATE ---

        rebuild_knot_ui(k_n)

        do_recalc()

        _auto_try_three_material_presets()

        bb.accepted.connect(on_keep)

        bb.rejected.connect(dlg.reject)

        lay.addWidget(bb)

        _apply_manual_spectrum_plot_range()

        return dlg.exec() == QDialog.DialogCode.Accepted

    def _query_pglobal_opt_in_within_3s(self) -> bool:

        """True si l'utilisateur active PGlobal dans les 3 s ; sinon False (défaut = sans PGlobal)."""

        return False

    def _on_run(self) -> None:

        # Reinitialisation de la securite retour de dialog

        self._preview_ret = None

        cfg = self._build_opt_config()

        if cfg is None:

            return

        cfg.gui_run_pglobal_opt_in = False

        cfg.spline_local_only = True

        if self.logger:

            self.logger.info("RUN local policy | spline_local_only=True")

        t_run_cfg = time.perf_counter()

        if self.logger:

            self.logger.info(

                "RUN config | n_seg=%s d=[%.2f,%.2f] wt=%.3f wr=%.3f profile=%s nk_interp=%s local_only=%s polish=%s sol3_p1=%s mono=%s n_lambda_rise_slack=%.4f",

                int(cfg.n_seg),

                float(cfg.d_lo),

                float(cfg.d_hi),

                float(cfg.weight_t),

                float(cfg.weight_r),

                str(self.cb_profilee.currentData() or "fast"),

                str(cfg.nk_profile_interp),

                bool(cfg.spline_local_only),

                int(cfg.polish_maxfun),

                int(sol3_phase1_maxfun_effective(cfg)),

                cfg.n_mono_band_nm,

                float(getattr(cfg, "n_lambda_rising_penalty_slack", 0.0) or 0.0),

            )

            log_structured_json_event(

                self.logger,

                "AUTO_BEST_JSON",

                "run_config",

                n_seg=int(cfg.n_seg),

                d_lo=float(cfg.d_lo),

                d_hi=float(cfg.d_hi),

                wt=float(cfg.weight_t),

                wr=float(cfg.weight_r),

                profile=str(self.cb_profilee.currentData() or "fast"),

                nk_profile_interp=str(cfg.nk_profile_interp),

                pg_iter=int(cfg.pglobal_max_iter),

                pg_feval=cfg.pglobal_max_feval,

                pg_time=cfg.pglobal_max_time,

                pg_local=cfg.pglobal_local_search_budget,

                polish=int(cfg.polish_maxfun),

                sol3_phase1_maxfun=int(sol3_phase1_maxfun_effective(cfg)),

                sol3_phase1_maxfun_raw=cfg.sol3_phase1_maxfun,

                n_mono_band=cfg.n_mono_band_nm,

                n_lambda_rising_slack=float(

                    getattr(cfg, "n_lambda_rising_penalty_slack", 0.0) or 0.0

                ),

            )

        # Toujours forcer le mode Smart Init en "Auto-Best"

        if getattr(self, "_auto_best_force_smart_init", True):

            if self.logger:

                self.logger.info("Auto-Best: Smart Init dialog interception active.")

        self._save_undo_state()

        self._stop_event = Event()

        self._cleanup_thread()

        reset_smart_init_preview_guard(cfg)

        self._best_live_rmse = float("inf")

        self._best_live_result = None

        self._last_worker_result = None

        self._corridor_rmse_manual_active = False

        self._corridor_rmse_manual_lo = float("nan")

        self._corridor_rmse_manual_hi = float("nan")

        self._log_prog_last = -1

        self._last_live_log_mono = 0.0

        self._log_optimization_header(cfg)

        cfg_run = replace(cfg, smart_init_preview_hook=self._smart_init_preview_hook)

        setattr(

            cfg_run,

            "gui_defer_corridor_profile_after_nl",

            bool(getattr(self, "chk_corridor_d", None) and self.chk_corridor_d.isChecked()),

        )

        setattr(cfg_run, "gui_use_nl_alpha_for_corridors", False)

        self._last_run_cfg = cfg_run

        # CRITICAL : dataclasses.replace() ne copie PAS les attributs dynamiques.

        # On les transfere manuallement pour que le worker voie l'injection manualle.

        for _attr in (

            "smart_preview_node_override",

            "smart_preview_exact_sigma_knots",

            "smart_preview_exact_n_L",

            "smart_preview_d_nm_override",

            "smart_preview_accepted_rmse",

            "spline_local_only",

            "smart_init_manual_force_restart",

            "gui_run_pglobal_opt_in",

        ):

            if hasattr(cfg, _attr):

                setattr(cfg_run, _attr, getattr(cfg, _attr))

        self._worker = GenericWorker(worker_spline_optimization, cfg_run, self._stop_event)

        self._worker_role = "main"

        self._worker.kwargs["progress_cb"] = self._worker.signals.progress.emit

        self._worker.kwargs["live_cb"] = self._worker.signals.live.emit

        self._worker.signals.progress.connect(self._on_progress)

        self._worker.signals.live.connect(self._on_live_update)

        self._worker.signals.finished.connect(self._on_worker_done)

        self._worker.signals.error.connect(self._on_worker_err)

        self._worker.signals.finished.connect(self._cleanup_thread)

        self._worker.signals.error.connect(self._cleanup_thread)

        self.btn_run.setEnabled(False)

        self.btn_stop.setEnabled(True)

        self._prog_ui_last = 0

        self._prog_reset_bar()

        # live_cb / _on_live_update : une seule connexion (évite double _plot_result → UI qui « gèle »).

        if self.logger:

            self.logger.info("RUN dispatch worker=%s prep_elapsed=%.3fs", getattr(self._worker.func, "__name__", "?"), time.perf_counter() - t_run_cfg)

        # one-shot gate: force smart init by default next time too (or rely on the fact it's permanent for auto-best)

        self._auto_best_force_smart_init = True

        self._worker.start()

    def _prog_reset_bar(self) -> None:

        anim = getattr(self, "_prog_anim", None)

        if anim is not None and anim.state() == QAbstractAnimation.State.Running:

            anim.stop()

        self.prog.setValue(0)

    def _cleanup_thread(self) -> None:

        if self._worker is not None:

            if self._worker.isRunning():

                self._worker.stop()

                self._worker.wait(100)

            self._worker.deleteLater()

            self._worker = None

    def _on_progress(self, v: int, msg: str) -> None:

        # ``v`` : centi-pourcents 0..10000 (10000 = 100 %), voir ``_WorkerProgressCoordinator``.

        raw = int(v)

        # Signaux Qt en file : ignorer une valeur < dernier % affiché (évite « 31 %% PGlobal » après fin PGlobal).

        if raw < self._prog_ui_last:

            return

        self._prog_ui_last = raw

        cur = int(self.prog.value())

        if raw >= 9800 or raw - cur < 12:

            if self._prog_anim.state() == QAbstractAnimation.State.Running:

                self._prog_anim.stop()

            self.prog.setValue(raw)

        else:

            if self._prog_anim.state() == QAbstractAnimation.State.Running:

                self._prog_anim.stop()

            self._prog_anim.setStartValue(cur)

            self._prog_anim.setEndValue(raw)

            span = raw - cur

            self._prog_anim.setDuration(int(min(420, max(110, span // 18))))

            self._prog_anim.start()

        st = msg

        if np.isfinite(self._best_live_rmse) and self._best_live_rmse < 1e90:

            st = f"{msg} | best displayed RMSE={self._best_live_rmse:.6f}"

        self.lbl_status.setText(st)

        if self.logger and (

            raw <= 800

            or raw >= 9800

            or raw >= self._log_prog_last + 700

            or self._log_prog_last < 0

        ):

            self._log_prog_last = raw

            # logger is already handled in backend spline_pipeline to avoid dual log.

    def _on_live_update(self, result: dict) -> None:

        """Refresh during calculation: graphs = always the best RMSE snapshot (copied arrays)."""

        if not isinstance(result, dict) or "lam_nm" not in result:

            return

        current_rmse = self._rmse_from_result_dict(result)

        improved = False

        if np.isfinite(current_rmse) and (

            self._best_live_result is None or current_rmse < self._best_live_rmse

        ):

            self._best_live_rmse = current_rmse

            self._best_live_result = _snap_spline_visual_dict(result)

            improved = True

        to_plot = (

            self._best_live_result

            if self._best_live_result is not None

            else _snap_spline_visual_dict(result)

        )

        now = time.monotonic()

        remind = (now - self._last_live_log_mono) >= self._LIVE_LOG_REMINDER_S

        if self.logger and self._best_live_result is not None and np.isfinite(self._best_live_rmse):

            if improved:

                self._last_live_log_mono = now

                _log_index_spline_best_config(

                    self.logger,

                    self._best_live_result,

                    float(self._best_live_rmse),

                    title="[BEST RMSE  new record]",

                )

            elif remind:

                self._last_live_log_mono = now

                sk = self._best_live_result.get("sigma_knots")

                k_sigma = int(np.asarray(sk, dtype=np.float64).size) if sk is not None else 0

                d_nm = float(self._best_live_result.get("d_nm", float("nan")))

                self.logger.info(

                    "[BEST DISPLAYED] reminder (~2 s) RMSE=%.6f | d_nm=%.2f | K_sigma=%d (detail: last record above)",

                    float(self._best_live_rmse),

                    d_nm,

                    k_sigma,

                )

        self._plot_result(to_plot)

        self._refresh_data_table(result_override=to_plot)

        try:

            lam_u = np.asarray(to_plot.get("lam_nm", []), dtype=np.float64).ravel()

            n_u = np.asarray(to_plot.get("n_lam", []), dtype=np.float64).ravel()

            k_u = np.asarray(to_plot.get("k_lam", []), dtype=np.float64).ravel()

            if lam_u.size and n_u.size == lam_u.size and k_u.size == lam_u.size:

                self._update_persistent_nk_monitor(

                    lam_u, n_u, k_u, float(to_plot.get("d_nm", float("nan")))

                )

        except Exception:

            logger.debug("nk monitor update in _on_live_update failed", exc_info=True)

    @staticmethod

    def _rmse_from_result_dict(d: dict) -> float:

        """RMSE displayable for comparison (priority to 'rmse' key, else √MSE)."""

        r = d.get("rmse")

        if r is not None and np.isfinite(float(r)):

            return float(r)

        m = float(d.get("mse", float("nan")))

        if np.isfinite(m):

            return float(np.sqrt(max(m, 0.0)))

        return float("inf")

    @staticmethod

    def _strip_worker_final_fields_inconsistent_with_live_merge(merged: dict[str, Any]) -> None:

        """Removes fields from the **final** worker dict that no longer describe the displayed curves after merging

        with the best live snapshot (n_lam/k_lam/d/x come from the live one).

        Without this: corridors, bootstrap, reg scan, polish spline sigma (seg_spline_sigma), spectral RMSEs

        and ln_k_lam would remain aligned with the final solution - resulting in wrong Excel export / metadata."""

        _variant_nk = (

            "n_lam_seg_spline_sigma",

            "k_lam_seg_spline_sigma",

        )

        for k in list(merged.keys()):

            if k.startswith("profile_d_"):

                merged.pop(k, None)

            elif k.startswith("corridor_"):

                merged.pop(k, None)

            elif k.startswith("boot_"):

                merged.pop(k, None)

            elif k.startswith("reg_sens"):

                merged.pop(k, None)

            elif k.startswith("spectral_rmse_"):

                merged.pop(k, None)

        for k in _variant_nk:

            merged.pop(k, None)

        merged.pop("ln_k_lam", None)

        merged.pop("spectral_rmse", None)

        merged.pop("d_nm_seg_spline_sigma", None)

    def _display_result_prefer_best_live(self, result: dict) -> dict:

        """

        If a live snapshot recorded strictly better RMSE than the worker’s final dict,

        merge: plots / Data / Excel use that best snapshot while keeping metadata

        present only in the final result (keys missing from live).

        """

        rmse_fin = self._rmse_from_result_dict(result)

        live = self._best_live_result

        if live is None or not isinstance(live, dict):

            return result

        rmse_live = self._rmse_from_result_dict(live)

        if not (np.isfinite(rmse_live) and np.isfinite(rmse_fin)):

            return result

        tol = max(1e-12, 1e-10 * max(abs(rmse_fin), 1.0))

        if rmse_live + tol >= rmse_fin:

            return result

        snap = _snap_spline_visual_dict(live)

        merged = dict(result)

        for k, v in snap.items():

            merged[k] = v

        self._strip_worker_final_fields_inconsistent_with_live_merge(merged)

        merged["gui_display_from_best_live"] = True

        merged["gui_worker_raw_rmse"] = float(rmse_fin)

        merged["gui_best_live_rmse"] = float(rmse_live)

        if self.logger:

            self.logger.info(

                "INDEX_SPLINE GUI: spectrum / indices / Data / export aligned on the **best** live "

                "snapshot (RMSE=%.8f) - final worker dict had RMSE=%.8f. "

                "Removing inconsistent keys (corridors, bootstrap, reg_sens, polish spline sigma variants, "

                "spectral_rmse_*, residual ln_k_lam).",

                rmse_live,

                rmse_fin,

            )

        return merged

    def _result_needs_deferred_corridors(self, result: dict) -> bool:

        if not isinstance(result, dict):

            return False

        if not bool(getattr(self, "chk_corridor_d", None) and self.chk_corridor_d.isChecked()):

            return False

        if bool(result.get("profile_d_enabled", False)) or result.get("profile_d_values_nm") is not None:

            return False

        return True

    def _result_can_offer_nl_alpha_for_corridors(self, result: dict) -> bool:

        if not self._result_needs_deferred_corridors(result):

            return False

        try:

            alpha_nl = result.get("nl_alpha_opt")

            x_nl = result.get("x_nl")

            d_nl = result.get("d_nm_nl")

            return (

                alpha_nl is not None

                and np.isfinite(float(alpha_nl))

                and x_nl is not None

                and d_nl is not None

            )

        except (TypeError, ValueError):

            return False

    def _prompt_use_nl_alpha_for_corridors(self) -> bool:

        dlg = QDialog(self)

        dlg.setWindowTitle("Corridors after NL alpha")

        dlg.setModal(True)

        lay = QVBoxLayout(dlg)

        lab = QLabel(

            "Use the NL alpha result as the base for the next corridor step?<br><br>"

            "Default: <b>No</b>. This window closes automatically after 3 s."

        )

        lab.setWordWrap(True)

        lay.addWidget(lab)

        bb = QDialogButtonBox(dlg)

        btn_yes = bb.addButton("Yes", QDialogButtonBox.ButtonRole.AcceptRole)

        btn_no = bb.addButton("No", QDialogButtonBox.ButtonRole.RejectRole)

        btn_no.setDefault(True)

        choice = {"use_nl": False}

        close_reason = {"kind": "timeout_default_no"}

        if self.logger:

            self.logger.info(
                "Corridors after NL alpha: prompt opened | default=no | auto_close_s=3"
            )

        def _accept_yes() -> None:

            choice["use_nl"] = True

            close_reason["kind"] = "user_yes"

            dlg.accept()

        def _reject_no() -> None:

            choice["use_nl"] = False

            if close_reason["kind"] == "timeout_default_no":

                close_reason["kind"] = "user_no"

            dlg.reject()

        def _timeout_reject_no() -> None:

            choice["use_nl"] = False

            close_reason["kind"] = "timeout_default_no"

            dlg.reject()

        btn_yes.clicked.connect(_accept_yes)

        btn_no.clicked.connect(_reject_no)

        lay.addWidget(bb)

        QTimer.singleShot(3000, _timeout_reject_no)

        dlg.exec()

        if self.logger:

            self.logger.info(
                "Corridors after NL alpha: prompt closed | choice_use_nl=%s | reason=%s",
                "yes" if bool(choice["use_nl"]) else "no",
                str(close_reason["kind"]),
            )

        return bool(choice["use_nl"])

    def _start_deferred_corridor_worker(self, result: dict, *, use_nl_alpha: bool) -> bool:

        cfg_base = self._last_run_cfg

        if cfg_base is None:

            cfg_base = self._build_opt_config(notify=False)

        if cfg_base is None:

            return False

        self._stop_event = Event()

        self._cleanup_thread()

        self._best_live_rmse = float("inf")

        self._best_live_result = None

        self._last_live_log_mono = 0.0

        cfg_corr = replace(cfg_base)

        setattr(cfg_corr, "gui_defer_corridor_profile_after_nl", False)

        setattr(cfg_corr, "gui_use_nl_alpha_for_corridors", bool(use_nl_alpha))

        self._worker = GenericWorker(worker_run_corridor_profile_after_nl_choice, cfg_corr, dict(result), self._stop_event)

        def _corr_progress(p: float | int, m: str) -> None:

            pv = int(round(float(p) * 100.0))

            self._worker.signals.progress.emit(max(0, min(10000, pv)), m)

        self._worker.kwargs["progress_cb"] = _corr_progress

        self._worker.signals.progress.connect(self._on_progress)

        self._worker.signals.finished.connect(self._on_worker_done)

        self._worker.signals.error.connect(self._on_worker_err)

        self._worker.signals.finished.connect(self._cleanup_thread)

        self._worker.signals.error.connect(self._cleanup_thread)

        self._worker_role = "corridors"

        if self.logger:

            self.logger.info(
                "Corridors after NL alpha: launching deferred corridor worker | use_nl_alpha=%s",
                "yes" if bool(use_nl_alpha) else "no",
            )

        self.btn_run.setEnabled(False)

        self.btn_stop.setEnabled(True)

        self._prog_ui_last = 0

        self._prog_reset_bar()

        self.lbl_status.setText(
            f"Corridors: running post-NL profiling (base={'NL alpha' if use_nl_alpha else 'standard'})..."
        )

        self._worker.start()

        return True

    def _on_worker_err(self, msg) -> None:

        self.btn_run.setEnabled(True)

        self.btn_stop.setEnabled(False)

        if isinstance(msg, tuple) and len(msg) == 3:

            s_msg = str(msg[1])

            logger.error(msg[2])

            if self.logger:

                self.logger.error("Optimization: %s", s_msg)

        else:

            s_msg = str(msg)

            logger.error(s_msg)

            if self.logger:

                self.logger.error("Optimization: %s", s_msg)

        QMessageBox.critical(self, "Optimization error", s_msg)

    def _on_worker_done(self, result: object) -> None:

        worker_role = str(getattr(self, "_worker_role", "main") or "main")

        self._worker_role = "idle"

        self.btn_run.setEnabled(True)

        self.btn_stop.setEnabled(False)

        if not isinstance(result, dict):

            self.lbl_status.setText("Canceled or no result (dict)")

            if self.logger:

                if result is None:

                    self.logger.warning(

                        "INDEX_SPLINE GUI: the worker finished without dict (None value). "

                        "Common causes: Stop button during calculation, thread closure/interruption, "

                        "or silent worker-side exception. Graphs are not updated since this signal."

                    )

                else:

                    self.logger.warning(

                        "INDEX_SPLINE GUI: the worker returned a %s instead of a dict - result ignored.",

                        type(result).__name__,

                    )

            return

        if self.logger:

            _wm = result.get("pipeline_best_rmse_watermark")

            _wms = result.get("pipeline_best_rmse_stage")

            _wm_hint = ""

            if _wm is not None and np.isfinite(float(_wm)):

                _wm_hint = (

                    f" | pipeline watermark (best RMSE seen during run): {_wm:.6f} "

                    f"(@ {_wms!s})"

                )

            self.logger.info(

                "INDEX_SPLINE GUI: result dict received - dict RMSE (current n/k curves) = %.6f | "

                "d = %.4f nm | worker = %s%s",

                float(np.sqrt(max(float(result.get("mse", 0.0)), 0.0))),

                float(result.get("d_nm", float("nan"))),

                getattr(self._worker.func, "__name__", "?") if self._worker is not None else "?",

                _wm_hint,

            )

            self.logger.info(

                "INDEX_SPLINE GUI: worker detail | mse=%.6e | flags split=%s continuous=%s adaptive=%s",

                float(result.get("mse", float("nan"))),

                bool(result.get("split_knots_refine")),

                bool(result.get("continuous_model")),

                bool(result.get("adaptive_mesh")),

            )

            log_structured_json_event(

                self.logger,

                "AUTO_BEST_JSON",

                "worker_done",

                worker=str(getattr(self._worker.func, "__name__", "?") if self._worker is not None else "?"),

                mse=float(result.get("mse", float("nan"))),

                rmse=float(np.sqrt(max(float(result.get("mse", 0.0)), 0.0))),

                rmse_convention="sqrt(max(mse,0))",

                d_nm=float(result.get("d_nm", float("nan"))),

                split=bool(result.get("split_knots_refine")),

                continuous=bool(result.get("continuous_model")),

                adaptive=bool(result.get("adaptive_mesh")),

            )

        # Auto-Best: declencher une 2e passe locale (knots libres split n/logk) after la 1ere passe warm.

        if self._auto_best_two_stage_refine:

            cfg2 = self._build_opt_config()

            if cfg2 is not None:

                self._auto_best_second_stage_pending = {

                    "seed": dict(result),

                    "cfg": cfg2,

                }

                self._auto_best_two_stage_refine = False

                self.log(

                    "Auto-Best: launching local pass 2 (knots sigma separes pour n et ln k, puis polish).",

                    "INFO",

                )

                self.lbl_status.setText("Auto-Best pass 2: preparing...")

                QTimer.singleShot(0, self._start_auto_best_second_stage)

                return

            self._auto_best_two_stage_refine = False

        self._last_worker_result = dict(result)

        self._corridor_rmse_manual_active = False

        self._corridor_rmse_manual_lo = float("nan")

        self._corridor_rmse_manual_hi = float("nan")

        display = result if worker_role == "corridors" else self._display_result_prefer_best_live(result)

        self._last_result = display

        rmse_tag = (

            "RMSE (bande lambda)"

            if display.get("rmse_fit_lambda_nm") is not None

            else "RMSE (plein spectre)"

        )

        mse_d = float(display.get("mse", 0.0))

        d_nm_d = float(display.get("d_nm", float("nan")))

        nfev_d = int(display.get("nit_polish", result.get("nit_polish", 0)) or 0)

        st = (

            f"{rmse_tag}  {np.sqrt(max(mse_d, 0.0)):.6f} | d={d_nm_d:.2f} nm | "

            f"solver evals≈{nfev_d}"

        )

        if display.get("adaptive_mesh"):

            st = "Maillage adaptatif | " + st

        if display.get("auto_knot_stages") and "sigma_knots" in display:

            kfin = int(np.asarray(display["sigma_knots"], dtype=np.float64).size)

            kbest = display.get("auto_knots_K_best")

            if kbest is not None and int(kbest) != kfin:

                st = f"K retenu={int(kbest)} (last K={kfin}) etapes={len(display['auto_knot_stages'])} | " + st

            else:

                st = f"K={kfin} etapes={len(display['auto_knot_stages'])} | " + st

        if display.get("gui_display_from_best_live"):

            st = "Meilleur RMSE (live) | " + st

        self.lbl_status.setText(st)

        if self.logger:

            self.logger.info("End optimization: %s", st)

            rmse_fin = float(

                display.get(

                    "rmse",

                    float(np.sqrt(max(float(display.get("mse", 0.0)), 0.0))),

                )

            )

            _log_index_spline_best_config(

                self.logger, display, rmse_fin, title="[FIN OPTIM  affichage / export]"

            )

        self._plot_result(display)

        self._refresh_data_table()

        if worker_role != "corridors" and self._result_needs_deferred_corridors(display):

            use_nl_alpha = False

            if self._result_can_offer_nl_alpha_for_corridors(display):

                use_nl_alpha = self._prompt_use_nl_alpha_for_corridors()

            elif self.logger:

                self.logger.info(
                    "Corridors after NL alpha: deferred run without NL-alpha prompt | NL result unavailable as corridor base"
                )

            if self._start_deferred_corridor_worker(display, use_nl_alpha=use_nl_alpha):

                return

        self.export_excel(auto=True)

    def _start_auto_best_second_stage(self) -> None:

        pend = self._auto_best_second_stage_pending

        self._auto_best_second_stage_pending = None

        if not isinstance(pend, dict):

            return

        seed = pend.get("seed")

        cfg2 = pend.get("cfg")

        if not isinstance(seed, dict) or cfg2 is None:

            return

        if self.logger:

            self.logger.info(

                "AUTO_BEST stage2 start | seed_rmse=%.6f seed_d=%.4f",

                float(np.sqrt(max(float(seed.get("mse", 0.0)), 0.0))),

                float(seed.get("d_nm", float("nan"))),

            )

        self._stop_event = Event()

        self._cleanup_thread()

        self._best_live_rmse = float("inf")

        self._best_live_result = None

        self._log_prog_last = -1

        self._last_live_log_mono = 0.0

        self._worker = GenericWorker(

            worker_auto_best_split_knot_refinement,

            seed,

            cfg2,

            self._stop_event,

        )

        _wsig_ab = self._worker.signals

        def _ab_progress(p: float | int, m: str) -> None:

            pv = int(round(float(p) * 100.0))

            _wsig_ab.progress.emit(max(0, min(10000, pv)), m)

        self._worker.kwargs["progress_cb"] = _ab_progress

        self._worker_role = "auto_best"

        self._worker.signals.progress.connect(self._on_progress)

        self._worker.signals.finished.connect(self._on_worker_done)

        self._worker.signals.error.connect(self._on_worker_err)

        self._worker.signals.finished.connect(self._cleanup_thread)

        self._worker.signals.error.connect(self._cleanup_thread)

        self._worker.signals.live.connect(self._on_live_update)

        self._worker.kwargs["live_cb"] = self._worker.signals.live.emit

        self.btn_run.setEnabled(False)

        self.btn_stop.setEnabled(True)

        self._prog_ui_last = 0

        self._prog_reset_bar()

        self._worker.start()

    def export_excel(self, auto: bool = False) -> None:

        """Automatic saving of results to Excel (like CERTUS_DESIGN).

        Generates a timestamped file containing:

        - Spectrum: modèle final n/k ; colonnes polish spectral spline cubique sigma si ``n_lam_seg_spline_sigma``.

          Lines sorted by increasing lambda. Corridors / boot in same order.

        - Corridors_nk: sorted lambda grid - profiling ref, corridor bounds, log₁₀ k; bootstrap if aligned.

        - Comparaison_RMSE_indices: spectral RMSE (solver ref, sigma-spline polish) + best polished model.

        - Parameters: exported x vector.

        - Resume / Best indices: RMSE dict, sigma-spline polish, selected model.

        """

        result = self._last_result

        if result is None:

            if auto:

                return  # Pas de result, rien a exporter

            QMessageBox.warning(self, "Error", "No result to export.")

            return

        import datetime

        # Determiner le dossier et le nom du file (epaisseur en angstrom dans le nom)

        base_dir = os.path.dirname(getattr(self, "_last_spectrum_path", "")) or _SCRIPT_DIR

        ts = datetime.datetime.now().strftime("%Y%m%d_%Hh%M")

        d_nm_fn = result.get("d_nm")

        if isinstance(d_nm_fn, (int, float)) and np.isfinite(float(d_nm_fn)):

            d_ang_int = int(round(float(d_nm_fn) * 10.0))

            fname = f"IndexSpline_Result_{ts}_d{d_ang_int}Ang.xlsx"

        else:

            fname = f"IndexSpline_Result_{ts}_dNA_Ang.xlsx"

        out_path = os.path.join(base_dir, fname)

        try:

            lam_src_full = np.asarray(result["lam_nm"], dtype=np.float64).ravel()

            n_res_full = np.asarray(result["n_lam"], dtype=np.float64).ravel()

            k_res_full = np.asarray(result["k_lam"], dtype=np.float64).ravel()

            t_theo_raw = np.asarray(result.get("t_theo", []), dtype=np.float64).ravel()

            if t_theo_raw.size == lam_src_full.size:

                t_theo_full = t_theo_raw

            else:

                t_theo_full = np.full(lam_src_full.shape, np.nan, dtype=np.float64)

                n_tt = int(min(t_theo_raw.size, lam_src_full.size))

                if n_tt > 0:

                    t_theo_full[:n_tt] = t_theo_raw[:n_tt]

                if lam_src_full.size and t_theo_raw.size != lam_src_full.size:

                    logger.warning(

                        "Export Excel: len(t_theo)=%d ≠ len(lam_nm)=%d - padded with NaN.",

                        int(t_theo_raw.size),

                        int(lam_src_full.size),

                    )

            def _align_to_lam(a: np.ndarray, name: str) -> np.ndarray:

                v = np.asarray(a, dtype=np.float64).ravel()

                if v.size == lam_src_full.size:

                    return v

                out = np.full(lam_src_full.shape, np.nan, dtype=np.float64)

                n_m = int(min(v.size, lam_src_full.size))

                if n_m > 0:

                    out[:n_m] = v[:n_m]

                if lam_src_full.size and v.size != lam_src_full.size:

                    logger.warning(

                        "Export Excel: len(%s)=%d ≠ len(lam_nm)=%d - padded with NaN.",

                        name,

                        int(v.size),

                        int(lam_src_full.size),

                    )

                return out

            n_res_full = _align_to_lam(n_res_full, "n_lam")

            k_res_full = _align_to_lam(k_res_full, "k_lam")

            # Calculer le ratio experimental sur le mesh complet du result

            ratio_exp_pct_full = np.full_like(lam_src_full, np.nan)

            if self.df is not None and "T" in self.df.columns:

                lam_raw = ensure_lam_nm_array(self.df["lambda"].to_numpy(dtype=np.float64))

                t_raw_all = _to_fraction_T(self.df["T"].to_numpy(dtype=np.float64))

                t_raw_interp = np.interp(lam_src_full, lam_raw, t_raw_all)

                if result.get("t_is_ratio", self.chk_trel.isChecked()):

                    ratio_exp_pct_full = t_raw_interp * 100.0

                else:

                    sub_name = str(self.cb_sub.currentData() or self.cb_sub.currentText())

                    sub_id = substrate_id_from_name(sub_name)

                    n_sub = get_n_substrate_array_by_id(sub_id, lam_src_full)

                    t_sub = calculate_T_substrate_array(lam_src_full, n_sub)

                    ratio_exp_pct_full = (t_raw_interp / np.maximum(t_sub, 1e-6)) * 100.0

            ratio_theo_pct_full = t_theo_full * 100.0

            rw_rep = self._rmse_fit_lambda_tuple_for_report()

            m_obj = self._smart_mesh_objective_lam_mask_float(lam_src_full)

            keep = m_obj > 0.5

            spectre_filtre = bool(rw_rep is not None and np.any(~keep) and np.count_nonzero(keep) > 0)

            export_fallback_lam = False

            if np.count_nonzero(keep) == 0:

                logger.warning(

                    "Export Excel: aucun point dans le masque objectif - export de tous les lambda finis."

                )

                keep = np.isfinite(lam_src_full)

                spectre_filtre = False

                export_fallback_lam = True

            cfg_ex = self._build_opt_config(notify=False)

            k_clip_lo = float(getattr(cfg_ex, "k_clip_lo", 1e-5)) if cfg_ex is not None else 1e-5

            k_clip_hi = (

                float(getattr(cfg_ex, "k_clip_hi", min(0.99, float(K_MAX_LIMIT))))

                if cfg_ex is not None

                else min(0.99, float(K_MAX_LIMIT))

            )

            k_hi = float(min(max(k_clip_hi, k_clip_lo * 1.0001), float(K_MAX_LIMIT)))

            x_res = np.asarray(result.get("x", np.zeros(19)), dtype=np.float64)

            d_nm_c = float(result["d_nm"]) if isinstance(result.get("d_nm"), (int, float)) and np.isfinite(float(result["d_nm"])) else (

                float(x_res[0]) if x_res.size >= 1 and np.isfinite(float(x_res[0])) else float("nan")

            )

            # sigma-spline mesh polish: only if curves present (same length as lambda).

            n_spl_full = np.full(lam_src_full.shape, np.nan, dtype=np.float64)

            k_spl_full = np.full(lam_src_full.shape, np.nan, dtype=np.float64)

            n_sp = result.get("n_lam_seg_spline_sigma")

            k_sp = result.get("k_lam_seg_spline_sigma")

            if (

                n_sp is not None

                and k_sp is not None

                and np.asarray(n_sp).size == lam_src_full.size

                and np.asarray(k_sp).size == lam_src_full.size

            ):

                n_spl_full = np.asarray(n_sp, dtype=np.float64).ravel()

                k_spl_full = np.asarray(k_sp, dtype=np.float64).ravel()

            def _spectral_rmse_export(

                n_arr: np.ndarray,

                k_arr: np.ndarray,

                *,

                d_nm_use: float | None = None,

            ) -> tuple[str, float]:

                d_eff = (

                    float(d_nm_use)

                    if d_nm_use is not None and np.isfinite(float(d_nm_use))

                    else float(d_nm_c)

                )

                if cfg_ex is None or not np.isfinite(d_eff):

                    return "N/A", float("nan")

                g = build_spline_objective_masked_grid(cfg_ex)

                if g is None:

                    return "N/A", float("nan")

                lam_f, _sig_f, n_sub_f, w, inv_npix, t_exp_f, r_exp_f = g

                n_sub_eff = np.asarray(n_sub_f, dtype=np.float64)

                ls = np.asarray(lam_src_full, dtype=np.float64).ravel()

                na = np.asarray(n_arr, dtype=np.float64).ravel()

                ka = np.asarray(k_arr, dtype=np.float64).ravel()

                if na.size != ls.size or ka.size != ls.size:

                    return "N/A", float("nan")

                ord_i = np.argsort(ls, kind="mergesort")

                ls_s = ls[ord_i]

                n_f = np.interp(lam_f, ls_s, na[ord_i], left=np.nan, right=np.nan)

                k_f = np.interp(lam_f, ls_s, ka[ord_i], left=np.nan, right=np.nan)

                if not np.all(np.isfinite(n_f) & np.isfinite(k_f)):

                    return "N/A", float("nan")

                try:

                    mse_v = spline_objective_mse_on_masked_grid(

                        cfg_ex,

                        lam_f=lam_f,

                        n_sub_f=n_sub_eff,

                        w=w,

                        inv_npix=inv_npix,

                        t_exp_f=t_exp_f,

                        r_exp_f=r_exp_f,

                        n_l=n_f,

                        k_l=k_f,

                        d=float(d_eff),

                    )

                    if np.isfinite(mse_v) and mse_v < 1e29:

                        r = float(np.sqrt(mse_v))

                        return f"{r:.6f}", r

                except Exception:

                    logger.exception("Spectral RMSE Excel export (model comparison)")

                return "N/A", float("nan")

            d_spl_x = result.get("d_nm_seg_spline_sigma")

            d_spl_f = (

                float(d_spl_x)

                if isinstance(d_spl_x, (int, float)) and np.isfinite(float(d_spl_x))

                else float("nan")

            )

            def _rmse_pref_result(

                key: str, n_a: np.ndarray, k_a: np.ndarray, d_alt: float

            ) -> tuple[str, float]:

                v = result.get(key)

                if v is not None and np.isfinite(float(v)):

                    fv = float(v)

                    return f"{fv:.6f}", fv

                d_use = d_alt if np.isfinite(d_alt) else None

                return _spectral_rmse_export(n_a, k_a, d_nm_use=d_use)

            rmse_spl_txt, _ = _rmse_pref_result(

                "spectral_rmse_seg_spline_sigma", n_spl_full, k_spl_full, d_spl_f

            )

            rmse_solver_txt = "N/A"

            srv = result.get("spectral_rmse_segments")

            if srv is not None and np.isfinite(float(srv)):

                rmse_solver_txt = f"{float(srv):.6f}"

            has_spl_cols = bool(np.any(np.isfinite(n_spl_full)) and np.any(np.isfinite(k_spl_full)))

            best_lbl = str(result.get("spectral_rmse_best_label") or "").strip()

            best_v = result.get("spectral_rmse_best_value")

            best_pretty = {

                "Spline_cubique_sigma": "Polish maillage spline cubique sigma",

            }.get(best_lbl, best_lbl or "-")

            best_line = (

                f"{best_pretty} - RMSE={float(best_v):.6f}"

                if best_v is not None and np.isfinite(float(best_v)) and best_lbl

                else "N/A (voir colonnes RMSE)"

            )

            if best_lbl == "Spline_cubique_sigma" and not has_spl_cols:

                best_line = "N/A ('sigma spline' label without n_lam_seg_spline_sigma in dict)"

            if best_lbl == "Spline_cubique_sigma" and has_spl_cols:

                n_best_src = np.asarray(n_spl_full, dtype=np.float64).ravel().copy()

                k_best_src = np.asarray(k_spl_full, dtype=np.float64).ravel().copy()

                d_best_export = float(d_spl_f) if np.isfinite(d_spl_f) else float(d_nm_c)

            else:

                n_best_src = np.asarray(n_res_full, dtype=np.float64).ravel().copy()

                k_best_src = np.asarray(k_res_full, dtype=np.float64).ravel().copy()

                d_best_export = (

                    float(d_nm_c)

                    if isinstance(d_nm_c, (int, float)) and np.isfinite(float(d_nm_c))

                    else float("nan")

                )

            rmse_best_recalc_txt, _ = _spectral_rmse_export(

                n_best_src,

                k_best_src,

                d_nm_use=d_best_export if np.isfinite(d_best_export) else None,

            )

            compare_note = (

                "L-BFGS-B spectral polish on sigma mesh (cubic spline between nodes, same objective mask). "

                f"Solver reference (before mesh polish): RMSE={rmse_solver_txt}. "

                f"Polished model: {best_line}."

            )

            if not has_spl_cols:

                compare_note += (

                    " 'Spectrum' sheet sigma-spline polish columns not filled "

                    "(n_lam_seg_spline_sigma / k_lam_seg_spline_sigma absentes ou NaN)."

                )

            lam = lam_src_full[keep]

            n_lam = n_res_full[keep]

            k_lam = k_res_full[keep]

            ratio_exp_pct = ratio_exp_pct_full[keep]

            ratio_theo_pct = ratio_theo_pct_full[keep]

            n_spl_spec = n_spl_full[keep]

            k_spl_spec = k_spl_full[keep]

            ord_ex = np.argsort(lam, kind="mergesort") if lam.size else np.arange(0, dtype=np.intp)

            if lam.size:

                lam = lam[ord_ex]

                n_lam = n_lam[ord_ex]

                k_lam = k_lam[ord_ex]

                ratio_exp_pct = ratio_exp_pct[ord_ex]

                ratio_theo_pct = ratio_theo_pct[ord_ex]

                n_spl_spec = n_spl_spec[ord_ex]

                k_spl_spec = k_spl_spec[ord_ex]

            def _log10_k_safe(kv: np.ndarray) -> np.ndarray:

                return np.log10(np.maximum(np.asarray(kv, dtype=np.float64).ravel(), 1e-300))

            cn_lo_f = np.asarray(result.get("corridor_n_lo", []), dtype=np.float64).ravel()

            cn_hi_f = np.asarray(result.get("corridor_n_hi", []), dtype=np.float64).ravel()

            ck_lo_f = np.asarray(result.get("corridor_k_lo", []), dtype=np.float64).ravel()

            ck_hi_f = np.asarray(result.get("corridor_k_hi", []), dtype=np.float64).ravel()

            cn_ref_f = np.asarray(result.get("corridor_reference_n_lam", []), dtype=np.float64).ravel()

            ck_ref_f = np.asarray(result.get("corridor_reference_k_lam", []), dtype=np.float64).ravel()

            # Corridor sheets: tables aligned on lam_nm (not only profile_d_enabled bool).

            corr_grid_ok = (

                cn_lo_f.size == lam_src_full.size

                and cn_hi_f.size == lam_src_full.size

                and ck_lo_f.size == lam_src_full.size

                and ck_hi_f.size == lam_src_full.size

                and cn_lo_f.size > 0

            )

            if cn_ref_f.size != lam_src_full.size:

                cn_ref_f = np.array([], dtype=np.float64)

            if ck_ref_f.size != lam_src_full.size:

                ck_ref_f = np.array([], dtype=np.float64)

            spec_rows: dict[str, Any] = {

                "Wavelength (nm)": lam,

                "n_film (final model)": n_lam,

                "k_film (final model)": k_lam,

                "n_cubic_spline_sigma_spectral_polish": n_spl_spec,

                "k_cubic_spline_sigma_spectral_polish": k_spl_spec,

                "Ratio_Exp (%)": ratio_exp_pct,

                "Ratio_Theo (%)": ratio_theo_pct,

            }

            if corr_grid_ok:

                cnk = cn_ref_f[keep][ord_ex] if cn_ref_f.size else np.full(lam.shape, np.nan)

                ckk = ck_ref_f[keep][ord_ex] if ck_ref_f.size else np.full(lam.shape, np.nan)

                spec_rows["n_corridor_ref (d profiling)"] = cnk

                spec_rows["k_corridor_ref (d profiling)"] = ckk

                if cn_ref_f.size and ck_ref_f.size:

                    spec_rows["log10_k_corridor_ref"] = _log10_k_safe(ck_ref_f[keep][ord_ex])

                else:

                    spec_rows["log10_k_corridor_ref"] = np.full(lam.shape, np.nan)

                spec_rows["n_corridor_lo"] = cn_lo_f[keep][ord_ex]

                spec_rows["n_corridor_hi"] = cn_hi_f[keep][ord_ex]

                spec_rows["k_corridor_lo"] = ck_lo_f[keep][ord_ex]

                spec_rows["k_corridor_hi"] = ck_hi_f[keep][ord_ex]

                spec_rows["log10_k_corridor_lo"] = _log10_k_safe(ck_lo_f[keep][ord_ex])

                spec_rows["log10_k_corridor_hi"] = _log10_k_safe(ck_hi_f[keep][ord_ex])

            bsn_lo = np.asarray(result.get("boot_corridor_n_lo", []), dtype=np.float64).ravel()

            bsn_hi = np.asarray(result.get("boot_corridor_n_hi", []), dtype=np.float64).ravel()

            bsk_lo = np.asarray(result.get("boot_corridor_k_lo", []), dtype=np.float64).ravel()

            bsk_hi = np.asarray(result.get("boot_corridor_k_hi", []), dtype=np.float64).ravel()

            boot_spec_ok = (

                bsn_lo.size == lam_src_full.size

                and bsn_hi.size == lam_src_full.size

                and bsk_lo.size == lam_src_full.size

                and bsk_hi.size == lam_src_full.size

                and bsn_lo.size > 0

            )

            if boot_spec_ok:

                spec_rows["boot_n_lo"] = bsn_lo[keep][ord_ex]

                spec_rows["boot_n_hi"] = bsn_hi[keep][ord_ex]

                spec_rows["boot_k_lo"] = bsk_lo[keep][ord_ex]

                spec_rows["boot_k_hi"] = bsk_hi[keep][ord_ex]

                spec_rows["boot_log10_k_lo"] = _log10_k_safe(bsk_lo[keep][ord_ex])

                spec_rows["boot_log10_k_hi"] = _log10_k_safe(bsk_hi[keep][ord_ex])

            # lambda order for “full grid” sheets (same permutation everywhere).

            ord_lam_full = _mergesort_order_lambda(lam_src_full)

            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

                # Spectrum sheet: final model + polish spectral spline cubique sigma (seg_spline_sigma) on same lambda

                pd.DataFrame(spec_rows).to_excel(writer, sheet_name="Spectre", index=False)

                # Parameters sheet (piecewise mesh x vector)

                xb = np.asarray(result.get("x", np.zeros(1)), dtype=np.float64)

                labels = ["Thickness d (nm)"]

                labels += [f"Coeff_n_{i}" for i in range(1, 10)]

                labels += [f"Coeff_logk_{i}" for i in range(1, 10)]

                while len(labels) < len(xb):

                    labels.append(f"Param_{len(labels)}")

                pd.DataFrame({

                    "Index": np.arange(len(xb)),

                    "Meaning": labels[:len(xb)],

                    "Value": xb,

                }).to_excel(writer, sheet_name="Mesh_Parameters", index=False)

                # Normalized variable

                sig_knots = result.get("sigma_knots", np.array([0, 1]))

                smin, smax = float(sig_knots[0]), float(sig_knots[-1])

                if rw_rep is not None:

                    lo_r, hi_r = float(rw_rep[0]), float(rw_rep[1])

                    fen_txt = f"[{lo_r:.2f}, {hi_r:.2f}]"

                    if export_fallback_lam:

                        spec_txt = "Fallback: all finite lambda (empty objective mask)"

                    elif spectre_filtre:

                        spec_txt = "Only lambda in objective mask (RMSE window + valid data)"

                    else:

                        spec_txt = "All lambda from result (window covers grid or no excluded points)"

                else:

                    fen_txt = "- (full objective spectrum)"

                    spec_txt = (

                        "Fallback: all finite lambda (mask error)"

                        if export_fallback_lam

                        else "All lambda points from result"

                    )

                gui_live = bool(result.get("gui_display_from_best_live"))

                rmse_w_fin = result.get("gui_worker_raw_rmse")

                rmse_live_gui = result.get("gui_best_live_rmse")

                live_note = (

                    f"Yes - displayed RMSE={float(rmse_live_gui):.6f}, final worker dict RMSE={float(rmse_w_fin):.6f}"

                    if gui_live

                    and rmse_w_fin is not None

                    and rmse_live_gui is not None

                    and np.isfinite(float(rmse_w_fin))

                    and np.isfinite(float(rmse_live_gui))

                    else ("Yes (details: gui_best_live_rmse / gui_worker_raw_rmse)" if gui_live else "No")

                )

                spectre_ordre = "increasing lambda (mergesort, aligned with Data table / Spectrum tab)"

                pd.DataFrame({

                    "Indicateur": [

                        "Final RMSE (result dict)",

                        "Spectral RMSE - solver ref (mesh, before mesh polish)",

                        "Spectral RMSE - cubic spline sigma mesh (spectral polish)",

                        "Best model (mesh polish spline sigma)",

                        "Thickness (nm) final model",

                        "Display = best live snapshot (GUI)",

                        "Spectrum sheet - Wavelength order",

                        "RMSE lambda window (nm)",

                        "Spectrum sheet (lambda lines)",

                        "Variable u",

                        "sigma_min (1/nm)",

                        "sigma_max (1/nm)",

                        "Export Date",

                    ],

                    "Valeur": [

                        f"{result.get('rmse', 'N/A'):.6f}" if isinstance(result.get('rmse'), (int, float)) else "N/A",

                        rmse_solver_txt,

                        rmse_spl_txt,

                        best_line,

                        f"{result.get('d_nm', 'N/A'):.2f}" if isinstance(result.get('d_nm'), (int, float)) else "N/A",

                        live_note,

                        spectre_ordre,

                        fen_txt,

                        spec_txt,

                        f"u = (sigma - {smin:.6e}) / ({smax:.6e} - {smin:.6e}), sigma = 1/lambda",

                        f"{smin:.6e}",

                        f"{smax:.6e}",

                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                    ],

                }).to_excel(writer, sheet_name="Resume", index=False)

                pd.DataFrame({

                    "Grandeur": [

                        "Spectral RMSE - solver ref (before mesh polish)",

                        "Spectral RMSE - cubic spline sigma (polish)",

                        "Best (internal label)",

                        "Criteria",

                        "Note",

                    ],

                    "Valeur": [

                        rmse_solver_txt,

                        rmse_spl_txt,

                        str(best_lbl) if best_lbl else "-",

                        "Same mask and weights as spline objective (build_spline_objective_masked_grid).",

                        compare_note,

                    ],

                }).to_excel(writer, sheet_name="Comparaison_RMSE_indices", index=False)

                # Corridors n/k: full lambda grid (aligned on result lam_nm), same as GUI tab.

                if corr_grid_ok:

                    try:

                        ord_cf = ord_lam_full

                        lam_cf = np.asarray(lam_src_full, dtype=np.float64).ravel()[ord_cf]

                        nref_c = (

                            cn_ref_f[ord_cf].copy()

                            if cn_ref_f.size == lam_src_full.size

                            else np.full(lam_cf.shape, np.nan, dtype=np.float64)

                        )

                        kref_c = (

                            ck_ref_f[ord_cf].copy()

                            if ck_ref_f.size == lam_src_full.size

                            else np.full(lam_cf.shape, np.nan, dtype=np.float64)

                        )

                        log10_k_ref_col = np.full(lam_cf.shape, np.nan, dtype=np.float64)

                        if ck_ref_f.size == lam_src_full.size and np.any(np.isfinite(ck_ref_f)):

                            log10_k_ref_col = _log10_k_safe(ck_ref_f[ord_cf])

                        corr_full: dict[str, Any] = {

                            "Wavelength (nm)": lam_cf,

                            "n_corridor_ref (d profiling)": nref_c,

                            "k_corridor_ref (d profiling)": kref_c,

                            "log10_k_corridor_ref": log10_k_ref_col,

                            "n_corridor_lo": cn_lo_f[ord_cf],

                            "n_corridor_hi": cn_hi_f[ord_cf],

                            "k_corridor_lo": ck_lo_f[ord_cf],

                            "k_corridor_hi": ck_hi_f[ord_cf],

                            "log10_k_corridor_lo": _log10_k_safe(ck_lo_f[ord_cf]),

                            "log10_k_corridor_hi": _log10_k_safe(ck_hi_f[ord_cf]),

                        }

                        bn_lo = np.asarray(result.get("boot_corridor_n_lo", []), dtype=np.float64).ravel()

                        bn_hi = np.asarray(result.get("boot_corridor_n_hi", []), dtype=np.float64).ravel()

                        bk_lo = np.asarray(result.get("boot_corridor_k_lo", []), dtype=np.float64).ravel()

                        bk_hi = np.asarray(result.get("boot_corridor_k_hi", []), dtype=np.float64).ravel()

                        if (

                            bn_lo.size == lam_src_full.size

                            and bn_hi.size == lam_src_full.size

                            and bk_lo.size == lam_src_full.size

                            and bk_hi.size == lam_src_full.size

                        ):

                            corr_full["boot_n_lo"] = bn_lo[ord_cf]

                            corr_full["boot_n_hi"] = bn_hi[ord_cf]

                            corr_full["boot_k_lo"] = bk_lo[ord_cf]

                            corr_full["boot_k_hi"] = bk_hi[ord_cf]

                            corr_full["boot_log10_k_lo"] = _log10_k_safe(bk_lo[ord_cf])

                            corr_full["boot_log10_k_hi"] = _log10_k_safe(bk_hi[ord_cf])

                        pd.DataFrame(corr_full).to_excel(

                            writer, sheet_name="Corridors_nk", index=False

                        )

                    except Exception:

                        logger.exception("Export Excel: feuille Corridors_nk")

                else:

                    try:

                        bn_lo2 = np.asarray(

                            result.get("boot_corridor_n_lo", []), dtype=np.float64

                        ).ravel()

                        bn_hi2 = np.asarray(

                            result.get("boot_corridor_n_hi", []), dtype=np.float64

                        ).ravel()

                        bk_lo2 = np.asarray(

                            result.get("boot_corridor_k_lo", []), dtype=np.float64

                        ).ravel()

                        bk_hi2 = np.asarray(

                            result.get("boot_corridor_k_hi", []), dtype=np.float64

                        ).ravel()

                        if (

                            bn_lo2.size == lam_src_full.size

                            and bn_hi2.size == lam_src_full.size

                            and bk_lo2.size == lam_src_full.size

                            and bk_hi2.size == lam_src_full.size

                            and bn_lo2.size > 0

                        ):

                            ord_b = ord_lam_full

                            lam_b = np.asarray(lam_src_full, dtype=np.float64).ravel()[ord_b]

                            pd.DataFrame(

                                {

                                    "Wavelength (nm)": lam_b,

                                    "boot_n_lo": bn_lo2[ord_b],

                                    "boot_n_hi": bn_hi2[ord_b],

                                    "boot_k_lo": bk_lo2[ord_b],

                                    "boot_k_hi": bk_hi2[ord_b],

                                    "boot_log10_k_lo": _log10_k_safe(bk_lo2[ord_b]),

                                    "boot_log10_k_hi": _log10_k_safe(bk_hi2[ord_b]),

                                }

                            ).to_excel(

                                writer, sheet_name="Corridors_bootstrap", index=False

                            )

                    except Exception:

                        logger.exception("Export Excel: Corridors_bootstrap sheet")

                # Best indices: n, k from the model with minimal spectral RMSE (sigma-mesh spline polish),

                # interpolated on uniform 2 nm, 5 nm, 10 nm grids (same [lambda_min, lambda_max] range).

                lo = float(np.nanmin(lam_src_full[np.isfinite(lam_src_full)])) if np.any(np.isfinite(lam_src_full)) else float("nan")

                hi = float(np.nanmax(lam_src_full[np.isfinite(lam_src_full)])) if np.any(np.isfinite(lam_src_full)) else float("nan")

                def _lam_uniform_grid(lo_h: float, hi_h: float, step: float) -> np.ndarray:

                    if not (np.isfinite(lo_h) and np.isfinite(hi_h) and hi_h > lo_h):

                        return np.array([], dtype=np.float64)

                    st = float(np.ceil(lo_h / step) * step)

                    en = float(np.floor(hi_h / step) * step)

                    if en < st - 1e-9:

                        return np.array([0.5 * (lo_h + hi_h)], dtype=np.float64)

                    if abs(en - st) < 1e-9:

                        return np.array([st], dtype=np.float64)

                    return np.arange(st, en + 1e-9, step, dtype=np.float64)

                ls_src = np.asarray(lam_src_full, dtype=np.float64).ravel()

                n_bs = np.asarray(n_best_src, dtype=np.float64).ravel()

                k_bs = np.asarray(k_best_src, dtype=np.float64).ravel()

                grid_parts: list[pd.DataFrame] = []

                if n_bs.size == ls_src.size and k_bs.size == ls_src.size and ls_src.size > 0:

                    ord_i = np.argsort(ls_src, kind="mergesort")

                    ls_s = ls_src[ord_i]

                    n_s = n_bs[ord_i]

                    k_s = k_bs[ord_i]

                    for step in (2.0, 5.0, 10.0):

                        lam_g = _lam_uniform_grid(lo, hi, step)

                        if lam_g.size == 0:

                            continue

                        n_g = np.interp(lam_g, ls_s, n_s, left=np.nan, right=np.nan)

                        k_g = np.interp(lam_g, ls_s, k_s, left=np.nan, right=np.nan)

                        grid_parts.append(

                            pd.DataFrame(

                                {

                                    "Step (nm)": np.full(lam_g.size, step, dtype=np.float64),

                                    "Wavelength (nm)": lam_g,

                                    "n_best": n_g,

                                    "k_best": k_g,

                                }

                            )

                        )

                df_grids = (

                    pd.concat(grid_parts, ignore_index=True)

                    if grid_parts

                    else pd.DataFrame(columns=["Step (nm)", "Wavelength (nm)", "n_best", "k_best"])

                )

                desc_rows = [

                    "Source / method",

                    "Chosen model (spectral RMSE mesh polish - spline cubique sigma)",

                    "Internal label",

                    "Spectral RMSE (chosen value)",

                    "Spectral RMSE (control, result mesh + objective mask)",

                    "Thickness d associated with chosen model (nm)",

                    "n/k Corridors (d profiling): active",

                    "Corridors: d interval (nm)",

                    "Corridors: mode",

                    "Corridors: conf (LR)",

                    "Corridors: Deltachi2 (LR)",

                    "Corridors: sigma_T (LR)",

                    "Corridors: sigma_R (LR)",

                    "Corridors: alpha (RMSE <= alpha * RMSE_opt)",

                    "Corridors: RMSE_opt (threshold reference)",

                    "Corridors: ref RMSE source (spectral_rmse_segments | dict_rmse | recalc_objective)",

                    "Corridors: RMSE_threshold",

                    "Table: 2 nm then 5 nm then 10 nm grids",

                    "Spectral RMSE - solver ref (before mesh polish)",

                    "Spectral RMSE - cubic spline sigma (polish)",

                ]

                def _result_float(key: str) -> float:

                    """float(result[key]) tolerating missing key or None (e.g. alpha mode -> LR N/A)."""

                    v = result.get(key)

                    if v is None:

                        return float("nan")

                    try:

                        return float(v)

                    except (TypeError, ValueError):

                        return float("nan")

                val_rows = [

                    "numpy.interp on lambda (sorted result mesh) from n(lambda), k(lambda) "

                    "curves of the best polish; uniform sub-sampling steps 2, 5 and 10 nm on [lambda_min, lambda_max].",

                    best_pretty,

                    str(best_lbl) if best_lbl else "-",

                    f"{float(best_v):.6f}" if best_v is not None and np.isfinite(float(best_v)) else "N/A",

                    rmse_best_recalc_txt,

                    f"{d_best_export:.4f}" if np.isfinite(d_best_export) else "N/A",

                    "Yes (corridor_* vectors present, aligned with lam_nm)" if corr_grid_ok else "No",

                    (

                        f"[{float(result.get('profile_d_interval_nm')[0]):.3f}, {float(result.get('profile_d_interval_nm')[1]):.3f}]"

                        if corr_grid_ok

                        and isinstance(result.get("profile_d_interval_nm"), (tuple, list))

                        and len(result.get("profile_d_interval_nm")) == 2

                        else "-"

                    ),

                    str(result.get("profile_d_mode", "-")),

                    f"{_result_float('profile_d_lr_conf'):.3f}"

                    if np.isfinite(_result_float("profile_d_lr_conf"))

                    else "-",

                    f"{_result_float('profile_d_lr_delta_chi2'):.6f}"

                    if np.isfinite(_result_float("profile_d_lr_delta_chi2"))

                    else "-",

                    f"{_result_float('profile_d_sigma_t'):.6g}"

                    if np.isfinite(_result_float("profile_d_sigma_t"))

                    else "-",

                    f"{_result_float('profile_d_sigma_r'):.6g}"

                    if np.isfinite(_result_float("profile_d_sigma_r"))

                    else "-",

                    f"{_result_float('profile_d_rmse_alpha'):.3f}"

                    if np.isfinite(_result_float("profile_d_rmse_alpha"))

                    else "-",

                    f"{_result_float('profile_d_rmse_opt'):.6f}"

                    if np.isfinite(_result_float("profile_d_rmse_opt"))

                    else "-",

                    str(result.get("profile_d_rmse_ref_source", "-") or "-"),

                    f"{_result_float('profile_d_rmse_thresh'):.6f}"

                    if np.isfinite(_result_float("profile_d_rmse_thresh"))

                    else "-",

                    "Column “Step (nm)” separates the three blocks; same spectral interval.",

                    rmse_solver_txt,

                    rmse_spl_txt,

                ]

                df_head = pd.DataFrame({"Description": desc_rows, "Value": val_rows})

                sheet_best = "Best indices"

                df_head.to_excel(writer, sheet_name=sheet_best, index=False)

                if not df_grids.empty:

                    df_grids.to_excel(

                        writer,

                        sheet_name=sheet_best,

                        index=False,

                        startrow=len(df_head) + 2,

                    )

                # RMSE(d) profile (if available)

                d_prof = np.asarray(result.get("profile_d_values_nm", []), dtype=np.float64).ravel()

                r_prof = np.asarray(result.get("profile_d_rmse_values", []), dtype=np.float64).ravel()

                if d_prof.size and r_prof.size == d_prof.size:

                    od = np.argsort(d_prof, kind="mergesort")

                    pd.DataFrame({"d_nm": d_prof[od], "rmse": r_prof[od]}).to_excel(

                        writer, sheet_name="Profil_d_RMSE", index=False

                    )

                c_prof = np.asarray(result.get("profile_d_chi2_values", []), dtype=np.float64).ravel()

                if d_prof.size and c_prof.size == d_prof.size:

                    od = np.argsort(d_prof, kind="mergesort")

                    pd.DataFrame({"d_nm": d_prof[od], "chi2": c_prof[od]}).to_excel(

                        writer, sheet_name="Profil_d_CHI2", index=False

                    )

                # V2.3: regularization sensitivity (if available)

                w_reg = np.asarray(result.get("reg_sens_weights", []), dtype=np.float64).ravel()

                d_lo = np.asarray(result.get("reg_sens_d_lo_nm", []), dtype=np.float64).ravel()

                d_hi = np.asarray(result.get("reg_sens_d_hi_nm", []), dtype=np.float64).ravel()

                w_n = np.asarray(result.get("reg_sens_mean_width_n", []), dtype=np.float64).ravel()

                w_k = np.asarray(result.get("reg_sens_mean_width_k", []), dtype=np.float64).ravel()

                n_v = np.asarray(result.get("reg_sens_n_valid", []), dtype=np.int64).ravel()

                if w_reg.size and d_lo.size == w_reg.size and d_hi.size == w_reg.size:

                    pd.DataFrame(

                        {

                            "reg_weight_lnk": w_reg,

                            "d_lo_nm": d_lo,

                            "d_hi_nm": d_hi,

                            "mean_width_n": w_n if w_n.size == w_reg.size else np.full_like(w_reg, np.nan),

                            "mean_width_k": w_k if w_k.size == w_reg.size else np.full_like(w_reg, np.nan),

                            "n_valid": n_v if n_v.size == w_reg.size else np.zeros_like(w_reg, dtype=np.int64),

                        }

                    ).to_excel(writer, sheet_name="Sensibilite_reg", index=False)

                # V2.4: bootstrap - only if metadata or samples present (avoids empty rows after live strip).

                _boot_meta_present = (

                    result.get("boot_n") is not None

                    or np.asarray(result.get("boot_d_lo_samples_nm", []), dtype=np.float64).size > 0

                    or np.asarray(result.get("boot_runs_b", []), dtype=np.int64).size > 0

                )

                if _boot_meta_present:

                    try:

                        df_boot = pd.DataFrame(

                            {

                                "boot_n": [int(result.get("boot_n", 0) or 0)],

                                "boot_n_ok": [int(result.get("boot_n_ok", 0) or 0)],

                                "boot_seed": [int(result.get("boot_seed", 0) or 0)],

                                "boot_mode": [str(result.get("boot_mode", "-"))],

                                "boot_block_len": [int(result.get("boot_block_len", 1) or 1)],

                                "boot_percentile": [_result_float("boot_percentile")],

                                "boot_sigma_t": [_result_float("boot_sigma_t")],

                                "boot_sigma_r": [_result_float("boot_sigma_r")],

                                "boot_d_lo_q_nm": [_result_float("boot_d_lo_q_nm")],

                                "boot_d_hi_q_nm": [_result_float("boot_d_hi_q_nm")],

                            }

                        )

                        df_boot.to_excel(writer, sheet_name="Bootstrap_resume", index=False)

                        dls = np.asarray(result.get("boot_d_lo_samples_nm", []), dtype=np.float64).ravel()

                        dhs = np.asarray(result.get("boot_d_hi_samples_nm", []), dtype=np.float64).ravel()

                        if dls.size and dhs.size == dls.size:

                            pd.DataFrame({"d_lo_nm": dls, "d_hi_nm": dhs}).to_excel(

                                writer, sheet_name="Bootstrap_d_samples", index=False

                            )

                        # Table d'audit par run

                        rb = np.asarray(result.get("boot_runs_b", []), dtype=np.int64).ravel()

                        rok = np.asarray(result.get("boot_runs_ok", []), dtype=np.int64).ravel()

                        rd0 = np.asarray(result.get("boot_runs_d_lo_nm", []), dtype=np.float64).ravel()

                        rd1 = np.asarray(result.get("boot_runs_d_hi_nm", []), dtype=np.float64).ravel()

                        rnv = np.asarray(result.get("boot_runs_n_valid", []), dtype=np.int64).ravel()

                        if rb.size and rok.size == rb.size and rd0.size == rb.size and rd1.size == rb.size:

                            pd.DataFrame(

                                {

                                    "b": rb,

                                    "ok": rok,

                                    "d_lo_nm": rd0,

                                    "d_hi_nm": rd1,

                                    "n_valid": rnv if rnv.size == rb.size else np.zeros_like(rb),

                                }

                            ).to_excel(writer, sheet_name="Bootstrap_runs", index=False)

                    except Exception:

                        logger.exception("Export Excel: bootstrap")

            self.log(f"Resultats exportes -> {fname}", "SUCCESS")

            logger.info(f"Export Excel: {out_path}")

        except Exception as e:

            self.log(f"Error export Excel: {e}", "ERROR")

            logger.exception("Export Excel failed")

    def _plot_corridor_tab(

        self,

        r: dict,

        lam_s: np.ndarray,

        n_s: np.ndarray,

        k_s: np.ndarray,

        *,

        spectral_sort_order: np.ndarray | None = None,

    ) -> None:

        """'n/k Corridors' tab: central curves + envelopes; auto-focus if bands present.

        Corridors in the dict are indexed as ``r['lam_nm']`` (file/model order). Spectral display

        uses ``lam_s`` sorted by lambda (_spectral_display_align): same indices must be permuted on

        corridor_* otherwise envelope and center refit are decorrelated from the n,k plot (nominal or 'horizontal' band outside corridor).

        Same convention as ``_mergesort_order_lambda`` / Excel export.

        """

        if not hasattr(self, "plot_n_corridor"):

            return

        try:

            self.plot_n_corridor.clear()

            self.plot_lgk_corridor.clear()

        except (AttributeError, RuntimeError):

            return

        nu = int(np.asarray(lam_s, dtype=np.float64).ravel().size)

        ord_s = (

            np.asarray(spectral_sort_order, dtype=np.intp).ravel()

            if spectral_sort_order is not None

            else np.array([], dtype=np.intp)

        )

        use_order = ord_s.size == nu and nu > 0

        def _corridor_aligned_to_display(a: np.ndarray) -> np.ndarray:

            aa = np.asarray(a, dtype=np.float64).ravel()

            if not use_order:

                return aa

            if aa.size < nu:

                return aa

            return aa[:nu][ord_s]

        # Bold curves = nominal n_lam / k_lam (same as "n & log₁₀ k" tab and Excel export). Shaded band =

        # min/max over d-profiling refits. Dashed overlay (if any): center-d refit used as first row of the stack.

        n_ref = _corridor_aligned_to_display(np.asarray(r.get("corridor_reference_n_lam", []), dtype=np.float64))

        k_ref = _corridor_aligned_to_display(np.asarray(r.get("corridor_reference_k_lam", []), dtype=np.float64))

        has_ref = n_ref.size == lam_s.size and k_ref.size == lam_s.size

        n_c = np.asarray(n_s, dtype=np.float64).ravel()

        k_c = np.asarray(k_s, dtype=np.float64).ravel()

        has_profile = False

        has_boot = False

        try:

            n_lo = _corridor_aligned_to_display(np.asarray(r.get("corridor_n_lo", []), dtype=np.float64))

            n_hi = _corridor_aligned_to_display(np.asarray(r.get("corridor_n_hi", []), dtype=np.float64))

            k_lo = _corridor_aligned_to_display(np.asarray(r.get("corridor_k_lo", []), dtype=np.float64))

            k_hi = _corridor_aligned_to_display(np.asarray(r.get("corridor_k_hi", []), dtype=np.float64))

            if (

                n_lo.size == lam_s.size

                and n_hi.size == lam_s.size

                and k_lo.size == lam_s.size

                and k_hi.size == lam_s.size

            ):

                has_profile = True

                pen_n = pg.mkPen((0, 87, 255, 35), width=1)

                cu_n = pg.PlotCurveItem(lam_s, n_hi, pen=pen_n)

                cl_n = pg.PlotCurveItem(lam_s, n_lo, pen=pen_n)

                self.plot_n_corridor.addItem(cu_n)

                self.plot_n_corridor.addItem(cl_n)

                self.plot_n_corridor.addItem(

                    pg.FillBetweenItem(cl_n, cu_n, brush=pg.mkBrush(0, 87, 255, 35))

                )

                lk_lo = np.full(k_lo.shape, np.nan, dtype=np.float64)

                lk_hi = np.full(k_hi.shape, np.nan, dtype=np.float64)

                mlo = np.isfinite(k_lo) & (k_lo > 0.0)

                mhi = np.isfinite(k_hi) & (k_hi > 0.0)

                lk_lo[mlo] = np.log10(np.maximum(k_lo[mlo], 1e-30))

                lk_hi[mhi] = np.log10(np.maximum(k_hi[mhi], 1e-30))

                pen_k = pg.mkPen((255, 90, 0, 35), width=1)

                cu_k = pg.PlotCurveItem(lam_s, lk_hi, pen=pen_k)

                cl_k = pg.PlotCurveItem(lam_s, lk_lo, pen=pen_k)

                self.plot_lgk_corridor.addItem(cu_k)

                self.plot_lgk_corridor.addItem(cl_k)

                self.plot_lgk_corridor.addItem(

                    pg.FillBetweenItem(cl_k, cu_k, brush=pg.mkBrush(255, 90, 0, 35))

                )

            bn_lo = _corridor_aligned_to_display(np.asarray(r.get("boot_corridor_n_lo", []), dtype=np.float64))

            bn_hi = _corridor_aligned_to_display(np.asarray(r.get("boot_corridor_n_hi", []), dtype=np.float64))

            bk_lo = _corridor_aligned_to_display(np.asarray(r.get("boot_corridor_k_lo", []), dtype=np.float64))

            bk_hi = _corridor_aligned_to_display(np.asarray(r.get("boot_corridor_k_hi", []), dtype=np.float64))

            if (

                bn_lo.size == lam_s.size

                and bn_hi.size == lam_s.size

                and bk_lo.size == lam_s.size

                and bk_hi.size == lam_s.size

            ):

                has_boot = True

                pen_bn = pg.mkPen((0, 160, 80, 90), width=1, style=Qt.PenStyle.DashLine)

                cu_bn = pg.PlotCurveItem(lam_s, bn_hi, pen=pen_bn)

                cl_bn = pg.PlotCurveItem(lam_s, bn_lo, pen=pen_bn)

                self.plot_n_corridor.addItem(cu_bn)

                self.plot_n_corridor.addItem(cl_bn)

                self.plot_n_corridor.addItem(

                    pg.FillBetweenItem(cl_bn, cu_bn, brush=pg.mkBrush(0, 160, 80, 28))

                )

                blk_lo = np.full(bk_lo.shape, np.nan, dtype=np.float64)

                blk_hi = np.full(bk_hi.shape, np.nan, dtype=np.float64)

                mblo = np.isfinite(bk_lo) & (bk_lo > 0.0)

                mbhi = np.isfinite(bk_hi) & (bk_hi > 0.0)

                blk_lo[mblo] = np.log10(np.maximum(bk_lo[mblo], 1e-30))

                blk_hi[mbhi] = np.log10(np.maximum(bk_hi[mbhi], 1e-30))

                pen_bk = pg.mkPen((120, 0, 180, 90), width=1, style=Qt.PenStyle.DashLine)

                cu_bk = pg.PlotCurveItem(lam_s, blk_hi, pen=pen_bk)

                cl_bk = pg.PlotCurveItem(lam_s, blk_lo, pen=pen_bk)

                self.plot_lgk_corridor.addItem(cu_bk)

                self.plot_lgk_corridor.addItem(cl_bk)

                self.plot_lgk_corridor.addItem(

                    pg.FillBetweenItem(cl_bk, cu_bk, brush=pg.mkBrush(120, 0, 180, 28))

                )

        except Exception:

            logger.debug("Corridor bootstrap band plot failed", exc_info=True)

        # Nominal curves on top of fills (crosshair snaps to these).

        # Envelope k_lo/k_hi: linear k min/max; log₁₀ view can look asymmetric (see tab hint).

        lk = np.full(k_c.shape, np.nan, dtype=np.float64)

        mk = np.isfinite(k_c) & (k_c > 0.0)

        lk[mk] = np.log10(np.maximum(k_c[mk], 1e-30))

        self._add_curve(

            self.plot_n_corridor, lam_s, n_c, "#0057ff", "n", crosshair_primary=True

        )

        self._add_curve(

            self.plot_lgk_corridor, lam_s, lk, "#ff5a00", "log10 k", crosshair_primary=True

        )

        show_prof_alt = False

        if has_ref:

            dn = float(np.nanmax(np.abs(n_ref - n_c))) if n_c.size == n_ref.size else 0.0

            mk2 = np.isfinite(k_c) & np.isfinite(k_ref) & (k_c > 0.0) & (k_ref > 0.0)

            dlk = (

                float(np.nanmax(np.abs(np.log10(k_ref[mk2]) - np.log10(k_c[mk2]))))

                if np.any(mk2)

                else 0.0

            )

            if dn > 1e-5 or dlk > 0.02:

                show_prof_alt = True

                plot_widget_plot_finite(

                    self.plot_n_corridor,

                    lam_s,

                    n_ref,

                    pen=pg.mkPen("#88aacc", width=2, style=Qt.PenStyle.DashLine),

                    name="n (center-d refit)",

                )

                lk_r = np.full(k_ref.shape, np.nan, dtype=np.float64)

                mk_r = np.isfinite(k_ref) & (k_ref > 0.0)

                lk_r[mk_r] = np.log10(np.maximum(k_ref[mk_r], 1e-30))

                plot_widget_plot_finite(

                    self.plot_lgk_corridor,

                    lam_s,

                    lk_r,

                    pen=pg.mkPen("#cc8844", width=2, style=Qt.PenStyle.DashLine),

                    name="log10 k (center-d refit)",

                )

        d_nm = float(r.get("d_nm", float("nan")))

        d_txt = f"d = {d_nm:.1f} nm" if np.isfinite(d_nm) else "d = "

        ref_sub = (

            " - bold: exported n,k; dashed: center-d refit if it differs"

            if show_prof_alt

            else " - bold: same n,k as main tab"

        )

        ref_sub_k = (

            ref_sub

            + "; shade ⊇ reported k + refits (asymmetric in log₁₀)"

            if has_profile

            else ref_sub

        )

        try:

            self.plot_n_corridor.plotItem.setTitle(

                f"n(lambda) + corridors  {d_txt}{ref_sub}", color=CertusTheme.PRIMARY, size="10pt"

            )

            self.plot_lgk_corridor.plotItem.setTitle(

                f"log₁₀ k(lambda) + corridors  {d_txt}{ref_sub_k}", color=CertusTheme.PRIMARY, size="10pt"

            )

            if has_profile:

                self.plot_lgk_corridor.setToolTip(

                    "Bold orange: k from the result dict (same as main tab). "

                    "Shaded band: min/max linear k over accepted d-refits, enlarged so the bold curve stays inside. "

                    "Dashed orange (if shown): center-d refit when it differs from the bold line. "

                    "Crosshair y follows the bold curve at the cursor lambda when possible."

                )

            else:

                self.plot_lgk_corridor.setToolTip("")

        except (AttributeError, RuntimeError):

            logger.debug("Corridor plot title set failed", exc_info=True)

        self.plot_n_corridor.autoRange()

        self.plot_lgk_corridor.autoRange()

        if lam_s.size > 0:

            span_lo = float(np.nanmin(lam_s))

            span_hi = float(np.nanmax(lam_s))

            if np.isfinite(span_lo) and np.isfinite(span_hi) and span_hi > span_lo:

                pad = 0.02 * (span_hi - span_lo)

                self.plot_n_corridor.plotItem.setXRange(

                    span_lo - pad, span_hi + pad, padding=0.0

                )

                self.plot_lgk_corridor.plotItem.setXRange(

                    span_lo - pad, span_hi + pad, padding=0.0

                )

        if has_profile or has_boot:

            if hasattr(self, "tabs_main") and hasattr(self, "_idx_tab_corridor"):

                self.tabs_main.setCurrentIndex(int(self._idx_tab_corridor))

    def _plot_corridor_rmse_tab(self, r: dict) -> None:

        """Tab 'Corridor RMSE(d)': profile points + best sampled thickness marker."""

        if not hasattr(self, "plot_corridor_rmse_d"):

            return

        try:

            self.plot_corridor_rmse_d.clear()

        except (AttributeError, RuntimeError):

            return

        src = r

        d_prof = np.asarray(src.get("profile_d_values_nm", []), dtype=np.float64).ravel()

        rmse_prof = np.asarray(src.get("profile_d_rmse_values", []), dtype=np.float64).ravel()

        if (d_prof.size == 0 or rmse_prof.size != d_prof.size) and self._corridor_profile_source_result() is not None:

            src = self._corridor_profile_source_result() or r

            d_prof = np.asarray(src.get("profile_d_values_nm", []), dtype=np.float64).ravel()

            rmse_prof = np.asarray(src.get("profile_d_rmse_values", []), dtype=np.float64).ravel()

        if d_prof.size == 0 or rmse_prof.size != d_prof.size:

            if hasattr(self, "lbl_corridor_rmse_summary"):

                self.lbl_corridor_rmse_summary.setText("No corridor RMSE profile available for this run.")

            self._reset_corridor_manual_controls()

            return

        m = np.isfinite(d_prof) & np.isfinite(rmse_prof)

        d_prof = d_prof[m]

        rmse_prof = rmse_prof[m]

        if d_prof.size == 0:

            if hasattr(self, "lbl_corridor_rmse_summary"):

                self.lbl_corridor_rmse_summary.setText("No finite corridor RMSE profile points.")

            self._reset_corridor_manual_controls()

            return

        order = np.argsort(d_prof)

        d_s = d_prof[order]

        r_s = rmse_prof[order]

        self._corridor_rmse_d_vals = d_s.copy()

        self._corridor_rmse_vals = r_s.copy()

        self._add_curve(self.plot_corridor_rmse_d, d_s, r_s, CertusTheme.PRIMARY, "RMSE(d)")

        self.plot_corridor_rmse_d.addItem(

            pg.ScatterPlotItem(

                d_s,

                r_s,

                pen=pg.mkPen(0, 87, 255, 80),

                brush=pg.mkBrush(0, 87, 255, 80),

                size=6,

                symbol="o",

            )

        )

        i_best = int(np.argmin(r_s))

        self._corridor_rmse_best_idx = i_best

        d_best = float(d_s[i_best])

        rmse_best = float(r_s[i_best])

        self.plot_corridor_rmse_d.addItem(

            pg.InfiniteLine(

                pos=d_best,

                angle=90,

                movable=False,

                pen=pg.mkPen("#17a673", width=2, style=Qt.PenStyle.DashLine),

            )

        )

        self.plot_corridor_rmse_d.addItem(

            pg.ScatterPlotItem(

                [d_best],

                [rmse_best],

                pen=pg.mkPen("#0a5f42", width=1),

                brush=pg.mkBrush("#20c997"),

                size=10,

                symbol="o",

            )

        )

        rmse_thr = src.get("profile_d_rmse_thresh")

        if rmse_thr is not None and np.isfinite(float(rmse_thr)):

            thr = float(rmse_thr)

            self.plot_corridor_rmse_d.addItem(

                pg.InfiniteLine(

                    pos=thr,

                    angle=0,

                    movable=False,

                    pen=pg.mkPen(CertusTheme.DANGER, width=1, style=Qt.PenStyle.DashLine),

                )

            )

        delta_rb = (

            float(self.sp_corridor_rmse_delta.value())

            if hasattr(self, "sp_corridor_rmse_delta")

            else 2e-4

        )

        win_rb = (

            int(self.sp_corridor_rmse_win.value())

            if hasattr(self, "sp_corridor_rmse_win")

            else 3

        )

        rb_ok, d_lo_rb, d_hi_rb, slope_b, _curv_b = self._robust_interval_from_local_quadratic(

            d_s, r_s, i_best, delta_rb, win_rb

        )

        self._corridor_rmse_robust_ok = bool(rb_ok)

        self._corridor_rmse_robust_lo = float(d_lo_rb)

        self._corridor_rmse_robust_hi = float(d_hi_rb)

        if rb_ok:

            pen_rb = pg.mkPen("#7a3cff", width=1, style=Qt.PenStyle.DashLine)

            self.plot_corridor_rmse_d.addItem(

                pg.InfiniteLine(pos=float(d_lo_rb), angle=90, movable=False, pen=pen_rb)

            )

            self.plot_corridor_rmse_d.addItem(

                pg.InfiniteLine(pos=float(d_hi_rb), angle=90, movable=False, pen=pen_rb)

            )

        self._sync_corridor_manual_controls(d_s, i_best)

        d_lo_man = float(getattr(self, "_corridor_rmse_manual_lo", float("nan")))

        d_hi_man = float(getattr(self, "_corridor_rmse_manual_hi", float("nan")))

        if np.isfinite(d_lo_man) and np.isfinite(d_hi_man) and d_hi_man >= d_lo_man:

            pen_man = pg.mkPen("#ff4d4f", width=1, style=Qt.PenStyle.DashLine)

            self.plot_corridor_rmse_d.addItem(

                pg.InfiniteLine(pos=float(d_lo_man), angle=90, movable=False, pen=pen_man)

            )

            self.plot_corridor_rmse_d.addItem(

                pg.InfiniteLine(pos=float(d_hi_man), angle=90, movable=False, pen=pen_man)

            )

        if hasattr(self, "lbl_corridor_rmse_summary"):

            txt = (

                f"Best computed thickness: d* = {d_best:.3f} nm | RMSE(d*) = {rmse_best:.6f} | "

                f"samples = {int(d_s.size)}"

            )

            if rb_ok:

                txt += (

                    f" | robust Δ={delta_rb:.6f} -> interval ≈ [{float(d_lo_rb):.3f}, {float(d_hi_rb):.3f}] nm"

                    f" | slope@d*≈{float(slope_b):+.2e} /nm"

                )

            else:

                txt += " | robust interval unavailable (insufficient local convex fit)"

            if np.isfinite(d_lo_man) and np.isfinite(d_hi_man):

                man_state = "active" if bool(getattr(self, "_corridor_rmse_manual_active", False)) else "preview"

                txt += (

                    f" | manual {man_state} ≈ [{float(d_lo_man):.3f}, {float(d_hi_man):.3f}] nm"

                )

                if bool(src.get("manual_corridor_active", False)):

                    txt += f" ({int(src.get('manual_corridor_selected_count', 0))} profiled points)"

            self.lbl_corridor_rmse_summary.setText(txt)

        try:

            self.plot_corridor_rmse_d.plotItem.setTitle(

                f"Corridor profile RMSE(d) — best d* = {d_best:.3f} nm (RMSE {rmse_best:.6f})",

                color=CertusTheme.PRIMARY,

                size="10pt",

            )

        except (AttributeError, RuntimeError):

            logger.debug("Corridor RMSE(d) title set failed", exc_info=True)

        self.plot_corridor_rmse_d.autoRange()

    def _plot_result(self, r: dict) -> None:

        lam0 = np.asarray(r["lam_nm"], dtype=np.float64).ravel()

        tt0 = np.asarray(r["t_theo"], dtype=np.float64).ravel()

        n0 = np.asarray(r["n_lam"], dtype=np.float64).ravel()

        k0 = np.asarray(r["k_lam"], dtype=np.float64).ravel()

        lam_exp: np.ndarray | None = None

        if self.df is not None and "lambda" in self.df.columns:

            lam_exp = ensure_lam_nm_array(self.df["lambda"].to_numpy(dtype=np.float64))

        plot_r_model = (

            r.get("r_theo") is not None

            and self.df is not None

            and "R" in self.df.columns

            and lam_exp is not None

            and lam_exp.size > 0

        )

        if plot_r_model:

            rt0 = np.asarray(r["r_theo"], dtype=np.float64).ravel()

            lam_s, pack, order = _spectral_display_align(lam0, tt0, n0, k0, rt0)

            tt_s, n_s, k_s, rt_s = pack[0], pack[1], pack[2], pack[3]

        else:

            lam_s, pack, order = _spectral_display_align(lam0, tt0, n0, k0)

            tt_s, n_s, k_s = pack[0], pack[1], pack[2]

            rt_s = None

        x_mod, x_lbl = self._transform_spectrum_x(lam_s)

        self.plot_T.clear()

        self.plot_n.clear()

        self.plot_lgk.clear()

        x_exp: np.ndarray | None = None

        if lam_exp is not None and lam_exp.size:

            x_exp, _ = self._transform_spectrum_x(lam_exp)

        logger.debug("plot result: model_lambda=%d points d=%s", lam_s.size, r.get("d_nm"))

        if self.df is not None and "T" in self.df.columns and lam_exp is not None and lam_exp.size:

            ye_raw = _to_fraction_T(self.df["T"].to_numpy(dtype=np.float64))

            ye = ye_raw

            ne = "T/Tsub exp" if bool(r.get("t_is_ratio", False)) else "T exp"

            self._add_curve(self.plot_T, x_exp, ye, CertusTheme.TEXT_SUB, ne, True)

        nm = "T/Tsub model" if bool(r.get("t_is_ratio", False)) else "T model"

        self._add_curve(self.plot_T, x_mod, tt_s, CertusTheme.PRIMARY, nm)

        if plot_r_model and rt_s is not None:

            ye_raw = _to_fraction_T(self.df["R"].to_numpy(dtype=np.float64))

            ye = ye_raw

            r_ne = "R/Tsub exp" if bool(r.get("t_is_ratio", False)) else "R exp"

            self._add_curve(self.plot_T, x_exp, ye, "#888888", r_ne, True)

            r_nm = "R/Tsub model" if bool(r.get("t_is_ratio", False)) else "R model"

            self._add_curve(self.plot_T, x_mod, rt_s, CertusTheme.SECONDARY, r_nm)

        lk = np.full(k_s.shape, np.nan, dtype=np.float64)

        mk = np.isfinite(k_s) & (k_s >= 0.0)

        lk[mk] = np.log10(np.maximum(k_s[mk], 1e-30))

        self._add_curve(self.plot_n, lam_s, n_s, "#0057ff", "n", crosshair_primary=True)

        self._add_curve(self.plot_lgk, lam_s, lk, "#ff5a00", "log10 k", crosshair_primary=True)

        self._plot_corridor_tab(r, lam_s, n_s, k_s, spectral_sort_order=order)

        self._plot_corridor_rmse_tab(r)

        self._plot_nl_tab(r, lam_s, n_s, k_s)

        d_nm = float(r.get("d_nm", float("nan")))

        d_txt = f"d = {d_nm:.1f} nm" if np.isfinite(d_nm) else "d = "

        try:

            self.plot_n.plotItem.setTitle(f"n(lambda)  {d_txt}", color=CertusTheme.PRIMARY, size="10pt")

            self.plot_lgk.plotItem.setTitle(f"log10 k(lambda)  {d_txt}", color=CertusTheme.PRIMARY, size="10pt")

        except (AttributeError, RuntimeError):

            logger.debug("Index plot title set failed", exc_info=True)

        self.plot_T.autoRange()

        self._apply_spectrum_x_axis_label(x_lbl)

        self.plot_n.autoRange()

        self.plot_lgk.autoRange()

        if lam0.size > 0:

            span_lo = float(np.nanmin(lam0))

            span_hi = float(np.nanmax(lam0))

            if np.isfinite(span_lo) and np.isfinite(span_hi) and span_hi > span_lo:

                pad = 0.02 * (span_hi - span_lo)

                self.plot_n.plotItem.setXRange(span_lo - pad, span_hi + pad, padding=0.0)

                self.plot_lgk.plotItem.setXRange(span_lo - pad, span_hi + pad, padding=0.0)

        self._apply_spectrum_plot_title(r)

        self._update_rmse_fit_region_overlay()

        # Auto-refresh corridor RMSE profile window if open
        win = getattr(self, "_corridor_rmse_profile_win", None)
        if win is not None and win.isVisible():
            d_prof = np.asarray(r.get("profile_d_values_nm", []), dtype=np.float64)
            r_prof = np.asarray(r.get("profile_d_rmse_values", []), dtype=np.float64)
            if d_prof.size > 0 and r_prof.size == d_prof.size:
                rmse_thresh = r.get("profile_d_rmse_thresh")
                win.update_profile(d_prof, r_prof, rmse_thresh)


def main() -> None:

    import multiprocessing

    multiprocessing.freeze_support()

    setup_logging(log_file="certus_index_spline.log")

    if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):

        QApplication.setHighDpiScaleFactorRoundingPolicy(

            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough

        )

    app = QApplication(sys.argv)

    init_certus_app("CERTUS-INDEX-SPLINE", app=app)

    win = CertusIndexSplineApp()

    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":

    main()
