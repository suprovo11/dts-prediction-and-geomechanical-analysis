#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 research_pipeline.py
 Machine-Learning Synthetic Shear-Sonic (DTS) Prediction + Geomechanics
 Well 15/9-F-1 A  (Volve field)  —  reproducible end-to-end research pipeline
================================================================================

One command reproduces the whole study:

    python research_pipeline.py --input Research_Data.xlsx --outdir results

Stages (run all, or one at a time with --stage):
    data      load + clean + leak-free feature engineering        -> features.parquet
    train     benchmark models under depth-ordered blocked CV     -> metrics.csv, best_model.joblib
    geomech   propagate predicted DTS -> fracture gradient / MWW   -> geomech_*.csv
    figures   render all diagnostic figures                       -> *.png
    all       everything above, in order (default)

Design principles (why this is trustworthy for a paper):
  * DEPTH-ORDERED blocked validation — never random splits; adjacent depth
    samples are autocorrelated and would leak between train/test.
  * NAIVE EMPIRICAL BASELINES (DTC-only, DTC+GR) so every ML gain is measured
    against a physically motivated reference.
  * LEAK-FREE FEATURES — rolling stats are trailing (causal) only; target-derived
    columns (Poisson, Vp, Vs) are excluded from the feature set.
  * OUT-OF-SAMPLE geomechanics — model trained on shallow depths only; fracture
    gradient evaluated on an unseen deep block. Overburden uses measured density
    in both branches so measured-vs-predicted isolates the DTS effect.

Dependencies: numpy, pandas, scikit-learn, scipy, matplotlib, joblib, openpyxl,
pyarrow (for parquet). XGBoost / LightGBM are used automatically if installed,
else the pipeline falls back to sklearn's HistGradientBoostingRegressor.

Author: TrendinTools (Md. Momin Ali)   |   Target venue: ICERIE 2027
================================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

log = logging.getLogger("pipeline")


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    # --- I/O ---
    input: str = "Research_Data.xlsx"
    outdir: str = "results"
    sheet: object = 0
    header_row: int = 5            # 0-indexed row holding column names
    drop_first_data_row: bool = True   # the units row directly under the header

    # --- columns ---
    target: str = "DTS"
    depth_col: str = "MD"
    base_features: tuple = ("DTC", "GR", "DEN", "NEU",
                            "RES-SHT", "RES-MED", "RES-DEP", "Caliper")
    log_transform: tuple = ("RES-SHT", "RES-MED", "RES-DEP")
    drop_always: tuple = ("Hole Size",)
    leaky_cols: tuple = ("Poisson", "Vp (Measured)", "Vs (Measured)")

    # --- leak-free feature engineering ---
    add_rolling: bool = True
    rolling_logs: tuple = ("DTC", "GR", "DEN", "NEU")
    rolling_window: int = 7

    # --- depth-ordered evaluation ---
    n_cv_splits: int = 5
    holdout_frac: float = 0.20

    # --- geomechanics constants / assumptions ---
    slow_to_vel: float = 304.8      # DTx[us/ft] -> velocity[km/s]
    psi_ft_per_sg: float = 0.4335   # gradient of 1.0 s.g. fluid
    ppg_to_psi_ft: float = 0.05195  # 1 ppg == 0.05195 psi/ft
    obg_cap_grad: float = 0.90      # psi/ft, avg overburden gradient above top of log
    pp_grad: float = 0.465          # psi/ft, assumed hydrostatic pore pressure
    biot_alpha: float = 1.0

    def paths(self) -> dict:
        o = Path(self.outdir)
        return {
            "outdir": o,
            "features": o / "features.parquet",
            "metrics": o / "metrics.csv",
            "importance": o / "feature_importance.csv",
            "best_model": o / "best_model.joblib",
            "synth_dts": o / "synthetic_DTS_full_well.csv",
            "geomech": o / "geomech_fracture_gradient_holdout.csv",
            "geomech_summary": o / "geomech_summary.json",
            "summary": o / "run_summary.json",
            "manifest": o / "manifest.json",
        }


# =============================================================================
# STAGE 1 — DATA + LEAK-FREE FEATURES
# =============================================================================
def load_clean(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.input, sheet_name=cfg.sheet, header=cfg.header_row)
    if cfg.drop_first_data_row:
        df = df.drop(index=0).reset_index(drop=True)          # units row
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cfg.drop_always:
        df = df.drop(columns=c, errors="ignore")
    df = df.dropna(subset=[cfg.target, cfg.depth_col])
    df = df.sort_values(cfg.depth_col).reset_index(drop=True)
    log.info("loaded %d rows | depth %.1f–%.1f ft | cols=%s",
             len(df), df[cfg.depth_col].min(), df[cfg.depth_col].max(), list(df.columns))
    return df


def engineer_features(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    out, feats = df.copy(), []
    for c in cfg.base_features:
        if c not in out.columns:
            continue
        if c in cfg.log_transform:
            name = f"{c}_log"
            out[name] = np.log10(out[c].clip(lower=1e-3))
            feats.append(name)
        else:
            feats.append(c)
    if cfg.add_rolling:
        w = cfg.rolling_window
        for c in cfg.rolling_logs:
            if c in out.columns:
                name = f"{c}_rmean{w}"
                out[name] = out[c].rolling(window=w, min_periods=1).mean()  # trailing/causal
                feats.append(name)
    out = out.dropna(subset=feats + [cfg.target]).reset_index(drop=True)
    log.info("engineered %d features: %s", len(feats), feats)
    # keep raw DTC/DEN/MD for the geomech stage + a feature registry column
    out.attrs["features"] = feats
    return out, feats


def stage_data(cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    df = load_clean(cfg)
    df, feats = engineer_features(df, cfg)
    path = cfg.paths()["features"]
    try:
        df.to_parquet(path)
        # persist feature list alongside (parquet attrs are not always preserved)
        (cfg.paths()["outdir"] / "features_columns.json").write_text(json.dumps(feats))
    except Exception as e:
        log.warning("parquet save failed (%s); falling back to csv", e)
        df.to_csv(path.with_suffix(".csv"), index=False)
    log.info("data stage complete -> %s", path)
    return df, feats


def _load_features(cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    p = cfg.paths()["features"]
    if p.exists():
        df = pd.read_parquet(p)
    elif p.with_suffix(".csv").exists():
        df = pd.read_csv(p.with_suffix(".csv"))
    else:
        return stage_data(cfg)
    fcol = cfg.paths()["outdir"] / "features_columns.json"
    feats = json.loads(fcol.read_text()) if fcol.exists() else df.attrs.get("features", [])
    return df, feats


# =============================================================================
# SPLIT + MODEL ZOO
# =============================================================================
def shallow_deep_holdout(n: int, frac: float) -> tuple[np.ndarray, np.ndarray]:
    cut = int(round(n * (1 - frac)))
    return np.arange(0, cut), np.arange(cut, n)


def make_boosting():
    try:
        from xgboost import XGBRegressor
        return (XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=RANDOM_STATE, n_jobs=-1), "XGBoost")
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        return (LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=48,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, n_jobs=-1), "LightGBM")
    except Exception:
        pass
    from sklearn.ensemble import HistGradientBoostingRegressor
    return (HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                          l2_regularization=1.0,
                                          random_state=RANDOM_STATE),
            "HistGradientBoosting")


def build_models(feats: list[str]):
    scaled = lambda est: Pipeline([("scaler", StandardScaler()), ("est", est)])
    gbm, gbm_name = make_boosting()
    models = {
        "Baseline_DTC":    (scaled(LinearRegression()), ["DTC"]),
        "Baseline_DTC_GR": (scaled(LinearRegression()), ["DTC", "GR"]),
        "RandomForest":    (RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                                  n_jobs=-1, random_state=RANDOM_STATE), feats),
        gbm_name:          (gbm, feats),
        "MLP":             (scaled(MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                                alpha=1e-3, max_iter=600,
                                                early_stopping=True, n_iter_no_change=25,
                                                random_state=RANDOM_STATE)), feats),
    }
    return models, gbm_name


def score(y_true, y_pred) -> dict:
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred))}


def blocked_cv_rmse(pipe, X, y, n_splits: int) -> tuple[float, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for tr, te in tscv.split(X):
        p = clone(pipe); p.fit(X.iloc[tr], y.iloc[tr])
        rmses.append(np.sqrt(mean_squared_error(y.iloc[te], p.predict(X.iloc[te]))))
    return float(np.mean(rmses)), float(np.std(rmses))


# =============================================================================
# STAGE 2 — TRAIN + EVALUATE
# =============================================================================
def stage_train(cfg: Config, df: pd.DataFrame, feats: list[str]) -> dict:
    P = cfg.paths()
    X, y = df[feats].copy(), df[cfg.target].copy()
    depth = df[cfg.depth_col].values
    tr, te = shallow_deep_holdout(len(df), cfg.holdout_frac)
    log.info("split: train=%d (<=%.0f ft)  holdout=%d (>=%.0f ft)",
             len(tr), depth[tr[-1]], len(te), depth[te[0]])

    models, gbm_name = build_models(feats)
    rows, fitted = [], {}
    for name, (pipe, cols) in models.items():
        cv_m, cv_s = blocked_cv_rmse(pipe, X[cols].iloc[tr], y.iloc[tr], cfg.n_cv_splits)
        m = clone(pipe); m.fit(X[cols].iloc[tr], y.iloc[tr])
        pred = m.predict(X[cols].iloc[te])
        s = score(y.iloc[te].values, pred)
        rows.append({"model": name, "n_features": len(cols),
                     "cv_rmse_mean": cv_m, "cv_rmse_std": cv_s,
                     "holdout_RMSE": s["RMSE"], "holdout_MAE": s["MAE"],
                     "holdout_R2": s["R2"]})
        fitted[name] = (m, cols, pred)
        log.info("%-22s CV=%.3f±%.3f  holdout RMSE=%.3f  R2=%.4f",
                 name, cv_m, cv_s, s["RMSE"], s["R2"])

    results = pd.DataFrame(rows).sort_values("holdout_RMSE").reset_index(drop=True)
    results.to_csv(P["metrics"], index=False)

    ml = results[~results["model"].str.startswith("Baseline")]
    best_name = ml.iloc[0]["model"]
    best_model, best_cols, best_pred = fitted[best_name]
    log.info("best ML model: %s (holdout RMSE=%.3f)", best_name, ml.iloc[0]["holdout_RMSE"])

    # feature importance (permutation on the holdout — model-agnostic)
    perm = permutation_importance(best_model, X[best_cols].iloc[te],
                                  y.iloc[te].values, n_repeats=10,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    est = best_model.named_steps["est"] if hasattr(best_model, "named_steps") else best_model
    imp = est.feature_importances_ if hasattr(est, "feature_importances_") else [np.nan]*len(best_cols)
    pd.DataFrame({"feature": best_cols, "impurity": imp,
                  "permutation": perm.importances_mean}
                 ).sort_values("permutation", ascending=False).to_csv(P["importance"], index=False)

    # full-well synthetic DTS + saved model (retrained on all data)
    final = clone(dict(models)[best_name][0]); final.fit(X[best_cols], y)
    synth = df[[cfg.depth_col, cfg.target]].copy()
    synth["DTS_pred"] = final.predict(X[best_cols])
    synth.to_csv(P["synth_dts"], index=False)
    joblib.dump({"model": final, "features": best_cols, "config": asdict(cfg)}, P["best_model"])

    return {"results": results, "best_name": best_name, "best_cols": best_cols,
            "holdout_idx": te, "train_idx": tr, "best_pred_holdout": best_pred,
            "gbm_name": gbm_name, "perm": perm.importances_mean}


# =============================================================================
# STAGE 3 — GEOMECHANICS (predicted DTS -> fracture gradient / mud window)
# =============================================================================
def elastic_props(dtc, dts, den, k):
    dtc, dts, den = map(lambda a: np.asarray(a, float), (dtc, dts, den))
    vp, vs = k / dtc, k / dts
    vpvs = vp / vs
    nu = (vpvs**2 - 2.0) / (2.0 * (vpvs**2 - 1.0))
    g = den * vs**2
    return dict(Vp=vp, Vs=vs, VpVs=vpvs, Poisson=nu, G_GPa=g, E_GPa=2*g*(1+nu))


def overburden_psi(depth, den, cfg):
    depth, den = np.asarray(depth, float), np.asarray(den, float)
    sv = np.empty_like(depth)
    sv[0] = cfg.obg_cap_grad * depth[0]
    dz = np.diff(depth)
    sv[1:] = sv[0] + np.cumsum(cfg.psi_ft_per_sg * 0.5 * (den[1:] + den[:-1]) * dz)
    return sv


def stresses(depth, sv, nu, cfg):
    depth = np.asarray(depth, float)
    pp = cfg.pp_grad * depth
    obg_grad = sv / depth
    k = nu / (1.0 - nu)
    fg_grad = k * (obg_grad - cfg.pp_grad) + cfg.pp_grad
    sh = k * (sv - cfg.biot_alpha * pp) + cfg.biot_alpha * pp
    to_ppg = lambda g: g / cfg.ppg_to_psi_ft
    return pd.DataFrame({
        "Sv_psi": sv, "Pp_psi": pp, "Shmin_psi": sh,
        "OBG_ppg": to_ppg(obg_grad), "Pp_ppg": to_ppg(cfg.pp_grad*np.ones_like(depth)),
        "FG_ppg": to_ppg(fg_grad), "Shmin_ppg": to_ppg(sh/depth),
        "MudWindow_ppg": to_ppg(fg_grad) - to_ppg(cfg.pp_grad*np.ones_like(depth)),
    })


def stage_geomech(cfg: Config, df: pd.DataFrame, feats: list[str], trained: dict | None) -> dict:
    P = cfg.paths()
    # obtain out-of-sample DTS on the deep holdout
    if trained is None:
        bundle = joblib.load(P["best_model"])
        model, best_cols = bundle["model"], bundle["features"]
        tr, te = shallow_deep_holdout(len(df), cfg.holdout_frac)
        m = clone(model); m.fit(df[best_cols].iloc[tr], df[cfg.target].iloc[tr])
        dts_pred = m.predict(df[best_cols].iloc[te])
    else:
        te = trained["holdout_idx"]; dts_pred = trained["best_pred_holdout"]

    depth = df[cfg.depth_col].values
    den, dtc, dts_meas = df["DEN"].values, df["DTC"].values, df[cfg.target].values
    sv = overburden_psi(depth, den, cfg)

    ep_m = elastic_props(dtc[te], dts_meas[te], den[te], cfg.slow_to_vel)
    ep_p = elastic_props(dtc[te], dts_pred,     den[te], cfg.slow_to_vel)
    g_m = stresses(depth[te], sv[te], ep_m["Poisson"], cfg)
    g_p = stresses(depth[te], sv[te], ep_p["Poisson"], cfg)

    rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
    mae  = lambda a, b: float(mean_absolute_error(a, b))
    res = {
        "n_holdout": int(len(te)),
        "holdout_depth_ft": [float(depth[te].min()), float(depth[te].max())],
        "DTS_rmse_us_ft": rmse(dts_meas[te], dts_pred),
        "Poisson_rmse": rmse(ep_m["Poisson"], ep_p["Poisson"]),
        "E_GPa_rmse": rmse(ep_m["E_GPa"], ep_p["E_GPa"]),
        "FG_rmse_ppg": rmse(g_m["FG_ppg"], g_p["FG_ppg"]),
        "FG_mae_ppg": mae(g_m["FG_ppg"], g_p["FG_ppg"]),
        "FG_mean_measured_ppg": float(g_m["FG_ppg"].mean()),
        "mud_window_mean_ppg": float(g_m["MudWindow_ppg"].mean()),
        "OBG_mean_ppg": float(g_m["OBG_ppg"].mean()),
        "assumptions": {"obg_cap_grad": cfg.obg_cap_grad, "pp_grad": cfg.pp_grad,
                        "biot_alpha": cfg.biot_alpha, "TVD": "approx = MD"},
    }
    out = pd.DataFrame({"MD": depth[te], "DTS_meas": dts_meas[te], "DTS_pred": dts_pred,
                        "Poisson_meas": ep_m["Poisson"], "Poisson_pred": ep_p["Poisson"],
                        "E_GPa_meas": ep_m["E_GPa"], "E_GPa_pred": ep_p["E_GPa"],
                        "FG_meas_ppg": g_m["FG_ppg"], "FG_pred_ppg": g_p["FG_ppg"],
                        "Pp_ppg": g_m["Pp_ppg"], "OBG_ppg": g_m["OBG_ppg"],
                        "MudWindow_meas_ppg": g_m["MudWindow_ppg"]})
    out.to_csv(P["geomech"], index=False)
    P["geomech_summary"].write_text(json.dumps(res, indent=2))
    log.info("geomech: DTS RMSE=%.2f  Poisson RMSE=%.3f  FG RMSE=%.2f ppg",
             res["DTS_rmse_us_ft"], res["Poisson_rmse"], res["FG_rmse_ppg"])
    return {"summary": res, "table": out}


# =============================================================================
# STAGE 4 — FIGURES
# =============================================================================
def stage_figures(cfg: Config, df: pd.DataFrame, feats: list[str],
                  trained: dict, geo: dict):
    o = cfg.paths()["outdir"]
    target = cfg.target

    # correlation
    cols = feats + [target]; corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation matrix"); fig.tight_layout()
    fig.savefig(o / "01_correlation.png", dpi=130); plt.close(fig)

    te = trained["holdout_idx"]
    y_te = df[target].values[te]; pred = trained["best_pred_holdout"]
    depth_te = df[cfg.depth_col].values[te]; name = trained["best_name"]

    # pred vs actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_te, pred, s=6, alpha=0.35, edgecolors="none")
    lo, hi = min(y_te.min(), pred.min()), max(y_te.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="1:1")
    ax.set_xlabel("Measured DTS (µs/ft)"); ax.set_ylabel("Predicted DTS (µs/ft)")
    ax.set_title(f"{name}: predicted vs measured DTS"); ax.legend()
    fig.tight_layout(); fig.savefig(o / "02_pred_vs_actual.png", dpi=130); plt.close(fig)

    # depth track
    fig, ax = plt.subplots(figsize=(5, 11))
    ax.plot(y_te, depth_te, lw=0.8, color="black", label="Measured DTS")
    ax.plot(pred, depth_te, lw=0.8, color="crimson", alpha=0.8, label=f"Predicted ({name})")
    ax.invert_yaxis(); ax.set_xlabel("DTS (µs/ft)"); ax.set_ylabel("Depth MD (ft)")
    ax.set_title("Deep holdout interval"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(o / "03_depth_track.png", dpi=130); plt.close(fig)

    # permutation importance
    order = np.argsort(trained["perm"])
    fig, ax = plt.subplots(figsize=(8, 0.4*len(trained["best_cols"]) + 1.5))
    ax.barh(np.array(trained["best_cols"])[order], np.array(trained["perm"])[order],
            color="#2a6f97")
    ax.set_xlabel("Permutation importance"); ax.set_title(f"{name}: feature importance")
    fig.tight_layout(); fig.savefig(o / "04_importance.png", dpi=130); plt.close(fig)

    # mud-weight window
    g = geo["table"]
    fig, ax = plt.subplots(figsize=(6.2, 11))
    ax.fill_betweenx(g["MD"], g["Pp_ppg"], g["FG_meas_ppg"], color="#a8dadc",
                     alpha=0.5, label="Safe window (measured)")
    ax.plot(g["OBG_ppg"], g["MD"], color="#6c757d", lw=1.0, label="Overburden")
    ax.plot(g["Pp_ppg"], g["MD"], color="#1d3557", lw=1.2, label="Pore pressure")
    ax.plot(g["FG_meas_ppg"], g["MD"], color="black", lw=1.4, label="FG (measured DTS)")
    ax.plot(g["FG_pred_ppg"], g["MD"], color="crimson", lw=1.1, ls="--",
            label="FG (predicted DTS)")
    ax.invert_yaxis(); ax.set_xlabel("Equivalent mud weight (ppg)")
    ax.set_ylabel("Depth MD (ft)")
    ax.set_title("Mud-weight window & fracture gradient\n(out-of-sample)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(o / "05_mud_weight_window.png", dpi=130); plt.close(fig)

    log.info("figures written to %s", o)


# =============================================================================
# ORCHESTRATION
# =============================================================================
def run(cfg: Config, stage: str):
    P = cfg.paths(); P["outdir"].mkdir(parents=True, exist_ok=True)
    df = feats = trained = geo = None

    if stage in ("all", "data"):
        df, feats = stage_data(cfg)
    if stage in ("all", "train"):
        if df is None: df, feats = _load_features(cfg)
        trained = stage_train(cfg, df, feats)
    if stage in ("all", "geomech"):
        if df is None: df, feats = _load_features(cfg)
        geo = stage_geomech(cfg, df, feats, trained)
    if stage in ("all", "figures"):
        if df is None: df, feats = _load_features(cfg)
        if trained is None: trained = stage_train(cfg, df, feats)
        if geo is None: geo = stage_geomech(cfg, df, feats, trained)
        stage_figures(cfg, df, feats, trained, geo)

    if stage == "all":
        summary = {
            "n_samples": int(len(df)), "n_features": len(feats), "features": feats,
            "boosting_backend": trained["gbm_name"], "best_model": trained["best_name"],
            "holdout_frac": cfg.holdout_frac,
            "metrics": trained["results"].to_dict(orient="records"),
            "geomechanics": geo["summary"],
        }
        P["summary"].write_text(json.dumps(summary, indent=2))
        artifacts = sorted(str(p.name) for p in P["outdir"].glob("*")
                           if p.is_file())
        P["manifest"].write_text(json.dumps({"artifacts": artifacts}, indent=2))
        print("\n===================== LEADERBOARD (deep holdout) =====================")
        print(trained["results"].to_string(index=False, float_format="%.4f"))
        print(f"\nAll artifacts -> {P['outdir'].resolve()}")


def parse_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(description="DTS prediction + geomechanics research pipeline")
    ap.add_argument("--input", default="Research_Data.xlsx", help="input .xlsx well-log file")
    ap.add_argument("--outdir", default="results", help="output directory")
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "train", "geomech", "figures"])
    ap.add_argument("--holdout-frac", type=float, default=0.20)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--no-rolling", action="store_true", help="disable rolling features")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    return Config(input=a.input, outdir=a.outdir, holdout_frac=a.holdout_frac,
                  n_cv_splits=a.cv_splits, add_rolling=not a.no_rolling), a.stage


if __name__ == "__main__":
    cfg, stage = parse_args()
    run(cfg, stage)
