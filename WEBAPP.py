"""
Streamlit Web App — Darrieus VAWT cINN Blade Design Tool
=========================================================
Run locally:
    pip install streamlit
    streamlit run streamlit_app.py

Deploy free:
    1. Push this file + cinn_tsr_robust_v2.py + dmst_dataset.csv to GitHub
    2. Go to https://share.streamlit.io → "New app" → select your repo
    3. Main file: streamlit_app.py
    4. Done — you get a public URL for your QR code.

requirements.txt needed on deployment:
    numpy pandas torch scikit-learn matplotlib streamlit
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import all model/training functions from the main module
from triasl_user import (
    load_data, train_cinn, train_surrogate,
    generate_at_target, screen_robustness,
    cluster_shortlist, predict_cp, plot_results,
    NACA_MAP, NACA_MAP_INV,
    DATASET_PATH, EPOCHS_CINN, EPOCHS_SURROGATE,
    BATCH_SIZE, LR, TEST_SIZE, N_SAMPLES, Z_STD, N_BAND_PTS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VAWT Blade Designer",
    page_icon="🌀",
    layout="wide",
)

st.title("🌀 Darrieus VAWT — cINN Inverse Blade Design")
st.markdown("""
This tool uses a **Conditional Invertible Neural Network (cINN)** to find blade
geometries (chord, solidity, NACA airfoil, blade count) that achieve your
target power coefficient **Cp** at a chosen tip-speed ratio **TSR**, while
remaining robust across a TSR operating band.
""")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Design Inputs")

    st.subheader("Performance Target")
    Cp_TARGET  = st.slider("Target Cp",  0.10, 0.55, 0.35, 0.01,
                           help="Desired power coefficient. Betz limit = 0.593.")
    TSR_TARGET = st.slider("Target TSR", 1.0,  7.0,  4.0,  0.1,
                           help="Tip-Speed Ratio = ω·R / V_wind")

    st.subheader("Turbine Geometry")
    R_TARGET = st.slider("Rotor Radius R (m)", 0.2, 10.0, 1.0, 0.1)
    H_TARGET = st.slider("Rotor Height H (m)", 0.2, 20.0, 2.0, 0.1)

    st.subheader("Robustness Band")
    TSR_DELTA  = st.slider("TSR half-band Δ", 0.0, 3.0, 1.0, 0.1,
                           help="Band = [TSR ± Δ]. Set 0 to skip robustness check.")
    Cp_FLOOR   = st.slider("Cp floor",        0.05, float(Cp_TARGET)-0.01, 0.15, 0.01,
                           help="Minimum Cp required across the entire band.")
    Cp_DROPOFF = st.slider("Max Cp drop",     0.05, 0.45, 0.20, 0.01,
                           help="Max allowed fall in Cp across the band.")
    N_CLUSTERS = st.slider("Designs to return", 1, 10, 5,
                           help="Number of distinct shortlisted designs.")

    run = st.button("▶  Run Design", type="primary", use_container_width=True)

# ── Main area: explanation while waiting ─────────────────────────────────────
if not run:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**① Set your targets** in the left panel — Cp, TSR, radius, height.")
    with c2:
        st.info("**② Set the robustness band** — how much TSR variation to tolerate.")
    with c3:
        st.info("**③ Click Run Design** — results and plots appear here.")

    st.markdown("---")
    st.markdown("### 📖 Parameter guide")
    st.markdown("""
| Parameter | Symbol | Meaning | Typical range |
|-----------|--------|---------|---------------|
| Power Coefficient | Cp | Fraction of wind power extracted | 0.20 – 0.45 |
| Tip-Speed Ratio | TSR | Blade-tip speed / wind speed | 2 – 5 |
| Rotor Radius | R | Half the turbine diameter | 0.5 – 5 m |
| Rotor Height | H | Blade span | 1 – 10 m |
| Betz limit | — | Theoretical maximum Cp = 16/27 | 0.593 |
| Solidity | σ = N·c/R | Fraction of swept area covered by blades | 0.1 – 0.8 |
""")
    st.stop()

# ── Run pipeline ─────────────────────────────────────────────────────────────
# Inject globals so plot_results and screen_robustness can read Cp_FLOOR etc.
import triasl_user as _mod
_mod.Cp_FLOOR   = Cp_FLOOR
_mod.Cp_DROPOFF = Cp_DROPOFF

with st.spinner("Loading dataset …"):
    X_raw, Y_raw, df_full = load_data(DATASET_PATH)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_raw).astype(np.float32)
    Y_scaled = y_scaler.fit_transform(Y_raw).astype(np.float32)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=TEST_SIZE, random_state=42)

with st.spinner("Training cINN inverse model …"):
    cinn_model = train_cinn(X_train, Y_train,
                            X_scaled.shape[1], Y_scaled.shape[1])

with st.spinner("Training Cp surrogate …"):
    surrogate, sc_x_surr, sc_y_surr = train_surrogate(df_full)

with st.spinner(f"Generating candidates at TSR = {TSR_TARGET} …"):
    candidates = generate_at_target(
        cinn_model, x_scaler, y_scaler,
        Cp_TARGET, TSR_TARGET, R_TARGET, H_TARGET, n=N_SAMPLES)

if len(candidates) == 0:
    st.error("No candidates generated. Try relaxing your inputs.")
    st.stop()

with st.spinner("Screening robustness across TSR band …"):
    if TSR_DELTA == 0:
        survivors = candidates
        scores    = np.ones(len(candidates))
        tsr_band  = np.array([TSR_TARGET], dtype=np.float32)
        cp_mat    = predict_cp(surrogate, sc_x_surr, sc_y_surr,
                               candidates, tsr_band, R_TARGET, H_TARGET)
    else:
        survivors, scores, cp_mat, tsr_band = screen_robustness(
            candidates, surrogate, sc_x_surr, sc_y_surr,
            TSR_TARGET, TSR_DELTA, N_BAND_PTS,
            Cp_FLOOR, Cp_DROPOFF, R_TARGET, H_TARGET)

rep_df, rep_cps = cluster_shortlist(survivors, scores, cp_mat, N_CLUSTERS)

# ── Results table ─────────────────────────────────────────────────────────────
st.success(f"✅  Found {len(rep_df)} robust design(s).")

if len(rep_df) > 0:
    display_df = rep_df[["NACA_label", "num_blades", "chord",
                          "solidity", "robustness_score"]].copy()
    display_df.index = [f"R{i+1}" for i in range(len(display_df))]
    display_df.columns = ["NACA Profile", "Blades", "Chord c (m)",
                          "Solidity σ", "Robustness Score"]
    st.dataframe(display_df.style.format({
        "Chord c (m)": "{:.4f}",
        "Solidity σ":  "{:.4f}",
        "Robustness Score": "{:.4f}",
    }), use_container_width=True)

    csv_bytes = rep_df.to_csv(index=False).encode()
    st.download_button("⬇ Download results CSV", csv_bytes,
                       "robust_designs.csv", "text/csv")
else:
    st.warning("No designs survived. Try ↑ Max Cp drop  |  ↓ Cp floor  |  ↓ TSR Δ")

# ── Plot ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Design Plots")

fig = plot_results.__wrapped__ if hasattr(plot_results, "__wrapped__") else None

# Call plot_results but capture the figure instead of saving
# (We re-implement just the figure creation inline for Streamlit)
BETZ   = 16 / 27
colors = plt.cm.tab10(np.linspace(0, 0.8, max(len(rep_df), 1)))

fig = plt.figure(figsize=(20, 13))
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.55, wspace=0.38,
                        top=0.91, bottom=0.08, left=0.06, right=0.97)
fig.suptitle(
    f"Darrieus VAWT — cINN Inverse Design     "
    f"Cp={Cp_TARGET}  |  TSR={TSR_TARGET}  |  R={R_TARGET} m  |  H={H_TARGET} m  |  "
    f"Band TSR∈[{tsr_band[0]:.1f},{tsr_band[-1]:.1f}]",
    fontsize=12, fontweight="bold")

# Panel 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(candidates[:, 2], candidates[:, 0], s=14, alpha=0.25, color="#90CAF9",
            label=f"All candidates (n={len(candidates)})")
if len(survivors):
    ax1.scatter(survivors[:, 2], survivors[:, 0], s=22, alpha=0.65, color="#1565C0",
                label=f"Passed robustness (n={len(survivors)})")
if len(rep_df):
    ax1.scatter(rep_df["solidity"], rep_df["chord"], s=200, marker="*",
                color="#FF6F00", edgecolors="white", lw=0.8, zorder=5,
                label=f"Shortlisted (n={len(rep_df)})")
    for idx, row in rep_df.iterrows():
        ax1.annotate(f" R{idx+1}", (row["solidity"], row["chord"]),
                     fontsize=8, color="#BF360C", fontweight="bold", va="center")
ax1.set_xlabel("Solidity σ = N·c/R", fontsize=9)
ax1.set_ylabel("Chord c (m)", fontsize=9)
ax1.set_title("① Design Space Funnel", fontsize=10, fontweight="bold", pad=6)
ax1.legend(fontsize=8, loc="lower right", framealpha=0.85)
ax1.grid(True, alpha=0.3)

# Panel 2
ax2 = fig.add_subplot(gs[0, 1:])
ax2.axhspan(Cp_FLOOR, BETZ, alpha=0.06, color="green", label="Feasible zone")
target_idx = np.argmin(np.abs(tsr_band - TSR_TARGET))
if len(rep_df) and len(rep_cps):
    for i, (idx, row) in enumerate(rep_df.iterrows()):
        ax2.plot(tsr_band, rep_cps[i], color=colors[i], lw=2.2,
                 label=f"R{i+1}: {row['NACA_label']}, N={row['num_blades']}, "
                       f"c={row['chord']:.3f} m, σ={row['solidity']:.3f}")
        cp_val = rep_cps[i][target_idx]
        y_off  = 0.012 * (1 if i % 2 == 0 else -1)
        ax2.annotate(f"Cp={cp_val:.3f}", xy=(TSR_TARGET, cp_val),
                     xytext=(TSR_TARGET + 0.08, cp_val + y_off),
                     fontsize=7.5, color=colors[i],
                     arrowprops=dict(arrowstyle="-", color=colors[i], lw=0.6, alpha=0.6))
ax2.axvline(TSR_TARGET, color="red",    lw=1.8, ls="--", label=f"Design TSR={TSR_TARGET}")
ax2.axvline(tsr_band[0],  color="#777", lw=0.9, ls=":", label=f"Band edges")
ax2.axvline(tsr_band[-1], color="#777", lw=0.9, ls=":")
ax2.axhline(Cp_TARGET,  color="#2E7D32", lw=1.2, ls="--", label=f"Cp target={Cp_TARGET}")
ax2.axhline(BETZ,       color="#B71C1C", lw=1.2, ls="-.", label=f"Betz limit={BETZ:.3f}")
ax2.axhline(Cp_FLOOR,   color="#E65100", lw=0.9, ls=":", label=f"Cp floor={Cp_FLOOR}")
ax2.axvspan(tsr_band[0], tsr_band[-1], alpha=0.04, color="blue")
ax2.set_xlabel("Tip-Speed Ratio  TSR = ω·R / V_wind", fontsize=9)
ax2.set_ylabel("Power Coefficient  Cp = P / (½ρAV³)", fontsize=9)
ax2.set_title("② Cp vs TSR — Shortlisted Designs", fontsize=10, fontweight="bold", pad=6)
ax2.text(0.02, 0.03, "Cp clamped to Betz limit (16/27 ≈ 0.593)",
         transform=ax2.transAxes, fontsize=7.5, va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="#ccc"))
ax2.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=1)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, BETZ + 0.06)

# Panel 3
ax3 = fig.add_subplot(gs[1, 0])
if len(rep_df):
    x, w = np.arange(len(rep_df)), 0.35
    b1 = ax3.bar(x-w/2, rep_df["chord"],    w, color="#42A5F5", edgecolor="k", lw=0.4, label="Chord c (m)")
    b2 = ax3.bar(x+w/2, rep_df["solidity"], w, color="#FFA726", edgecolor="k", lw=0.4, label="Solidity σ")
    for bar in b1:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar in b2:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"R{i+1}" for i in range(len(rep_df))], fontsize=9)
    ax3.set_ylabel("Value", fontsize=9)
    ax3.set_title("③ Blade Geometry", fontsize=10, fontweight="bold", pad=6)
    ax3.legend(fontsize=8.5, loc="upper left")
    ax3.grid(True, alpha=0.3, axis="y")

# Panel 4
ax4 = fig.add_subplot(gs[1, 1])
if len(survivors):
    nc = pd.Series(survivors[:, 3].astype(int)).value_counts().sort_index()
    lbls = [NACA_MAP_INV.get(i, str(i)) for i in nc.index]
    bars4 = ax4.bar(lbls, nc.values, color="#66BB6A", edgecolor="k", lw=0.5)
    tot = nc.sum()
    for bar, val in zip(bars4, nc.values):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{val}  ({100*val/tot:.0f}%)", ha="center", va="bottom", fontsize=8)
    ax4.set_ylabel("Count (surviving designs)", fontsize=9)
    ax4.set_xlabel("NACA profile  (XX = thickness %)", fontsize=9)
    ax4.set_title("④ Airfoil Preference", fontsize=10, fontweight="bold", pad=6)
    ax4.grid(True, alpha=0.3, axis="y")
    plt.setp(ax4.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)

# Panel 5
ax5 = fig.add_subplot(gs[1, 2])
if len(rep_df):
    bars5 = ax5.barh([f"R{i+1}" for i in range(len(rep_df))],
                     rep_df["robustness_score"],
                     color=colors[:len(rep_df)], edgecolor="k", lw=0.5)
    for bar in bars5:
        ax5.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                 f"{bar.get_width():.3f}", va="center", fontsize=8.5)
    detail = ["Score = Cp(target TSR) − 0.5×norm.Cp-drop", ""]
    for i, (idx, row) in enumerate(rep_df.iterrows()):
        detail.append(f"R{i+1}: {row['NACA_label']}, N={row['num_blades']}, c={row['chord']:.3f} m")
    ax5.text(0.97, 0.03, "\n".join(detail), transform=ax5.transAxes,
             fontsize=7, va="bottom", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.88, ec="#ccc"))
    ax5.set_xlabel("Robustness Score", fontsize=9)
    ax5.set_title("⑤ Robustness Ranking", fontsize=10, fontweight="bold", pad=6)
    ax5.invert_yaxis()
    ax5.grid(True, alpha=0.3, axis="x")

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
buf.seek(0)
st.image(buf, use_column_width=True)
st.download_button("⬇ Download plot (PNG)", buf, "vawt_design_results.png", "image/png")

plt.close(fig)