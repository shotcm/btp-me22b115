"""
cINN Inverse Model — Input-Driven TSR-Robust Blade Design
==========================================================
Workflow
--------
  1.  User provides a TARGET TSR (e.g. 4.0) → model generates candidate
      blade designs optimised for that exact operating point.

  2.  Each candidate is forward-evaluated across a ROBUSTNESS BAND
      [TSR_target − delta, TSR_target + delta]  (default ±1)
      using a trained MLP forward surrogate  f(design, TSR) → Cp.

  3.  Candidates are scored on:
        • Cp_target  : predicted Cp at the target TSR          (primary)
        • Cp_floor   : minimum Cp across the robustness band   (must-pass)
        • Cp_dropoff : how much Cp degrades at the band edges  (lower = better)

  4.  Survivors are clustered → a compact shortlist of DISTINCT robust designs.

INPUT SECTION
-------------
  All user-facing parameters live in the  ── USER INPUT ──  block below.
  Hard-coded for now; will become a GUI later.

REQUIREMENTS
------------
    pip install numpy pandas torch scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =============================================================================
#  ── USER INPUT ──  (hard-coded; replace with GUI widgets later)
# =============================================================================

# Primary design target
Cp_TARGET  = None   # set at runtime by get_user_inputs()
TSR_TARGET = None
R_TARGET   = None
H_TARGET   = None
TSR_DELTA  = None
N_BAND_PTS = 9
Cp_FLOOR   = None
Cp_DROPOFF = None
N_CLUSTERS = 5

# =============================================================================
#  CONFIG  (model / training)
# =============================================================================

DATASET_PATH      = "dmst_dataset.csv"
EPOCHS_CINN       = 100
EPOCHS_SURROGATE  = 80
BATCH_SIZE        = 256
LR                = 1e-3
TEST_SIZE         = 0.2
N_SAMPLES         = 1500    # inverse samples at TSR_TARGET
Z_STD             = 0.3

NACA_MAP = {
    "NACA_0012": 0, "NACA_0015": 1, "NACA_0017": 2,
    "NACA_0018": 3, "NACA_0021": 4, "NACA_0030": 5,
}
NACA_MAP_INV = {v: k for k, v in NACA_MAP.items()}

# X → what user specifies (model condition)
INPUT_COLS  = ["Cp", "TSR", "radius", "height"]
# Y → blade design variables (model output)
OUTPUT_COLS = ["chord", "num_blades", "solidity", "NACA_index"]

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)
print(f"[Device] {DEVICE}")


# =============================================================================
#  DATA LOADING
# =============================================================================

def load_data(path):
    df = pd.read_csv(path)
    print(f"\n[Data] {len(df)} rows | columns: {list(df.columns)}")

    missing = [c for c in INPUT_COLS + OUTPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df["NACA_index"] = df["NACA_index"].replace(NACA_MAP)

    df = df[INPUT_COLS + OUTPUT_COLS].dropna()
    print(f"[Data] {len(df)} rows after dropna")

    X = df[INPUT_COLS].values.astype(np.float32)
    Y = df[OUTPUT_COLS].values.astype(np.float32)
    return X, Y, df


# =============================================================================
#  cINN  (identical architecture to original)
# =============================================================================

class AffineCoupling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        self.s = nn.Sequential(
            nn.Linear(half, 128), nn.ReLU(),
            nn.Linear(128, 128),  nn.ReLU(),
            nn.Linear(128, half), nn.Tanh()
        )
        self.t = nn.Sequential(
            nn.Linear(half, 128), nn.ReLU(),
            nn.Linear(128, 128),  nn.ReLU(),
            nn.Linear(128, half)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        s = self.s(x2);  t = self.t(x2)
        y1 = x1 * torch.exp(s) + t
        s = self.s(y1);  t = self.t(y1)
        y2 = x2 * torch.exp(s) + t
        return torch.cat([y1, y2], dim=1), s.sum(dim=1)

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        s = self.s(y1);  t = self.t(y1)
        x2 = (y2 - t) * torch.exp(-s)
        s = self.s(x2);  t = self.t(x2)
        x1 = (y1 - t) * torch.exp(-s)
        return torch.cat([x1, x2], dim=1)


class cINN(nn.Module):
    def __init__(self, cond_dim, design_dim, n_blocks=8):
        super().__init__()
        self.design_dim = design_dim
        self.blocks = nn.ModuleList(
            [AffineCoupling(design_dim * 2) for _ in range(n_blocks)]
        )
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),      nn.ReLU(),
            nn.Linear(128, design_dim)
        )

    def forward(self, y, x_cond):
        h = torch.cat([y, self.cond_net(x_cond)], dim=1)
        logdet = 0
        for block in self.blocks:
            h, ld = block(h); logdet = logdet + ld
        return h, logdet

    def inverse(self, x_cond, z_noise):
        h = torch.cat([z_noise, self.cond_net(x_cond)], dim=1)
        for block in reversed(self.blocks):
            h = block.inverse(h)
        return h[:, :self.design_dim]


# =============================================================================
#  FORWARD SURROGATE  ← NEW
#  f(chord, num_blades, solidity, NACA_index, TSR, radius, height) → Cp
#  Trained on the same dataset; used for robustness band evaluation.
# =============================================================================

class CpSurrogate(nn.Module):
    """
    Lightweight MLP: (design vars + TSR + R + H) → Cp
    Input dimension: 4 design + 3 context = 7
    """
    def __init__(self, in_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(),
            nn.Linear(256, 256),    nn.SiLU(),
            nn.Linear(256, 128),    nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_surrogate(df_full):
    """
    Build training data: input = [chord, num_blades, solidity, NACA_index,
                                   TSR, radius, height]
                         output = Cp
    """
    surr_in_cols  = ["chord", "num_blades", "solidity", "NACA_index",
                     "TSR", "radius", "height"]
    surr_out_col  = "Cp"

    X_s = df_full[surr_in_cols].values.astype(np.float32)
    y_s = df_full[surr_out_col].values.astype(np.float32)

    sc_x = StandardScaler(); sc_y = StandardScaler()
    X_s  = sc_x.fit_transform(X_s)
    y_s  = sc_y.fit_transform(y_s.reshape(-1, 1)).ravel()

    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_s, y_s, test_size=0.15, random_state=0
    )

    model = CpSurrogate(in_dim=X_s.shape[1]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_SURROGATE)

    print(f"\n[Surrogate] Training Cp surrogate for {EPOCHS_SURROGATE} epochs …")
    for epoch in range(EPOCHS_SURROGATE):
        model.train()
        perm = np.random.permutation(len(Xs_tr))
        total, nb = 0.0, 0
        for i in range(0, len(Xs_tr), BATCH_SIZE):
            xb = torch.tensor(Xs_tr[perm[i:i+BATCH_SIZE]], device=DEVICE)
            yb = torch.tensor(ys_tr[perm[i:i+BATCH_SIZE]], device=DEVICE)
            loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        sched.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            # validation
            model.eval()
            with torch.no_grad():
                xv = torch.tensor(Xs_te, device=DEVICE)
                yv = torch.tensor(ys_te, device=DEVICE)
                val_loss = nn.functional.mse_loss(model(xv), yv).item()
            print(f"  Epoch {epoch+1:3d}/{EPOCHS_SURROGATE}  "
                  f"train MSE={total/nb:.5f}  val MSE={val_loss:.5f}")

    print("[Surrogate] Done.")
    return model, sc_x, sc_y


def predict_cp(surrogate, sc_x_surr, sc_y_surr,
               designs_raw, tsr_values, R, H):
    """
    Evaluate Cp for every (design × TSR) combination.

    designs_raw : np.ndarray  (N, 4)  [chord, num_blades, solidity, NACA_index]
    tsr_values  : np.ndarray  (T,)

    Returns
    -------
    cp_matrix   : np.ndarray  (N, T)   predicted Cp
    """
    surrogate.eval()
    N, T = len(designs_raw), len(tsr_values)

    # Build input matrix: repeat each design T times, tile TSR N times
    design_rep  = np.repeat(designs_raw, T, axis=0)       # (N*T, 4)
    tsr_rep     = np.tile(tsr_values, N).reshape(-1, 1)   # (N*T, 1)
    R_col       = np.full((N * T, 1), R, dtype=np.float32)
    H_col       = np.full((N * T, 1), H, dtype=np.float32)

    X_surr = np.hstack([design_rep, tsr_rep, R_col, H_col]).astype(np.float32)
    X_surr = sc_x_surr.transform(X_surr)

    with torch.no_grad():
        Cp_scaled = surrogate(
            torch.tensor(X_surr, device=DEVICE)
        ).cpu().numpy()

    Cp_raw = sc_y_surr.inverse_transform(Cp_scaled.reshape(-1, 1)).ravel()
    # Clamp to physically valid range: Cp ∈ [0, Betz limit = 16/27 ≈ 0.593]
    Cp_raw = np.clip(Cp_raw, 0.0, 16 / 27)
    return Cp_raw.reshape(N, T)


# =============================================================================
#  cINN TRAINING
# =============================================================================

def train_cinn(X_train, Y_train, cond_dim, design_dim):
    model = cINN(cond_dim, design_dim, n_blocks=8).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_CINN)
    prior = torch.distributions.Normal(0, 1)

    print(f"\n[cINN] Training for {EPOCHS_CINN} epochs …")
    for epoch in range(EPOCHS_CINN):
        model.train()
        perm = np.random.permutation(len(X_train))
        Xs, Ys = X_train[perm], Y_train[perm]
        total, nb = 0.0, 0
        for i in range(0, len(Xs), BATCH_SIZE):
            xb = torch.tensor(Xs[i:i+BATCH_SIZE], device=DEVICE)
            yb = torch.tensor(Ys[i:i+BATCH_SIZE], device=DEVICE)
            h, logdet = model(yb, xb)
            loss = -(prior.log_prob(h).sum(1) + logdet).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item(); nb += 1
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{EPOCHS_CINN}  "
                  f"NLL={total/nb:.4f}  lr={sched.get_last_lr()[0]:.2e}")
    return model


# =============================================================================
#  INVERSE GENERATION AT TARGET TSR
# =============================================================================

def generate_at_target(cinn_model, x_scaler, y_scaler,
                       Cp_target, TSR_target, R, H, n=N_SAMPLES):
    """
    Run inverse pass at exactly TSR_target.

    Returns raw (unscaled) design candidates: (N_valid, 4)
    """
    cinn_model.eval()

    query_raw    = np.array([[Cp_target, TSR_target, R, H]], dtype=np.float32)
    query_scaled = x_scaler.transform(query_raw)
    x_tensor     = torch.tensor(query_scaled).repeat(n, 1).to(DEVICE)
    z_noise      = torch.randn(n, cinn_model.design_dim, device=DEVICE) * Z_STD

    with torch.no_grad():
        y_scaled = cinn_model.inverse(x_tensor, z_noise).cpu().numpy()

    y_raw = y_scaler.inverse_transform(y_scaled)

    # Snap discrete columns
    y_raw[:, 3] = np.clip(np.round(y_raw[:, 3]), 0, len(NACA_MAP) - 1)
    valid_blades = np.array([2, 3, 4])
    y_raw[:, 1] = valid_blades[
        np.argmin(np.abs(y_raw[:, 1, None] - valid_blades), axis=1)
    ]

    # Physical validity filter
    mask = (
        (y_raw[:, 0] > 0.03) & (y_raw[:, 0] < 0.35) &
        (y_raw[:, 2] > 0.0)  & (y_raw[:, 2] < 1.0)  &
        (np.abs(y_raw[:, 2] - y_raw[:, 1] * y_raw[:, 0] / R) < 0.15)
    )
    valid = y_raw[mask]
    print(f"\n[Generate] {mask.sum()} / {n} candidates passed physical filters "
          f"(at TSR={TSR_target})")
    return valid


# =============================================================================
#  ROBUSTNESS SCREENING  ← CORE NEW LOGIC
# =============================================================================

def screen_robustness(candidates_raw,
                      surrogate, sc_x_surr, sc_y_surr,
                      TSR_target, TSR_delta, n_band_pts,
                      Cp_floor, Cp_dropoff_limit,
                      R, H):
    """
    For each candidate design evaluate Cp across the TSR band and apply
    robustness criteria.

    Criteria
    --------
    1. Cp_floor   : Cp(TSR) >= Cp_floor   for ALL band points
    2. Cp_dropoff : max(Cp_band) - min(Cp_band) <= Cp_dropoff_limit

    Returns
    -------
    survivors    : np.ndarray (M, 4)  — designs that passed
    scores       : np.ndarray (M,)    — robustness score (higher = better)
    cp_matrix    : np.ndarray (M, T)  — Cp at each band TSR (for plotting)
    tsr_band     : np.ndarray (T,)    — band TSR values
    """
    tsr_band = np.linspace(
        TSR_target - TSR_delta,
        TSR_target + TSR_delta,
        n_band_pts,
        dtype=np.float32
    )

    print(f"\n[Robustness] Evaluating {len(candidates_raw)} candidates across "
          f"TSR band [{tsr_band[0]:.2f}, {tsr_band[-1]:.2f}] "
          f"({n_band_pts} points) …")

    cp_mat = predict_cp(surrogate, sc_x_surr, sc_y_surr,
                        candidates_raw, tsr_band, R, H)  # (N, T)

    # ── Criterion 1: floor check
    floor_pass   = (cp_mat >= Cp_floor).all(axis=1)
    # ── Criterion 2: dropoff check
    dropoff      = cp_mat.max(axis=1) - cp_mat.min(axis=1)
    dropoff_pass = dropoff <= Cp_dropoff_limit

    mask = floor_pass & dropoff_pass

    survivors  = candidates_raw[mask]
    cp_mat_s   = cp_mat[mask]
    dropoff_s  = dropoff[mask]

    # Robustness score: high Cp at target + low dropoff (normalised)
    target_idx  = np.argmin(np.abs(tsr_band - TSR_target))
    Cp_at_tgt   = cp_mat_s[:, target_idx]
    score       = Cp_at_tgt - 0.5 * (dropoff_s / (Cp_dropoff_limit + 1e-8))

    print(f"[Robustness] {mask.sum()} / {len(candidates_raw)} passed all criteria")
    print(f"             floor ≥ {Cp_floor}  |  "
          f"dropoff ≤ {Cp_dropoff_limit}  |  "
          f"dropoff stats: mean={dropoff.mean():.3f}, "
          f"min={dropoff.min():.3f}, max={dropoff.max():.3f}")

    return survivors, score, cp_mat_s, tsr_band


# =============================================================================
#  CLUSTERING → SHORTLIST
# =============================================================================

def cluster_shortlist(survivors, scores, cp_matrix, n_clusters=N_CLUSTERS):
    """
    KMeans on (chord, solidity) → pick best-scoring member per cluster.
    """
    if len(survivors) == 0:
        return pd.DataFrame(), np.empty((0, cp_matrix.shape[1]))

    k = min(n_clusters, len(survivors))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(survivors[:, [0, 2]])   # chord, solidity

    rep_rows, rep_cps = [], []
    for c in range(k):
        idx  = np.where(labels == c)[0]
        best = idx[np.argmax(scores[idx])]
        rep_rows.append(list(survivors[best]) + [scores[best]])
        rep_cps.append(cp_matrix[best])

    df = pd.DataFrame(rep_rows, columns=OUTPUT_COLS + ["robustness_score"])
    df["NACA_label"] = df["NACA_index"].astype(int).map(NACA_MAP_INV)
    df["num_blades"] = df["num_blades"].astype(int)
    df["NACA_index"] = df["NACA_index"].astype(int)
    df = df.sort_values("robustness_score", ascending=False).reset_index(drop=True)

    return df, np.array(rep_cps)


# =============================================================================
#  PLOTTING
# =============================================================================

def plot_results(candidates_raw, survivors, rep_df, rep_cps,
                 tsr_band, TSR_target, Cp_target,
                 R, H):
    """
    Five-panel figure — clean layout, no overlapping text.

    Layout:
        Row 0:  [Panel 1 — design scatter]  [Panel 2 — Cp vs TSR  (2 cols)]
        Row 1:  [Panel 3 — blade geo bar]   [Panel 4 — NACA freq] [Panel 5 — robustness rank]

    Text discipline:
        • Panel titles: one short line only — no subtitle in title.
        • Descriptions go as ax.text() annotations INSIDE the plot area
          where there is empty white space, not above it.
        • Axis labels: one line, concise, with units in parentheses.
        • Legend entries: short — no newlines inside a label.
    """
    BETZ   = 16 / 27
    colors = plt.cm.tab10(np.linspace(0, 0.8, max(len(rep_df), 1)))

    # Tall figure so rows have breathing room
    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.55,   # vertical gap between rows
        wspace=0.38,   # horizontal gap between columns
        top=0.91, bottom=0.08, left=0.06, right=0.97
    )

    # ── Global title (one line) ───────────────────────────────────────────
    fig.suptitle(
        "Darrieus VAWT — cINN Inverse Design + TSR-Robustness Screening     "
        f"Cp={Cp_target}  |  TSR={TSR_target}  |  R={R} m  |  H={H} m  |  "
        f"Band TSR∈[{tsr_band[0]:.1f},{tsr_band[-1]:.1f}]",
        fontsize=12, fontweight="bold"
    )

    # =========================================================================
    # PANEL 1 — Design space scatter
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.scatter(candidates_raw[:, 2], candidates_raw[:, 0],
                s=14, alpha=0.25, color="#90CAF9",
                label=f"All candidates  (n={len(candidates_raw)})")
    if len(survivors) > 0:
        ax1.scatter(survivors[:, 2], survivors[:, 0],
                    s=22, alpha=0.65, color="#1565C0",
                    label=f"Passed robustness  (n={len(survivors)})")
    if len(rep_df) > 0:
        ax1.scatter(rep_df["solidity"], rep_df["chord"],
                    s=200, marker="*", color="#FF6F00",
                    edgecolors="white", linewidths=0.8, zorder=5,
                    label=f"Shortlisted  (n={len(rep_df)})")
        for idx, row in rep_df.iterrows():
            ax1.annotate(
                f" R{idx+1}",
                (row["solidity"], row["chord"]),
                fontsize=8, color="#BF360C", fontweight="bold", va="center"
            )

    ax1.set_xlabel("Solidity  σ = N·c/R", fontsize=9)
    ax1.set_ylabel("Chord  c  (m)", fontsize=9)
    ax1.set_title("① Design Space Funnel", fontsize=10, fontweight="bold", pad=6)

    # Description as in-plot text (top-right empty corner)
    ax1.text(0.97, 0.97,
             "cINN inverse model → physics filter\n→ robustness screen → KMeans cluster",
             transform=ax1.transAxes, fontsize=7.5,
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="#ccc"))

    ax1.legend(fontsize=8, loc="lower right", framealpha=0.85, edgecolor="#ccc")
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # PANEL 2 — Cp vs TSR  [spans columns 1 & 2]
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1:])

    ax2.axhspan(Cp_FLOOR, BETZ, alpha=0.06, color="green", label="Feasible zone")

    target_idx = np.argmin(np.abs(tsr_band - TSR_target))

    if len(rep_df) > 0 and len(rep_cps) > 0:
        for i, (idx, row) in enumerate(rep_df.iterrows()):
            lbl = (f"R{i+1}: {row['NACA_label']},  N={row['num_blades']},  "
                   f"c={row['chord']:.3f} m,  σ={row['solidity']:.3f}")
            ax2.plot(tsr_band, rep_cps[i], color=colors[i], lw=2.2, label=lbl)

            # Annotate Cp at target TSR — offset alternating up/down to avoid pile-up
            cp_val  = rep_cps[i][target_idx]
            y_off   = 0.012 * (1 if i % 2 == 0 else -1)
            ax2.annotate(
                f"Cp={cp_val:.3f}",
                xy=(TSR_target, cp_val),
                xytext=(TSR_target + 0.08, cp_val + y_off),
                fontsize=7.5, color=colors[i],
                arrowprops=dict(arrowstyle="-", color=colors[i],
                                lw=0.6, alpha=0.6)
            )

    # Reference lines — minimal labels
    ax2.axvline(TSR_target, color="red",     lw=1.8, ls="--",
                label=f"Design TSR = {TSR_target}")
    ax2.axvline(tsr_band[0],  color="#777",  lw=0.9, ls=":",
                label=f"Band edges ({tsr_band[0]:.1f} / {tsr_band[-1]:.1f})")
    ax2.axvline(tsr_band[-1], color="#777",  lw=0.9, ls=":")
    ax2.axhline(Cp_target,  color="#2E7D32", lw=1.2, ls="--",
                label=f"Cp target = {Cp_target}")
    ax2.axhline(BETZ,       color="#B71C1C", lw=1.2, ls="-.",
                label=f"Betz limit = {BETZ:.3f}")
    ax2.axhline(Cp_FLOOR,   color="#E65100", lw=0.9, ls=":",
                label=f"Cp floor = {Cp_FLOOR}")

    ax2.axvspan(tsr_band[0], tsr_band[-1], alpha=0.04, color="blue")

    ax2.set_xlabel("Tip-Speed Ratio  TSR = ω·R / V_wind", fontsize=9)
    ax2.set_ylabel("Power Coefficient  Cp = P / (½ρAV³)", fontsize=9)
    ax2.set_title("② Cp vs TSR — Shortlisted Designs", fontsize=10,
                  fontweight="bold", pad=6)

    # Surrogate note inside plot (top-left)
    ax2.text(0.02, 0.03,
             "Cp predicted by MLP surrogate · clamped to Betz limit (16/27 ≈ 0.593)",
             transform=ax2.transAxes, fontsize=7.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="#ccc"))

    ax2.legend(fontsize=8, loc="upper right", framealpha=0.9,
               edgecolor="#ccc", ncol=1)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, BETZ + 0.06)

    # =========================================================================
    # PANEL 3 — Chord & Solidity bars
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    if len(rep_df) > 0:
        x     = np.arange(len(rep_df))
        width = 0.35
        b1 = ax3.bar(x - width/2, rep_df["chord"],    width,
                     color="#42A5F5", edgecolor="k", lw=0.4, label="Chord c (m)")
        b2 = ax3.bar(x + width/2, rep_df["solidity"], width,
                     color="#FFA726", edgecolor="k", lw=0.4, label="Solidity σ")

        for bar in b1:
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.004,
                     f"{bar.get_height():.3f}",
                     ha="center", va="bottom", fontsize=7.5)
        for bar in b2:
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.004,
                     f"{bar.get_height():.3f}",
                     ha="center", va="bottom", fontsize=7.5)

        # Clean x-tick labels — just R1/R2 etc; NACA shown in legend of panel 5
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"R{i+1}" for i in range(len(rep_df))], fontsize=9)
        ax3.set_ylabel("Value", fontsize=9)
        ax3.set_title("③ Blade Geometry", fontsize=10, fontweight="bold", pad=6)

        ax3.text(0.98, 0.97,
                 "σ = N·c/R  (solidity)\nc = blade chord width",
                 transform=ax3.transAxes, fontsize=7.5,
                 va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="#ccc"))

        ax3.legend(fontsize=8.5, loc="upper left")
        ax3.grid(True, alpha=0.3, axis="y")

    # =========================================================================
    # PANEL 4 — NACA profile frequency
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    if len(survivors) > 0:
        naca_counts = (pd.Series(survivors[:, 3].astype(int))
                       .value_counts().sort_index())
        labels_nc   = [NACA_MAP_INV.get(i, str(i)) for i in naca_counts.index]
        bars_nc     = ax4.bar(labels_nc, naca_counts.values,
                              color="#66BB6A", edgecolor="k", lw=0.5)

        total_surv = naca_counts.sum()
        for bar, val in zip(bars_nc, naca_counts.values):
            ax4.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f"{val}  ({100*val/total_surv:.0f}%)",
                     ha="center", va="bottom", fontsize=8)

        ax4.set_ylabel("Count  (surviving designs)", fontsize=9)
        ax4.set_xlabel("NACA profile  (XX = thickness %)", fontsize=9)
        ax4.set_title("④ Airfoil Preference", fontsize=10, fontweight="bold", pad=6)
        ax4.grid(True, alpha=0.3, axis="y")
        plt.setp(ax4.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)

        ax4.text(0.98, 0.97,
                 "Among all robustness-screened designs",
                 transform=ax4.transAxes, fontsize=7.5,
                 va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="#ccc"))

    # =========================================================================
    # PANEL 5 — Robustness ranking
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    if len(rep_df) > 0:
        # Short y-axis labels: just R1, R2 … — details in annotation box
        short_labels = [f"R{i+1}" for i in range(len(rep_df))]
        bars_r = ax5.barh(short_labels, rep_df["robustness_score"],
                          color=colors[:len(rep_df)], edgecolor="k", lw=0.5)

        for bar in bars_r:
            xval = bar.get_width()
            ax5.text(xval + 0.003,
                     bar.get_y() + bar.get_height()/2,
                     f"{xval:.3f}",
                     va="center", fontsize=8.5)

        ax5.set_xlabel("Robustness Score", fontsize=9)
        ax5.set_title("⑤ Robustness Ranking", fontsize=10, fontweight="bold", pad=6)
        ax5.invert_yaxis()
        ax5.grid(True, alpha=0.3, axis="x")

        # Score formula + design legend in a text box inside the plot
        detail_lines = ["Score = Cp(target TSR) − 0.5×norm.Cp-drop", ""]
        for i, (idx, row) in enumerate(rep_df.iterrows()):
            detail_lines.append(
                f"R{i+1}: {row['NACA_label']},  N={row['num_blades']},  "
                f"c={row['chord']:.3f} m"
            )
        ax5.text(0.97, 0.03,
                 "\n".join(detail_lines),
                 transform=ax5.transAxes, fontsize=7,
                 va="bottom", ha="right",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.88, ec="#ccc"))

    plt.savefig("cinn_robust_results.png", dpi=160, bbox_inches="tight")
    print("[Plot] Saved → cinn_robust_results.png")
    plt.show(block=False)


# =============================================================================
#  TERMINAL UI
# =============================================================================

def _ask(prompt, default, cast, validate=None, hint=""):
    """
    Print a single prompt line, show the default, read input.
    Re-asks on invalid values. Returns the cast value.
    """
    while True:
        raw = input(f"  {prompt} [{default}]{':' if not hint else '  (' + hint + ')'} "
                    ).strip()
        val = raw if raw else str(default)
        try:
            result = cast(val)
        except ValueError:
            print(f"    ✗  Please enter a valid {cast.__name__}.\n")
            continue
        if validate and not validate(result):
            print(f"    ✗  Value out of allowed range.\n")
            continue
        return result


def _section(title):
    W = 58
    print()
    print("┌" + "─" * W + "┐")
    print("│" + f"  {title}".ljust(W) + "│")
    print("└" + "─" * W + "┘")


def get_user_inputs():
    """
    Guided terminal session. Returns a dict of all design parameters.
    """
    W = 60
    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  Darrieus VAWT — cINN Inverse Blade Design Tool".center(W) + "║")
    print("║" + "  TSR-Robust Design via Conditional Invertible NN ".center(W) + "║")
    print("╚" + "═" * W + "╝")
    print("""
  This tool finds blade geometries (chord, solidity, NACA
  airfoil, blade count) that achieve your target Cp at a
  chosen TSR, while staying reasonable across a ±TSR band.

  Press ENTER to accept the default shown in [brackets].
""")

    # ── Section 1: Performance target ──────────────────────────────────
    _section("1 / 3  —  Performance Target")
    print("""
  Cp   = Power Coefficient = P / (½ρAV³)
         Fraction of wind energy the turbine extracts.
         Betz limit (theoretical max) = 0.593.
         Typical real VAWTs: 0.20 – 0.45.

  TSR  = Tip-Speed Ratio = ω·R / V_wind
         Blade-tip speed divided by wind speed.
         Higher TSR → faster rotation relative to wind.
         Typical range: 2 – 5.
""")
    Cp_t  = _ask("Target Cp",  0.35, float,
                 lambda v: 0.05 < v < 0.593,
                 "0.05 – 0.593")
    TSR_t = _ask("Target TSR", 4.0,  float,
                 lambda v: 1.0 < v < 8.0,
                 "1.0 – 8.0")

    # ── Section 2: Turbine geometry ─────────────────────────────────────
    _section("2 / 3  —  Turbine Geometry")
    print("""
  R    = Rotor radius [m]   — half the turbine diameter.
  H    = Rotor height [m]   — blade span from top to bottom.
""")
    R_t = _ask("Rotor radius R (m)", 1.0, float,
               lambda v: 0.1 < v < 20.0, "0.1 – 20 m")
    H_t = _ask("Rotor height H (m)", 2.0, float,
               lambda v: 0.1 < v < 50.0, "0.1 – 50 m")

    # ── Section 3: Robustness band ──────────────────────────────────────
    _section("3 / 3  —  Robustness Band")
    print(f"""
  Wind speed varies throughout the day, so the actual
  operating TSR drifts around your target.

  TSR band = [target − Δ, target + Δ]
  With target={TSR_t} and Δ=1 → band = [{TSR_t-1:.1f}, {TSR_t+1:.1f}]

  Cp floor   = minimum acceptable Cp anywhere in the band.
  Cp drop    = max allowed Cp drop from peak within the band.
               (lower = stricter flatness requirement)
""")
    delta   = _ask("TSR half-band Δ",   1.0,  float,
                   lambda v: 0.0 <= v <= 3.0,  "0 to skip robustness check")
    cp_fl   = _ask("Cp floor",          0.15,  float,
                   lambda v: 0.0 < v < Cp_t,   f"0 – {Cp_t:.2f}")
    cp_drop = _ask("Max Cp drop",       0.20,  float,
                   lambda v: 0.0 < v < 0.5,    "0.05 – 0.50")
    n_res   = _ask("Designs to return", 5,     int,
                   lambda v: 1 <= v <= 10,      "1 – 10")

    # ── Confirmation summary ─────────────────────────────────────────────
    print()
    print("┌" + "─" * 58 + "┐")
    print("│  YOUR DESIGN SPECIFICATION".ljust(59) + "│")
    print("├" + "─" * 58 + "┤")
    rows = [
        ("Target Cp",         f"{Cp_t}"),
        ("Target TSR",        f"{TSR_t}"),
        ("Rotor radius R",    f"{R_t} m"),
        ("Rotor height H",    f"{H_t} m"),
        ("Robustness band",   f"TSR ∈ [{TSR_t-delta:.2f}, {TSR_t+delta:.2f}]"
                               if delta > 0 else "disabled (Δ=0)"),
        ("Cp floor",          f"{cp_fl}"),
        ("Max Cp drop",       f"{cp_drop}"),
        ("Designs returned",  f"{n_res}"),
    ]
    for label, val in rows:
        print(f"│  {label:<22}  {val:<32}│")
    print("└" + "─" * 58 + "┘")
    print()

    confirm = input("  Looks good? Press ENTER to run, or 'r' to re-enter:  ").strip()
    if confirm.lower() == "r":
        return get_user_inputs()   # recursive restart

    return dict(
        Cp_TARGET  = Cp_t,
        TSR_TARGET = TSR_t,
        R_TARGET   = R_t,
        H_TARGET   = H_t,
        TSR_DELTA  = delta,
        Cp_FLOOR   = cp_fl,
        Cp_DROPOFF = cp_drop,
        N_CLUSTERS = n_res,
    )


def print_results_table(rep_df, TSR_TARGET, Cp_TARGET, tsr_band):
    """Pretty-print the shortlist to the terminal."""
    W = 74
    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  SHORTLISTED TSR-ROBUST DESIGNS".center(W) + "║")
    print("╠" + "═" * W + "╣")
    hdr = (f"  {'#':<4} {'NACA':<12} {'Blades':>6}  {'Chord c':>8}  "
           f"{'Solidity σ':>10}  {'Score':>7}")
    print("║" + hdr.ljust(W) + "║")
    print("╠" + "─" * W + "╣")

    if len(rep_df) == 0:
        print("║" + "  No designs survived.".ljust(W) + "║")
        print("║" + "  Try: ↑ Max Cp drop  |  ↓ Cp floor  |  ↓ TSR Δ".ljust(W) + "║")
    else:
        for i, row in rep_df.iterrows():
            line = (f"  R{i+1:<3} {row['NACA_label']:<12} "
                    f"{row['num_blades']:>6}  "
                    f"{row['chord']:>7.4f} m  "
                    f"{row['solidity']:>10.4f}  "
                    f"{row['robustness_score']:>7.4f}")
            print("║" + line.ljust(W) + "║")

    print("╠" + "─" * W + "╣")
    print("║" + f"  Target: Cp={Cp_TARGET}  TSR={TSR_TARGET}  "
                f"Band TSR∈[{tsr_band[0]:.1f},{tsr_band[-1]:.1f}]".ljust(W) + "║")
    print("║" + "  Score = Cp(target TSR) − 0.5 × normalised Cp-drop"
                "  (higher = better)".ljust(W) + "║")
    print("╚" + "═" * W + "╝")
    print()


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":

    # ── 1. Collect user inputs ────────────────────────────────────────────
    params = get_user_inputs()

    # Unpack into module-level globals so plot helpers can read them
    Cp_TARGET  = params["Cp_TARGET"]
    TSR_TARGET = params["TSR_TARGET"]
    R_TARGET   = params["R_TARGET"]
    H_TARGET   = params["H_TARGET"]
    TSR_DELTA  = params["TSR_DELTA"]
    Cp_FLOOR   = params["Cp_FLOOR"]
    Cp_DROPOFF = params["Cp_DROPOFF"]
    N_CLUSTERS = params["N_CLUSTERS"]

    # ── 2. Load & scale data ──────────────────────────────────────────────
    print("\n  Loading dataset …")
    X_raw, Y_raw, df_full = load_data(DATASET_PATH)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_raw).astype(np.float32)
    Y_scaled = y_scaler.fit_transform(Y_raw).astype(np.float32)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=TEST_SIZE, random_state=42
    )

    # ── 3. Train models ───────────────────────────────────────────────────
    print("\n  Training cINN inverse model …")
    cond_dim   = X_scaled.shape[1]
    design_dim = Y_scaled.shape[1]
    cinn_model = train_cinn(X_train, Y_train, cond_dim, design_dim)

    print("\n  Training Cp surrogate (forward model) …")
    surrogate, sc_x_surr, sc_y_surr = train_surrogate(df_full)

    # ── 4. Generate candidates at target TSR ─────────────────────────────
    print(f"\n  Generating candidates at TSR = {TSR_TARGET} …")
    candidates = generate_at_target(
        cinn_model, x_scaler, y_scaler,
        Cp_TARGET, TSR_TARGET, R_TARGET, H_TARGET,
        n=N_SAMPLES
    )

    if len(candidates) == 0:
        print("\n  ✗  No candidates generated.")
        print("     Check that DATASET_PATH is correct and covers your input range.")
        exit(1)

    # ── 5. Robustness screening ───────────────────────────────────────────
    if TSR_DELTA == 0:
        print("\n  Robustness band disabled (Δ=0) — skipping screen.")
        survivors = candidates
        scores    = np.ones(len(candidates))
        tsr_band  = np.array([TSR_TARGET], dtype=np.float32)
        cp_mat    = predict_cp(surrogate, sc_x_surr, sc_y_surr,
                               candidates, tsr_band, R_TARGET, H_TARGET)
    else:
        survivors, scores, cp_mat, tsr_band = screen_robustness(
            candidates, surrogate, sc_x_surr, sc_y_surr,
            TSR_TARGET, TSR_DELTA, N_BAND_PTS,
            Cp_FLOOR, Cp_DROPOFF,
            R_TARGET, H_TARGET
        )

    # ── 6. Cluster → shortlist ────────────────────────────────────────────
    rep_df, rep_cps = cluster_shortlist(survivors, scores, cp_mat, N_CLUSTERS)

    # ── 7. Print results ──────────────────────────────────────────────────
    print_results_table(rep_df, TSR_TARGET, Cp_TARGET, tsr_band)

    if len(rep_df) > 0:
        rep_df.to_csv("robust_designs.csv", index=False)
        print("  Results saved → robust_designs.csv")

    # ── 8. Plot ───────────────────────────────────────────────────────────
    print("  Generating plots …")
    plot_results(candidates, survivors, rep_df, rep_cps,
                 tsr_band, TSR_TARGET, Cp_TARGET,
                 R_TARGET, H_TARGET)

    input("\n  Press ENTER to exit … ")
    plt.close("all")
