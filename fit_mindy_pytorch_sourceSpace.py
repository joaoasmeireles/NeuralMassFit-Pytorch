"""
MINDy-BPKF Source-Space — GPU Accelerated — v2 (anti-hub / anti-idiosyncrasy)
================================================================================
Integrated source-space version, with the sensor-space v7 hardening PORTED and
the technical-report recommendations APPLIED (stronger in source space, where the
BEM inversion has already undone part of the volume conduction).

CHANGES vs previous version (source-space "v1"):

  TRACK A — ported from sensor-space v7 (low risk, already validated):
    A1. Saves allW / allS + region metadata  ........... unlocks interpretability
    A2. State warmup before collecting loss  ........... removes x=0 initial noise
    A3. Learnable R (log-param), fixed Q  .............. channel for measurement noise
    A4. Parameter groups (weight_decay=0)  ............. does not push S to linear regime
    A5. Divergence handling + row cap + initial-P fix (corrected no-op)

  TRACK B — report recommendations:
    B1. Anatomical mask + distance decay on W_EE  ...... anti-hub lever #1
    B2. Per-channel WHITENED loss (diagonal Mahalanobis)  ... prevents Volt^2 capture
    B3. Sparsity (L1/group-lasso) + Laplacian smoothness  ... L2 only diffuses, not sparsifies
    B4. Longer free-sim (N_REC_STEPS 3 -> 8)  ......... constrains the GENERATOR
    B5. Dynamic features at the OPERATING POINT + D_PR  ... state, not W magnitude
    B6. N_ITERS 10000 -> 3500  ......................... implicit regularization

Source-space notes (vs sensors):
  - Volume conduction is already partially undone by the upstream BEM inversion, so
    hubs here arise from ill-posedness + loss capture. The right levers are B1/B2/B3
    (the auxiliary-wPLI trick from sensor space is NOT needed here).
  - The anatomical mask uses the Desikan MNI coords (DESIKAN_MNI / _get_region_coords
    from visualizacao_cerebral.py). If unavailable, the pipeline runs in dense mode
    (legacy behavior) with a warning — plug in the coords to enable B1.

Requirements:
  pip install torch numpy scipy matplotlib h5py

Usage:
  python fit_mindy_pytorch_sourceSpace.py --subject all --device cuda
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import io as sio
from scipy.signal import welch
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis as sp_kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

if sys.platform == "linux":
    BASE_DIR = Path("YOUR PATH")
else:
    BASE_DIR = Path(r"YOUR PATH")
DATA_ROOT = BASE_DIR / "MINDy_source"
OUT_ROOT  = BASE_DIR / "results_source"

SESSIONS = ['ses-S1', 'ses-S2', 'ses-S3']
CONDITIONS = ['0-Back', '1-Back', '2-Back']
COND_FILES = ['zeroBACK', 'oneBACK', 'twoBACK']

WINDOW_SEC = 30
OVERLAP = 0.5
SRATE = 250
WIN_SAMP = WINDOW_SEC * SRATE
STEP_SAMP = int(WIN_SAMP * (1 - OVERLAP))

# Training
N_ITERS = 3500         # B6: 10000 -> 3500 (implicit anti-overfit regularization)
LR = 2e-4              # C-stab: 5e-4 -> 2e-4 (smaller steps; kills loss oscillation)
GRAD_CLIP_NORM = 1.0   # C-estab: 5.0 -> 1.0 (corta picos de gradiente)
N_STACKS = 100         # C-stab: 50 -> 100 (less noisy gradient per iteration)
STACK_SHIFT = 5        # spacing between segments

# Horizonte temporal por segmento (A2: warmup adicionado)
N_WARMUP_STEPS = 8     # A2: KF sem loss, deixa x convergir
N_KF_STEPS = 15        # Kalman filtering steps (com loss)
N_REC_STEPS = 8        # B4: 3 -> 8 (free-running prediction; restringe o gerador)
HUBER_DELTA = 2.0

# Fixed parameters
FIXED_D = 0.80         # C-estab: 0.85 -> 0.80 (mais margem de estabilidade)
FIXED_C = 0.0
FIXED_V = 0.0
INIT_S = 2.5           # B-regime: starts in nonlinear regime
INIT_W_SCALE = 0.05

# A4 + nonlinear regime: S clamp [S_MIN, S_MAX]. S_MIN=2.0 keeps W*tanh(Sx)
# longe do regime quase-linear onde o W perde significado E/I.
S_MIN = 2.0
S_MAX = 8.0

# Noise covariances
Q_DIAG = 0.2           # process noise (FIXO, buffer)
INIT_R_DIAG = 0.2      # initial measurement noise (A3: now LEARNABLE)
LOG_R_MIN = -3.0       # R >= 0.05
LOG_R_MAX = 0.5        # R <= 1.65 (tight, avoids saturation -> late NaN)

# Kalman smoothing
DEC_P = 0.9

# Numerical jitter
JITTER = 1e-6

# -- B1: anatomical mask (distance) ------------------------------------------
TAU_MM    = 40.0       # escala do kernel de proximidade exp(-dist/TAU_MM)
KEEP_FRAC = 0.25       # fraction of admissible long-range E->E connections (~Singh)
APPLY_MASK_EI = False  # apply mask also to the E->I block (long-range)

# -- B3: structural regularization of W --------------------------------------
LAMBDA_L1     = 1e-4   # esparsidade (L1 / group-lasso) sobre o bloco EE
LAMBDA_SMOOTH = 1e-4   # suavidade Laplaciana espacial sobre o bloco EE

# -- C-stab: dynamic-instability penalty -------------------------------------
# sigma_max(J) of J = W*diag(S) + diag(D) at x~=0 (sech^2=1 -> max gain = worst case),
# medido EXATO via shift espectral + power iteration com v destacado (estilo
# spectral-norm). The penalty is a SOFT HINGE (guide, not wall): it allows the
# operating point to be marginally off-equilibrium (oscillatory attractor a la
# Singh, specR may sit ~40% above sigma_max for non-normal matrices), without
# engessar o erro de dados.
LAMBDA_STAB = 0.3      # guide, not wall (was 1.0 -> stalled the data error)
SPEC_TARGET = 1.0      # permite marginalidade (era 0.95); suba p/ 1.1 se sufocar
K_POWER     = 20       # 8 -> 20 (+ shift): sigma_max ~exact, removes numerical doubt

# DIRECT control of free-simulation growth -- kills loss spikes (13.5)
# at the origin, without forbidding healthy oscillation (penalizes only growth > GROWTH_FACTOR x).
LAMBDA_GROWTH = 1.0
GROWTH_FACTOR = 2.5    # tolerates free trajectory up to 2.5x the post-filtering amplitude

# -- A5: soft cap on W row-norm ----------------------------------------------
W_ROW_NORM_MAX = 1.0   # C-estab: 5.0 -> 1.0 (o teto de 5 era decorativo: max|W|~0.25)
DIVERGENCE_MAX_X = 1e6

# Validation
PLOT_SEC = 3

FEATURE_NAMES = [
    'eiRatio', 'meanWEE', 'meanWEI',
    'meanBetaIE', 'meanBetaII',
    'meanD_exc', 'meanD_inh', 'meanS_exc', 'meanS_inh',
    'stdWEE', 'stdWEI', 'normWEE', 'normWEI', 'maxAbsW',
    'meanAbsBetaIE', 'meanAbsBetaII', 'localDistalRat',
    'strongEE', 'strongEI', 'kurtWEE',
    # B5: dynamic features (Jacobian at the operating point) + manifold
    'specRadius', 'domEigReal', 'domEigImag', 'nUnstable',
    'eiNormRatio', 'specRadius_EE', 'dimPR',
]


# ============================================================================
# B1: ANATOMICAL PRIOR (admissibility mask + distance decay)
# ============================================================================

def build_distance_prior(region_coords, n_exc, n_inh,
                         tau_mm=TAU_MM, keep_frac=KEEP_FRAC):
    """
    Builds an anatomical prior for the E->E block from the Desikan MNI coords.

    region_coords : (n_regions, 3) MNI coordinates of the regions.
    keep_frac     : fraction of admissible long-range connections (sparsity ~Singh).

    Returns (at the POPULATION level, already kron-expanded, dtype float32):
      adm_EE  : (n_e, n_e) binary mask of admissible E->E (0/1)
      prox_EE : (n_e, n_e) prior de proximidade em [0,1]
      L_pop   : (n_e, n_e) Laplaciano do grafo de proximidade (p/ suavidade)
    """
    Rd = cdist(region_coords, region_coords).astype(np.float32)   # (n_reg, n_reg) mm
    prox = np.exp(-Rd / tau_mm).astype(np.float32)
    np.fill_diagonal(prox, 0.0)

    off = prox[~np.eye(len(prox), dtype=bool)]
    thr = np.quantile(off, 1.0 - keep_frac)                       # keep top keep_frac
    adm = (prox >= thr).astype(np.float32)

    # Laplacian L = Diag(degree) - A, using the admissible weighted adjacency
    A = (prox * adm).astype(np.float32)
    Lap = (np.diag(A.sum(axis=1)) - A).astype(np.float32)

    ones_ee = np.ones((n_exc, n_exc), dtype=np.float32)
    adm_EE  = np.kron(ones_ee, adm)
    prox_EE = np.kron(ones_ee, prox)
    L_pop   = np.kron(np.eye(n_exc, dtype=np.float32), Lap)        # smoothness intra-pop
    return adm_EE, prox_EE, L_pop


def resolve_region_coords(D, n_regions):
    """
    Attempts to obtain MNI region coords for the anatomical mask.
      1) D['region_coords'] no .mat (se o forward model salvou).
      2) DESIKAN_MNI / _get_region_coords de visualizacao_cerebral.py.
      3) None -> pipeline roda em modo denso (com aviso).
    """
    if n_regions is None:
        return None
    # 1) coords in the .mat itself
    if 'region_coords' in D:
        coords = np.asarray(D['region_coords'], dtype=float)
        if coords.shape[0] == n_regions and coords.shape[1] == 3:
            return coords
    # 2) Desikan catalog from the visualization module
    try:
        from visualizacao_cerebral import _get_region_coords, DESIKAN_MNI
        dk_names = ([f'{r}-lh' for r in DESIKAN_MNI] +
                    [f'{r}-rh' for r in DESIKAN_MNI])
        coords = np.asarray(_get_region_coords(dk_names), dtype=float)
        if coords.shape[0] >= n_regions:
            return coords[:n_regions]
    except Exception as e:
        warnings.warn(f"resolve_region_coords: Desikan catalog unavailable ({e}). "
                      f"Anatomical mask disabled (dense mode).", RuntimeWarning)
    return None


# ============================================================================
# MINDy MODEL
# ============================================================================

class MINDyModel(nn.Module):
    """
    MINDy neural mass model: x_{t+1} = W*tanh(S*x + V) + D*x + C + eps,  eps~N(0,Q)
    Measurement:             y      = H·x + η,                        η~N(0,R)

    Learnable: W (connectivity), S (sigmoid gain), log_r_diag (R)
    Fixos:       D, C, V, Q, H (leadfield BEM)
    """

    def __init__(self, n_ch, n_exc, n_inh, device='cpu', n_regions=None,
                 anat_adm=None, prox_EE=None, L_pop=None):
        super().__init__()
        self.n_ch = n_ch
        self.n_exc = n_exc
        self.n_inh = n_inh
        # Source-space: n_pop = n_regions*(n_exc+n_inh); Sensor-space: n_ch*(...)
        if n_regions is not None:
            self.n_e = n_regions * n_exc
            self.n_i = n_regions * n_inh
        else:
            self.n_e = n_ch * n_exc
            self.n_i = n_ch * n_inh
        self.n_pop = self.n_e + self.n_i
        self.device = device

        nE, nI = self.n_e, self.n_i

        # --- Learnable: connectivity ---
        W_init = torch.randn(self.n_pop, self.n_pop, device=device) * INIT_W_SCALE
        self.W = nn.Parameter(W_init)

        # --- Learnable: sigmoid gain ---
        S_init = torch.full((self.n_pop,), INIT_S, device=device)
        S_init += torch.randn(self.n_pop, device=device) * 0.02
        self.S = nn.Parameter(S_init)

        # --- A3: Learnable measurement noise (log-diagonal) ---
        self.log_r_diag = nn.Parameter(
            torch.full((n_ch,), float(np.log(INIT_R_DIAG)), device=device)
        )

        # --- Fixed buffers (D, C, V, Q) ---
        self.register_buffer('D', torch.full((self.n_pop,), FIXED_D, device=device))
        self.register_buffer('C', torch.full((self.n_pop,), FIXED_C, device=device))
        self.register_buffer('V', torch.full((self.n_pop,), FIXED_V, device=device))
        self.register_buffer('Q', torch.eye(self.n_pop, device=device) * Q_DIAG)
        self.register_buffer('rtQ', torch.eye(self.n_pop, device=device) * (Q_DIAG ** 0.5))

        # --- Masks (sign structure) ---
        Wmask = torch.zeros(self.n_pop, self.n_pop, device=device)
        # EE: full off-diagonal, positive
        Wmask[:nE, :nE] = 1.0
        Wmask[:nE, :nE] -= torch.eye(nE, device=device)
        # EI: full, positive
        Wmask[nE:, :nE] = 1.0
        # IE: diagonal, negative
        Wmask[:nE, nE:] = -torch.eye(nE, nI, device=device)
        # II: diagonal, negative
        Wmask[nE:, nE:] = -torch.eye(nI, device=device)

        # --- B1: apply anatomical admissibility to the EE block (and opt. EI) ---
        if anat_adm is not None:
            adm = torch.from_numpy(anat_adm).to(device)
            Wmask[:nE, :nE] = Wmask[:nE, :nE] * adm
            if APPLY_MASK_EI:
                Wmask[nE:, :nE] = Wmask[nE:, :nE] * adm

        # Derive the sign buffers AFTER the anatomical mask
        self.register_buffer('Wmask', Wmask)
        self.register_buffer('W_sign_pos', (Wmask > 0).float())
        self.register_buffer('W_sign_neg', (Wmask < 0).float())
        self.register_buffer('W_zero', (Wmask == 0).float())

        # --- B3: Laplaciano para suavidade (zeros se sem coords) ---
        if L_pop is not None:
            self.register_buffer('L_pop', torch.from_numpy(L_pop).to(device))
        else:
            self.register_buffer('L_pop', torch.zeros(nE, nE, device=device))

        # --- Measurement matrix H (placeholder; sobrescrito pelo BEM) ---
        H = torch.zeros(n_ch, self.n_pop, device=device)
        for i in range(n_ch):
            for e in range(n_exc):
                if i + e * n_ch < self.n_pop:
                    H[i, i + e * n_ch] = 1.0
        self.register_buffer('H', H)

        # --- Kalman prior + reusables ---
        self.register_buffer('Pfix', torch.eye(self.n_pop, device=device))
        self.register_buffer('I_pop', torch.eye(self.n_pop, device=device))
        self.register_buffer('I_ch', torch.eye(n_ch, device=device))

        self.project_signs()

    # --- Computed noise matrices ---
    @property
    def R(self):
        """Measurement noise covariance (n_ch × n_ch, diagonal). Learnable (A3)."""
        return torch.diag(torch.exp(self.log_r_diag))

    @property
    def R_diag(self):
        return torch.exp(self.log_r_diag)

    def project_signs(self):
        """Enforce constraints after each optimizer step."""
        with torch.no_grad():
            self.W.data *= (1.0 - self.W_zero)
            pos_mask = self.W_sign_pos.bool()
            self.W.data[pos_mask] = torch.clamp(self.W.data[pos_mask], min=0)
            neg_mask = self.W_sign_neg.bool()
            self.W.data[neg_mask] = torch.clamp(self.W.data[neg_mask], max=0)
            self.S.data.clamp_(S_MIN, S_MAX)
            self.log_r_diag.data.clamp_(LOG_R_MIN, LOG_R_MAX)


# ============================================================================
# BPKF TRAINING LOOP  (monolithic compile + warmup + whitened loss + reg)
# ============================================================================

def bpkf_train_window(model, data, n_iters=N_ITERS, lr=LR, device='cpu'):
    """
    BPKF training. Retorna: (model, losses_cpu, success).
    """
    n_ch, n_time = data.shape
    n_pop = model.n_pop
    nE = model.n_e
    B = N_STACKS

    H = model.H
    H_T = H.T
    I_pop = model.I_pop
    I_ch_jit = model.I_ch * JITTER
    L_pop = model.L_pop

    total_steps = N_WARMUP_STEPS + N_KF_STEPS + N_REC_STEPS
    seg_len = total_steps * STACK_SHIFT
    max_start = max(1, n_time - seg_len)

    # A4: parameter groups -- weight_decay=0 (B3 handles W regularization)
    optimizer = torch.optim.NAdam([
        {'params': [model.W],          'weight_decay': 0.0},
        {'params': [model.S],          'weight_decay': 0.0},
        {'params': [model.log_r_diag], 'weight_decay': 0.0},
    ], lr=lr, betas=(0.95, 0.99))

    losses = []
    nan_count = 0
    success = True

    offsets = torch.arange(total_steps, device=device) * STACK_SHIFT

    # B2: per-channel whitening scale (1/std), precomputed outside compile
    ch_scale = (1.0 / (data.std(dim=1) + 1e-3)).to(device)   # (n_ch,)

    # C-stab: fixed seed vector for sigma_max power iteration (avoids RNG in compile)
    stab_v0 = torch.randn(n_pop, device=device)
    stab_v0 = stab_v0 / (stab_v0.norm() + 1e-8)

    @torch.compile(mode="reduce-overhead", dynamic=False)
    def train_step(starts):
        idx_grid = starts.unsqueeze(1) + offsets.unsqueeze(0)
        idx_grid = idx_grid.clamp(0, n_time - 1)
        batch_data = data[:, idx_grid].permute(1, 2, 0)   # (B, total_steps, n_ch)

        x = torch.zeros(B, n_pop, device=device)
        P = model.Pfix.clone()                            # A5: P inicial correto
        Q = model.Q
        R = model.R                                       # A3: property (grad flui)
        total_loss = torch.zeros((), device=device)

        # --- Phase 1: warmup (sem loss) ---
        for k in range(N_WARMUP_STEPS):
            y_actual = batch_data[:, k, :]
            psi = torch.tanh(model.S.unsqueeze(0) * x + model.V.unsqueeze(0))
            x_pred = psi @ model.W.T + model.D.unsqueeze(0) * x + model.C.unsqueeze(0)
            y_pred = x_pred @ H_T
            innov = y_actual - y_pred

            PHt = P @ H_T
            S_inn = H @ PHt + R + I_ch_jit
            K = PHt @ torch.inverse(S_inn)
            x = x_pred + innov @ K.T

            with torch.no_grad():
                sech2 = 1.0 - torch.tanh(model.S * x.mean(0).detach() + model.V) ** 2
                J = model.W * (model.S * sech2).unsqueeze(0) + torch.diag(model.D)
                P_pred = J @ P @ J.T + Q
                P = (I_pop - K @ H) @ P_pred
                P = DEC_P * P + (1.0 - DEC_P) * P_pred
                P = 0.5 * (P + P.T)

        # --- Phase 2: KF com loss (whitened) ---
        for k in range(N_KF_STEPS):
            step_idx = N_WARMUP_STEPS + k
            y_actual = batch_data[:, step_idx, :]
            psi = torch.tanh(model.S.unsqueeze(0) * x + model.V.unsqueeze(0))
            x_pred = psi @ model.W.T + model.D.unsqueeze(0) * x + model.C.unsqueeze(0)
            y_pred = x_pred @ H_T
            innov = y_actual - y_pred

            PHt = P @ H_T
            S_inn = H @ PHt + R + I_ch_jit
            K = PHt @ torch.inverse(S_inn)
            x = x_pred + innov @ K.T

            # B2: whitening -- scales both sides by the inverse of the channel std
            yp = y_pred * ch_scale.unsqueeze(0)
            ya = y_actual * ch_scale.unsqueeze(0)
            total_loss = total_loss + torch.nn.functional.huber_loss(
                yp, ya, delta=HUBER_DELTA, reduction='mean')

            with torch.no_grad():
                sech2 = 1.0 - torch.tanh(model.S * x.mean(0).detach() + model.V) ** 2
                J = model.W * (model.S * sech2).unsqueeze(0) + torch.diag(model.D)
                P_pred = J @ P @ J.T + Q
                P = (I_pop - K @ H) @ P_pred
                P = DEC_P * P + (1.0 - DEC_P) * P_pred
                P = 0.5 * (P + P.T)

        # post-filtering amplitude reference (for the growth control)
        x_ref = x.detach().norm(dim=1, keepdim=True) + 1e-6      # (B,1)
        growth_pen = torch.zeros((), device=device)

        # --- Phase 3: free simulation (whitened) + controle de crescimento ---
        for r in range(N_REC_STEPS):
            step_idx = N_WARMUP_STEPS + N_KF_STEPS + r
            y_actual = batch_data[:, step_idx, :]
            psi = torch.tanh(model.S.unsqueeze(0) * x + model.V.unsqueeze(0))
            x = psi @ model.W.T + model.D.unsqueeze(0) * x + model.C.unsqueeze(0)
            y_pred = x @ H_T

            yp = y_pred * ch_scale.unsqueeze(0)
            ya = y_actual * ch_scale.unsqueeze(0)
            total_loss = total_loss + torch.nn.functional.huber_loss(
                yp, ya, delta=HUBER_DELTA, reduction='mean')

            # penalize only if the trajectory grows beyond GROWTH_FACTOR x the initial amplitude
            ratio = x.norm(dim=1, keepdim=True) / x_ref
            growth_pen = growth_pen + (torch.relu(ratio - GROWTH_FACTOR) ** 2).mean()

        data_loss = total_loss / (N_KF_STEPS + N_REC_STEPS)

        # B3: structural regularization of the EE block
        WEE = model.W[:nE, :nE]
        reg_l1 = LAMBDA_L1 * WEE.abs().sum()
        reg_smooth = LAMBDA_SMOOTH * torch.trace(WEE.T @ L_pop @ WEE)

        # C-stab: EXACT sigma_max(J) at x~=0 (gain = S, worst case). Spectral shift
        # M = J^T J - mu*I (mu = 0.5*<D>^2) speeds convergence; power iteration under
        # no_grad (v detached) -> clean gradient via ||J0 v|| (Miyato style).
        # Penalty is a SOFT HINGE (guide, not wall).
        J0 = model.W * model.S.unsqueeze(0) + torch.diag(model.D)   # (n_pop, n_pop)
        mu = 0.5 * (model.D.mean() ** 2)
        with torch.no_grad():
            v = stab_v0
            M = J0.t() @ J0 - mu * I_pop
            for _ in range(K_POWER):
                v = M @ v
                v = v / (v.norm() + 1e-8)
        sigma_max = (J0 @ v).norm()
        reg_stab = LAMBDA_STAB * torch.relu(sigma_max - SPEC_TARGET) ** 2
        reg_growth = LAMBDA_GROWTH * growth_pen / N_REC_STEPS

        loss = data_loss + reg_l1 + reg_smooth + reg_stab + reg_growth

        loss.backward()
        return loss, sigma_max.detach()

    for it in range(n_iters):
        optimizer.zero_grad()
        starts = torch.randint(0, max_start, (B,), device=device)

        try:
            loss, sigma_max = train_step(starts)
        except Exception as e:
            print(f"\n    [ERROR iter{it+1}] train_step falhou: {e}")
            nan_count += 1
            if nan_count > 10:
                success = False
                break
            continue

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count > 10:
                print(f"    iter {it+1}: NaN demais, abortando janela")
                success = False
                break
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        model.project_signs()

        # A5: soft renormalization of W rows that grow too much
        with torch.no_grad():
            row_norms = torch.norm(model.W, dim=1, keepdim=True)
            scale = torch.clamp(W_ROW_NORM_MAX / (row_norms + 1e-10), max=1.0)
            model.W.mul_(scale)

        losses.append(loss.detach().clone())

        # C-stab: print sigma_max every 500 iters to watch stabilization in real time
        if (it + 1) % 500 == 0:
            print(f"    iter {it+1}/{n_iters}, loss={loss.item():.4f}, "
                  f"σmax={sigma_max.item():.3f}, "
                  f"⟨R⟩={model.R_diag.mean().item():.3f}, "
                  f"⟨S⟩={model.S.mean().item():.2f}, "
                  f"max|W|={torch.abs(model.W).max().item():.3f}")

    losses_cpu = [l.item() for l in losses]
    if len(losses_cpu) < n_iters * 0.1:
        success = False
    return model, losses_cpu, success


# ============================================================================
# FEATURE EXTRACTION  (20 magnitude + 7 dynamic -- B5)
# ============================================================================

def _operating_point(W, S, D, V, C, n_pop, n_burn=400):
    """Operating point x*: deterministic iteration (no noise), tail mean."""
    try:
        x = np.zeros(n_pop)
        traj = np.zeros((n_burn, n_pop))
        for t in range(n_burn):
            x = W @ np.tanh(S * x + V) + D * x + C
            if not np.all(np.isfinite(x)):
                return np.zeros(n_pop)
            x = np.clip(x, -50.0, 50.0)
            traj[t] = x
        return traj[n_burn // 2:].mean(axis=0)
    except Exception:
        return np.zeros(n_pop)


def _dynamics_features(W, S, D, V, nE, x_star):
    """6 features of the Jacobian linearized at the real OPERATING POINT (not at 0)."""
    try:
        gain = S * (1.0 - np.tanh(S * x_star + V) ** 2)     # sech² no ponto real
        J = W * gain[None, :] + np.diag(D)
        eig = np.linalg.eigvals(J)
        mag = np.abs(eig)
        k = int(np.argmax(mag))
        WEE = W[:nE, :nE]; WEI = W[nE:, :nE]
        Jee = WEE * gain[:nE][None, :]
        spec_ee = float(np.max(np.abs(np.linalg.eigvals(Jee))))
        ei_norm = float(np.linalg.norm(WEE, 'fro') /
                        (np.linalg.norm(WEI, 'fro') + 1e-10))
        return np.array([
            float(mag[k]),                  # specRadius
            float(np.real(eig[k])),         # domEigReal
            float(np.abs(np.imag(eig[k]))), # domEigImag (intrinsic freq.)
            float(np.sum(mag > 1.0)),       # nUnstable
            ei_norm,                        # eiNormRatio
            spec_ee,                        # specRadius_EE
        ], dtype=np.float64)
    except Exception:
        return np.zeros(6, dtype=np.float64)


def _participation_ratio(W, S, D, V, C, n_pop, n_steps=2000, seed=0):
    """Manifold dimensionality (Singh D_PR) of the noise-simulated trajectories."""
    try:
        rng = np.random.default_rng(seed)
        rtq = np.sqrt(Q_DIAG)
        x = np.zeros(n_pop)
        X = np.zeros((n_steps, n_pop))
        for t in range(n_steps):
            x = W @ np.tanh(S * x + V) + D * x + C + rtq * rng.standard_normal(n_pop)
            if not np.all(np.isfinite(x)):
                return 0.0
            x = np.clip(x, -50.0, 50.0)
            X[t] = x
        Xc = X[n_steps // 2:]
        Xc = Xc - Xc.mean(0)
        cov = Xc.T @ Xc / len(Xc)
        lam = np.linalg.eigvalsh(cov)
        lam = lam[lam > 0]
        return float((lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-20))
    except Exception:
        return 0.0


def extract_features(model):
    """Extract 20 magnitude features + 7 dynamics features (B5) = 27 total."""
    W = model.W.detach().cpu().numpy()
    D = model.D.detach().cpu().numpy()
    S = model.S.detach().cpu().numpy()
    V = model.V.detach().cpu().numpy()
    C = model.C.detach().cpu().numpy()
    nE = model.n_e
    n_pop = model.n_pop

    WEE = W[:nE, :nE]
    WEI = W[nE:, :nE]
    betaIE = np.diag(W[:nE, nE:])
    betaII = np.diag(W[nE:, nE:])

    wee_nz = WEE[WEE != 0]
    wei_nz = WEI[WEI != 0]
    if len(wee_nz) == 0: wee_nz = np.array([0.0])
    if len(wei_nz) == 0: wei_nz = np.array([0.0])

    mEE = np.mean(np.abs(wee_nz))
    mEI = np.mean(np.abs(wei_nz))
    eiRat = mEE / (mEI + 1e-10)

    feat = np.array([
        eiRat,                                   #  1 eiRatio
        mEE,                                     #  2 meanWEE
        mEI,                                     #  3 meanWEI
        np.mean(betaIE),                         #  4 meanBetaIE
        np.mean(betaII),                         #  5 meanBetaII
        np.mean(D[:nE]),                         #  6 meanD_exc
        np.mean(D[nE:]),                         #  7 meanD_inh
        np.mean(S[:nE]),                         #  8 meanS_exc
        np.mean(S[nE:]),                         #  9 meanS_inh
        np.std(wee_nz),                          # 10 stdWEE
        np.std(wei_nz),                          # 11 stdWEI
        np.linalg.norm(WEE, 'fro'),              # 12 normWEE
        np.linalg.norm(WEI, 'fro'),              # 13 normWEI
        np.max(np.abs(W)),                       # 14 maxAbsW
        np.mean(np.abs(betaIE)),                 # 15 meanAbsBetaIE
        np.mean(np.abs(betaII)),                 # 16 meanAbsBetaII
        np.mean(np.abs(betaIE)) / (mEE + 1e-10), # 17 localDistalRat
        np.sum(np.abs(wee_nz) > 0.1),            # 18 strongEE
        np.sum(np.abs(wei_nz) > 0.1),            # 19 strongEI
        sp_kurtosis(wee_nz, fisher=False),       # 20 kurtWEE
    ], dtype=np.float64)

    # B5: dynamics at the operating point + manifold
    x_star = _operating_point(W, S, D, V, C, n_pop)
    dyn = _dynamics_features(W, S, D, V, nE, x_star)
    dpr = np.array([_participation_ratio(W, S, D, V, C, n_pop)], dtype=np.float64)
    return np.concatenate([feat, dyn, dpr])


# ============================================================================
# SIMULATION & VALIDATION
# ============================================================================

def simulate_model(model, n_steps, calibrate_rms=None):
    """Simulate the model with noise, return observed signal."""
    n_ch = model.n_ch
    n_pop = model.n_pop

    W = model.W.detach().cpu().numpy()
    D = model.D.detach().cpu().numpy()
    S = model.S.detach().cpu().numpy()
    C = model.C.detach().cpu().numpy()
    V = model.V.detach().cpu().numpy()
    H = model.H.detach().cpu().numpy()
    rtQ = model.rtQ.detach().cpu().numpy()

    x = np.zeros(n_pop)
    simY = np.zeros((n_ch, n_steps))
    for t in range(n_steps):
        psi = np.tanh(S * x + V)
        x = W @ psi + D * x + C + rtQ @ np.random.randn(n_pop)
        simY[:, t] = H @ x

    if calibrate_rms is not None:
        rms_sim = np.sqrt(np.mean(simY ** 2))
        scale = calibrate_rms / (rms_sim + 1e-10)
        x = np.zeros(n_pop)
        rtQ_scaled = rtQ * scale
        simY = np.zeros((n_ch, n_steps))
        for t in range(n_steps):
            psi = np.tanh(S * x + V)
            x = W @ psi + D * x + C + rtQ_scaled @ np.random.randn(n_pop)
            simY[:, t] = H @ x

    return simY


def compute_validation_metrics(real_data, sim_data, srate):
    """Compute all validation metrics."""
    n_ch, n_samp = real_data.shape
    nfft = min(512, n_samp)

    psd_corr_norm = []
    psd_corr_log = []
    for ch in range(n_ch):
        fR, pR = welch(real_data[ch], fs=srate, nperseg=nfft)
        fS, pS = welch(sim_data[ch], fs=srate, nperseg=nfft)
        rlog = np.corrcoef(np.log10(pR + 1e-20), np.log10(pS + 1e-20))[0, 1]
        psd_corr_log.append(rlog)
        pR_n = pR / (np.sum(pR) + 1e-20)
        pS_n = pS / (np.sum(pS) + 1e-20)
        rnorm = np.corrcoef(pR_n, pS_n)[0, 1]
        psd_corr_norm.append(rnorm)

    corr_real = np.corrcoef(real_data)
    corr_sim = np.corrcoef(sim_data)
    tri_idx = np.triu_indices(n_ch, k=1)
    spatial_r = np.corrcoef(corr_real[tri_idx], corr_sim[tri_idx])[0, 1]

    max_lag = min(100, n_samp - 1)
    acf_errs = []
    acf_real_avg = np.zeros(max_lag + 1)
    acf_sim_avg = np.zeros(max_lag + 1)
    for ch in range(n_ch):
        acf_r = np.correlate(real_data[ch] - np.mean(real_data[ch]),
                             real_data[ch] - np.mean(real_data[ch]), mode='full')
        acf_r = acf_r[n_samp - 1:n_samp + max_lag] / (acf_r[n_samp - 1] + 1e-20)
        acf_s = np.correlate(sim_data[ch] - np.mean(sim_data[ch]),
                             sim_data[ch] - np.mean(sim_data[ch]), mode='full')
        acf_s = acf_s[n_samp - 1:n_samp + max_lag] / (acf_s[n_samp - 1] + 1e-20)
        acf_errs.append(np.sqrt(np.mean((acf_r - acf_s) ** 2)))
        acf_real_avg += acf_r / n_ch
        acf_sim_avg += acf_s / n_ch

    kurt_real = sp_kurtosis(real_data, axis=1, fisher=False)
    kurt_sim = sp_kurtosis(sim_data, axis=1, fisher=False)
    kurt_corr = np.corrcoef(kurt_real, kurt_sim)[0, 1]

    return {
        'psd_log': np.mean(psd_corr_log),
        'psd_norm': np.mean(psd_corr_norm),
        'spatial_r': spatial_r,
        'acf_rmse': np.mean(acf_errs),
        'kurt_corr': kurt_corr,
        'corr_real': corr_real,
        'corr_sim': corr_sim,
        'acf_real': acf_real_avg,
        'acf_sim': acf_sim_avg,
        'kurt_real': kurt_real,
        'kurt_sim': kurt_sim,
    }


def save_validation_plots(real_data, sim_data, metrics, ch_names,
                          subject, session, fig_dir, srate=SRATE):
    """Save comprehensive validation figures."""
    n_ch = real_data.shape[0]
    nfft = min(512, real_data.shape[1])

    # --- Figure 1: PSD + Timeseries ---
    fig, axes = plt.subplots(4, 2, figsize=(14, 8), facecolor='black')
    ch_idx = np.linspace(0, n_ch - 1, 4).astype(int)
    plot_n = min(PLOT_SEC * srate, real_data.shape[1])
    t_ax = np.arange(plot_n) / srate

    for p, ci in enumerate(ch_idx):
        name = ch_names[ci] if ci < len(ch_names) else f'Ch{ci}'

        ax = axes[p, 0]
        ax.plot(t_ax, real_data[ci, :plot_n], color='#3399ff', linewidth=0.7, label='Real')
        ax.plot(t_ax, sim_data[ci, :plot_n], color='#ff6633', linewidth=0.7, label='Model')
        ax.set_ylabel(name, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.set_xlim(0, PLOT_SEC)
        if p == 0:
            ax.set_title('Timeseries', color='#66aaff')
            ax.legend(facecolor='black', labelcolor='white', fontsize=7)
        if p == 3:
            ax.set_xlabel('Time (s)', color='white')

        ax = axes[p, 1]
        fR, pR = welch(real_data[ci], fs=srate, nperseg=nfft)
        fS, pS = welch(sim_data[ci], fs=srate, nperseg=nfft)
        pR_n = pR / (np.sum(pR) + 1e-20)
        pS_n = pS / (np.sum(pS) + 1e-20)
        ax.semilogy(fR, pR_n, color='#3399ff', linewidth=1.2)
        ax.semilogy(fS, pS_n, color='#ff6633', linewidth=1.2)
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.set_xlim(0, 30)
        if p == 0:
            ax.set_title(f"Norm PSD (r={metrics['psd_norm']:.3f})", color='#66aaff')
        if p == 3:
            ax.set_xlabel('Freq (Hz)', color='white')

    fig.suptitle(f'{subject} / {session} — PSD Validation', color='#66aaff')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'val_psd_{session}.png'),
                facecolor='black', dpi=120)
    plt.close(fig)

    # --- Figure 2: Dynamics validation ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), facecolor='black')

    ax = axes[0, 0]
    ax.imshow(metrics['corr_real'], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Spatial Corr: Real', color='#66aaff')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')

    ax = axes[0, 1]
    ax.imshow(metrics['corr_sim'], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Spatial Corr: Model', color='#66aaff')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')

    ax = axes[0, 2]
    tri = np.triu_indices(n_ch, k=1)
    ax.scatter(metrics['corr_real'][tri], metrics['corr_sim'][tri],
               s=2, alpha=0.3, color='#3399ff')
    ax.plot([-1, 1], [-1, 1], '--', color='gray', linewidth=1)
    ax.set_xlabel('Real', color='white')
    ax.set_ylabel('Model', color='white')
    ax.set_title(f"Spatial r={metrics['spatial_r']:.3f}", color='#66aaff')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax = axes[1, 0]
    max_lag = len(metrics['acf_real']) - 1
    lag_ms = np.arange(max_lag + 1) / srate * 1000
    ax.plot(lag_ms, metrics['acf_real'], color='#3399ff', linewidth=1.5, label='Real')
    ax.plot(lag_ms, metrics['acf_sim'], color='#ff6633', linewidth=1.5, label='Model')
    ax.set_xlabel('Lag (ms)', color='white')
    ax.set_ylabel('ACF', color='white')
    ax.set_title(f"ACF (RMSE={metrics['acf_rmse']:.4f})", color='#66aaff')
    ax.legend(facecolor='black', labelcolor='white', fontsize=7)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')

    ax = axes[1, 1]
    edges = np.linspace(-6, 6, 60)
    ax.hist(real_data.ravel(), bins=edges, alpha=0.5, color='#3399ff',
            label='Real', density=True)
    ax.hist(sim_data.ravel(), bins=edges, alpha=0.5, color='#ff6633',
            label='Model', density=True)
    ax.set_xlabel('Amplitude', color='white')
    ax.set_title(f"Distribution (kurt r={metrics['kurt_corr']:.3f})", color='#66aaff')
    ax.legend(facecolor='black', labelcolor='white', fontsize=7)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')

    ax = axes[1, 2]
    ax.scatter(metrics['kurt_real'], metrics['kurt_sim'], s=10, color='#3399ff')
    mn = min(metrics['kurt_real'].min(), metrics['kurt_sim'].min())
    mx = max(metrics['kurt_real'].max(), metrics['kurt_sim'].max())
    ax.plot([mn, mx], [mn, mx], '--', color='gray')
    ax.set_xlabel('Real kurtosis', color='white')
    ax.set_ylabel('Model kurtosis', color='white')
    ax.set_title('Kurtosis per channel', color='#66aaff')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')

    fig.suptitle(f'{subject} / {session} — Dynamics Validation', color='#66aaff')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'val_dynamics_{session}.png'),
                facecolor='black', dpi=120)
    plt.close(fig)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_subject(subject, device='cpu'):
    """Process one subject: all sessions × conditions × windows."""
    sub_data_dir = DATA_ROOT / subject
    sub_out_dir = OUT_ROOT / subject
    fig_dir = sub_out_dir / 'figures'
    feat_file = sub_out_dir / 'window_features.mat'

    if feat_file.exists():
        print(f"[CHECKPOINT] {subject} done — skipping")
        return True

    if not sub_data_dir.exists():
        print(f"[SKIP] {subject} — no data")
        return False

    sub_out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"  SUBJECT: {subject}")
    print(f"  Device: {device}")
    print(f"{'#' * 60}")

    sub_start = time.time()

    all_features = []
    all_labels = []
    all_sessions = []
    all_windows = []
    all_W = []          # A1: matriz de conectividade completa por janela
    all_S = []          # A1: vetor de ganho por janela
    all_R_diag = []     # A3: learned measurement noise per window
    model_count = 0
    last_meta = None

    for i_sess, session in enumerate(SESSIONS):
        sess_plot_done = False

        for i_cond, (cond_name, cond_file) in enumerate(zip(CONDITIONS, COND_FILES)):
            fname = sub_data_dir / f"{session}_{cond_file}.mat"
            if not fname.exists():
                print(f"  [SKIP] {session}/{cond_name}")
                continue

            print(f"\n--- {subject} {session} {cond_name} ---")

            D = sio.loadmat(str(fname))
            n_ch      = int(D['n_channels'].item())
            n_pop     = int(D['n_populations'].item())
            n_exc     = int(D['n_exc_per_ch'].item())
            n_inh     = int(D['n_inh_per_ch'].item())
            full_data = D['MeasData'].astype(np.float32)
            n_total   = full_data.shape[1]
            ch_names  = [str(c).strip() for c in D['channel_names'].flatten()]

            n_regions = int(D['n_regions'].item()) if 'n_regions' in D else None
            region_names = ([str(r).strip() for r in D['region_names'].flatten()]
                            if 'region_names' in D else
                            [f'region_{i:02d}' for i in range(n_regions or 0)])

            print(f"  {n_ch}ch {n_pop}pop "
                  f"{'(' + str(n_regions) + ' regions) ' if n_regions else ''}"
                  f"{n_total/SRATE:.0f}s")

            # B1: anatomical prior (once per condition; coords do not change)
            region_coords = resolve_region_coords(D, n_regions)
            if region_coords is not None:
                anat_adm, prox_EE, L_pop = build_distance_prior(
                    region_coords, n_exc, n_inh)
                dens = float(anat_adm.mean())
                print(f"  [B1] anatomical mask active -- E->E density={dens:.2f}")
            else:
                anat_adm = prox_EE = L_pop = None
                print(f"  [B1] coords unavailable -- DENSE mode (no mask)")

            # H from the BEM forward model -- normalized to O(1)
            H_loaded = torch.from_numpy(D['H'].astype(np.float32)).to(device)
            H_norm   = H_loaded / (H_loaded.abs().max() + 1e-10)

            last_meta = (n_ch, n_exc, n_inh, n_regions, ch_names, region_names)

            win_starts = list(range(0, n_total - WIN_SAMP + 1, STEP_SAMP))
            n_win = len(win_starts)
            print(f"  {n_win} windows × {WINDOW_SEC}s")

            for i_win, w_start in enumerate(win_starts):
                w_end = w_start + WIN_SAMP
                win_data = full_data[:, w_start:w_end]
                data_tensor = torch.from_numpy(win_data).to(device)

                model = MINDyModel(n_ch, n_exc, n_inh, device=device,
                                   n_regions=n_regions,
                                   anat_adm=anat_adm, prox_EE=prox_EE, L_pop=L_pop)
                model.H.copy_(H_norm)

                t0 = time.time()
                print(f"  W{i_win + 1}/{n_win} [{w_start // SRATE}-{w_end // SRATE}s] ",
                      end='', flush=True)

                try:
                    model, losses, success = bpkf_train_window(
                        model, data_tensor, n_iters=N_ITERS, lr=LR, device=device)
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

                if not success:
                    print(f"    [DIVERGIU] janela descartada")
                    continue

                elapsed = time.time() - t0

                feat = extract_features(model)
                W_np = model.W.detach().cpu().numpy().astype(np.float32)
                S_np = model.S.detach().cpu().numpy().astype(np.float32)
                R_np = model.R_diag.detach().cpu().numpy().astype(np.float32)

                all_features.append(feat)
                all_labels.append(i_cond)
                all_sessions.append(i_sess + 1)
                all_windows.append(i_win + 1)
                all_W.append(W_np)
                all_S.append(S_np)
                all_R_diag.append(R_np)
                model_count += 1

                print(f"{elapsed:.1f}s E/I={feat[0]:.3f} "
                      f"specR={feat[20]:.3f} dimPR={feat[26]:.1f} "
                      f"⟨S⟩={S_np.mean():.2f} ⟨R⟩={R_np.mean():.3f}")

                if not sess_plot_done:
                    rms_real = np.sqrt(np.mean(win_data ** 2))
                    sim_data = simulate_model(model, WIN_SAMP, calibrate_rms=rms_real)
                    metrics = compute_validation_metrics(win_data, sim_data, SRATE)
                    print(f"    PSD(norm)={metrics['psd_norm']:.3f} "
                          f"Spatial={metrics['spatial_r']:.3f} "
                          f"ACF={metrics['acf_rmse']:.4f} "
                          f"Kurt={metrics['kurt_corr']:.3f}")
                    save_validation_plots(win_data, sim_data, metrics, ch_names,
                                          subject, session, str(fig_dir))
                    sess_plot_done = True

    sub_time = time.time() - sub_start

    if model_count > 0:
        n_ch_s, n_exc_s, n_inh_s, n_reg_s, ch_names_s, region_names_s = last_meta
        n_pop_s = (n_exc_s + n_inh_s) * (n_reg_s if n_reg_s else n_ch_s)
        n_units = n_reg_s if n_reg_s else n_ch_s

        readme = (
            f"MINDy source-space v2 fit. Cada linha = uma janela fitada.\n"
            f"Janela {WINDOW_SEC}s, overlap {OVERLAP}, srate {SRATE}Hz.\n\n"
            f"W layout (n_pop={n_pop_s}); population order:\n"
            f"  indices 0..{n_units-1}: E_pop_1 across {n_units} regions/channels\n"
            f"  ... E_pop_{n_exc_s} ... depois I_pop_1 ... I_pop_{n_inh_s}\n\n"
            f"Inter-region connectivity E->E (nxn, n={n_units}):\n"
            f"  W_EE[i,j] = soma_{{e1,e2}} W[e1*n + i, e2*n + j], i!=j\n"
            f"(compatible with aggregate_W_blocks from visualizacao_cerebral.py)."
        )

        save_dict = {
            'allFeatures':  np.array(all_features),
            'allLabels':    np.array(all_labels).reshape(-1, 1),
            'allSessions':  np.array(all_sessions).reshape(-1, 1),
            'allWindows':   np.array(all_windows).reshape(-1, 1),
            'featureNames': np.array(FEATURE_NAMES, dtype=object),
            # A1: full model parameters for connectivity analysis
            'allW':         np.array(all_W),
            'allS':         np.array(all_S),
            'allR_diag':    np.array(all_R_diag),
            'n_ch':         n_ch_s,
            'n_exc':        n_exc_s,
            'n_inh':        n_inh_s,
            'n_pop':        n_pop_s,
            'n_regions':    n_reg_s if n_reg_s else 0,
            'channel_names': np.array(ch_names_s, dtype=object),
            'region_names':  np.array(region_names_s, dtype=object),
            'popOrder': f'[E1..E{n_exc_s}, I1..I{n_inh_s}] each × {n_units}',
            'README':       readme,
            # Reprodutibilidade
            'WINDOW_SEC':   WINDOW_SEC,
            'OVERLAP':      OVERLAP,
            'SRATE':        SRATE,
            'SUBJECT':      subject,
            'N_ITERS':      N_ITERS,
            'LR':           LR,
            'N_WARMUP_STEPS': N_WARMUP_STEPS,
            'N_KF_STEPS':   N_KF_STEPS,
            'N_REC_STEPS':  N_REC_STEPS,
            'FIXED_D':      FIXED_D,
            'INIT_S':       INIT_S,
            'S_MIN':        S_MIN,
            'S_MAX':        S_MAX,
            'Q_DIAG':       Q_DIAG,
            'R_learnable':  True,
            'TAU_MM':       TAU_MM,
            'KEEP_FRAC':    KEEP_FRAC,
            'LAMBDA_L1':    LAMBDA_L1,
            'LAMBDA_SMOOTH': LAMBDA_SMOOTH,
            'LAMBDA_STAB':  LAMBDA_STAB,
            'SPEC_TARGET':  SPEC_TARGET,
            'K_POWER':      K_POWER,
            'LAMBDA_GROWTH': LAMBDA_GROWTH,
            'GROWTH_FACTOR': GROWTH_FACTOR,
            'N_STACKS':     N_STACKS,
            'GRAD_CLIP_NORM': GRAD_CLIP_NORM,
            'W_ROW_NORM_MAX': W_ROW_NORM_MAX,
            'modelCount':   model_count,
            'codeVersion':  'source-v2',
            'source_space': True,
        }
        sio.savemat(str(feat_file), save_dict, do_compression=True)
        print(f"\n  {subject}: {model_count} models, {sub_time / 60:.1f} min")
        print(f"  Saved: {feat_file}")
        print(f"  allW shape: {np.array(all_W).shape}")
        return True
    else:
        print(f"\n  {subject}: no models fitted")
        return False


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MINDy-BPKF source-space v2')
    parser.add_argument('--subject', type=str, default='all',
                        help='Subject ID (e.g., sub-01) or "all"')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"MINDy-BPKF source-space v2 — Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if not DATA_ROOT.exists():
        print(f"\n  ERROR: Data directory not found: {DATA_ROOT}")
        sys.exit(1)

    if args.subject == 'all':
        subjects = [f"sub-{i:02d}" for i in range(1, 30)]
    else:
        subjects = [args.subject]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    global_start = time.time()
    n_processed = 0
    n_skipped = 0

    for subject in subjects:
        result = process_subject(subject, device=device)
        if result:
            n_processed += 1
        else:
            n_skipped += 1

    total_hours = (time.time() - global_start) / 3600
    print(f"\n{'#' * 60}")
    print(f"  COMPLETE: {n_processed} processed, {n_skipped} skipped")
    print(f"  Total: {total_hours:.1f} hours")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
