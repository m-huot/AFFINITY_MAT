import os
import warnings
import numpy as np
import pandas as pd
import torch
import copy
import argparse
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import scipy.linalg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning, message="KMP_DUPLICATE_LIB_OK")

from fokker_planck import (
    run_fp,
    plot_fp_density_time_dim,
    N_I,
    LAM,
    KBT,
    E_A,
    N_MAX,
    V_ADV,
    D,
)


def make_grad_Gamma(fp_ctx):
    times, grads = fp_ctx["times"], fp_ctx["grad_list"]

    def grad(h, t):
        if t <= times[0]:
            return grads[0](h)
        if t >= times[-1]:
            return grads[-1](h)
        hi = np.searchsorted(times, t)
        lo = hi - 1
        w = (t - times[lo]) / (times[hi] - times[lo])
        return (1.0 - w) * grads[lo](h) + w * grads[hi](h)

    return grad


def solve_optimal_trajectory(hf, fp_ctx, *, mesh=1000, max_nodes=20_000, plot=False):
    """
    Computes the least-action trajectory from h(t0)=0 to h(tf)=hf.
    Works for arbitrary dimensionality N = fp_ctx["S_mat"].shape[0].
    """
    # ---------------- basic info ----------------
    times = fp_ctx["times"]
    grad_G = make_grad_Gamma(fp_ctx)
    t0, tf = times[0], times[-1]
    N = fp_ctx["S_mat"].shape[0]  # number of h coordinates

    # ------------- final boundary vector -------------
    if np.isscalar(hf):
        hf = np.repeat(float(hf), N)  # same target for every dim
    else:
        hf = np.asarray(hf, dtype=float)
        if hf.size != N:
            raise ValueError(
                f"hf must have length {N} (one per shape dim); got {hf.size}"
            )

    # ----------------- ODE system --------------------
    # state y = (h, v)  ⇒  size 2N
    def ode(t, y):
        h, v = y[:N], y[N:]  # each is shape (N, len(t))
        a = np.empty_like(h)
        for j, tj in enumerate(t):  # loop over mesh points
            a[:, j] = -D * grad_G(h[:, j], tj)  # acceleration
        return np.vstack((v, a))

    # ------------- boundary conditions --------------
    def bc(ya, yb):
        # ya[:N]  : h(t0)  ;  yb[:N] : h(tf)
        return np.hstack((ya[:N], yb[:N] - hf))

    # ------------- initial mesh & guess -------------
    t_mesh = np.linspace(t0, tf, mesh)
    y_guess = np.zeros((2 * N, mesh))
    for i in range(N):  # linear interpolation guess
        y_guess[i] = hf[i] * (t_mesh - t0) / (tf - t0)

    # ------------- solve BVP ------------------------
    sol = solve_bvp(ode, bc, t_mesh, y_guess, max_nodes=max_nodes)
    if not sol.success:
        raise RuntimeError(sol.message)

    # ------------- optional plot --------------------
    if plot:
        plt.figure(figsize=(6, 3))
        for i in range(N):
            plt.plot(sol.x, sol.y[i], label=r"$Affinity (-E(h))$")
        plt.xlabel("GC rounds (12h)")
        plt.ylabel("shape coordinate")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return sol


def compute_least_action(sol, fp_ctx, *, baseline=True, verbose=False):
    """
    Return S_opt   (and S_lin when baseline=True)
    Compatible with any number N = S_mat.shape[0] of shape coordinates.
    """
    # ------------------------------------------------------------------
    times = fp_ctx["times"]
    omegas = fp_ctx["omegas"]
    Phi_bar = fp_ctx["Phi_bar"]
    get_C = fp_ctx["get_C"]
    S_mat = fp_ctx["S_mat"]
    n_A = S_mat.shape[1]
    N = S_mat.shape[0]  # number of h coordinates

    # constants (assumed already defined in the notebook)
    v_vec = V_ADV * np.ones(N)  # deterministic drift term

    # --- torch version of S_mat for Γ --------------------------------
    S_T = torch.tensor(S_mat, dtype=torch.float64)

    # -------------------- Γ(t, h) ------------------------------------
    def Γ(t, h_np):
        """
        Growth rate minus omega at time t and shape h (numpy array length N).
        Returns a scalar float.
        """
        # match given t to nearest entry in 'times'
        k = int(np.argmin(np.abs(times - t)))
        omega = omegas[k]
        Phi_b = Phi_bar[k]

        C_vec = np.array([get_C(float(times[k]), v) for v in range(n_A)], dtype=float)

        # torch variables
        h_t = torch.tensor(h_np, dtype=torch.float64)
        C_t = torch.tensor(C_vec, dtype=torch.float64)

        # energy of each antigen
        E_v = torch.mv(S_T.T, h_t)  # shape (n_A,)

        # binding probability to target antigen
        PAg = (C_t * torch.exp((E_v - E_A) / KBT)).sum()
        PAg = PAg / (1.0 + PAg)

        # total binding probability (any antigen)
        Phi = (torch.exp(E_v / KBT)).sum()
        PT = Phi / (Phi + Phi_b / C_t.sum())

        return (LAM + torch.log(PAg) + torch.log(PT) - omega).item()

    # ----------------- extract optimal traj ---------------------------
    t_mesh = sol.x  # shape (m,)
    h_opt = sol.y[:N].T  # shape (m, N)
    v_opt = sol.y[N : 2 * N].T  # shape (m, N)

    # ----------------- action functional ------------------------------
    def S_of(h, v):
        g = np.array([Γ(ti, hi) for ti, hi in zip(t_mesh, h)])
        kin = np.sum((v - v_vec) ** 2, axis=1) / (2 * D)
        return -np.trapz(g - kin, t_mesh)

    S_opt = S_of(h_opt, v_opt)
    if not baseline:
        if verbose:
            print(f"S_opt = {S_opt:.6e}")
        return S_opt

    # ---------------- linear baseline trajectory ----------------------
    slope = h_opt[-1] / (t_mesh[-1] - t_mesh[0])  # vector length N
    h_lin = (t_mesh[:, None] - t_mesh[0]) * slope[None, :]
    v_lin = np.tile(slope, (len(t_mesh), 1))
    S_lin = S_of(h_lin, v_lin)

    if verbose:
        print(f"S_opt = {S_opt:.6e}   S_lin = {S_lin:.6e}   ΔS = {S_lin - S_opt:.6e}")

    return S_opt, S_lin


def compute_global_hessian_and_det(sol, fp_ctx, verbose=True):
    """
    Constructs the full Block-Tridiagonal Hessian matrix for the Action
    and computes its determinant.

    The variable vector is the flattened trajectory:
    X = [h(t_1), h(t_2), ..., h(t_{m-2})]
    (Excluding fixed endpoints t_0 and t_{m-1} for Dirichlet boundary conditions)
    """
    # ------------------------------------------------------------------
    # 1. Unpack Context & Constants
    # ------------------------------------------------------------------
    times = fp_ctx["times"]
    omegas = fp_ctx["omegas"]
    Phi_bar = fp_ctx["Phi_bar"]
    get_C = fp_ctx["get_C"]
    S_mat = fp_ctx["S_mat"]

    # Physical Constants

    # Dimensions
    n_A = S_mat.shape[1]
    N = S_mat.shape[0]  # Dimension of h space
    S_T = torch.tensor(S_mat, dtype=torch.float64)

    # ------------------------------------------------------------------
    # 2. Helper: Pure Torch Gamma (Second Derivative Ready)
    # ------------------------------------------------------------------
    def gamma_torch_func(h_tensor, t_scalar):
        # Time-dependent parameters
        t_val = t_scalar.item()
        k = int(np.argmin(np.abs(times - t_val)))

        omega = omegas[k]
        Phi_b = Phi_bar[k]
        C_vec_np = np.array(
            [get_C(float(times[k]), v) for v in range(n_A)], dtype=float
        )
        C_t = torch.tensor(C_vec_np, dtype=torch.float64)

        # Physics Logic (same as before, strictly torch)
        E_v = torch.mv(S_T.T, h_tensor)
        args_Ag = (E_v - E_A) / KBT
        numerator_Ag = (C_t * torch.exp(args_Ag)).sum()
        log_PAg = torch.log(numerator_Ag) - torch.log(1.0 + numerator_Ag)

        Phi = (torch.exp(E_v / KBT)).sum()
        denom_PT = Phi + (Phi_b / C_t.sum())
        log_PT = torch.log(Phi) - torch.log(denom_PT)
        return LAM + log_PAg + log_PT - omega

    # ------------------------------------------------------------------
    # 3. Trajectory Setup
    # ------------------------------------------------------------------
    t_mesh = sol.x
    h_opt = sol.y[:N].T  # Shape (m, N)

    M = len(t_mesh)
    dt = t_mesh[1] - t_mesh[0]  # Assuming uniform grid for standard formula

    # We solve for fluctuations of inner points only (fixed start/end)
    # Number of variable time steps
    n_steps = M - 2

    # Total size of the Hessian Matrix
    total_dim = n_steps * N

    # Initialize the Global Hessian (Sparse structure, but dense array for print)
    H_global = np.zeros((total_dim, total_dim))

    # ------------------------------------------------------------------
    # 4. Assemble Matrix Blocks
    # ------------------------------------------------------------------
    # Kinetic Factors
    # Second derivative of Kinetic term ~ (h_{k+1} - h_k)^2 / (2 D dt)
    # d^2/dh_k^2   =  2 / (D * dt)
    # d^2/dh_k dh_{k+1} = -1 / (D * dt)

    diag_kin_val = 2.0  # / (D * dt)
    off_kin_val = -1.0  # / (D * dt)

    if verbose:
        print(f"--- Constructing Hessian ---")
        print(
            f"Time steps: {M}, Variables: {n_steps}, Total Matrix Size: {total_dim}x{total_dim}"
        )
        print(f"Kinetic Diagonal Base: {diag_kin_val:.2f}")

    # Loop over variable time indices (from 1 to M-2 in 0-based indexing)
    for k in range(1, M - 1):
        # Map time index k to matrix block index i = k - 1
        i = k - 1

        # A. Compute Local Gamma Hessian
        t_val = t_mesh[k]
        h_val = h_opt[k]
        h_tensor = torch.tensor(h_val, dtype=torch.float64, requires_grad=True)
        t_tensor = torch.tensor(t_val, dtype=torch.float64)

        # Get local hessian of Gamma (N x N)
        func_h = lambda h: gamma_torch_func(h, t_tensor)
        gamma_hess_local = (
            torch.autograd.functional.hessian(func_h, h_tensor).detach().numpy()
        )

        if i > 0:
            block_diag = np.eye(N) * diag_kin_val - gamma_hess_local * D * dt**2
        else:
            block_diag = np.eye(N) * diag_kin_val / 2 - gamma_hess_local * D * dt**2

        # Place Diagonal Block
        row_start = i * N
        row_end = (i + 1) * N
        H_global[row_start:row_end, row_start:row_end] = block_diag

        # C. Construct Off-Diagonal Blocks (Kinetic connection)
        # Connects i to i+1
        if i < n_steps - 1:
            block_off = np.eye(N) * off_kin_val

            # Right neighbor
            col_start_next = (i + 1) * N
            col_end_next = (i + 2) * N

            H_global[row_start:row_end, col_start_next:col_end_next] = block_off
            # Symmetric (Left neighbor for the next row)
            H_global[col_start_next:col_end_next, row_start:row_end] = block_off

    # ------------------------------------------------------------------
    # 5. Print Hessian Snippet
    # ------------------------------------------------------------------
    if verbose:
        print("\nGlobal Hessian (Top-Left 10x10 snippet):")
        with np.printoptions(precision=2, suppress=True, linewidth=120):
            print(H_global[:10, :10])

        print("\nGlobal Hessian (Bottom-Right 6x6 snippet):")
        with np.printoptions(precision=2, suppress=True, linewidth=120):
            print(H_global[-6:, -6:])

    # ------------------------------------------------------------------
    # 6. Compute Determinant
    # ------------------------------------------------------------------
    # Use slogdet for stability (returns sign, log_abs_det)
    sign, logdet = np.linalg.slogdet(H_global)

    if verbose:
        print("-" * 40)
        print(f"Log Determinant: {logdet:.6f}")
        print(f"Sign: {sign}")
        print("-" * 40)

    # Return the log determinant
    return logdet, H_global
