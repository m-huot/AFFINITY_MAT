import os
import warnings
import numpy as np
import pandas as pd
import torch
import copy
import argparse
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

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


def compute_action_and_gradients(
    C_inj, t_inj, target_hf, tau=20.0, T=140.0, basal_C=1e-12, epsilon=1
):
    """
    Computes the least action S and its finite difference gradients with respect to
    vaccine doses (C_inj) and injection times (t_inj).

    Returns:
        action (float): The least action S.
        grad_C_inj (np.ndarray): Finite difference gradient of S with respect to doses.
        grad_t_inj (np.ndarray): Finite difference gradient of S with respect to injection times.
    """
    C_inj = np.array(C_inj, dtype=float)
    t_inj = np.array(t_inj, dtype=float)

    # Helper function to compute just the action for a given set of parameters
    def get_action_for_params(c_params, t_params):
        def get_C_func(t, v=0, *args):
            t_arr = np.atleast_1d(t)
            C_total = np.full_like(t_arr, basal_C, dtype=float)
            for ci, ti in zip(c_params, t_params):
                mask = t_arr >= ti
                C_total[mask] += ci * np.exp(-(t_arr[mask] - ti) / tau)
            return C_total[0] if np.isscalar(t) else C_total

        # Compute population evolution
        fp_ctx = run_fp(get_C=get_C_func, T=T, verbose=False)
        fp_ctx_safe = copy.deepcopy(fp_ctx)

        # Compute least action trajectory
        sol = solve_optimal_trajectory([target_hf], fp_ctx_safe)
        action_val, _ = compute_least_action(sol, fp_ctx_safe, verbose=False)

        return action_val

    # 1. Compute Base Action
    base_action = get_action_for_params(C_inj, t_inj)

    grad_C_inj = np.zeros_like(C_inj)
    grad_t_inj = np.zeros_like(t_inj)

    # 2. Compute Finite Difference Gradients for C_inj
    for i in range(len(C_inj)):
        C_inj_eps = np.copy(C_inj)
        C_inj_eps[i] += epsilon

        action_eps = get_action_for_params(C_inj_eps, t_inj)
        grad_C_inj[i] = (action_eps - base_action) / epsilon

    # 3. Compute Finite Difference Gradients for t_inj
    for i in range(len(t_inj)):
        if i == 0:
            grad_t_inj[i] = 0
            continue
        t_inj_eps = np.copy(t_inj)
        t_inj_eps[i] += epsilon

        action_eps = get_action_for_params(C_inj, t_inj_eps)
        grad_t_inj[i] = (action_eps - base_action) / epsilon

    return base_action, grad_C_inj, grad_t_inj


def run_optimization_loop(
    C_inj_init,
    t_inj_init,
    target_hf,
    num_iterations=50,
    lr_C=5,
    lr_t=1,
    tau=20.0,
    T=140.0,
    basal_C=1e-12,
):
    """
    Optimizes vaccine doses (C_inj) and injection times (t_inj) using gradient descent,
    calling the standalone compute_action_and_gradients function at each step.

    Returns:
        C_inj (np.ndarray): Optimized doses.
        t_inj (np.ndarray): Optimized injection times.
        history (list): History of action values to track convergence.
    """
    C_inj = np.array(C_inj_init, dtype=float)
    t_inj = np.array(t_inj_init, dtype=float)

    history = []

    for step in range(num_iterations):
        print(f"--- Iteration {step + 1}/{num_iterations} ---")
        print(f"Current Doses: {C_inj}")
        print(f"Current Times: {t_inj}")

        # 1. Compute action and gradients using the encapsulated function
        action, grad_C_inj, grad_t_inj = compute_action_and_gradients(
            C_inj, t_inj, target_hf, tau=tau, T=T, basal_C=basal_C
        )

        # Store action and copies of current parameters
        history.append({"action": action, "C_inj": C_inj.copy(), "t_inj": t_inj.copy()})

        print(f"Least action S = {action:.6e}")
        print(f"Gradient w.r.t C_inj: {grad_C_inj}")
        print(f"Gradient w.r.t t_inj: {grad_t_inj}")

        # 2. Safeguards: Clip gradients to prevent massive updates and stiffness
        # grad_C_inj_clipped = np.clip(grad_C_inj, -1.0, 1.0)
        # grad_t_inj_clipped = np.clip(grad_t_inj, -1.0, 1.0)

        # 3. Apply parameter updates
        C_inj -= lr_C * grad_C_inj
        t_inj -= lr_t * grad_t_inj

        # 4. Impose constraints and physical boundaries
        # Keep the first injection time strictly fixed
        t_inj[0] = t_inj_init[0]

        # Impose physical boundaries (safely above 0 to avoid 1/C explosions)
        C_inj = np.maximum(C_inj, 1e-3)
        t_inj = np.clip(t_inj, 0.0, T)

    print("\n" + "=" * 40)
    print("=== OPTIMIZATION COMPLETE ===")
    print("=" * 40)
    print(f"Final Doses: {C_inj}")
    print(f"Final Times: {t_inj}")

    return C_inj, t_inj, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize vaccine doses and injection times."
    )
    parser.add_argument(
        "--C_inj_init",
        type=float,
        nargs="+",
        default=[91.24023, 44.56984],
        help="Initial guesses for doses (space-separated values).",
    )
    parser.add_argument(
        "--t_inj_init",
        type=float,
        nargs="+",
        default=[0.0, 76.80806],
        help="Initial guesses for injection times (space-separated values).",
    )
    parser.add_argument(
        "--target_hf", type=float, default=30.0, help="Target affinity value."
    )

    parser.add_argument(
        "--tau", type=float, default=20.0, help="Decay time constant for vaccine."
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=20,
        help="Number of optimization iterations.",
    )
    parser.add_argument(
        "--lr_C", type=float, default=20.0, help="Learning rate for doses."
    )
    parser.add_argument(
        "--lr_t", type=float, default=1.0, help="Learning rate for injection times."
    )

    # add dossier
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vaccine_optim_results",
        help="Directory to save optimization results.",
    )

    args = parser.parse_args()

    C_inj_initial = np.array(args.C_inj_init)
    t_inj_initial = np.array(args.t_inj_init)
    target_hf = args.target_hf
    n_injections = len(C_inj_initial)

    # Ensure lengths of initial arrays match
    if len(C_inj_initial) != len(t_inj_initial):
        raise ValueError(
            "C_inj_init and t_inj_init must have the same number of elements."
        )

    print(f"Starting optimization with parameters:")
    print(f"  C_inj_init: {C_inj_initial}")
    print(f"  t_inj_init: {t_inj_initial}")
    print(f"  target_hf: {target_hf}")
    print(f"  num_iterations: {args.num_iterations}")
    print(f"  lr_C: {args.lr_C}")
    print(f"  lr_t: {args.lr_t}")
    print("-" * 40)

    # Call the optimization loop
    optimized_C, optimized_t, history = run_optimization_loop(
        C_inj_init=C_inj_initial,
        t_inj_init=t_inj_initial,
        target_hf=target_hf,
        num_iterations=args.num_iterations,
        lr_C=args.lr_C,
        lr_t=args.lr_t,
        tau=args.tau,
        T=140.0,
    )

    print("Optimized Doses:", optimized_C)
    print("Optimized Injection Times:", optimized_t)
    # --- ADD THIS BEFORE SAVING FILES ---
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Save final parameter results
    # ---------------------------------------------------------
    params_filename = os.path.join(
        args.output_dir,
        f"optimized_params_hf_{target_hf}_ninj_{n_injections}_tau_{args.tau}.csv",
    )

    history_filename = os.path.join(
        args.output_dir,
        f"optimization_history_hf_{target_hf}_ninj_{n_injections}_tau_{args.tau}.csv",
    )
    df_params = pd.DataFrame(
        {
            "injection_index": range(n_injections),
            "dose_C": optimized_C,
            "time_t": optimized_t,
        }
    )
    df_params.to_csv(params_filename, index=False)
    print(f"Saved final parameters to {params_filename}")

    # ---------------------------------------------------------
    # Save optimization history
    # ---------------------------------------------------------

    history_records = []
    for i, h in enumerate(history):
        record = {"iteration": i + 1, "action": h["action"]}
        for j in range(n_injections):
            record[f"C_inj_{j}"] = h["C_inj"][j]
            record[f"t_inj_{j}"] = h["t_inj"][j]
        history_records.append(record)

    df_history = pd.DataFrame(history_records)
    df_history.to_csv(history_filename, index=False)
    print(f"Saved optimization history to {history_filename}")
