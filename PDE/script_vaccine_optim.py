import os
import warnings
import numpy as np
import pandas as pd
import torch
import copy
import argparse
import matplotlib.pyplot as plt

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

from least_action import solve_optimal_trajectory, compute_least_action


def compute_action_and_gradients(
    C_inj,
    t_inj,
    target_hf,
    tau=20.0,
    T=140.0,
    basal_C=1e-12,
    epsilon=1,
    alpha_extinction=1000.0,
):
    """
    Computes the least action S and its finite difference gradients with respect to
    vaccine doses (C_inj) and injection times (t_inj). Includes a penalty
    for intermediate population extinction.

    Returns:
        action (float): The least action S including extinction penalty.
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

        # ---------------- Extinction Penalty Calculation ----------------
        axes = fp_ctx_safe["axes"]
        d_h = axes[0][1] - axes[0][0]
        dH = d_h ** len(axes)

        penalty = 0.0
        # Iterate over population densities at each time step
        for rho in fp_ctx_safe["rho_t"]:
            N_t = np.sum(rho) * dH
            # Penalize if population drops below the single-cell threshold
            Nlim = 1
            if N_t < Nlim:
                N_t = max(N_t, 1e-6)  # Avoid division by zero
                penalty += ((Nlim / N_t) - 1) ** 2
        if penalty > 0:
            print(f"Extinction penalty applied: {penalty:.6e}")
        # ----------------------------------------------------------------

        return action_val + alpha_extinction * penalty

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
    alpha_extinction=1000.0,
    clipbound=1,
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
            C_inj,
            t_inj,
            target_hf,
            tau=tau,
            T=T,
            basal_C=basal_C,
            epsilon=1,
            alpha_extinction=alpha_extinction,
        )

        # Store action and copies of current parameters
        history.append({"action": action, "C_inj": C_inj.copy(), "t_inj": t_inj.copy()})

        print(f"Least action S = {action:.6e}")
        print(f"Gradient w.r.t C_inj: {grad_C_inj}")
        print(f"Gradient w.r.t t_inj: {grad_t_inj}")

        # 2. Safeguards: Clip gradients to prevent massive updates and stiffness
        # max of init c_inj
        max_cinj = np.max(C_inj)
        grad_C_inj = np.clip(grad_C_inj, -max_cinj * clipbound, max_cinj * clipbound)
        grad_t_inj = np.clip(grad_t_inj, -5.0, 5.0)

        # 3. Apply parameter updates
        C_inj -= lr_C * grad_C_inj
        t_inj -= lr_t * grad_t_inj

        # 4. Impose constraints and physical boundaries
        # Keep the first injection time strictly fixed
        t_inj[0] = t_inj_init[0]

        # Impose physical boundaries (safely above 0 to avoid 1/C explosions)
        C_inj = np.maximum(C_inj, 1e-12)
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
        default=[951],
        help="Initial guesses for doses (space-separated values).",
    )
    parser.add_argument(
        "--t_inj_init",
        type=float,
        nargs="+",
        default=[0.0],
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
        default=30,
        help="Number of optimization iterations.",
    )
    parser.add_argument(
        "--lr_C", type=float, default=1.0, help="Learning rate for doses."
    )
    parser.add_argument(
        "--lr_t", type=float, default=1.0, help="Learning rate for injection times."
    )

    parser.add_argument(
        "--alpha_extinction",
        type=float,
        default=10,
        help="Penalty weight to apply when population density falls below 1.0.",
    )

    parser.add_argument(
        "--clipbound",
        type=float,
        default=1,
        help="Bound on gradient values.",
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
    clipbound = args.clipbound
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
    print(f"  alpha_extinction: {args.alpha_extinction}")
    print(f"  clipbound: {clipbound}")
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
        alpha_extinction=args.alpha_extinction,
        clipbound=clipbound,
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
