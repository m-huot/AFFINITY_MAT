import matplotlib.pyplot as plt

# from __future__ import annotations
import numpy as np
from numpy.random import default_rng
import pandas as pd
from collections import defaultdict
import matplotlib.ticker as mtick


"""
Germinal‑centre B‑cell maturation with multiple antigens
========================================================

• State of each cycling B‑cell: h ∈ ℝ^{N_h} (residue binding‑energy vector)
• Antigens:            S_mat ∈ {‑1, +1}^{N_h × n_A}
                       ε_v(h) = Σ_i s_{iv} h_i    for every antigen v
• One round:           duplication → SHM → Ag‑gate → T‑gate → differentiation
• Output:              `history[step]` is a NumPy array of shape
                       (n_cells_alive_that_step, N_h) with the h‑vectors

The implementation below follows exactly the gates used in your FP solver.
"""


mu = 0.001 * 3 * 40
p_sil = 0.5 * (mu) + (1 - mu)
p_let = 0.3 * (mu)
p_aa = 0.2 * (mu)
E_a = np.log(40)
T = 140
T_steps = 140

kBT = 1  # thermal factor
N_i = 2500
N_max = 2500
mu_i = 0.0
sigma_i = 0.001
p_diff = 0.10
c = 170

# Average site effects from DMS data
db = pd.read_csv("../data/cdr_site_effects.csv")
cdr_positions_relative = db["cdr_positions_relative"].to_numpy()
mu_arr = db["mu"].to_numpy()
sigma_arr = db["sigma"].to_numpy()
S_mat = np.array([[1] for pos in range(len(cdr_positions_relative))])
N_h = S_mat.shape[0]


# -------------------------- helper functions ---------------------------
def energies(pop: np.ndarray, S_mat: np.ndarray) -> np.ndarray:
    """
    Compute energies for every cell and every antigen.

    Parameters
    ----------
    pop   : (N_cells, N_h) ndarray
    S_mat : (N_h, n_A)     ndarray

    Returns
    -------
    eps : (N_cells, n_A) ndarray   with eps[c, v] = ε_v(h_c)
    """
    return pop @ S_mat  # broadcasting handles dot‑product


def P_Ag(eps: np.ndarray, C_vec: np.ndarray) -> np.ndarray:
    """
    Antigen‑binding survival gate   P_Ag(h)

    eps   : (N_cells, n_A) energies
    C_vec : (n_A,)        antigen concentrations at current round
    """
    exp_term = np.exp((eps - E_a) / kBT)  # same as FP code
    numer = (exp_term * C_vec).sum(axis=1)  # Σ_v C_v e^{(ε_v - E_a)/kT}
    return numer / (1.0 + numer)


def P_T(eps: np.ndarray, C_vec: np.ndarray) -> np.ndarray:
    """
    T‑cell‑help survival gate   P_T(h | pop)

    eps   : (N_cells, n_A) energies *after* Ag selection
    """
    C_tot = C_vec.sum()
    eE = np.exp(eps / kBT)  # e^{ε_v/kT}
    Phi = (eE * C_vec).sum(axis=1)  # Σ_v C_v e^{ε_v/kT}
    Phi_bar = Phi.mean()  # ⟨Φ⟩_pop
    return Phi / (Phi + Phi_bar / C_tot)


def gc_round(
    pop: np.ndarray,
    S_mat: np.ndarray,
    C_vec: np.ndarray,
    rng: np.random.Generator,
    mu_M: np.ndarray,
    sigma_M: np.ndarray,
    mutable_positions: np.ndarray,
    mut_sites: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        pop_next   : GC survivors after differentiation (n_t, N_h)
        mem_cells  : differentiated cells (n_mem, N_h)
        parent_idx : parent row index in previous gen (n_t,)
        uniq_counts: unique mutated-site counts aligned to pop_next (n_t,)
        mut_sites  : boolean mutated-sites mask aligned to pop_next (n_t, N_h)
    Counting rule: only the first mutation at a site counts for that lineage; repeated hits to the same site do not.
    """
    N_h = pop.shape[1]
    assert mu_M.shape == (N_h,) and sigma_M.shape == (N_h,), (
        "mu_M and sigma_M must be length N_h"
    )
    uniq_counts = None
    # init counters/masks for gen-0
    if uniq_counts is None:
        uniq_counts = np.zeros(pop.shape[0], dtype=int)
    if mut_sites is None:
        mut_sites = np.zeros((pop.shape[0], N_h), dtype=bool)

    # 1) Duplication: replicate state, counts, and masks; track parents
    prev_n = pop.shape[0]
    pop = np.repeat(pop, 2, axis=0)
    uniq_counts = np.repeat(uniq_counts, 2, axis=0)
    mut_sites = np.repeat(mut_sites, 2, axis=0)
    parent_idx = np.repeat(np.arange(prev_n, dtype=int), 2)

    # 2) SHM: at most one site per mutating cell this round
    N_cells = pop.shape[0]
    fate = rng.choice(["let", "aa", "sil"], size=N_cells, p=[p_let, p_aa, p_sil])
    alive_mask = fate != "let"
    aa_mask = fate == "aa"

    if aa_mask.any():
        rows = np.flatnonzero(aa_mask)
        if mutable_positions is None:
            idx_sites = rng.integers(0, N_h, size=rows.size)
        else:
            mp = np.asarray(mutable_positions, dtype=int)
            idx_sites = rng.choice(mp, size=rows.size)
        delta_h = rng.normal(mu_M[idx_sites], sigma_M[idx_sites])
        # apply mutation to h
        # pop[rows, idx_sites] = np.clip(pop[rows, idx_sites] + delta_h, -3*np.log(10), 1*np.log(10))
        pop[rows, idx_sites] = pop[rows, idx_sites] + delta_h
        # UNIQUE-SITE counting: only first hit to a site increments
        # (relative to this lineage's gen-0 founder; we encode that by the boolean mut_sites)
        for r, i in zip(rows, idx_sites):
            if not mut_sites[r, i]:
                uniq_counts[r] += 1
                mut_sites[r, i] = True

    # lethal removal
    pop = pop[alive_mask]
    parent_idx = parent_idx[alive_mask]
    uniq_counts = uniq_counts[alive_mask]
    mut_sites = mut_sites[alive_mask]
    if pop.size == 0:
        return (
            pop,
            np.empty((0, N_h)),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
            np.empty((0, N_h), dtype=bool),
        )

    # 3) Antigen-binding gate (shape-safe reduction over possible multi-C_vec outputs)
    eps = energies(pop, S_mat)
    P_ag = P_Ag(eps, C_vec)
    if isinstance(P_ag, np.ndarray) and P_ag.ndim == 2:
        P_ag = P_ag.max(axis=1)  # or .mean/.min per your policy
    P_ag = np.asarray(P_ag, dtype=float).reshape(-1)
    assert P_ag.shape[0] == pop.shape[0]
    survive = rng.random(pop.shape[0]) < P_ag

    pop = pop[survive]
    parent_idx = parent_idx[survive]
    uniq_counts = uniq_counts[survive]
    mut_sites = mut_sites[survive]
    if pop.size == 0:
        return (
            pop,
            np.empty((0, N_h)),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
            np.empty((0, N_h), dtype=bool),
        )

    # 4) T-cell-help gate (shape-safe)
    P_t = P_T(eps, C_vec)
    if isinstance(P_t, np.ndarray) and P_t.shape[0] != pop.shape[0]:
        eps_t = energies(pop, S_mat)
        P_t = P_T(eps_t, C_vec)
    if isinstance(P_t, np.ndarray) and P_t.ndim == 2:
        P_t = P_t.max(axis=1)
    P_t = np.asarray(P_t, dtype=float).reshape(-1)
    if P_t.shape[0] != pop.shape[0]:
        if P_t.size == 1:
            P_t = np.full(pop.shape[0], float(P_t))
        else:
            raise ValueError(f"P_T shape {P_t.shape} must match n_cells {pop.shape[0]}")
    survive = rng.random(pop.shape[0]) < P_t

    pop = pop[survive]
    parent_idx = parent_idx[survive]
    uniq_counts = uniq_counts[survive]
    mut_sites = mut_sites[survive]
    if pop.size == 0:
        return (
            pop,
            np.empty((0, N_h)),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
            np.empty((0, N_h), dtype=bool),
        )

    # 5) Differentiation
    diff_mask = rng.random(size=pop.shape[0]) < p_diff
    mem_cells = pop[diff_mask]
    keep_mask = ~diff_mask
    pop = pop[keep_mask]
    parent_idx = parent_idx[keep_mask]
    uniq_counts = uniq_counts[keep_mask]
    mut_sites = mut_sites[keep_mask]

    return pop, mem_cells, parent_idx, uniq_counts, mut_sites


def plot_h_dimension(
    history, dim_idx=0, h_min=-4, h_max=10, n_hbins=100, t_clip=50, cmap="Greys_r"
):
    """
    Plot the density of B-cell h[dim_idx] values over rounds with fixed colorbar scale [0, 500].
    """
    time_points = []
    h_values = []
    for t, pop in enumerate(history):
        if pop.size == 0:
            continue
        time_points.extend([t] * pop.shape[0])
        h_values.extend(pop[:, dim_idx])

    rounds_arr = np.asarray(time_points)
    h_vals = np.asarray(h_values)

    n_rounds = len(history)
    round_bins = np.linspace(rounds_arr.min(), rounds_arr.max(), n_rounds + 1)
    h_bins = np.linspace(h_min, h_max, n_hbins + 1)

    H, xedges, yedges = np.histogram2d(rounds_arr, h_vals, bins=[round_bins, h_bins])

    plt.figure(figsize=(8, 5))
    im = plt.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        vmin=0.0,
        vmax=100,
    )  # fixed scale
    plt.xlabel("Round")
    plt.ylabel(rf"$h_{{{dim_idx + 1}}}$")
    plt.title(rf"Density of B-cells in $h_{{{dim_idx + 1}}}$ over GC rounds")
    plt.colorbar(im, label="Density")
    plt.ylim(h_min, h_max)
    plt.tight_layout()
    plt.show()


def simulate_gc_with_ancestry(
    C_schedule: callable,
    S_mat: np.ndarray = S_mat,
    rounds: int = T,
    N_init: int = N_i,
    N_max: int = N_max,
    mu_init: float = mu_i,
    sigma_init: float = sigma_i,
    mu_M: np.ndarray = mu_arr,
    sigma_M: np.ndarray = sigma_arr,
    mutable_positions=np.asarray(cdr_positions_relative, dtype=int),
    seed: int | None = None,
) -> list[dict]:
    """
    Simulates GC evolution and tracks ancestry using global unique IDs.

    Returns:
        history : list of dicts per generation (t=0..T). Keys:
                  'ids': global unique cell IDs
                  'parent_ids': global IDs of the parents (-1 for root)
                  'pop': cell states (N, N_h)
                  'uniq_counts': unique mutated-site count aligned to pop
    """
    rng = default_rng(seed)
    N_h = S_mat.shape[0]
    if mu_M is None or sigma_M is None:
        raise ValueError("Provide per-site mu_M and sigma_M of shape (N_h,)")

    pop = rng.normal(mu_init, sigma_init, size=(N_init, N_h))
    if mutable_positions is not None:
        mutable_positions = np.array(mutable_positions, dtype=int)
        all_positions = np.arange(N_h)
        non_mutable_positions = np.setdiff1d(all_positions, mutable_positions)
        pop[:, non_mutable_positions] = 0.0

    # Initialize ID tracking
    next_id = N_init
    initial_ids = np.arange(N_init, dtype=int)
    initial_parent_ids = np.full(N_init, -1, dtype=int)

    # Setup history as list of dicts
    uniq_counts = np.zeros(pop.shape[0], dtype=int)
    mut_sites = np.zeros((pop.shape[0], N_h), dtype=bool)

    history = [
        {
            "ids": initial_ids,
            "parent_ids": initial_parent_ids,
            "pop": pop.copy(),
            "uniq_counts": uniq_counts.copy(),
        }
    ]

    for r in range(rounds):
        C_vec = np.asarray(C_schedule(r), dtype=float)

        # Run GC round
        pop_next, _mem, parent_idx, uniq_counts, mut_sites = gc_round(
            pop,
            S_mat,
            C_vec,
            rng,
            mu_M,
            sigma_M,
            mutable_positions=mutable_positions,
            # uniq_counts=uniq_counts,
            mut_sites=mut_sites,
        )

        # Apply capacity cap consistently
        if pop_next.shape[0] > N_max:
            keep_idx = rng.choice(pop_next.shape[0], N_max, replace=False)
            pop_next = pop_next[keep_idx]
            parent_idx = parent_idx[keep_idx]
            uniq_counts = uniq_counts[keep_idx]
            mut_sites = mut_sites[keep_idx]

        # Map row indices to actual global parent IDs from the previous frame
        prev_ids = history[-1]["ids"]
        current_parent_ids = prev_ids[parent_idx]

        # Assign new global IDs to the survivors
        n_survivors = pop_next.shape[0]
        current_ids = np.arange(next_id, next_id + n_survivors, dtype=int)
        next_id += n_survivors

        # Store generation data
        history.append(
            {
                "ids": current_ids,
                "parent_ids": current_parent_ids,
                "pop": pop_next.copy(),
                "uniq_counts": uniq_counts.copy(),
            }
        )

        pop = pop_next

    return history


def trace_lineages(history, target_indices_at_end):
    """
    Reconstructs affinity trajectories for specific cells found in the last step.
    Affinity is defined as the sum of h_i across all positions for a given cell.

    Parameters
    ----------
    history : list of dicts
        The output from simulate_gc_with_ancestry.
    target_indices_at_end : array-like
        Indices in the *last* history frame of the cells you want to trace.

    Returns
    -------
    trajectories : list of numpy arrays
        A list where each element is a 1D array of shape (T+1,)
        representing the total affinity path of a single lineage.
    """
    if not history:
        return []

    # Get the last frame
    final_step = len(history) - 1
    last_frame = history[final_step]

    # These are the IDs we want to trace back
    current_ids_to_trace = last_frame["ids"][target_indices_at_end]

    # Map: Final_Unique_ID -> Current_Ancestor_ID
    active_traces = {uid: uid for uid in current_ids_to_trace}

    # Storage: Final_Unique_ID -> List of total affinities
    results = {uid: [] for uid in current_ids_to_trace}

    for t in range(final_step, -1, -1):
        frame = history[t]

        # Create lookup for this frame: ID -> (Index, Parent_ID, Vector)
        frame_lookup = {
            uid: (i, pid, vec)
            for i, (uid, pid, vec) in enumerate(
                zip(frame["ids"], frame["parent_ids"], frame["pop"])
            )
        }

        for final_uid, ancestor_id in list(active_traces.items()):
            if ancestor_id in frame_lookup:
                _, parent_id, vector = frame_lookup[ancestor_id]

                # --- MODIFICATION: Calculate total affinity (sum of h_i) ---
                total_affinity = np.sum(np.nan_to_num(vector, nan=0.0))
                results[final_uid].append(total_affinity)

                # Step back: looking for parent in previous frame
                if parent_id != -1:
                    active_traces[final_uid] = parent_id
                else:
                    # Reached root
                    del active_traces[final_uid]
            else:
                # Should not happen in an unbroken chain
                del active_traces[final_uid]

    # Reverse lists to get time 0 -> T
    trajectories = []
    for uid in current_ids_to_trace:
        traj = np.array(results[uid][::-1])
        trajectories.append(traj)

    return trajectories
