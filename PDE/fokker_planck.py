import numpy as np
import torch
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import os, warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning, message="KMP_DUPLICATE_LIB_OK")
import numpy as np
import matplotlib.pyplot as plt


mu = 0.001 * 3 * 40
p_sil = 0.5 * (mu) + (1 - mu)
p_let = 0.3 * (mu)
p_aa = 0.2 * (mu)
E_A = np.log(40)
T = 140
KBT = 1  # thermal factor
N_I = 2500
N_MAX = 2500
mu_i = 0.0
sigma_i = 0.001 * np.sqrt(40)
p_diff = 0.10
c = 170  # 800, 1120, 920, 510

mu_M = -0.8
# sigma_M=1.09
sigma_M = 1.68
# sigma_M=1.75


# ---------------- mutation statistics -------------------------------
f_aa = p_aa / (p_aa + p_sil)
f_sil = 1.0 - f_aa
V_ADV = f_aa * mu_M
D = f_aa * (sigma_M**2 + f_sil * mu_M**2)
LAM = np.log(2.0) + np.log1p(-p_let) + np.log1p(-p_diff)
S_mat = np.array([[1.0]])


# ---------------------------------------------------------------------
# 1.  Run FP simulation  +  build Γ & ∇Γ lists
# ---------------------------------------------------------------------
def fp_density_time_series_kD(
    # biological / model parameters ---------------------------------------
    get_C=lambda t, v: 5.0,  # user-supplied concentration C_v(t)
    get_S=lambda t: S_mat,  # shape (N_h, n_A) : s_{k,v} ∈ {±1}
    N_I=N_I,
    N_MAX=N_MAX,
    # grid / time ----------------------------------------------------------
    h_min=-1.0,
    h_max=40.0,
    d_h=0.1,  # d_h=0.05
    T=30.0,
    dt=0.001,
    # output ---------------------------------------------------------------
    snapshot_interval=0.005,
    verbose=False,
    rho_c=0,  # extinction threshold (cells per unit h-volume
):
    """
    Simulate k-dimensional GC Fokker–Planck dynamics in h-space.

    Parameters
    ----------
    S_mat : ndarray (N_h, n_A)
        Binary motif matrix: each column = antigen, each row = shape component.
    get_C : callable (t, v) → C_v(t)
        Returns surface concentration of antigen v at time t.
    """
    # ---------------- basic checks --------------------------------------
    if get_S is None:
        raise ValueError(
            "S_mat (binary motif matrix) must be provided for each time step"
        )
    N_h, n_A = get_S(0).shape  # k = dimension of h

    if D > 0.0:
        dt_max = d_h**2 / (2.0 * N_h * D)
        if dt > dt_max:
            raise ValueError(f"dt = {dt:.3g} exceeds CFL limit {dt_max:.3g}")

    # ---------------- k-dim Cartesian grid ------------------------------
    h_axis = np.arange(h_min, h_max + d_h, d_h)
    h_axes = (h_axis,) * N_h
    h_mesh = np.meshgrid(*h_axes, indexing="ij")  # list of length k
    dH = d_h**N_h

    # ---------------- Gaussian initial population -----------------------
    g1d = np.exp(-((h_axis - mu_i) ** 2) / (2.0 * sigma_i**2))
    rho = np.prod(np.meshgrid(*([g1d] * N_h), indexing="ij"), axis=0)
    rho *= N_I / (rho.sum() * dH)

    snapshots = [rho.copy()]
    times = [0.0]

    # ---------------- pre-allocate helper arrays ------------------------
    E_stack = np.zeros((n_A,) + rho.shape)  # energies  E_v(h)

    n_steps = int(round(T / dt))
    snap_every = max(1, int(round(snapshot_interval / dt)))
    omegas = [0]
    # -------------------------------------------------------------------
    for step in range(1, n_steps + 1):
        t_now = step * dt
        N_pop = rho.sum() * dH

        # --- time-dependent concentrations -----------------------------
        C_vec = np.array([get_C(t_now, v) for v in range(n_A)], dtype=float)
        C_tot = C_vec.sum()

        # If no antigen is present, skip selection (pure drift/diffusion)
        if C_tot == 0.0:
            if verbose and step % snap_every == 0:
                print(f"{step:5d}  t={t_now:6.2f}  (no antigen present)")
            if step % snap_every == 0:
                snapshots.append(rho.copy())
                times.append(t_now)
            continue

        # --- binding energies  E_v(h) ----------------------------------
        # S_mat.T : (n_A, N_h);  stacked h_mesh : (N_h, …grid…)
        S_mat = get_S(t_now)
        E_stack = np.tensordot(S_mat.T, np.stack(h_mesh, axis=0), axes=(1, 0))
        #            → shape (n_A, …grid…)

        # Broadcast C_v to all grid points
        C_broad = C_vec.reshape((n_A,) + (1,) * N_h)

        # --- antigen internalisation gate  P_Ag ------------------------
        exp_term = np.exp((E_stack - E_A) / KBT)
        numer_PAg = (C_broad * exp_term).sum(axis=0)
        P_Ag = numer_PAg / (1.0 + numer_PAg)

        # --- T‐cell help gate  P_T  (concentration-weighted) ----------
        eE = np.exp(E_stack / KBT)
        Phi = eE.sum(axis=0)  # Σ_v C_v e^{E_v/kBT}
        Phi_bar = (rho * Phi).sum() * dH / N_pop
        P_T = Phi / (Phi + Phi_bar / C_tot)

        # --- growth/decay rate Γ(h,t) ----------------------------------
        log_sel = np.log(P_Ag) + np.log(P_T)
        avg_log = (rho * log_sel).sum() * dH / N_pop
        Omega = 0.0 if N_pop < N_MAX + 0.1 else max(0.0, LAM + avg_log)
        Gamma = LAM + log_sel - Omega

        # --- conservative flux divergence ------------------------------
        div_total = np.zeros_like(rho)
        for ax in range(N_h):
            rho_fwd = np.roll(rho, -1, axis=ax)

            adv_flux = V_ADV * (rho if V_ADV >= 0 else rho_fwd)
            diff_flux = -0.5 * D * (rho_fwd - rho) / d_h
            J_int = adv_flux + diff_flux  # at cell interfaces

            # trim last cell (fwd neighbour outside domain)
            slc = [slice(None)] * N_h
            slc[ax] = slice(0, -1)
            J_int = J_int[tuple(slc)]

            # build full flux array with zero at boundaries
            J_shape = list(rho.shape)
            J_shape[ax] += 1
            J = np.zeros(J_shape)
            idx = [slice(None)] * N_h
            idx[ax] = slice(1, -1)
            J[tuple(idx)] = J_int

            div_total += np.diff(J, axis=ax) / d_h

        # --- explicit Euler update -------------------------------------
        rho += dt * (Gamma * rho - div_total)
        np.maximum(rho, 0.0, out=rho)  # clip negatives
        cell_density = 1.0 / dH
        rho[rho < rho_c * cell_density] = 0.0

        # --- book-keeping ----------------------------------------------
        if step % snap_every == 0:
            snapshots.append(rho.copy())
            times.append(t_now)
            omegas.append(Omega)

        if verbose and step % snap_every == 0:
            mean_Gamma = (rho * Gamma).sum() * dH / N_pop
            mean_h = [(rho * h_mesh[ax]).sum() * dH / N_pop for ax in range(N_h)]
            mean_h_str = "  ".join(
                f"⟨h_{i + 1}⟩={m:6.2f}" for i, m in enumerate(mean_h)
            )
            print(
                f"{step:5d}  t={t_now:6.2f}  N={N_pop:8.1f}  Ω={Omega:7.3f}  "
                f"⟨Γ⟩={mean_Gamma:8.3f}  {mean_h_str}"
            )

    return h_axes, np.array(times), np.stack(snapshots), np.array(omegas)


def run_fp(
    get_S=lambda t: S_mat,
    get_C=lambda t, v: 5.0,
    *,
    T=140.0,
    verbose=False,
    rho_c=0,
    N_MAX=N_MAX,
    N_I=N_I,
):
    """
    Returns a dict 'fp_ctx' with everything downstream functions need.
    """
    axes, times, rho_t, omegas = fp_density_time_series_kD(
        get_S=get_S,
        get_C=get_C,
        T=T,
        verbose=verbose,
        rho_c=rho_c,
        N_MAX=N_MAX,
        N_I=N_I,
    )

    # ---------- Φ̄(t_k) --------------------------------------------
    N_h, n_A = get_S(0).shape
    d_h = axes[0][1] - axes[0][0]
    dH = d_h**N_h
    h_mesh = np.meshgrid(*axes, indexing="ij")

    Phi_bar = np.zeros_like(times, dtype=float)
    for k, t_now in enumerate(times):
        E_full = np.tensordot(get_S(t_now).T, np.stack(h_mesh, axis=0), axes=(1, 0))

        C_vec = np.array([get_C(float(t_now), v) for v in range(n_A)])
        Phi = (np.exp(E_full / KBT)).sum(axis=0)
        rho = rho_t[k]
        Phi_bar[k] = (rho * Phi).sum() * dH / (rho.sum() * dH)

    # ---------- Γ & ∇Γ snapshots (autograd) ------------------------
    gamma_list, grad_list = [], []

    for k, t_now in enumerate(times):
        Phi_b = float(Phi_bar[k])
        S_torch = torch.tensor(get_S(t_now), dtype=torch.float64)

        omega = float(omegas[k])
        C_vec = np.array([get_C(float(t_now), v) for v in range(n_A)], dtype=float)
        C_t = torch.tensor(C_vec, dtype=torch.float64)

        def γ(h_t, C_t=C_t, Phi_b=Phi_b, omega=omega):
            E_v = torch.mv(S_torch.T, h_t)
            P_Ag = (C_t * torch.exp((E_v - E_A) / KBT)).sum()
            P_Ag = P_Ag / (1.0 + P_Ag)
            Phi = torch.exp(E_v / KBT).sum()
            P_T = Phi / (Phi + Phi_b / C_t.sum())
            return LAM + torch.log(P_Ag) + torch.log(P_T) - omega

        def gγ_np(h_np, γ=γ):  # capture γ itself
            h_t = torch.tensor(h_np, requires_grad=True, dtype=torch.float64)
            γ(h_t).backward()
            return h_t.grad.numpy()

        gamma_list.append(γ)
        grad_list.append(gγ_np)

    return dict(
        axes=axes,
        times=times,
        rho_t=rho_t,
        omegas=omegas,
        Phi_bar=Phi_bar,
        gamma_list=gamma_list,
        grad_list=grad_list,
        get_S=get_S,
        get_C=get_C,
        S_mat=S_mat,
        rho_c=rho_c,
    )


def plot_fp_density_time_dim(
    fp_ctx,
    dim=0,
    *,
    disp_step=5,
    cmap="Greys",
    sol=None,
    vmax_fixed=2000.0,
    ymax=None,
    leastaction_label=None,
):
    """
    Heat-map of ρ(t, h_dim) with a white background and shades of black.
    The color scale is fixed between 0 and 2500.
    """
    # Force default light style
    plt.style.use("default")

    axes, times, rho_t = fp_ctx["axes"], fp_ctx["times"], fp_ctx["rho_t"]
    N_h = len(axes)
    if not (0 <= dim < N_h):
        raise ValueError(f"dim must be in [0,{N_h - 1}]")

    h_axis = axes[dim]

    # --- marginal over all dims except `dim` --------------------------
    marginal_axes = tuple(i + 1 for i in range(N_h) if i != dim)

    if N_h > 1:
        steps = [(axes[i][1] - axes[i][0]) for i in range(N_h) if i != dim]
        dH = np.prod(steps)
    else:
        dH = 1.0

    rho_marg = rho_t.sum(axis=marginal_axes) * dH
    rho_marg[np.isnan(rho_marg)] = 0

    # --- plot settings ------------------------------------------------
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.figsize": (3.4, 2.6),
            "figure.dpi": 300,
            "font.family": "serif",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(facecolor="white")

    # 3. Plot Heatmap
    # vmin=0 and vmax=vmax_fixed (2500) ensures the scale is locked
    im = ax.imshow(
        rho_marg.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], h_axis[0], h_axis[-1]],
        cmap=cmap,
        vmin=0.0,
        vmax=vmax_fixed,
    )

    # 4. Add Colorbar with fixed range
    # Ticks are set at 0, 625, 1250, 1875, 2500
    cbar_ticks = np.linspace(0.0, vmax_fixed, 5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=cbar_ticks)
    cbar.set_label(r"$\rho(t,\epsilon)$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # 5. Labels and Ticks
    ax.set_xlabel(r"GC rounds (12h)")
    ax.set_ylabel(r"Affinity $\epsilon$")
    ax.set_xticks(np.arange(times[0], times[-1] + 1e-9, disp_step))

    ax.tick_params(direction="in", top=True, right=True, colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_visible(True)

    # 6. Overlay Trajectory
    if sol is not None:
        if leastaction_label is None:
            ax.plot(
                sol.x,
                sol.y[dim],
                color="#D0021B",
                lw=1.2,
                linestyle="--",
                label=r"Least-action",
            )
        else:
            ax.plot(
                sol.x,
                sol.y[dim],
                color="#D0021B",
                lw=1.2,
                linestyle="--",
                label=leastaction_label,
            )

        legend = ax.legend(frameon=True, loc="upper right", fontsize=7)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")
    if ymax is not None:
        ax.set_ylim(0, ymax)

    plt.tight_layout(pad=0.2)
    plt.show()
