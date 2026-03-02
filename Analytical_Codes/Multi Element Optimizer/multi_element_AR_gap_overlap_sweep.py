# Eppler_model_gap_opti_top10.py
# Modified from Eppler_model_gap_opti.py
# - AR sweep: 8.5 - 11.5 (step 0.5)
# - Adds Pareto & ranking selection to return top-10 CL/CD pairs
# Author: ChatGPT (patch for Team 37 / Shanky)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os

# ----------------------------- Core physics ------------------------------

def calculate_wake_properties(Re: float, chord_main: float = 1.0) -> Tuple[float, float]:
    """Estimate wake (momentum) thickness and a reference profile Cd.

    Simple empirical relations used in engineering practice:
      Cf ~ 0.074 / Re^0.2   (turbulent flat plate correlation)
      Cd_profile ~ 2.2 * Cf  (engineering proxy)
      delta_wake ~ 0.37 * c * sqrt(Cd_profile)

    Returns:
      (delta_wake, Cd_main_profile)
    """
    Cf = 0.074 / (Re ** 0.2)
    Cd_main_profile = 2.2 * Cf
    delta_wake = 0.37 * chord_main * (Cd_main_profile ** 0.5)
    return delta_wake, Cd_main_profile


def finite_wing_lift_slope(AR: float, e: float = 0.85) -> float:
    """Approximate finite-wing lift slope per radian using lifting-line.

    a_2d = 2*pi (thin-airfoil). Finite-wing slope:
      a = a_2d / (1 + a_2d/(pi*AR*e))

    Returned value is per radian. Convert to per-degree outside if needed.
    """
    a_2d = 2.0 * np.pi
    a = a_2d / (1.0 + a_2d / (np.pi * AR * e))
    return a


def estimate_aerodynamics(
    gap: float,
    overlap: float,
    flap_angle_deg: float,
    alpha_system_deg: float,
    delta_wake: float,
    chord_main: float = 1.0,
    chord_flap: float = 0.4,
    AR: float = 8.5,
    e: float = 0.82,
    Re: float = 5e5,
) -> Dict[str, float]:
    """Estimate CL, CD and intermediate breakdowns for a two-element (main+flap)
    configuration using a patched empirical + lifting-line approach.

    Input angles are in degrees.

    Returns dict with keys: CL, CD, CL_main, CL_flap, CD_induced, CD_profile,
    slot_efficiency, stalled (bool)
    """
    # ------------------ slot / wake efficiency (empirical) -----------------
    opt_gap = 1.3 * delta_wake
    # gaussian-shaped sensitivity to gap around opt_gap
    gap_sigma = max(0.5 * delta_wake, 1e-6)
    gap_quality = np.exp(-((gap - opt_gap) ** 2) / (2.0 * gap_sigma ** 2))
    overlap_factor = np.log10(1.0 + overlap * 18.0)
    overlap_quality = np.clip(overlap_factor, 0.0, 1.0)
    slot_efficiency = 0.6 * gap_quality + 0.4 * overlap_quality
    slot_efficiency = float(np.clip(slot_efficiency, 0.0, 0.98))

    # ------------------ main wing lift (finite-wing corrected) -------------
    a_rad = finite_wing_lift_slope(AR, e)
    # convert to per-degree: a_deg = a_rad * (rad -> deg) = a_rad * pi/180
    a_deg = a_rad * (np.pi / 180.0)
    cl_main_zero = 0.4  # empirical zero-lift offset (keeps values sensible)

    # stall limit scaled by slot efficiency (heuristic)
    main_stall_limit = 14.0 + (6.0 * slot_efficiency)
    if alpha_system_deg > main_stall_limit:
        cl_main = (a_deg * main_stall_limit + cl_main_zero) * 0.7
        main_stalled = True
    else:
        cl_main = a_deg * alpha_system_deg + cl_main_zero
        main_stalled = False

    # ------------------ flap lift estimate (empirical potential) ------------
    flap_efficiency = 0.62 + 0.18 * slot_efficiency
    flap_potential = 2.0 * np.pi * (chord_flap / chord_main) * np.sin(
        np.radians(flap_angle_deg)
    )
    cl_flap = max(flap_potential * flap_efficiency, 0.0)

    flap_stall_limit = 18.0 + 18.0 * slot_efficiency
    if flap_angle_deg > flap_stall_limit:
        cl_flap *= 0.5
        flap_stalled = True
    else:
        flap_stalled = False

    cl_total = cl_main + cl_flap
    stalled = main_stalled or flap_stalled

    # ------------------ drag model (profile + flap + interference + induced) --
    # base profile drag (small alpha dependence) and weak Re scaling
    cd_base_0 = 0.015
    cd_alpha_term = 0.0006 * (alpha_system_deg ** 2)
    Re_ref = 1e6
    re_factor = (Re_ref / Re) ** 0.06
    cd_profile = (cd_base_0 + cd_alpha_term) * re_factor

    # flap deflection penalty (quadratic in radians) scaled by flap chord
    cd_flap_deflection = 0.035 * (np.radians(flap_angle_deg) ** 2) * (chord_flap / chord_main)
    # interference penalty reduced by better slots
    cd_interference = 0.02 * (1.0 - slot_efficiency)
    # stall penalty (crude)
    cd_stall = 0.12 if stalled else 0.0
    # induced drag (finite wing)
    cd_induced = (cl_total ** 2) / (np.pi * AR * e + 1e-12)

    cd_total = cd_profile + cd_flap_deflection + cd_interference + cd_stall + cd_induced

    return {
        'CL': float(cl_total),
        'CD': float(cd_total),
        'CL_main': float(cl_main),
        'CL_flap': float(cl_flap),
        'CD_induced': float(cd_induced),
        'CD_profile': float(cd_profile),
        'Slot_eff': float(slot_efficiency),
        'Stalled': bool(stalled),
    }

# ----------------------------- Utilities --------------------------------


def sweep_params(
    AR_list=(8.5, 9.0, 9.5),
    gaps=(0.04, 0.05, 0.06),
    overlaps=(0.06, 0.08, 0.10),
    flap_angles=(20, 25, 30, 35),
    alphas=(8, 10, 12, 14, 16),
    chord_main=1.0,
    chord_flap=0.4,
    e=0.82,
    Re=5e5,
) -> pd.DataFrame:
    """Run a param sweep and return a pandas DataFrame of results."""
    delta_wake, _ = calculate_wake_properties(Re, chord_main)
    rows = []
    for AR in AR_list:
        for g in gaps:
            for ov in overlaps:
                for ang in flap_angles:
                    for a in alphas:
                        out = estimate_aerodynamics(
                            g, ov, ang, a, delta_wake,
                            chord_main=chord_main,
                            chord_flap=chord_flap,
                            AR=AR, e=e, Re=Re,
                        )
                        rows.append({
                            'AR': AR,
                            'Gap (%)': g * 100.0,
                            'Overlap (%)': ov * 100.0,
                            'Flap_deg': ang,
                            'AoA_deg': a,
                            'CL': out['CL'],
                            'CD': out['CD'],
                            'L/D': out['CL'] / out['CD'] if out['CD'] > 0 else np.nan,
                            'Stalled': out['Stalled'],
                            'Slot_eff': out['Slot_eff'],
                            'CL_main': out['CL_main'],
                            'CL_flap': out['CL_flap'],
                            'CD_induced': out['CD_induced'],
                            'CD_profile': out['CD_profile'],
                        })
    return pd.DataFrame(rows)


def summarize_and_save(df: pd.DataFrame, csv_path: str = 'sweep_results.csv') -> None:
    """Save sweep results and print a short summary. Also plot CL vs CD frontier."""
    df.to_csv(csv_path, index=False)
    print(f"Saved sweep results to: {csv_path}")

    # basic summary
    print("--- Sweep summary ---")
    print(f"Total cases: {len(df)}")
    feasible = df[(df['Stalled'] == False)]
    print(f"Non-stalled cases: {len(feasible)}")
    best_cl = feasible.sort_values('CL', ascending=False).head(5)
    best_ld = feasible.sort_values('L/D', ascending=False).head(5)
    print("Top 5 by CL (non-stalled):")
    print(best_cl[['AR','Gap (%)','Overlap (%)','Flap_deg','AoA_deg','CL','CD','L/D']].to_string(index=False, float_format="%.3f"))
    print("\nTop 5 by L/D (non-stalled):")
    print(best_ld[['AR','Gap (%)','Overlap (%)','Flap_deg','AoA_deg','CL','CD','L/D']].to_string(index=False, float_format="%.3f"))

    # Pareto-ish plot CL vs CD
    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(feasible['CD'], feasible['CL'], c=feasible['AR'], cmap='viridis', alpha=0.85)
    ax.set_xlabel('C_D')
    ax.set_ylabel('C_L')
    ax.set_title('CL vs CD (non-stalled cases). Color -> AR')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('AR')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('cl_vs_cd.png', dpi=200)
    print("Saved CL vs CD plot: cl_vs_cd.png")


# ------------------- New: Pareto & Top-10 CL/CD selection utils -----------------

def pareto_front_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Return boolean mask (len N) marking Pareto-optimal rows for the multi-objective:
    maximize CL, minimize CD.
    """
    # Use array where we minimize both dimensions: (-CL) and CD
    arr = np.vstack([-df['CL'].values, df['CD'].values]).T  # shape (N,2)
    N = arr.shape[0]
    is_pareto = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_pareto[i]:
            continue
        # any point that dominates i will set is_pareto[i] = False
        # j dominates i if arr[j] <= arr[i] in all dims and < in at least one
        better = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
        is_pareto[better] = False
    return is_pareto

def normalize_series(s: pd.Series) -> pd.Series:
    if s.max() == s.min():
        return pd.Series(np.ones_like(s) * 0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def select_top10_cl_cd(df: pd.DataFrame, top_n: int = 10, use_pareto: bool = True) -> pd.DataFrame:
    """
    Select top configurations balancing high CL and low CD.
    Steps:
      1. filter non-stalled rows & drop NaNs
      2. find Pareto front (if use_pareto True); if front smaller than top_n, use all
      3. normalize CL and CD, compute combined score = CL_norm + (1 - CD_norm)
      4. also compute distance to ideal (CL_norm=1, CD_norm=0) in normalized space
      5. rank by combined score and pick top_n unique configs
    Returns DataFrame of top_n rows (preserves original columns).
    """
    dff = df[df['Stalled'] == False].copy()
    dff = dff.dropna(subset=['CL','CD']).reset_index(drop=True)
    if dff.empty:
        raise RuntimeError("No valid (non-stalled) rows available for selection.")

    # Pareto mask
    if use_pareto:
        mask = pareto_front_mask(dff)
        pareto_df = dff[mask].copy()
    else:
        pareto_df = dff.copy()

    pool = pareto_df if len(pareto_df) >= top_n else dff.copy()

    # normalization
    pool['CL_norm'] = normalize_series(pool['CL'])
    pool['CD_norm'] = normalize_series(pool['CD'])
    pool['CD_norm_inv'] = 1.0 - pool['CD_norm']  # higher is better

    # combined score: want CL up and CD down
    pool['combined_score'] = pool['CL_norm'] + pool['CD_norm_inv']

    # distance to ideal (1,0) in normalized CL/CD space
    CLn_for_dist = pool['CL_norm']
    CDn_for_dist = pool['CD_norm']
    pool['dist_to_ideal'] = np.sqrt((1.0 - CLn_for_dist) ** 2 + (0.0 - CDn_for_dist) ** 2)

    # Rank: prefer combined_score (desc), tie-breaker dist_to_ideal (asc)
    pool_sorted = pool.sort_values(['combined_score', 'dist_to_ideal'], ascending=[False, True])

    # Keep unique configurations by key identifying fields (AR, Gap (%), Overlap (%), Flap_deg, AoA_deg)
    unique_keys = []
    selected_rows = []
    for _, row in pool_sorted.iterrows():
        key = (row['AR'], row['Gap (%)'], row['Overlap (%)'], row['Flap_deg'], row['AoA_deg'])
        if key not in unique_keys:
            unique_keys.append(key)
            selected_rows.append(row)
        if len(selected_rows) >= top_n:
            break

    top_df = pd.DataFrame(selected_rows).reset_index(drop=True)

    # tidy numeric rounding
    for c in ['CL','CD','L/D','CL_main','CL_flap','CD_induced','CD_profile','Slot_eff']:
        if c in top_df.columns:
            top_df[c] = top_df[c].round(5)

    return top_df

# ------------------------------- Script runner ---------------------------

if __name__ == '__main__':
    # AR_list changed to cover 8.5 through 11.5 with 0.5 step
    AR_list = list(np.arange(8.5, 11.6, 0.5))  # [8.5, 9.0, ..., 11.5]
    gaps = [0.04, 0.05, 0.06]
    overlaps = [0.06, 0.08, 0.10]
    flap_angles = [20, 25, 30, 35]
    alphas = [8, 10, 12, 14, 16]

    # Run sweep
    df = sweep_params(
        AR_list=AR_list,
        gaps=gaps,
        overlaps=overlaps,
        flap_angles=flap_angles,
        alphas=alphas,
        chord_main=1.0,
        chord_flap=0.4,
        e=0.82,
        Re=5e5,
    )

    # Save full sweep and CL vs CD plot
    summarize_and_save(df, csv_path='sweep_results.csv')

    # Select top-10 CL/CD pairs (Pareto-based & ranked)
    try:
        top10 = select_top10_cl_cd(df, top_n=10, use_pareto=True)
        top10.to_csv('top10_cl_cd_pairs.csv', index=False)
        print("\nSaved top-10 CL/CD paired configs to: top10_cl_cd_pairs.csv")
        print(top10[['AR','Gap (%)','Overlap (%)','Flap_deg','AoA_deg','CL','CD','L/D']].to_string(index=False))
        # Save pareto set as well for inspection
        pareto_mask = pareto_front_mask(df[df['Stalled']==False].reset_index(drop=True))
        pareto_df = df[df['Stalled']==False].reset_index(drop=True)[pareto_mask]
        if len(pareto_df) > 0:
            pareto_df.to_csv('top10_cl_cd_pareto.csv', index=False)
            print(f"Saved Pareto set (size={len(pareto_df)}) to: top10_cl_cd_pareto.csv")
    except Exception as e:
        print("Error during top10 selection:", str(e))

# End of file
