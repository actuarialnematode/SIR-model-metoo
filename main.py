# import libraries needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from scipy.integrate import odeint
import networkx as nx 

# import definitions from other files
from sir_models import sir_derivatives, get_model_incidence, neg_log_likelihood_fixed_gamma
from network_sim import generate_powerlaw_network, simulate_network_sir_branching
import visualizations as viz
import stats_analysis as stats

# plotting setup so graphs dont look all different
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 12

import warnings
warnings.filterwarnings('ignore') # ignore warnings for now

# create figures folder
figs_dir = Path("figures")
figs_dir.mkdir(parents=True, exist_ok=True)
print(f"--- Figures will be saved to: {figs_dir.resolve()} ---")


# block 2 data preprocessing
# data living in CSV folder, loop through all
csv_dir = Path("CSV")
csv_files = sorted(csv_dir.glob("*.csv"))

if not csv_files:
    raise ValueError(f"no csv files found in {csv_dir}")

df_list = []
for f in csv_files:
    try:
        tmp = pd.read_csv(f)
        tmp["time"] = pd.to_datetime(tmp["time"], errors="coerce")
        df_list.append(tmp)
    except Exception as e:
        print(f"Error reading {f}: {e}")

df_all = pd.concat(df_list, ignore_index=True)

if "hash_vector" not in df_all.columns:
    raise ValueError("column 'hash_vector' missing from data")

# only care about metoo tweets and drop na values
df_all = df_all[df_all["hash_vector"].str.contains("metoo", case=False, na=False)]
df_all = df_all.dropna(subset=["user_id", "time"])

print(f"total #metoo tweets: {len(df_all)}")
print(f"total unique users: {df_all['user_id'].nunique()}")


# block 3 process daily incidence

df_sorted = df_all.sort_values("time")

# grab just the first time a user tweets. SIR assumes you get infected once and thats it
# no reinfection in this model so drop duplicates
first_adopters = df_sorted.drop_duplicates(subset="user_id", keep="first").copy()
first_adopters["date"] = first_adopters["time"].dt.normalize()

daily_incidence = first_adopters.groupby("date")["user_id"].count()

# padding dates with zeros for days with no tweets
full_date_range = pd.date_range(start="2017-10-01", end="2019-12-31", freq="D")
daily_incidence = daily_incidence.reindex(full_date_range, fill_value=0)

print("--- daily incidence ---")
print(f"Data covers {len(daily_incidence)} days")
print(f"Max daily cases: {daily_incidence.max()}")

viz.plot_daily_incidence(daily_incidence, save_path=figs_dir / "01_daily_incidence.png")

# check how many times people actually use the tag
user_activity = df_all['user_id'].value_counts()
frequency_dist = user_activity.value_counts().sort_index()

viz.plot_frequency_distributions(frequency_dist, save_dir=figs_dir)

# checking variance vs mean. if var > mean then overdispersion present
mean_inc = daily_incidence.mean()
var_inc  = daily_incidence.var()

print("mean daily incidence:", mean_inc)
print("variance of daily incidence:", var_inc)
print("variance / mean (dispersion ratio):", var_inc / mean_inc)


# PART A) DETERMINISTIC SIR MODEL PARAMETER ESTIMATION

# block 6 profile likelihood scan

# define windows to slice the data in 2 months chunks
window_starts = pd.date_range(start="2017-10-01", end="2019-11-01", freq="2MS")

# find best gamma by trying [0.2, 0.8] first
gamma_grid = np.linspace(0, 0.4, 20) 
profile_scores = []

print(f"--- Running Profile Likelihood on {len(gamma_grid)} Gamma points ---")

for g_val in gamma_grid:
    total_nll = 0
    valid_windows = 0
    
    # loops over every window
    for start_date in window_starts:
        end_date = start_date + pd.DateOffset(months=2)
        mask = (daily_incidence.index >= start_date) & (daily_incidence.index < end_date)
        window_data = daily_incidence[mask]
        
        # skip empty windows
        if len(window_data) < 10 or window_data.sum() < 50:
            continue
            
        obs_data = window_data.values
        days = np.arange(len(obs_data))
        
        # setting N slightly higher than observed sum. 1.5x factor is a heuristic
        # assumes hidden population
        N_window = np.sum(obs_data) * 1.5
        I0_window = max(obs_data[0], 1)
        
        # optimize beta and k for THIS window and THIS fixed gamma
        res = minimize(
            neg_log_likelihood_fixed_gamma,
            [0.5, 1.0], 
            args=(days, obs_data, N_window, I0_window, g_val),
            method='Nelder-Mead',
            bounds=[(1e-5, 5.0), (1e-2, 100.0)]
        )
        
        if res.success:
            total_nll += res.fun
            valid_windows += 1
            
    profile_scores.append(total_nll)
    print(f"Gamma: {g_val:.3f} | Total NLL: {total_nll:.2f}")

# search for optimal gamma based on total score
best_idx = np.argmin(profile_scores)
best_gamma = gamma_grid[best_idx]

print(f"\n--- Optimization Complete ---")
print(f"Optimal Global Gamma: {best_gamma:.4f}")
print(f"Implied Infectious Period: {1/best_gamma:.2f} days")

viz.plot_profile_likelihood(gamma_grid, profile_scores, best_gamma, save_path=figs_dir / "03_profile_likelihood.png")


# block 7 run final fit using the optimal gamma found above

print(f"--- Running Final Window-by-Window Fit with Gamma = {best_gamma:.4f} ---")

results = []

for start_date in window_starts:
    end_date = start_date + pd.DateOffset(months=2)

    mask = (daily_incidence.index >= start_date) & (daily_incidence.index < end_date)
    window_data = daily_incidence[mask]

    # same filtering as before
    if len(window_data) < 10 or window_data.sum() < 50:
        continue

    obs_data = window_data.values
    days = np.arange(len(obs_data))

    # initial conditions setup
    N_window = np.sum(obs_data) * 1.5
    I0_window = max(obs_data[0], 1)

    initial_guess = [0.5, 1.0]

    # re-optimizing but keeping gamma fixed to the global best
    res = minimize(
        neg_log_likelihood_fixed_gamma,
        initial_guess,
        args=(days, obs_data, N_window, I0_window, best_gamma),
        method='Nelder-Mead',
        bounds=[(1e-5, 5.0), (1e-2, 100.0)]
    )

    if res.success:
        beta_est, k_est = res.x
        r0_est = beta_est / best_gamma
        
        # saving all metrics for plotting later
        results.append({
            'start_date': start_date,
            'end_date': end_date,
            'beta': beta_est,
            'gamma': best_gamma, 
            'k': k_est,
            'R0': r0_est,
            'total_cases': np.sum(obs_data)
        })
        print(f"Window {start_date.date()}: Beta={beta_est:.3f}, R0={r0_est:.2f}")

results_df = pd.DataFrame(results)


# block 8 visualize R0 over windows

if not results_df.empty:
    viz.plot_r0_evolution(results_df, best_gamma, save_path=figs_dir / "04_r0_evolution.png")
    # quick peek at the table
    print(results_df[['start_date', 'beta', 'R0', 'k']])
else:
    print("No valid results to plot. something went wrong")


# block 9: comparison of fits

if not results_df.empty:
    viz.plot_comparison_of_fits(results_df, daily_incidence, best_gamma, save_path=figs_dir / "05_comparison_fits.png")
else:
    print("No results_df available to plot.")


# block 10: two example SIR compartments

viz.plot_example_compartments(results_df, daily_incidence, best_gamma, save_path=figs_dir / "06_example_compartments.png")
viz.plot_sir_compartments_with_annotations(results_df, daily_incidence, best_gamma, save_path=figs_dir / "06b_annotated_compartments.png")

# block 12b: statistical comparison of specific events (Wald Test)

print("--- Event-Based Statistical Comparison ---")

# specific dates we care about. weinstein start vs kavanaugh vs the quiet period in between
target_events = {
    "Weinstein (Oct '17)": pd.Timestamp("2017-10-01"),
    "Période Calme (Avr '18)": pd.Timestamp("2018-04-01"),
    "Kavanaugh (Oct '18)": pd.Timestamp("2018-10-01")
}

# calc SE and stats. need covariance matrix from hessian probably handled inside helpar
event_stats = stats.calculate_event_statistics(
    target_events, 
    results_df, 
    daily_incidence, 
    best_gamma
)

# comparisons and Z tests to see if R0 diff is significant
comparisons = [
    ("Weinstein (Oct '17)", "Kavanaugh (Oct '18)"),
    ("Weinstein (Oct '17)", "Période Calme (Avr '18)"),
    ("Kavanaugh (Oct '18)", "Période Calme (Avr '18)")
]

stats.perform_z_tests(event_stats, comparisons)

# plot the bars with error whiskers
viz.plot_statistical_comparison(event_stats, save_path=figs_dir / "06c_statistical_comparison.png")


# PART B) STOCHASTIC NETWORK SIMULATIONS

# EXECUTION 1 - HOMOGENEOUS SIMULATION
# this is the naive approach assuming average degree describes everything (nope)

print("Starting Stochastic Network Simulation (Mean Field / Naive)...")

alpha_exp = 2.276 # pulled 2.276 from literature review
seed_net = 123
factor_scale = 1.0
num_simulations = 100 # run 100 times to get rough behavior
max_time_steps = 60
initial_pct = None

parameter_sets = [
    ("Set A (Weinstein-like)", 0.473145, 0.147, "2017-10-01"),
    ("Set B (Kavanaugh-like)", 0.313789, 0.147, "2018-10-01")
]

naive_results = []

for idx, (set_name, b_ode, g_ode, date_str) in enumerate(parameter_sets):
    
    start_date = pd.Timestamp(date_str)
    end_date = start_date + pd.DateOffset(months=2)
    
    # filter daily_incidence for this window
    # estimating N based on observed cases again
    if 'daily_incidence' in locals():
        mask = (daily_incidence.index >= start_date) & (daily_incidence.index < end_date)
        obs_count = daily_incidence[mask].sum()
        N_sim = int(obs_count * 1.5)
    else:
        print("Warning: daily_incidence not found. Using default N=20000.")
        N_sim = 20000
        obs_count = 13333
    
    print(f"\n--- {set_name} ---")
    print(f"  Window: {start_date.date()} to {end_date.date()}")
    print(f"  Observed Cases: {obs_count} -> Network Size (N): {N_sim}")

    # generate new network for N using config model
    G_twitter = generate_powerlaw_network(
        N=N_sim,
        gamma_exp=alpha_exp,
        min_degree=2,
        max_degree_frac=0.05,
        seed=seed_net 
    )

    avg_degree = 2 * G_twitter.number_of_edges() / G_twitter.number_of_nodes()
    print(f"  Network generated: {G_twitter.number_of_nodes()} nodes, avg degree ~ {avg_degree:.2f}")

    # figuring out who patient zero is
    # selecting from top 0.5% degree nodes. basically alyssa milano tier users
    all_degrees = np.array([d for n, d in G_twitter.degree()])
    threshold_deg = np.percentile(all_degrees, 99.5)
    hub_degrees = all_degrees[all_degrees >= threshold_deg]
    
    print(f"  [Patient Zero Analysis]")
    print(f"  Selection Pool: Top 0.5% (The Celebrity Tier)")
    print(f"  Threshold Degree: > {threshold_deg:.0f}")
    print(f"  Possible Degrees for Patient Zero: Min={hub_degrees.min()}, Max={hub_degrees.max()}, Mean={hub_degrees.mean():.1f}")

    # map ODE beta to per-edge infection prob
    # naive mapping: just dividing beta by avg degree
    beta_network = factor_scale * b_ode / max(avg_degree, 1.0)
    gamma_network = g_ode
    
    print(f"  Simulating: Beta_net={beta_network:.4f} (Naive Formulation)...")

    # run simulations
    rng_sim = np.random.default_rng(idx)
    
    S_trajectories = []
    I_trajectories = []
    R_trajectories = []

    for i in range(num_simulations):
        # branching process simulation on the graph
        S_net, I_net, R_net = simulate_network_sir_branching(
            G=G_twitter,
            beta=beta_network,
            gamma=gamma_network,
            max_steps=max_time_steps,
            rng=rng_sim,
            initial_fraction=initial_pct
        )
        S_trajectories.append(S_net)
        I_trajectories.append(I_net)
        R_trajectories.append(R_net)

    S_matrix = np.array(S_trajectories)
    I_matrix = np.array(I_trajectories)
    R_matrix = np.array(R_trajectories)
    
    naive_results.append((set_name, beta_network, S_matrix, I_matrix, R_matrix))

viz.plot_naive_simulation(naive_results, save_path=figs_dir / "07_simulation_naive.png")


# EXECUTION 2 - HETEROGENEOUS SIMULATION
# fixes the hub issue with hetero correction

print("Starting Stochastic Network Simulation (HeMF Corrected)...")

alpha_exp = 2.276
seed_net = 123
factor_scale = 1.0
num_simulations = 100
max_time_steps = 60
initial_pct = 0.01

parameter_sets = [
    ("Set A (Weinstein-like)", 0.473145, 0.147, "2017-10-01"),
    ("Set B (Kavanaugh-like)", 0.313789, 0.147, "2018-10-01")
]

# accumulator for simulation data
full_sir_results = []
i_only_results = []

for idx, (set_name, b_ode, g_ode, date_str) in enumerate(parameter_sets):
    
    start_date = pd.Timestamp(date_str)
    end_date = start_date + pd.DateOffset(months=2)
    
    # filter daily_incidence for this window
    if 'daily_incidence' in locals():
        mask = (daily_incidence.index >= start_date) & (daily_incidence.index < end_date)
        obs_count = daily_incidence[mask].sum()
        N_sim = int(obs_count * 1.5)
    else:
        print("Warning: daily_incidence not found. Using default N=20000.")
        N_sim = 20000
        obs_count = 13333
    
    print(f"\n--- {set_name} ---")
    print(f"  Window: {start_date.date()} to {end_date.date()}")
    print(f"  Observed Cases: {obs_count} -> Network Size (N): {N_sim}")

    # generate new network for N
    G_twitter = generate_powerlaw_network(
        N=N_sim,
        gamma_exp=alpha_exp,
        min_degree=2,
        max_degree_frac=0.05,
        seed=seed_net
    )

    all_degrees = np.array([d for n, d in G_twitter.degree()])
    moment_1 = np.mean(all_degrees) # <k>
    moment_2 = np.mean(all_degrees**2) # <k^2>
    
    # correct with kappa (connectivity factor)
    connectivity_factor = (moment_2 - moment_1) / moment_1
    
    print(f"  Correction Factor (Kappa): {connectivity_factor:.2f}")

    # map ODE beta to per-edge infection prob but using kappa this time
    beta_network = (factor_scale * b_ode) / connectivity_factor
    beta_network = min(beta_network, 1.0) # cant have prob > 1
    gamma_network = g_ode
    
    print(f"  Simulating: Beta_net={beta_network:.5f}")

    # run simulations AND COLLECT DATA
    rng_sim = np.random.default_rng(idx)
    
    # store trajectories
    S_trajectories = []
    I_trajectories = []
    R_trajectories = [] 

    for i in range(num_simulations):
        S_net, I_net, R_net = simulate_network_sir_branching(
            G=G_twitter,
            beta=beta_network,
            gamma=gamma_network,
            max_steps=max_time_steps,
            rng=rng_sim,
            initial_fraction=initial_pct 
        )
        S_trajectories.append(S_net)
        I_trajectories.append(I_net)
        R_trajectories.append(R_net)
    
    S_matrix = np.array(S_trajectories)
    I_matrix = np.array(I_trajectories)
    R_matrix = np.array(R_trajectories)
    full_sir_results.append((set_name, beta_network, S_matrix, I_matrix, R_matrix))
    
    i_only_results.append((set_name, I_matrix))


viz.plot_hemf_sir_components(full_sir_results, save_path=figs_dir / "08a_simulation_hemf_full_sir.png")
viz.plot_hemf_simulation(i_only_results, save_path=figs_dir / "08b_simulation_hemf_zoomed.png")