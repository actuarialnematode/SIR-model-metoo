import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime
from scipy.stats import norm
from sir_models import neg_log_likelihood_fixed_gamma

# calculates the hessian matrix for error bars
def get_hessian(func, x0, args):
    epsilon = 1e-4 # step size for finite diff
    n = len(x0)
    hessian = np.zeros((n, n))

    # need gradient first to compare changes
    grad = approx_fprime(x0, func, epsilon, *args)

    # finite diff for hessian
    for i in range(n):
        x_temp = x0.copy()
        x_temp[i] += epsilon # nudge just one param
        grad_temp = approx_fprime(x_temp, func, epsilon, *args)
        hessian[i, :] = (grad_temp - grad) / epsilon # rise over run basically

    return hessian

# this loop through events and gets the CIs
def calculate_event_statistics(target_events, results_df, daily_incidence, fixed_gamma):
    event_stats = {}

    print("Calculating Standard Errors using Hessian estimation...")

    for name, target_date in target_events.items():
        row = results_df[results_df['start_date'] == target_date]

        if row.empty:
            print(f"Warning: Could not find window starting {target_date} for {name}")
            continue
        
        # grabbing the optimized values from earlier
        beta_est = row.iloc[0]['beta']
        k_est = row.iloc[0]['k']
        
        # recreate the context for the likelihood function so we can derivate it
        start_date = row.iloc[0]['start_date']
        end_date = row.iloc[0]['end_date']
        mask = (daily_incidence.index >= start_date) & (daily_incidence.index < end_date)
        obs_data = daily_incidence[mask].values
        
        days = np.arange(len(obs_data))
        N_window = np.sum(obs_data) * 1.5
        I0_window = max(obs_data[0], 1)

        # calculate hessian at opt
        params = [beta_est, k_est]
        args = (days, obs_data, N_window, I0_window, fixed_gamma)

        try:
            H = get_hessian(neg_log_likelihood_fixed_gamma, params, args)

            # find covariance matrix
            # inverse of hessian gives covariance info linear algebra magic
            cov_matrix = np.linalg.inv(H)
            var_beta = cov_matrix[0, 0] # only really care about beta variance for R0

            # sometimes numerics give tiny neg numbers which ruins sqrt
            if var_beta < 0:
                print(f"Warning: Negative variance for {name}, fit might be unstable.")
                var_beta = 1e-6

            se_beta = np.sqrt(var_beta)

            # calculate R0 SE via error propagation
            # since gamma is fixed its just linear scaling
            r0_est = beta_est / fixed_gamma
            r0_se = se_beta / fixed_gamma

            event_stats[name] = {
                'R0': r0_est,
                'SE': r0_se,
                'N_cases': np.sum(obs_data)
            }
            print(f"  {name}: R0 = {r0_est:.3f} ± {1.96*r0_se:.3f} (95% CI)")

        except Exception as e:
            print(f"Standard Error calculation failed for {name}: {e}")
            
    return event_stats

# z-test for signficant differences
def perform_z_tests(event_stats, comparisons):
    print("\n--- Statistical Significance Tests (Z-Test) ---")

    if not event_stats:
        print("No events successfully analyzed.")
        return

    for name1, name2 in comparisons:
        if name1 in event_stats and name2 in event_stats:
            est1, se1 = event_stats[name1]['R0'], event_stats[name1]['SE']
            est2, se2 = event_stats[name2]['R0'], event_stats[name2]['SE']

            # standard wald test formula
            diff = est1 - est2
            se_diff = np.sqrt(se1**2 + se2**2) # pythagoras for errors
            z_score = diff / se_diff
            p_value = 2 * (1 - norm.cdf(abs(z_score))) # Two-tailed cause idk direction

            sig_label = "**SIGNIFICATIVE**" if p_value < 0.05 else "Non Significative"

            print(f"{name1} vs {name2}:")
            print(f"   Delta R0 = {diff:.3f}, p-value = {p_value:.4f} -> {sig_label}")