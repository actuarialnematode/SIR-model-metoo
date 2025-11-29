import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sir_models import sir_derivatives

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 12

def handle_plot(save_path):
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
        plt.close()
    else:
        plt.show()

# plotting the raw daily counts
def plot_daily_incidence(daily_incidence, save_path=None):
    plt.figure(figsize=(12,4))
    plt.plot(daily_incidence.index, daily_incidence.values, color='black', linewidth=1)
    plt.title("Incidence quotidienne des premiers adopteurs de #MeToo")
    plt.ylabel("Nouveaux utilisateurs")
    handle_plot(save_path)

# histoograms and log-log plots to check if user activity follows a power law
def plot_frequency_distributions(frequency_dist, save_dir=None):
    # barplot
    limit = 20
    plt.figure(figsize=(10, 6))
    plt.bar(frequency_dist.index[:limit], frequency_dist.values[:limit], color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(f"Distribution de la fréquence d'utilisation (1 à {limit} utilisations)")
    plt.xlabel("Nombre d'utilisations du hashtag")
    plt.ylabel("Nombre d'utilisateurs")
    plt.xticks(range(1, limit + 1))
    plt.grid(axis='y', alpha=0.3)
    
    if save_dir:
        handle_plot(save_dir / "02a_frequency_bar.png")
    else:
        plt.show()

    # power law plot
    plt.figure(figsize=(10, 6))
    plt.loglog(frequency_dist.index, frequency_dist.values, marker='.', linestyle='none', color='darkred', alpha=0.5)
    plt.title("Distribution de la fréquence d'utilisation (échelle log-log)")
    plt.xlabel("Nombre d'utilisations (log)")
    plt.ylabel("Nombre d'utilisateurs (log)")
    plt.grid(True, which="both", ls="-", alpha=0.2)

    if save_dir:
        handle_plot(save_dir / "02b_frequency_loglog.png")
    else:
        plt.show()

# convexity check for the likelihood scan to confirm where the optimal gamma actually sits
def plot_profile_likelihood(gamma_grid, profile_scores, best_gamma, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(gamma_grid, profile_scores, marker='o', color='darkblue')
    plt.axvline(best_gamma, color='red', linestyle='--', label=f'Gamma optimal : {best_gamma:.3f}')
    plt.title("Vraisemblance profilée pour gamma (agrégée sur toutes les fenêtres)")
    plt.xlabel("Gamma (taux de récupération)")
    plt.ylabel("Log-vraisemblance négative totale")
    plt.legend()
    handle_plot(save_path)

# plotting r0 changes over time with volume on the back to see correlations
def plot_r0_evolution(results_df, best_gamma, save_path=None):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    color = 'tab:red'
    ax1.set_xlabel('Date de début de fenêtre')
    ax1.set_ylabel('R0 estimé (Beta/Gamma)', color=color, fontsize=14)
    ax1.plot(results_df['start_date'], results_df['R0'], color=color, marker='o', linewidth=2, label=f'R0 (gamma fixé = {best_gamma:.3f})')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Volume total dans la fenêtre', color=color, fontsize=14)
    ax2.bar(results_df['start_date'], results_df['total_cases'], color=color, alpha=0.3, width=20, label='Volume de tweets')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Évolution de R0 avec gamma optimisé par profil ({best_gamma:.3f})")
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    handle_plot(save_path)

# comparing the shape of specific outbreaks side by side
def plot_comparison_of_fits(results_df, daily_incidence, best_gamma, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # plot 1: aligned infection curves
    key_events = {
        pd.Timestamp("2017-10-01"): {
            "label": "Weinstein", 
            "color": "tab:red", 
            "alpha": 1.0, 
            "zorder": 10,
            "offset": (-100, 0)
        },
        pd.Timestamp("2018-10-01"): {
            "label": "Kavanaugh", 
            "color": "tab:blue", 
            "alpha": 1.0, 
            "zorder": 10,
            "offset": (-100, 0)
        }
    }

    for index, row in results_df.iterrows():
        beta = row['beta']
        gamma = best_gamma 

        mask = (daily_incidence.index >= row['start_date']) & (daily_incidence.index < row['end_date'])
        obs_data = daily_incidence[mask].values
        days = np.arange(len(obs_data))
        
        if len(days) < 10: continue

        # quick model simulation
        N_window = np.sum(obs_data) * 1.5
        I0_window = max(obs_data[0], 1)
        S0_window = N_window - I0_window
        
        sol = odeint(sir_derivatives, [S0_window, I0_window, 0], days, args=(beta, gamma, N_window))
        active_cases = sol[:, 1] # I(t)
        
        start_dt = row['start_date']
        style = key_events.get(start_dt, {"label": None, "color": "gray", "alpha": 0.15, "zorder": 1})
        
        ax1.plot(days, active_cases, color=style['color'], alpha=style['alpha'], 
                 label=style['label'], linewidth=2 if style['label'] else 1, zorder=style['zorder'])

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax1.set_title("Comparaison des formes d'épidémie (alignées au jour 0)")
    ax1.set_xlabel("Jours depuis le début de la fenêtre")
    ax1.set_ylabel("Cas actifs modélisés (I)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 60)

    # plot 2 : volume and R0
    
    sizes = results_df['total_cases'] / results_df['total_cases'].max() * 500
    
    sc = ax2.scatter(results_df['total_cases'], results_df['R0'], 
                     s=sizes, c=results_df['beta'], cmap='plasma', alpha=0.7, edgecolors='black')
    
    for dt, props in key_events.items():
        row = results_df[results_df['start_date'] == dt]
        if not row.empty:
            ax2.annotate(props['label'], 
                         (row['total_cases'].values[0], row['R0'].values[0]),
                         xytext=props['offset'],
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
                         ha='right',
                         va='center',
                         fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=props['color'], alpha=0.8))

    ax2.set_xscale('log') 
    ax2.set_xlabel("Volume total de tweets (échelle log)")
    ax2.set_ylabel("Nombre de reproduction ($R_0$)")
    ax2.set_title("Carte d'intensité épidémique : viralité vs ampleur")
    ax2.axhline(1.0, linestyle='--', color='gray', label='Seuil ($R_0=1$)')
    ax2.grid(True, alpha=0.3, which="both")
    

    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label(r"Taux de transmission ($\beta$)")

    plt.tight_layout()
    handle_plot(save_path)

# plotting the fitted sir curves for the two main events to check model realism
def plot_example_compartments(results_df, daily_incidence, best_gamma, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # weinstein oct 2017
    target_date_1 = pd.Timestamp("2017-10-01")
    window_row_1 = results_df[results_df['start_date'] == target_date_1]

    if not window_row_1.empty:
        row = window_row_1.iloc[0]
        beta = row['beta']
        gamma = best_gamma
        
        mask = (daily_incidence.index >= row['start_date']) & (daily_incidence.index < row['end_date'])
        window_data = daily_incidence[mask]
        obs_data = window_data.values
        days = np.arange(len(obs_data))
        
        # quick simulation run
        N_window = np.sum(obs_data) * 1.5
        I0_window = max(obs_data[0], 1)
        S0_window = N_window - I0_window
        sol = odeint(sir_derivatives, [S0_window, I0_window, 0], days, args=(beta, gamma, N_window))
        
        ax1.plot(window_data.index, sol[:, 0], color='blue', label='Susceptibles', linestyle='--')
        ax1.plot(window_data.index, sol[:, 1], color='red', label='Infectés (actifs)', linewidth=3)
        ax1.plot(window_data.index, sol[:, 2], color='green', label='Rétablis', linestyle='--')
        
        ax1.set_title(f"Zoom : première vague (Weinstein)\n($R_0$={row['R0']:.2f}, $\\beta$={beta:.2f})")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Utilisateurs")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "Fenêtre Weinstein introuvable", ha='center', transform=ax1.transAxes)

    # kavanaugh oct 2018
    target_date_2 = pd.Timestamp("2018-10-01")
    window_row_2 = results_df[results_df['start_date'] == target_date_2]

    if not window_row_2.empty:
        row = window_row_2.iloc[0]
        beta = row['beta']
        gamma = best_gamma
        
        mask = (daily_incidence.index >= row['start_date']) & (daily_incidence.index < row['end_date'])
        window_data = daily_incidence[mask]
        obs_data = window_data.values
        days = np.arange(len(obs_data))
        
        # quick simulation
        N_window = np.sum(obs_data) * 1.5
        I0_window = max(obs_data[0], 1)
        S0_window = N_window - I0_window
        sol = odeint(sir_derivatives, [S0_window, I0_window, 0], days, args=(beta, gamma, N_window))
        
        ax2.plot(window_data.index, sol[:, 0], color='blue', label='Susceptibles', linestyle='--')
        ax2.plot(window_data.index, sol[:, 1], color='red', label='Infectés (actifs)', linewidth=3)
        ax2.plot(window_data.index, sol[:, 2], color='green', label='Rétablis', linestyle='--')
        
        ax2.set_title(f"Zoom : deuxième vague (Kavanaugh)\n($R_0$={row['R0']:.2f}, $\\beta$={beta:.2f})")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Utilisateurs")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Fenêtre Kavanaugh introuvable", ha='center', transform=ax2.transAxes)

    plt.tight_layout()
    handle_plot(save_path)

# plotting trajectories for the basic network model (homo)
def plot_naive_simulation(simulation_results, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for idx, (set_name, beta_network, S_matrix, I_matrix, R_matrix) in enumerate(simulation_results):
        ax = axes[idx]
        
        num_sims = S_matrix.shape[0]
        
        # plot individual runs
        for i in range(num_sims):
            ax.plot(I_matrix[i, :], color='red', alpha=0.15, linewidth=1)
            ax.plot(S_matrix[i, :], color='black', alpha=0.05, linewidth=1)
            ax.plot(R_matrix[i, :], color='limegreen', alpha=0.05, linewidth=1)

        # legend
        ax.plot([], [], color='red', linewidth=2, label='Infectés')
        ax.plot([], [], color='black', linewidth=1, label='Susceptibles')
        ax.plot([], [], color='limegreen', linewidth=1, label='Rétablis')
        
        ax.set_title(f"{set_name} (champ moyen naïf)\n(Démarrage = hub top 0,5 % , $\\beta_{{net}}$={beta_network:.4f})")
        ax.set_xlabel("Pas de temps")
        
        if idx == 0:
            ax.set_ylabel("Effectif de la population")
            
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    handle_plot(save_path)

# full sir curves for the heterogeneous model to see each compartment
def plot_hemf_sir_components(simulation_results, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # unpack beta for graph labels
    for idx, (set_name, beta_network, S_matrix, I_matrix, R_matrix) in enumerate(simulation_results):
        ax = axes[idx]
        
        num_sims_to_plot = min(S_matrix.shape[0], 100)

        # plot faint indivdual runs for SIR
        for i in range(num_sims_to_plot):
            ax.plot(I_matrix[i, :], color='red', alpha=0.15, linewidth=1)
            ax.plot(S_matrix[i, :], color='black', alpha=0.05, linewidth=1)
            ax.plot(R_matrix[i, :], color='limegreen', alpha=0.05, linewidth=1)
            
        ax.plot([], [], color='red', linewidth=2, label='Infectés')
        ax.plot([], [], color='black', linewidth=1, label='Susceptibles')
        ax.plot([], [], color='limegreen', linewidth=1, label='Rétablis')

        ax.set_title(f"{set_name} (HeMF corrigé)\n(Toutes les composantes SIR, $\\beta_{{net}}$={beta_network:.4f})")
        ax.set_xlabel("Pas de temps")
        
        if idx == 0:
            ax.set_ylabel("Effectif de la population")
            
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    handle_plot(save_path)

# plotting just the infection curves with confidence intervals for the hetero mixing model
def plot_hemf_simulation(simulation_results, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (set_name, I_matrix) in enumerate(simulation_results):
        ax = axes[idx]
        
        # calculate stats
        mean_I = np.mean(I_matrix, axis=0)
        p05_I = np.percentile(I_matrix, 5, axis=0)
        p95_I = np.percentile(I_matrix, 95, axis=0)
        
        # 1. plot individual runs
        for curve in I_matrix[:50]:
            ax.plot(curve, color='red', alpha=0.05, linewidth=1)
            
        # 2. plot mean
        ax.plot(mean_I, color='darkred', linewidth=2.5, label='Infectés moyens')
        
        # 3. plot CI
        ax.fill_between(range(len(mean_I)), p05_I, p95_I, color='red', alpha=0.1, label='IC 95 %')
        
        ax.set_title(f"{set_name} (HeMF corrigé)\n(Zoom sur l'infection - sans 'S')")
        ax.set_xlabel("Pas de temps")
        
        if idx == 0:
            ax.set_ylabel("Infections actives (effectif)")
            
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    

    plt.tight_layout()
    handle_plot(save_path)
    
# plot with text labels pointing out what real world event caused each spike
def plot_sir_compartments_with_annotations(results_df, daily_incidence, fixed_gamma, save_path=None):
    if results_df.empty:
        print("Aucun résultat pour gamma fixe à tracer.")
        return
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    print("Simulation de la dynamique des compartiments pour visualisation...")

    # just store the full infect curve here to find peaks dynamically later
    full_I_series = pd.Series(dtype=float)

    # loop through each window's results
    for index, row in results_df.iterrows():
        beta = row['beta']
        gamma = fixed_gamma 
        start_date = row['start_date']
        end_date = row['end_date']

        mask = (daily_incidence.index >= start_date) & (daily_incidence.index < end_date)
        window_data = daily_incidence[mask]
        if len(window_data) == 0: continue

        obs_data = window_data.values
        days = np.arange(len(obs_data))

        N_window = np.sum(obs_data) * 1.5
        I0_window = max(obs_data[0], 1)
        R0_state = 0
        S0_window = N_window - I0_window - R0_state

        # solve ODe for window
        y0 = [S0_window, I0_window, R0_state]
        sol = odeint(sir_derivatives, y0, days, args=(beta, gamma, N_window))

        window_dates = window_data.index
        
        current_I = pd.Series(sol[:, 1], index=window_dates)
        full_I_series = pd.concat([full_I_series, current_I])

        ax1.plot(window_dates, sol[:, 0], color='blue', alpha=0.6, linewidth=1) # S
        ax2.plot(window_dates, sol[:, 1], color='red', alpha=0.6, linewidth=1)  # I
        ax3.plot(window_dates, sol[:, 2], color='green', alpha=0.6, linewidth=1)# R

    ax1.set_ylabel("Susceptibles (S)")
    ax1.set_title(f"Dynamique de la population susceptible (ajustements, gamma fixé={fixed_gamma:.3f})")
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Infectés/Actifs (I)")
    ax2.set_title("Dynamique des tweeters actifs (infectés) avec événements clés")
    ax2.grid(True, alpha=0.3)

    ax3.set_ylabel("Rétablis (R)")
    ax3.set_title("Dynamique des rétablis/inactifs")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)

    # noteworthy event annotations
    def get_peak_in_range(start_str, end_str):
        subset = full_I_series[start_str:end_str]
        if subset.empty:
            return None, None
        
        peak_date = subset.idxmax()
        peak_val = subset.max()
        return peak_date, peak_val

    pk_date, pk_val = get_peak_in_range('2017-10-01', '2017-12-01')
    if pk_date:
        ax2.annotate('Allégations Weinstein\n(1er pic)',
                     xy=(pk_date, pk_val), 
                     xytext=(pd.Timestamp('2018-01-15'), 3800), 
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    pk_date, pk_val = get_peak_in_range('2018-01-01', '2018-02-01')
    if pk_date:
        ax2.annotate("Golden Globes",
                     xy=(pk_date, pk_val),
                     xytext=(pd.Timestamp('2018-03-01'), 1500),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    pk_date, pk_val = get_peak_in_range('2018-04-01', '2018-05-30')
    if pk_date:
        ax2.annotate('Condamnation Bill Cosby',
                     xy=(pk_date, pk_val),
                     xytext=(pd.Timestamp('2018-02-01'), 2000),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    pk_date, pk_val = get_peak_in_range('2018-09-01', '2018-11-01')
    if pk_date:
        ax2.annotate('Auditions Kavanaugh\n(2e pic)',
                     xy=(pk_date, pk_val),
                     xytext=(pd.Timestamp('2018-06-15'), 2500),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
    plt.tight_layout()
    handle_plot(save_path)

# bar chart with error whiskers to see if the r0 differences matter
def plot_statistical_comparison(event_stats, save_path=None):
    if not event_stats:
        print("Aucune donnée statistique à tracer.")
        return

    names = list(event_stats.keys())
    r0_values = [event_stats[n]['R0'] for n in names]
    r0_errors = [1.96 * event_stats[n]['SE'] for n in names] 

    plt.figure(figsize=(10, 6))

    colors = ['tab:red' if 'Calme' not in n else 'tab:gray' for n in names]
    
    bars = plt.bar(names, r0_values, yerr=r0_errors, capsize=10,
                   color=colors, alpha=0.7)

    plt.ylabel("R0 Estimé (avec IC 95%)")
    plt.title("Comparaison statistique de R0 (Phases clés #MeToo)")
    plt.axhline(1.0, color='black', linestyle='--', linewidth=1, label="Seuil épidémique (R0=1)")
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    handle_plot(save_path)