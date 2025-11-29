import numpy as np
from scipy.integrate import odeint
from scipy.special import gammaln

# definition of SIR ODEs
def sir_derivatives(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = (beta * S * I / N) - (gamma * I)
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# helper func to run simulations and extract the daily incidence
def get_model_incidence(params, t_grid, N, I0):
    beta, gamma = params
    
    # initial conditions setup assuming nobody is recovered at t=0 for simplicity
    R0_state = 0
    S0 = N - I0 - R0_state
    y0 = [S0, I0, R0_state]

    # ODE solver
    sol = odeint(sir_derivatives, y0, t_grid, args=(beta, gamma, N))
    S = sol[:, 0]

    # calculating incidence by checking how many people left S
    incidence = -np.diff(S)
    incidence = np.append(incidence, incidence[-1] if len(incidence) > 0 else 0)

    # avoid 0s
    return np.maximum(incidence, 1e-9)


# obj func to minimize with negbinom loglike
def neg_log_likelihood_fixed_gamma(params, t_grid, data, N, I0, gamma_fixed):
    beta, k = params

    # error catcher
    if beta < 0 or k <= 0:
        return 1e12

    # run the model with current params to get expected mean mu
    mu = get_model_incidence([beta, gamma_fixed], t_grid, N, I0)
    
    # numerical stability checks
    mu = np.maximum(mu, 1e-9)
    k = np.maximum(k, 1e-9)
    y = data + 1e-9

    # log of gamma factorial terms
    term1 = gammaln(y + k)
    term2 = gammaln(k)
    term3 = gammaln(y + 1)
    
    # log of prob terms derived from negbinom pdf
    term4 = k * np.log(k)
    term5 = y * np.log(mu)
    term6 = (k + y) * np.log(k + mu)

    log_lik = term1 - term2 - term3 + term4 + term5 - term6
    
    return -np.sum(log_lik)