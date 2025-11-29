import networkx as nx
import numpy as np

# generates the network topology under power law curve
def generate_powerlaw_network(N, gamma_exp, min_degree=1, max_degree_frac=0.1, seed=None):
    rng = np.random.default_rng(seed)

    # draw degrees from pareto
    raw = rng.pareto(gamma_exp - 1, size=N) + 1.0
    degrees = min_degree * raw

    # capping degrees
    k_max = int(max_degree_frac * N)
    degrees = np.clip(np.round(degrees).astype(int), min_degree, k_max)

    # ensure sum of degrees is even for stubbing latter
    if degrees.sum() % 2 == 1:
        degrees[0] += 1

    # config model connects stubs randomly
    G_multi = nx.configuration_model(degrees, seed=seed)

    # convert to simple graph to strip multi edges
    G = nx.Graph(G_multi)
    # remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


# runs the discerete  time actual sir dynamics on the the graph
def simulate_network_sir_branching(G, beta, gamma, max_steps=200, rng=None, initial_fraction=None): 
    if rng is None:
        rng = np.random.default_rng()

    nodes = np.array(G.nodes())
    N = len(nodes)
    
    # 0 = S, 1 = I, 2 = R
    status = np.zeros(N, dtype=np.int8)

    # initialization of patient 0
    if initial_fraction is None: 
        
        # find a "celebrity" patient zero
        degree_dict = dict(G.degree())
        degrees = np.array(list(degree_dict.values()))
        node_indices = np.array(list(degree_dict.keys()))
        
        # taking the top 0.5% connected users. basically the alyssa milanos of the graph
        threshold = np.percentile(degrees, 99.5) 
        celebrity_candidates = node_indices[degrees >= threshold]
        
        # finally pick someone at random from the elites
        patient_zero = rng.choice(celebrity_candidates)
        status[patient_zero] = 1
        
    else:
        num_initial = int(N * initial_fraction)
        
        if num_initial < 1: 
            num_initial = 1
            
        initial_patients = rng.choice(N, size=num_initial, replace=False)
        status[initial_patients] = 1

    # simulation loop
    
    S_hist, I_hist, R_hist = [], [], []

    # prebuild neighbor lists keyed by node index
    neighbors = {i: np.fromiter(G.neighbors(nodes[i]), dtype=int) for i in range(N)}

    for t in range(max_steps):
        # record counts for plotting later
        s_count = np.sum(status == 0)
        i_count = np.sum(status == 1)
        r_count = np.sum(status == 2)

        S_hist.append(s_count)
        I_hist.append(i_count)
        R_hist.append(r_count)

        # extinction check here
        if i_count == 0:
            break

        new_status = status.copy()
        infected_idx = np.where(status == 1)[0]

        # infection step
        # loop through everyone currently sick
        for u in infected_idx:
            neigh = neighbors[u]
            # find neighbors who are susceptible
            sus_neigh = neigh[status[neigh] == 0]
            if sus_neigh.size == 0:
                continue
            # infection prob of transmission
            inf_flags = rng.random(sus_neigh.size) < beta
            newly_inf = sus_neigh[inf_flags]
            new_status[newly_inf] = 1

        # recovery step
        rec_flags = rng.random(infected_idx.size) < gamma
        new_status[infected_idx[rec_flags]] = 2

        status = new_status

    # pad to max_steps for plotting purposes
    last_S, last_I, last_R = S_hist[-1], I_hist[-1], R_hist[-1]
    remaining = max_steps - len(S_hist)
    if remaining > 0:
        S_hist.extend([last_S] * remaining)
        I_hist.extend([last_I] * remaining)
        R_hist.extend([last_R] * remaining)

    return S_hist, I_hist, R_hist