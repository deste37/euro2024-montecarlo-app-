import numpy as np

def run_monte_carlo_simulation(shots_a_xg, shots_b_xg, n_simulations=100000):
    """
    Esegue la simulazione Monte Carlo vettorizzata sui tiri delle due squadre.
    """
    n_a = len(shots_a_xg)
    n_b = len(shots_b_xg)
    
    # Se una squadra non ha effettuato tiri
    if n_a == 0:
        goals_a = np.zeros(n_simulations, dtype=int)
    else:
        rand_a = np.random.uniform(0, 1, size=(n_simulations, n_a))
        goals_a = np.sum(rand_a <= np.array(shots_a_xg), axis=1)
        
    if n_b == 0:
        goals_b = np.zeros(n_simulations, dtype=int)
    else:
        rand_b = np.random.uniform(0, 1, size=(n_simulations, n_b))
        goals_b = np.sum(rand_b <= np.array(shots_b_xg), axis=1)
        
    # Calcolo delle probabilità d'esito
    win_a = np.mean(goals_a > goals_b) * 100
    draw = np.mean(goals_a == goals_b) * 100
    win_b = np.mean(goals_a < goals_b) * 100
    
    # Matrice dei risultati esatti (griglia 6x6 da 0 a 5+ gol)
    matrix = np.zeros((6, 6))
    for g_a, g_b in zip(goals_a, goals_b):
        idx_a = min(g_a, 5)
        idx_b = min(g_b, 5)
        matrix[idx_a, idx_b] += 1
        
    matrix = (matrix / n_simulations) * 100
    
    # xPTS (Expected Points)
    xpts_a = (win_a / 100 * 3) + (draw / 100 * 1)
    xpts_b = (win_b / 100 * 3) + (draw / 100 * 1)
    
    return {
        "p_win_a": round(win_a, 1),
        "p_draw": round(draw, 1),
        "p_win_b": round(win_b, 1),
        "xpts_a": round(xpts_a, 2),
        "xpts_b": round(xpts_b, 2),
        "matrix": np.round(matrix, 1),
        "xg_tot_a": round(sum(shots_a_xg), 2),
        "xg_tot_b": round(sum(shots_b_xg), 2)
    }
