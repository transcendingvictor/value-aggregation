# %% THE NEW MATRICIAL METHOD <3
import numpy as np
from scipy.optimize import minimize

# Define parameters
N = 5  # number of values
M = 10  # number of actions
Q = 4  # number of agents
# p = 2.5  # value of p

# Construct weight matrices of size N x Q
np.random.seed(20)  # for reproducibility
W = np.random.rand(N, Q)
W = W / W.sum(axis=0)  # normalize columns to sum to 1

# Construct judgement matrices of size N x M
G_plus = 2 * np.random.rand(N, M) - 1  # values between -1 and 1

# Construct G_minus such that it's not an exact opposite of G_plus
# but with the opposite sign constraint without explicit loops
G_minus = np.random.uniform(-1, 1, (N, M))
G_minus = np.where(G_plus * G_minus > 0, -G_minus, G_minus)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
# Print matrices
print(f"Weight matrix W ({N} x {Q}):")
print(W)
print(f"\nPositive judgement matrix G_plus ({N} values x {M} actions):")
print(G_plus)
print(f"\nNegative judgement matrix G_minus ({N} values x {M} actions):")
print(G_minus)
# %%
# Objective function for aggregation based on weights
p = 1  # value of p

def objective_weights(w_S, W, p): # w_S: (N,), W: (N, Q)
    residuals = W - w_S[:, np.newaxis]  # residuals: (N, Q)
    return np.sum(np.abs(residuals)**p)**(1/p)

# Initial guess for w_S
initial_guess = W.mean(axis=1)  # start with equal weights

# Constraint: weights should sum to 1
constraints = {'type': 'eq', 'fun': lambda w_S: np.sum(w_S) - 1}

# Bounds: weights should be non-negative
bounds = [(0, 1)] * N

# Solve the optimization problem
result_weights = minimize(objective_weights, initial_guess, args=(W, p), constraints=constraints,
                          bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
# Print the results
print(f"\nOptimal weights for aggregation based on weights (w_S): with p = {p}")
print(result_weights.x)
# %%
# Objective function for aggregation based on judgement of actions
def objective_judgement(w_S, W, G_plus, G_minus, p): 
    # w_S: (N,), W: (N, Q), G_plus: (N, M), G_minus: (N, M)
    
    G_plus_k = np.dot(W.T, G_plus)  # G_plus_k: (Q, M)
    G_minus_k = np.dot(W.T, G_minus)  # G_minus_k: (Q, M)
    G_plus_S = np.dot(w_S, G_plus)  # G_plus_S: (M,)
    G_minus_S = np.dot(w_S, G_minus)  # G_minus_S: (M,)
    residuals_plus = G_plus_k - G_plus_S  # residuals_plus: (Q, M)
    residuals_minus = G_minus_k - G_minus_S  # residuals_minus: (Q, M)
    return (np.sum(np.abs(residuals_plus)**p) + np.sum(np.abs(residuals_minus)**p))**(1/p)

# Solve the optimization problem
result_judgement = minimize(objective_judgement, initial_guess, args=(W, G_plus, G_minus, p),
                            constraints=constraints, bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})

# Print the results
print(f"\nOptimal weights for aggregation based on judgement of actions (w_S): with p = {p}")
print(result_judgement.x)

# %%
