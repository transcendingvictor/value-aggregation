#%%
import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective(vars):
    x, y = vars
    p = 222.2  # Replace with your desired value of p
    term_x = np.abs(x - 0.8)**p + np.abs(x - 0.35)**p + np.abs(x - 0.5)**p
    term_y = np.abs(y - 0.2)**p + np.abs(y - 0.65)**p + np.abs(y - 0.5)**p
    return (term_x + term_y)**(1/p)

# Initial guess for x and y
initial_guess = [0.9, 0.9]

# Run the optimization
result = minimize(objective, initial_guess, method='BFGS')

# Optimal values
x_optimal, y_optimal = result.x
print("Optimal x:", x_optimal)
print("Optimal y:", y_optimal)
# %%  now for 3 values
def objective(vars):
    x, y, z = vars
    p = 2.2  # Replace with your desired value of p
    term_x = np.abs(x - 0.6)**p + np.abs(x - 0.15)**p + np.abs(x - 0.3)**p
    term_y = np.abs(y - 0.2)**p + np.abs(y - 0.45)**p + np.abs(y - 0.3)**p
    term_z = np.abs(z - 0.2)**p + np.abs(z - 0.40)**p + np.abs(z - 0.4)**p
    return (term_x + term_y + term_z)**(1/p)

# Initial guess for x and y
initial_guess = [0.6, 0.3, 0.8]

# Run the optimization
result = minimize(objective, initial_guess, method='BFGS')

# Optimal values
x_optimal, y_optimal, z_optimal = result.x
print("Optimal x:", x_optimal)
print("Optimal y:", y_optimal)
print("Optimal z:", z_optimal)
# %% New method.
import numpy as np
from scipy.optimize import minimize

# Define the objective function for three variables
def objective(vars):
    x, y, z = vars
    p = 2.2  # Replace with your desired value of p
    weights = np.array([[0.6, 0.2, 0.2], [0.15, 0.45, 0.40], [0.3, 0.3, 0.4]])  # Example weights for agents
    term = 0
    for w in weights:
        term += np.abs(x - w[0])**p + np.abs(y - w[1])**p + np.abs(z - w[2])**p
    return term**(1/p)

# Initial guess for x, y, and z
initial_guess = [0.6, 0.3, 0.8]

# Define the constraint that weights sum to 1 (if applicable)
constraints = {'type': 'eq', 'fun': lambda vars: np.sum(vars) - 1}

# Run the optimization
# result = minimize(objective, initial_guess, method='SLSQP', constraints=[constraints])
result = minimize(objective, initial_guess, method='SLSQP')

# Optimal values
x_optimal, y_optimal, z_optimal = result.x
print("Optimal x:", x_optimal)
print("Optimal y:", y_optimal)
print("Optimal z:", z_optimal)