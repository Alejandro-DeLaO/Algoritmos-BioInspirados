import numpy as np

# Define the objective function
def objective(x):
    return x**2

# Define the neighborhood function
def perturb(x, step_size):
    return x + np.random.uniform(-step_size, step_size)

# Define the Metropolis acceptance criterion
def metropolis(current, proposed, temperature):
    if proposed < current:
        return True
    else:
        return np.exp((current - proposed) / temperature) > np.random.rand()

# Initialize parameters
initial_temperature = 1000
final_temperature = 1e-9
cooling_rate = 0.9
step_size = 0.1
max_iterations = 10000

# Initialize current solution and temperature
current_solution = np.random.uniform(-10, 10)
temperature = initial_temperature

# Perform simulated annealing
for iteration in range(max_iterations):
    proposed_solution = perturb(current_solution, step_size)
    if metropolis(objective(current_solution), objective(proposed_solution), temperature):
        current_solution = proposed_solution
    temperature *= cooling_rate
    if temperature < final_temperature:
        break

print("Best solution found: ", current_solution)