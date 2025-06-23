import numpy as np
import matplotlib.pyplot as plt

def initialize_charges(N):
    r = np.sqrt(np.random.rand(N))  # Uniform on disc
    theta = 2 * np.pi * np.random.rand(N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def compute_energy(x, y):
    energy = 0.0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            energy += 1.0 / np.hypot(dx, dy)
    return 0.5 * energy

def is_within_disc(x, y):
    return x**2 + y**2 <= 1.0

def propose_move(x, y, k, delta, constrain_to_circumference=False):
    if constrain_to_circumference:
        # Propose a move along the circumference
        theta = np.arctan2(y[k], x[k])
        theta_new = theta + delta * np.random.uniform(-1, 1)
        x_new = np.cos(theta_new)
        y_new = np.sin(theta_new)
    else:
        # Randomly choose x or y coordinate
        coord = np.random.choice(['x', 'y'])
        delta_move = delta * np.random.choice([-1, 1])
        
        # Propose new position
        if coord == 'x':
            x_new = x[k] + delta_move
            y_new = y[k]
        else:
            x_new = x[k]
            y_new = y[k] + delta_move
        
        # Check if the new position is within the disc
        if not is_within_disc(x_new, y_new):
            return x[k], y[k]  # Reject the move
    
    return x_new, y_new

def simulated_annealing(x, y, N, delta=0.1, initial_temp=1.0, cooling_rate=0.95, M=1000, constrain_to_circumference=False):
    T = initial_temp
    energy = compute_energy(x, y)
    
    while T > 1e-6:
        for _ in range(M):
            # Randomly select a charge
            k = np.random.randint(N)
            
            # Propose a new move
            x_new, y_new = propose_move(x, y, k, delta, constrain_to_circumference)
            
            # Compute the change in energy
            energy_old = compute_energy(x, y)
            x_old, y_old = x[k], y[k]  # Save old position
            x[k], y[k] = x_new, y_new  # Temporarily update position
            energy_new = compute_energy(x, y)
            delta_W = energy_new - energy_old
            
            # Accept or reject the move
            if delta_W < 0 or np.random.rand() < np.exp(-delta_W / T):
                energy = energy_new  # Accept the move
            else:
                x[k], y[k] = x_old, y_old  # Reject the move
        
        # Reduce the temperature
        T *= cooling_rate
    
    return x, y, energy

def plot_configuration(x, y, title="Charge Configuration"):
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', label="Disc Boundary")
    ax.add_artist(circle)
    ax.scatter(x, y, color='red', label="Charges")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.show()

# Main execution for N=9 (constrain to circumference)
N = 20
x, y = initialize_charges(N)
x_opt, y_opt, energy = simulated_annealing(x, y, N, constrain_to_circumference=False)
plot_configuration(x_opt, y_opt, title=f"Optimal Charge Configuration for N={N}")