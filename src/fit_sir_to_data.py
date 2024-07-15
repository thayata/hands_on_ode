import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the SIR model differential equations
def sir_model(y, t, beta, gamma):
    S, I, R = y
    N = S + I + R
    dS_dt = -beta * S * I / N
    dI_dt = beta * S * I / N - gamma * I
    dR_dt = gamma * I
    return dS_dt, dI_dt, dR_dt

# Function to integrate the SIR equations over time
def sir_odeint(t, beta, gamma, S0, I0, R0):
    y0 = S0, I0, R0
    return odeint(sir_model, y0, t, args=(beta, gamma)).T[1]  # We only need I(t)

# Generate synthetic data (for demonstration purposes)
def generate_synthetic_data(beta, gamma, S0, I0, R0, t):
    result = odeint(sir_model, (S0, I0, R0), t, args=(beta, gamma))
    return result.T[1]  # Return the number of infected individuals

# Define initial conditions
S0 = 990
I0 = 10
R0 = 0
N = S0 + I0 + R0
t = np.linspace(0, 1200, 1200)  # Time grid

# True parameters (for generating synthetic data)
beta_true = 0.3
gamma_true = 0.1

# Generate synthetic infected data
synthetic_infected_data = generate_synthetic_data(beta_true, gamma_true, S0, I0, R0, t)

# Load actual data from a CSV file
# Assume the CSV file has columns 'time' and 'infected'
actual_data = pd.read_csv('data20230505-20200120.csv')
actual_data = actual_data.sort_values('time')
actual_time = actual_data['time'].values
actual_infected_data = actual_data['infected'].values

# Function to fit the SIR model to the infected data
def fit_sir_model(t, infected_data, S0, I0, R0):
    popt, pcov = curve_fit(lambda t, beta, gamma: sir_odeint(t, beta, gamma, S0, I0, R0),
                           t, infected_data, bounds=(0, [1.0, 1.0]))
    beta, gamma = popt
    return beta, gamma

# Fit the model to the actual data
beta_fit, gamma_fit = fit_sir_model(actual_time, actual_infected_data, S0, I0, R0)

# Print the fitted parameters
print(f"Fitted beta: {beta_fit}")
print(f"Fitted gamma: {gamma_fit}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, synthetic_infected_data, 'o', label='Synthetic Data (Infected)')
plt.plot(actual_time, actual_infected_data, 'x', label='Actual Data (Infected)')
plt.plot(t, sir_odeint(t, beta_fit, gamma_fit, S0, I0, R0), '-', label='Fitted Model (Infected)')
plt.xlabel('Time')
plt.ylabel('Number of Infected Individuals')
plt.legend()
plt.show()
