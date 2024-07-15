import numpy as np
import pandas as pd #エラーが出る場合はpip install pandasでインストール
from scipy.integrate import odeint #エラーが出る場合はpip install scipyでインストール
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

# Load actual data from a CSV file
# Assume the CSV file has columns 'time' and 'infected'
actual_data = pd.read_csv('./data20230505-20200120.csv')
actual_data = actual_data.sort_values('time')
actual_time = actual_data['time'].values
actual_infected_data = actual_data['infected'].values

# Define the time interval for fitting
# 上手くフィットするように適切な値を探してください（それが課題です）
start_index = 0  # Start index for the interval
end_index = 1200    # End index for the interval

# Extract the time and infected data for the specified interval
fit_time = actual_time[start_index:end_index]
fit_infected_data = actual_infected_data[start_index:end_index]

# Define the initial guesses for beta, gamma, S0, I0, and R0
initial_guess = [0.3, 0.1, 990, 10, 0]

# Function to fit the SIR model to the infected data including initial conditions
def fit_sir_model(t, infected_data):
    def sir_fit(t, beta, gamma, S0, I0, R0):
        return sir_odeint(t, beta, gamma, S0, I0, R0)
    
    popt, pcov = curve_fit(sir_fit, t, infected_data, p0=initial_guess, bounds=(0, [1.0, 1.0, np.inf, np.inf, np.inf]))
    beta, gamma, S0, I0, R0 = popt
    return beta, gamma, S0, I0, R0

# Fit the model to the actual data in the specified interval
beta_fit, gamma_fit, S0_fit, I0_fit, R0_fit = fit_sir_model(fit_time, fit_infected_data)

# Print the fitted parameters
print(f"Fitted beta: {beta_fit}")
print(f"Fitted gamma: {gamma_fit}")
print(f"Fitted S0: {S0_fit}")
print(f"Fitted I0: {I0_fit}")
print(f"Fitted R0: {R0_fit}")

# Generate the fitted curve using fit_time
fitted_curve = sir_odeint(fit_time, beta_fit, gamma_fit, S0_fit, I0_fit, R0_fit)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(actual_time, actual_infected_data, 'x', label='Actual Data (Infected)')
plt.plot(fit_time, fit_infected_data, 'o', label='Fit Interval Data (Infected)')
plt.plot(fit_time, fitted_curve, '-', label='Fitted Model (Infected)')
plt.xlabel('Time')
plt.ylabel('Number of Infected Individuals')
plt.legend()
plt.show()
