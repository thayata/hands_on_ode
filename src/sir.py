import numpy as np
import pandas as pd #エラーが出る場合はpip install pandasでインストール
from scipy.integrate import odeint #エラーが出る場合はpip install scipyでインストール
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
class SIRModel:
    def __init__(self, start_index=0, end_index=1200, initial_guess=[0.3, 0.1, 990, 10, 0]):
        self.start_index = start_index
        self.end_index = end_index
        self.initial_guess = initial_guess
        self.load_data()
        
    def load_data(self):
        # Load and prepare data
        actual_data = pd.read_csv('./data20230505-20200120.csv')
        actual_data = actual_data.sort_values('time')
        self.actual_time = actual_data['time'].values
        self.actual_infected_data = actual_data['infected'].values
        self.fit_time = self.actual_time[self.start_index:self.end_index]
        self.fit_infected_data = self.actual_infected_data[self.start_index:self.end_index]

    @staticmethod
    def sir_model(y, t, beta, gamma):
        S, I, R = y
        N = S + I + R
        dS_dt = -beta * S * I / N
        dI_dt = beta * S * I / N - gamma * I
        dR_dt = gamma * I
        return dS_dt, dI_dt, dR_dt

    @staticmethod
    def sir_odeint(t, beta, gamma, S0, I0, R0):
        y0 = S0, I0, R0
        return odeint(SIRModel.sir_model, y0, t, args=(beta, gamma)).T[1]

    def fit_sir_model(self):
        def sir_fit(t, beta, gamma, S0, I0, R0):
            return self.sir_odeint(t, beta, gamma, S0, I0, R0)
        
        popt, pcov = curve_fit(sir_fit, self.fit_time, self.fit_infected_data, 
                              p0=self.initial_guess, 
                              bounds=(0, [1.0, 1.0, np.inf, np.inf, np.inf]))
        return popt

    def plot_results(self, fitted_params):
        beta_fit, gamma_fit, S0_fit, I0_fit, R0_fit = fitted_params
        fitted_curve = self.sir_odeint(self.fit_time, beta_fit, gamma_fit, S0_fit, I0_fit, R0_fit)

        plt.figure(figsize=(8, 6))
        plt.plot(self.actual_time, self.actual_infected_data, 'x', label='Actual Data (Infected)')
        #plt.plot(self.fit_time, self.fit_infected_data, 'o', label='Fit Interval Data (Infected)')
        plt.plot(self.fit_time, fitted_curve, '-', label='Fitted Model (Infected)')
        plt.xlabel('Time')
        plt.ylabel('Number of Infected Individuals')
        plt.legend()
        plt.show()

    def run_analysis(self):
        # Fit the model and get parameters
        fitted_params = self.fit_sir_model()
        beta_fit, gamma_fit, S0_fit, I0_fit, R0_fit = fitted_params

        # Print results
        print(f"Fitted beta: {beta_fit}")
        print(f"Fitted gamma: {gamma_fit}")
        print(f"Fitted S0: {S0_fit}")
        print(f"Fitted I0: {I0_fit}")
        print(f"Fitted R0: {R0_fit}")

        # Plot results
        self.plot_results(fitted_params)

if __name__ == '__main__':
    sir_model = SIRModel(start_index=0, end_index=1200, initial_guess=[0.3, 0.1, 990, 10, 0])
    sir_model.run_analysis()