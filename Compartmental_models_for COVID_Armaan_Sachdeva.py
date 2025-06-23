import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from scipy.integrate import solve_ivp 
from scipy.optimize import minimize
from scipy.stats import weibull_min, lognorm
from scipy.optimize import curve_fit
from scipy import signal

path = r"C:\Users\singh\Downloads\daily-new-confirmed-covid-19-cases-deaths-per-million-people.csv"
df = pd.read_csv(path)

# Rename columns by index
cols = df.columns.tolist()
cols[0] = 'country'
cols[1] = 'date'
cols[2] = 'daily cases per 1m'
cols[3] = 'daily confirmed deaths per 1m'
df.columns = cols

df.head()

#loading the datasets
vpath=r"C:\Users\singh\Downloads\daily-covid-19-vaccine-doses-administered.csv"
vaccine=pd.read_csv(vpath)
ipath=r"C:\Users\singh\Downloads\weekly-confirmed-covid-19-cases.csv"
cases=pd.read_csv(ipath)
dpath=r"C:\Users\singh\Downloads\weekly-confirmed-covid-19-deaths.csv"
deaths=pd.read_csv(dpath)
#renaming the columns by index
vclos=vaccine.columns.tolist()
vclos[0]="country"
vclos[1]="date"
vclos[2]="daily doses"

ccols=cases.columns.tolist()
ccols[0]="country"
ccols[1]="date"
ccols[2]="weekly cases"

dcols=deaths.columns.tolist()
dcols[0]="country"
dcols[1]="date"
dcols[2]="weekly deaths"
#renaming the columns
vaccine.columns=vclos
cases.columns=ccols
deaths.columns=dcols
#converting the date column to datetime format
vaccine['date']=pd.to_datetime(vaccine['date'])
cases['date']=pd.to_datetime(cases['date'])
deaths['date']=pd.to_datetime(deaths['date'])
#dropping the country column
vaccine=vaccine.drop(columns=["country"])
cases=cases.drop(columns=["country"])
deaths=deaths.drop(columns=["country"])


#plotting the data
plt.figure(figsize=(15, 5))
plt.plot(vaccine['date'],vaccine['daily doses'],label="Daily Doses",color="blue")
plt.plot(cases['date'],cases['weekly cases'],label="Weekly Cases",color="red")
plt.plot(deaths['date'],deaths['weekly deaths'],label="Weekly Deaths",color="green")
plt.title("Covid-19 Vaccination, Cases and Deaths")
plt.xlabel("Date")
plt.ylabel("Count")
plt.legend()
plt.grid()
plt.savefig("covid_vaccination_cases_deaths.png")  # Save the plot
plt.show()
plt.savefig("covid_vaccination_cases_deaths.png")  # Save the plot
# Plotting individual graphs with better layout and larger size
plt.figure(figsize=(18, 12))  

# First subplot - Vaccinations
plt.subplot(3, 1, 1)  
plt.plot(vaccine['date'], vaccine['daily doses'], label="Daily Doses", color="blue", linewidth=2)
plt.title("Covid-19 Vaccination", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Number of Daily Doses", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', labelsize=10)

# Second subplot - Cases
plt.subplot(3, 1, 2)  # 3 rows, 1 column, position 2
plt.plot(cases['date'], cases['weekly cases'], label="Weekly Cases", color="red", linewidth=2)
plt.title("Covid-19 Cases", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Number of Weekly Cases", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', labelsize=10)

# Third subplot - Deaths
plt.subplot(3, 1, 3)  # 3 rows, 1 column, position 3
plt.plot(deaths['date'], deaths['weekly deaths'], label="Weekly Deaths", color="green", linewidth=2)
plt.title("Covid-19 Deaths", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Number of Weekly Deaths", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', labelsize=10)

# Adjust spacing between subplots and ensure no overlap
plt.tight_layout(pad=3.0)
plt.savefig("covid_individual_graphs.png")  # Save the combined subplots
plt.show()

# Cell added to compare SIR model predictions with actual case data

cases['date'] = pd.to_datetime(cases['date'])
actual_weekly_cases_df = cases.set_index('date')
actual_weekly_cases = actual_weekly_cases_df['weekly cases']


# 1. Create dates for the SIR model time 't' aligned with actual data
start_date = actual_weekly_cases.index.min() # Use start date from actual data
if pd.isna(start_date):
    print("Error: Could not determine start date from actual cases data.")
else:
    # Ensure t is a numpy array for proper indexing
    t_days = np.array(t)
    # Generate date range based on the number of days in t
    sir_dates = pd.date_range(start=start_date, periods=len(t_days), freq='D')

    # 2. Calculate cumulative infections from SIR model
    cumulative_infections_sir = S0 - S

    # 3. Create a Series for cumulative infections indexed by date
    sir_cumulative_series = pd.Series(cumulative_infections_sir, index=sir_dates)

    # 4. Resample weekly and get cumulative count at week end
    sir_cumulative_weekly = sir_cumulative_series.resample('W-SUN').last()

    # 5. Calculate weekly new infections (difference in cumulative)
    sir_new_weekly = sir_cumulative_weekly.diff()
    # Fill the first NaN with the value from the first week's cumulative count
    sir_new_weekly.iloc[0] = sir_cumulative_weekly.iloc[0]

    # Combine actual and predicted data, aligning by date index
    comparison_df = pd.DataFrame({
        'actual': actual_weekly_cases,
        'predicted': sir_new_weekly
    })

    # Drop rows where either actual or predicted is NaN (handles non-overlapping periods)
    comparison_df.dropna(inplace=True)

    # Ensure there's data to compare
    if comparison_df.empty:
        print("Error: No overlapping data found between SIR model and actual cases after alignment.")
    else:
        actual = comparison_df['actual']
        predicted = comparison_df['predicted']

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)

        print(f"Comparison Period: {comparison_df.index.min().date()} to {comparison_df.index.max().date()}")
        print(f"RMSE: {rmse:,.2f}") # Format RMSE with commas
        print(f"R-squared: {r2:.4f}")

        plt.figure(figsize=(12, 6))
        plt.plot(comparison_df.index, actual, label='Actual Weekly Cases', marker='o', linestyle='-')
        plt.plot(comparison_df.index, predicted, label='SIR Predicted Weekly Infections', marker='x', linestyle='--')
        plt.title('SIR Model vs Actual Weekly COVID-19 Cases')
        plt.xlabel('Date')
        plt.ylabel('Number of Cases/Infections')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout() 
        plt.show()
        plt.savefig("sir_model_vs_actual_cases.png")  # Save the plot

def sir_model_solve_ivp(t, y, beta_func, gamma_fixed, N_fixed):
    S, I, R = y
    beta_t = beta_func(t) # Get beta value at time t
    
    # Ensure beta is non-negative
    beta_t = max(beta_t, 0)

    dSdt = -beta_t * S * I / N_fixed
    dIdt = beta_t * S * I / N_fixed - gamma_fixed * I
    dRdt = gamma_fixed * I
    return [dSdt, dIdt, dRdt]

def beta_fourier_notebook(t, params, period):
    """
    Calculates beta(t) using Fourier series + baseline.
    params = [a0, a1, b1, a2, b2, ...]
    beta(t) = a0/2 + sum[an*cos(2*pi*n*t/T) + bn*sin(2*pi*n*t/T)]
    """
    a0 = params[0]
    beta_val = a0 / 2.0 # Constant term
    num_terms = (len(params) - 1) // 2
    
    if period > 1e-6: # Avoid division by zero
        for n in range(1, num_terms + 1):
            an = params[2*n - 1]
            bn = params[2*n]
            omega = 2 * np.pi * n / period
            beta_val += an * np.cos(omega * t) + bn * np.sin(omega * t)

    return beta_val

def fit_sir_fourier_notebook(params, period, gamma_fixed, N_fixed, y0_fixed, t_eval_fixed, actual_cases_aligned, start_date_fixed):
    """
    Objective function to minimize RMSE between predicted and actual weekly cases.
    """
    # Define the specific beta function for this optimization step
    def current_beta_func(t):
        return beta_fourier_notebook(t, params, period)

    try:
        # Solve the SIR model using solve_ivp
        t_span = (t_eval_fixed[0], t_eval_fixed[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = solve_ivp(sir_model_solve_ivp, t_span, y0_fixed,
                                 args=(current_beta_func, gamma_fixed, N_fixed),
                                 t_eval=t_eval_fixed, method='RK45', max_step=1.0)

        if solution.status != 0: 
             return np.inf

        S_pred, I_pred, R_pred = solution.y
        
        # Check for NaNs in solution
        if np.isnan(S_pred).any() or np.isnan(I_pred).any() or np.isnan(R_pred).any():
            # print("NaNs in solver output") # Debug
            return np.inf

        # Calculate predicted weekly new infections (change in I+R)
        sim_dates = pd.date_range(start=start_date_fixed, periods=len(t_eval_fixed), freq='D')
        cumulative_infected_recovered = pd.Series(I_pred + R_pred, index=sim_dates)
        
        # Resample weekly (summing change over the week is tricky, using last value diff)
        cumulative_weekly = cumulative_infected_recovered.resample('W-SUN').last()
        predicted_new_weekly = cumulative_weekly.diff().fillna(0) # Fill first NaN
        # Alternative: Fill first week with its cumulative value if diff() is not appropriate
        if not predicted_new_weekly.empty:
             predicted_new_weekly.iloc[0] = cumulative_weekly.iloc[0] - y0_fixed[1] - y0_fixed[2] # Cases in first week

        # Align with actual data
        comparison_df = pd.DataFrame({'actual': actual_cases_aligned, 'predicted': predicted_new_weekly})
        comparison_df.dropna(inplace=True) # Align based on index (dates)

        if comparison_df.empty or len(comparison_df['actual']) < 2:
            # print("No overlapping data") # Debug
            return np.inf

        # Calculate RMSE on weekly new cases
        rmse = np.sqrt(mean_squared_error(comparison_df['actual'], comparison_df['predicted']))

        # Penalize if beta becomes negative during the simulation
        simulated_betas = [current_beta_func(t) for t in t_eval_fixed]
        # More aggressive penalty if beta goes significantly negative
        penalty = np.sum(np.maximum(0, -np.array(simulated_betas))) * 1e6 
        
        # print(f"Params: {params[:3]}... RMSE: {rmse:.2f} Penalty: {penalty:.2f}") # Debug
        
        if not np.isfinite(rmse + penalty):
             # print("Non-finite result") # Debug
             return np.inf
             
        return rmse + penalty

    except Exception as e:
        # print(f"Error in objective function: {e}") # Debug
        return np.inf # Return infinity if any error occurs

print("\n Fitting Fourier Series Beta Model")

try:
    # Check if necessary variables exist
    if 'N' not in locals() or 'gamma' not in locals() or 'y0' not in locals() or \
       't' not in locals() or 'actual_weekly_cases' not in locals() or \
       'start_date' not in locals():
        raise NameError("One or more required variables (N, gamma, y0, t, actual_weekly_cases, start_date) not defined")

    num_fourier_terms = 2 # Number of Fourier terms (k=1 to n_terms) 
    num_params_fourier = 2 * num_fourier_terms + 1 # a0, a1, b1, a2, b2, ...

    # Set the fundamental period - Use 365 days or data duration
    period_fourier = 365.0 # Or max(365.0, t[-1] - t[0])

    # Initial guess: [a0, a1, b1, a2, b2, ...]
    # Use previous beta0 * 2 as guess for a0? Or estimate average beta.
    # Let's guess a0 around 0.2, small values for others.
    initial_guess_fourier = [0.2] + [0.01] * (num_params_fourier - 1)

    # Bounds: [(a0_min, a0_max), (a1_min, a1_max), (b1_min, b1_max), ...]
    # Bound a0 (related to average beta) to be positive
    # Allow other coefficients to be negative/positive but maybe bounded
    bounds_fourier = [(0.01, 1.0)] + [(-0.5, 0.5)] * (num_params_fourier - 1)

    print(f"Optimizing {num_params_fourier} parameters ({num_fourier_terms} Fourier terms) with period {period_fourier} days...")
    print(f"Initial Guess: {initial_guess_fourier}")

    # Wrap the objective function
    objective_wrapped = lambda p: fit_sir_fourier_notebook(p, period_fourier, gamma, N, y0, t, actual_weekly_cases, start_date)

    # Run the minimizer (using L-BFGS-B as in snippet, might need Nelder-Mead if it fails)
    result_fourier = minimize(objective_wrapped,
                              initial_guess_fourier,
                              method='L-BFGS-B', # Or 'Nelder-Mead'
                              bounds=bounds_fourier, # Only used by L-BFGS-B, TNC, SLSQP, etc.
                              options={'maxiter': 500, 'ftol': 1e-7, 'disp': True}) # Show convergence messages

    if result_fourier.success or "CONVERGENCE" in result_fourier.message.upper(): # Accept convergence messages too
        optimal_params_fourier = result_fourier.x
        min_rmse_penalty = result_fourier.fun

        print(f"\nOptimization Successful (or converged)!")
        print(f"Optimal Fourier Params (a0, a1, b1,...): {optimal_params_fourier}")
        print(f"Final Objective Function Value (RMSE + Penalty): {min_rmse_penalty:.4f}")
        def optimal_beta_func_fourier(t):
            return beta_fourier_notebook(t, optimal_params_fourier, period_fourier)

        # Solve the SIR model with the optimal Fourier beta
        t_span_final = (t[0], t[-1])
        solution_final = solve_ivp(sir_model_solve_ivp, t_span_final, y0,
                                   args=(optimal_beta_func_fourier, gamma, N),
                                   t_eval=t, method='RK45', max_step=1.0)

        S_opt, I_opt, R_opt = solution_final.y

        # Calculate final predicted weekly infections
        sim_dates_opt = pd.date_range(start=start_date, periods=len(t), freq='D')
        cumulative_infected_recovered_opt = pd.Series(I_opt + R_opt, index=sim_dates_opt)
        cumulative_weekly_opt = cumulative_infected_recovered_opt.resample('W-SUN').last()
        predicted_new_weekly_opt = cumulative_weekly_opt.diff().fillna(0)
        if not predicted_new_weekly_opt.empty:
             predicted_new_weekly_opt.iloc[0] = cumulative_weekly_opt.iloc[0] - y0[1] - y0[2]

        # Align data
        comparison_df_opt = pd.DataFrame({'actual': actual_weekly_cases, 'predicted_optimal': predicted_new_weekly_opt})
        comparison_df_opt.dropna(inplace=True)

        if not comparison_df_opt.empty:
            actual_opt = comparison_df_opt['actual']
            predicted_opt = comparison_df_opt['predicted_optimal']
            rmse_final = np.sqrt(mean_squared_error(actual_opt, predicted_opt))
            r2_final = r2_score(actual_opt, predicted_opt)
            print(f"\n Metrics with Optimal Fourier Parameters ")
            print(f"RMSE (Weekly Cases): {rmse_final:,.2f}")
            print(f"R-squared: {r2_final:.4f}")

            # Plotting
            plt.figure(figsize=(14, 12))

            # Plot 1: Actual vs Model Weekly Cases
            plt.subplot(3, 1, 1)
            plt.plot(comparison_df_opt.index, actual_opt, label='Actual Weekly Cases', marker='o', linestyle='-', markersize=4)
            plt.plot(comparison_df_opt.index, predicted_opt, label=f'Optimized Fourier SIR (n={num_fourier_terms})', marker='x', linestyle='--', markersize=4)
            plt.title('Actual vs Optimized Fourier SIR Model (Weekly New Cases)')
            plt.ylabel('Weekly New Cases')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)

            # Plot 2: Optimized Beta(t)
            plt.subplot(3, 1, 2)
            simulated_beta_values_opt = [optimal_beta_func_fourier(ti) for ti in t]
            sim_dates_plot = pd.date_range(start=start_date, periods=len(t), freq='D') # Dates for plotting beta
            plt.plot(sim_dates_plot, simulated_beta_values_opt, color='green', label=f'Optimized Fourier Beta(t) (n={num_fourier_terms})')
            plt.title('Optimized Fourier Series Transmission Rate Beta(t)')
            plt.xlabel('Date')
            plt.ylabel('Beta(t)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)

            # Plot 3: Infected Compartment (Model)
            plt.subplot(3, 1, 3)
            plt.plot(sim_dates_plot, I_opt, label='Infected (Model)', color='orange')
            plt.title('Infected Population (Optimized Fourier Model)')
            plt.xlabel('Date')
            plt.ylabel('Number Infected')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()
            plt.savefig("optimized_fourier_sir_model.png")  # Save the plot

        else:
            print("\nError: No overlapping data found for final evaluation after optimization.")

    else:
        print(f"\nFourier Beta optimization failed: {result_fourier.message}")
        print(f"Final parameters attempted: {result_fourier.x}")
        print(f"Final function value: {result_fourier.fun}")

except NameError as e:
    print(f"Error: {e}. Please ensure all required variables and data are loaded/defined in previous cells.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
# === SEIRDV Model with Fourier Series for Beta and Mu Optimization ===

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

def seirdv_model_mu_opt(t, y, beta_func, sigma, gamma, mu_func, nu_func, N):
    """
    SEIRDV model with time-varying beta(t) and mu(t)
    S: Susceptible, E: Exposed, I: Infected, R: Recovered, D: Deceased, V: Vaccinated
    """
    S, E, I, R, D, V = y
    
    # Get current rates from time-varying functions
    beta_t = beta_func(t)
    mu_t = mu_func(t)
    nu_t = nu_func(t)
    
    # Model equations
    dSdt = -beta_t * S * I / N - nu_t * S
    dEdt = beta_t * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - mu_t * I
    dRdt = gamma * I
    dDdt = mu_t * I
    dVdt = nu_t * S
    
    return [dSdt, dEdt, dIdt, dRdt, dDdt, dVdt]


# 2a. Beta(t) - Fourier Series
def beta_fourier(t, params_beta):
    """
    Calculates time-varying transmission rate beta(t) using Fourier series.
    params_beta = [beta0, T_beta, A1, B1, A2, B2, ...]
    """
    beta0 = params_beta[0]
    T_beta = params_beta[1]
    coeffs = params_beta[2:]
    num_terms = len(coeffs) // 2
    
    fourier_sum = 0
    if T_beta > 1e-6 and num_terms > 0:
        An = coeffs[:num_terms]
        Bn = coeffs[num_terms:]
        for k in range(num_terms):
            n = k + 1
            angle = 2 * np.pi * n * t / T_beta
            fourier_sum += An[k] * np.sin(angle) + Bn[k] * np.cos(angle)
            
    beta_val = beta0 * (1 + fourier_sum)
    
    return max(0, beta_val)  # Ensure non-negative

# 2b. Mu(t) - Fourier Series (now using the same approach as beta)
def mu_fourier(t, params_mu):
    """
    Calculates time-varying death rate mu(t) using Fourier series.
    params_mu = [mu0, T_mu, C1, D1, C2, D2, C3, D3, ...]
    mu(t) = mu0 * (1 + sum[Cn*sin(2πnt/T) + Dn*cos(2πnt/T)])
    """
    mu0 = params_mu[0]  # Baseline death rate
    T_mu = params_mu[1]  # Period for death rate oscillations
    coeffs = params_mu[2:]
    num_terms = len(coeffs) // 2
    
    fourier_sum = 0
    if T_mu > 1e-6 and num_terms > 0:
        Cn = coeffs[:num_terms]  # Sine coefficients
        Dn = coeffs[num_terms:]  # Cosine coefficients
        for k in range(num_terms):
            n = k + 1
            angle = 2 * np.pi * n * t / T_mu
            fourier_sum += Cn[k] * np.sin(angle) + Dn[k] * np.cos(angle)
    
    mu_val = mu0 * (1 + fourier_sum)
    return max(0, mu_val)  # Ensure non-negative

def create_vaccination_func(vaccine_df, N):
    """
    Creates a vaccination rate function using actual vaccination data.
    vaccine_df: DataFrame with 'date' and 'daily doses' columns
    N: Total population size
    Returns a function that gives the vaccination rate at time t
    """
    def nu_func(t):
        # Convert time point t to date
        current_date = pd.Timestamp(start_date) + pd.Timedelta(days=int(t))
        
        # Get vaccination rate for this date
        if current_date in vaccine_df.index:
            daily_doses = vaccine_df.loc[current_date, 'daily doses']
            rate = daily_doses / N  # Convert to rate by dividing by total population
            return rate
        else:
            return 0.0  # Return 0 for dates without data
            
    return nu_func

# Prepare vaccination data
vaccine_df_processed = vaccine.set_index('date')
nu_func = create_vaccination_func(vaccine_df_processed, N)

def objective_combined(params_combined, n_beta_params, n_mu_params, sigma, gamma, nu_func, N, y0, t_eval, actual_cases, actual_deaths, start_date, norm_cases, norm_deaths):
    """
    Objective function for combined optimization of beta(t) and mu(t)
    Minimizes weighted sum of normalized RMSE for cases and deaths
    """
    try:
        # Split combined parameters
        params_beta = params_combined[:n_beta_params]
        params_mu = params_combined[n_beta_params:]
        
        # Define current beta and mu functions
        beta_func = lambda ti: beta_fourier(ti, params_beta)
        mu_func = lambda ti: mu_fourier(ti, params_mu)
        
        # Solve SEIRDV model
        t_span = (t_eval[0], t_eval[-1])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = solve_ivp(
                seirdv_model_mu_opt,
                t_span, y0,
                args=(beta_func, sigma, gamma, mu_func, nu_func, N),
                t_eval=t_eval, method='RK45', max_step=1.0
            )
        
        if solution.status != 0:
            return np.inf
        
        S, E, I, R, D, V = solution.y
        
        # Check for invalid solutions
        if np.isnan(E).any() or np.isnan(I).any():
            return np.inf
        
        # For cases (proportional to E compartment)
        daily_new_infections = sigma * E
        sim_dates = pd.date_range(start=start_date, periods=len(t_eval), freq='D')
        pred_inf_series = pd.Series(daily_new_infections, index=sim_dates)
        pred_weekly_cases = pred_inf_series
        
        # For deaths (mu(t) * I)
        mu_values = np.array([mu_func(ti) for ti in t_eval])
        daily_deaths = mu_values * I
        pred_death_series = pd.Series(daily_deaths, index=sim_dates)
        pred_weekly_deaths = pred_death_series
        
        # Align with actual data and calculate errors
        cases_df = pd.DataFrame({'actual': actual_cases, 'predicted': pred_weekly_cases}).dropna()
        deaths_df = pd.DataFrame({'actual': actual_deaths, 'predicted': pred_weekly_deaths}).dropna()
        
        if len(cases_df) < 2 or len(deaths_df) < 2:
            return np.inf
        
        rmse_cases = np.sqrt(mean_squared_error(cases_df['actual'], cases_df['predicted']))
        rmse_deaths = np.sqrt(mean_squared_error(deaths_df['actual'], deaths_df['predicted']))
        
        # Normalize errors
        norm_rmse_cases = rmse_cases / norm_cases
        norm_rmse_deaths = rmse_deaths / norm_deaths
        
        # Combined error (you can adjust weights if needed)
        combined_error = norm_rmse_cases + norm_rmse_deaths
        
        return combined_error
        
    except Exception as e:
        # print(f"Error in objective: {e}")  # Uncomment for debugging
        return np.inf


# Define model parameters
N = 67000000  # UK population
gamma = 1/14  # Recovery rate (14 days)
sigma = 1/5   # Incubation rate (5 days)

# Ensure we have the required data
# Check if actual weekly cases and deaths exist in the correct format
if 'actual_weekly_cases' not in locals() or not isinstance(actual_weekly_cases, pd.Series):
    print("Warning: Resampling weekly cases from original data")
    actual_weekly_cases = cases.set_index('date')['weekly cases'].copy()

if 'actual_weekly_deaths' not in locals() or not isinstance(actual_weekly_deaths, pd.Series):
    print("Warning: Resampling weekly deaths from original data")
    actual_weekly_deaths = deaths.set_index('date')['weekly deaths'].copy()

# Get normalization factors
norm_factor_cases = np.mean(actual_weekly_cases) if np.mean(actual_weekly_cases) > 0 else 1.0
norm_factor_deaths = np.mean(actual_weekly_deaths) if np.mean(actual_weekly_deaths) > 0 else 1.0

# Set up time vector
t = np.linspace(0, 365, 365)  # One year simulation
start_date = actual_weekly_cases.index.min()  # Use first date from actual data

# Initial conditions
I0 = 4680  # Initial infected (adjust based on actual data)
E0 = I0 * 1.5  # Initial exposed typically higher than infected
S0 = N - I0 - E0  # Initial susceptible
R0 = 0  # Initial recovered
D0 = 0  # Initial deaths
V0 = 0  # Initial vaccinated
y0_seirdv = [S0, E0, I0, R0, D0, V0]

# Define nu function (vaccination rate)
nu_func = create_vaccination_func(
    vaccine_df_processed,
    N
)

# Parameter counts
n_terms_beta = 3  # Number of Fourier terms for beta
n_beta_params = 2 + 2 * n_terms_beta  # beta0, T_beta, [A1...An, B1...Bn]

n_terms_mu = 3  # Number of Fourier terms for mu
n_mu_params = 2 + 2 * n_terms_mu  # mu0, T_mu, [C1...Cn, D1...Dn]

print(f"Optimizing {n_beta_params} parameters for beta(t) and {n_mu_params} for mu(t)")
print(f"Total parameters: {n_beta_params + n_mu_params}")

# Initial guesses - carefully chosen to help convergence
# Beta Fourier initial guesses
beta0_guess = 0.08  # Initial guess for baseline transmission
T_beta_guess = 365  # Period of one year
beta_coeff_guess = [0.1, 0.05, -0.05, -0.1, 0.0, 0.0]  # A1, A2, A3, B1, B2, B3
if len(beta_coeff_guess) != 2 * n_terms_beta:
    beta_coeff_guess = [0.0] * (2 * n_terms_beta)
    
initial_beta_params = [beta0_guess, T_beta_guess] + beta_coeff_guess

# Mu Fourier initial guesses
mu0_guess = 0.002  # Initial guess for baseline death rate (smaller than beta)
T_mu_guess = 365  # Period of one year
mu_coeff_guess = [0.2, 0.1, 0.0, 0.3, 0.1, 0.0]  # C1, C2, C3, D1, D2, D3
if len(mu_coeff_guess) != 2 * n_terms_mu:
    mu_coeff_guess = [0.0] * (2 * n_terms_mu)

initial_mu_params = [mu0_guess, T_mu_guess] + mu_coeff_guess

# Combine all initial guesses
initial_params_combined = initial_beta_params + initial_mu_params

# Define bounds - 
# Beta bounds
beta0_bounds = (0.07,0.3)   
T_beta_bounds = (200, 500)   
beta_coeff_bounds = [(-0.5, 0.5)] * (2 * n_terms_beta)  # Limit Fourier coefficients
beta_bounds = [beta0_bounds, T_beta_bounds] + beta_coeff_bounds

# Mu bounds
mu0_bounds = (0.01, 0.03)  # Baseline death rate (smaller than beta)
T_mu_bounds = (200, 500)  
mu_coeff_bounds = [(-0.5, 0.5)] * (2 * n_terms_mu)  
mu_bounds = [mu0_bounds, T_mu_bounds] + mu_coeff_bounds

# Combine all bounds
bounds_combined = beta_bounds + mu_bounds

# Wrap the objective function with fixed arguments
objective_wrapped = lambda p: objective_combined(
    p, n_beta_params, n_mu_params, sigma, gamma, nu_func, N, y0_seirdv, t, actual_weekly_cases, actual_weekly_deaths, start_date, norm_factor_cases, norm_factor_deaths
)


print("\nStarting optimization with L-BFGS-B method...")
print("This may take several minutes. Consider reducing maxiter for faster testing...")

result = minimize(
    objective_wrapped, 
    initial_params_combined,
    method='L-BFGS-B',
    bounds=bounds_combined,
    options={
        'maxiter': 1000,  
        'ftol': 1e-7,  
        'gtol': 1e-5,   
        'disp': True
    }
)

if result.success or "CONVERGENCE" in result.message.upper():
    print("\nOptimization successful or converged!")
    optimal_params = result.x
    
    # Split parameters
    optimal_beta_params = optimal_params[:n_beta_params]
    optimal_mu_params = optimal_params[n_beta_params:]
    
    print(f"\nOptimal beta parameters (Fourier):")
    print(f"beta0 = {optimal_beta_params[0]:.4f}")
    print(f"T_beta = {optimal_beta_params[1]:.1f}")
    print(f"Beta Fourier coefficients (A_n, B_n): {optimal_beta_params[2:].round(4)}")
    
    print(f"\nOptimal mu parameters (Fourier):")
    print(f"mu0 = {optimal_mu_params[0]:.6f}")
    print(f"T_mu = {optimal_mu_params[1]:.1f}")
    print(f"Mu Fourier coefficients (C_n, D_n): {optimal_mu_params[2:].round(4)}")
    
    # Create optimal functions
    optimal_beta_func = lambda ti: beta_fourier(ti, optimal_beta_params)
    optimal_mu_func = lambda ti: mu_fourier(ti, optimal_mu_params)
    
    # Run final simulation
    solution_final = solve_ivp(
        seirdv_model_mu_opt, 
        (t[0], t[-1]), y0_seirdv,
        args=(optimal_beta_func, sigma, gamma, optimal_mu_func, nu_func, N),
        t_eval=t, method='RK45'
    )
    
    S_opt, E_opt, I_opt, R_opt, D_opt, V_opt = solution_final.y
    sim_dates = pd.date_range(start=start_date, periods=len(t), freq='D')
    
    # Calculate metrics for cases and deaths
    daily_inf_opt = sigma * E_opt
    pred_inf_series = pd.Series(daily_inf_opt, index=sim_dates)
    pred_cases_opt = pred_inf_series
    
    mu_values_opt = np.array([optimal_mu_func(ti) for ti in t])
    daily_deaths_opt = mu_values_opt * I_opt
    pred_death_series = pd.Series(daily_deaths_opt, index=sim_dates)
    pred_deaths_opt = pred_death_series
    
    # Calculate fit metrics
    cases_comp = pd.DataFrame({'actual': actual_weekly_cases, 'predicted': pred_cases_opt}).dropna()
    deaths_comp = pd.DataFrame({'actual': actual_weekly_deaths, 'predicted': pred_deaths_opt}).dropna()
    
    rmse_cases = np.sqrt(mean_squared_error(cases_comp['actual'], cases_comp['predicted']))
    r2_cases = r2_score(cases_comp['actual'], cases_comp['predicted'])
    
    rmse_deaths = np.sqrt(mean_squared_error(deaths_comp['actual'], deaths_comp['predicted']))
    r2_deaths = r2_score(deaths_comp['actual'], deaths_comp['predicted'])
    
    print(f"\nFit Metrics:")
    print(f"Cases: RMSE = {rmse_cases:,.2f}, R² = {r2_cases:.4f}")
    print(f"Deaths: RMSE = {rmse_deaths:,.2f}, R² = {r2_deaths:.4f}")
    
    # === Plot results ===
    
    # Plot 1: SEIRDV compartments
    plt.figure(figsize=(14, 7))
    plt.plot(sim_dates, S_opt, label='Susceptible (S)', color='blue', alpha=0.7)
    plt.plot(sim_dates, E_opt, label='Exposed (E)', color='orange', alpha=0.7)
    plt.plot(sim_dates, I_opt, label='Infected (I)', color='red', alpha=0.7)
    plt.plot(sim_dates, R_opt, label='Recovered (R)', color='green', alpha=0.7)
    plt.plot(sim_dates, D_opt, label='Deceased (D)', color='black', alpha=0.7)
    plt.plot(sim_dates, V_opt, label='Vaccinated (V)', color='purple', alpha=0.7)
    plt.title('SEIRDV Model Simulation with Optimized Fourier Parameters')
    plt.xlabel('Date')
    plt.ylabel('Number of Individuals')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    plt.savefig("seirdv_model_simulation.png")  # Save the plot
    
    # Plot 2: Optimal parameter functions
    beta_values = [optimal_beta_func(ti) for ti in t]
    nu_values = [nu_func(ti) for ti in t]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Beta(t)
    axes[0].plot(sim_dates, beta_values, color='dodgerblue')
    axes[0].set_title('Optimal Time-Varying Parameters')
    axes[0].set_ylabel('Beta(t)\nTransmission Rate')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(bottom=0)
    
    # Mu(t)
    axes[1].plot(sim_dates, mu_values_opt, color='firebrick')
    axes[1].set_ylabel('Mu(t)\nDeath Rate')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim(bottom=0)
    
    # Nu(t)
    axes[2].plot(sim_dates, nu_values, color='darkorchid')
    axes[2].set_ylabel('Nu(t)\nVaccination Rate')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("optimal_parameters.png")  # Save the plot
    
    # Plot 3: Model vs. actual data
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Cases
    axes[0].plot(cases_comp.index, cases_comp['actual'], 'o-', label='Actual Weekly Cases', 
                 color='black', markersize=4, alpha=0.7)
    axes[0].plot(cases_comp.index, cases_comp['predicted'], 'x--', 
                 label=f'Model Prediction (R² = {r2_cases:.3f})', 
                 color='red', markersize=4)
    axes[0].set_title('Model Fit: COVID-19 Cases and Deaths (Fourier Series Model)')
    axes[0].set_ylabel('Weekly Cases')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Deaths
    axes[1].plot(deaths_comp.index, deaths_comp['actual'], 'o-', label='Actual Weekly Deaths', 
                color='black', markersize=4, alpha=0.7)
    axes[1].plot(deaths_comp.index, deaths_comp['predicted'], 'x--', 
                label=f'Model Prediction (R² = {r2_deaths:.3f})', 
                color='blue', markersize=4)
    axes[1].set_ylabel('Weekly Deaths')
    axes[1].set_xlabel('Date')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
    plt.savefig("model_fit_cases_deaths.png")  # Save the plot

else:
    print(f"\nOptimization failed: {result.message}")
    print(f"Final function value: {result.fun}")
    
    # === SEIRDV Model with Fourier Beta and Multi-Gaussian Mu Optimization ===

def seirdv_model_mu_opt(t, y, beta_func, sigma, gamma, mu_func, nu_func, N):
    """
    SEIRDV model with time-varying beta(t) and mu(t)
    S: Susceptible, E: Exposed, I: Infected, R: Recovered, D: Deceased, V: Vaccinated
    """
    S, E, I, R, D, V = y
    
    # Get current rates from time-varying functions
    beta_t = beta_func(t)
    mu_t = mu_func(t)
    nu_t = nu_func(t)
    
    # Model equations
    dSdt = -beta_t * S * I / N - nu_t * S
    dEdt = beta_t * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - mu_t * I
    dRdt = gamma * I
    dDdt = mu_t * I
    dVdt = nu_t * S
    
    return [dSdt, dEdt, dIdt, dRdt, dDdt, dVdt]


# 2a. Beta(t) - Fourier Series
def beta_fourier(t, params_beta):
    """
    Calculates time-varying transmission rate beta(t) using Fourier series.
    params_beta = [beta0, T_beta, A1, B1, A2, B2, ...]
    """
    beta0 = params_beta[0]
    T_beta = params_beta[1]
    coeffs = params_beta[2:]
    num_terms = len(coeffs) // 2
    
    fourier_sum = 0
    if T_beta > 1e-6 and num_terms > 0:
        An = coeffs[:num_terms]
        Bn = coeffs[num_terms:]
        for k in range(num_terms):
            n = k + 1
            angle = 2 * np.pi * n * t / T_beta
            fourier_sum += An[k] * np.sin(angle) + Bn[k] * np.cos(angle)
            
    beta_val = beta0 * (1 + fourier_sum)
    
    return max(0, beta_val)  # Ensure non-negative

# 2b. Mu(t) - Multi-Gaussian
def mu_multi_gaussian(t, params_mu):
    """
    Calculates time-varying death rate mu(t) using a sum of Gaussians.
    params_mu = [offset, amp1, mean1, std1, amp2, mean2, std2, ...]
    """
    offset = params_mu[0]
    gaussian_params = params_mu[1:]
    num_gaussians = len(gaussian_params) // 3
    
    mu_val = offset
    for i in range(num_gaussians):
        amp = gaussian_params[i * 3]
        mean = gaussian_params[i * 3 + 1]
        std = gaussian_params[i * 3 + 2]
        
        std = max(1.0, std)  # Min std dev for numerical stability
        amp = max(0, amp)    # Non-negative amplitude
        
        exponent = -0.5 * ((t - mean) / std) ** 2
        # Avoid underflow in exp calculation
        if exponent > -700:
             mu_val += amp * np.exp(exponent)
             
    return max(0, mu_val)  # Ensure non-negative


def create_vaccination_func(vaccine_df, N):
    """
    Creates a vaccination rate function using actual vaccination data.
    vaccine_df: DataFrame with 'date' and 'daily doses' columns
    N: Total population size
    Returns a function that gives the vaccination rate at time t
    """
    def nu_func(t):
        # Convert time point t to date
        current_date = pd.Timestamp(start_date) + pd.Timedelta(days=int(t))
        
        # Get vaccination rate for this date
        if current_date in vaccine_df.index:
            daily_doses = vaccine_df.loc[current_date, 'daily doses']
            rate = daily_doses / N  # Convert to rate by dividing by total population
            return rate
        else:
            return 0.0  # Return 0 for dates without data
            
    return nu_func

vaccine_df_processed = vaccine.set_index('date')
nu_func = create_vaccination_func(vaccine_df_processed, N)


def objective_combined(params_combined, n_beta_params, n_mu_params, sigma, gamma, nu_func, N, y0, t_eval, actual_cases, actual_deaths, start_date, norm_cases, norm_deaths):
    """
    Objective function for combined optimization of beta(t) and mu(t)
    Minimizes weighted sum of normalized RMSE for cases and deaths
    """
    try:
        # Split combined parameters
        params_beta = params_combined[:n_beta_params]
        params_mu = params_combined[n_beta_params:]
        
        # Define current beta and mu functions
        beta_func = lambda ti: beta_fourier(ti, params_beta)
        mu_func = lambda ti: mu_multi_gaussian(ti, params_mu)
        
        # Solve SEIRDV model
        t_span = (t_eval[0], t_eval[-1])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = solve_ivp(
                seirdv_model_mu_opt,
                t_span, y0,
                args=(beta_func, sigma, gamma, mu_func, nu_func, N),
                t_eval=t_eval, method='RK45', max_step=1.0
            )
        
        if solution.status != 0:
            return np.inf
        
        S, E, I, R, D, V = solution.y
        
        # Check for invalid solutions
        if np.isnan(E).any() or np.isnan(I).any():
            return np.inf
        
        # For cases (proportional to E compartment)
        daily_new_infections = sigma * E
        sim_dates = pd.date_range(start=start_date, periods=len(t_eval), freq='D')
        pred_inf_series = pd.Series(daily_new_infections, index=sim_dates)
        pred_weekly_cases = pred_inf_series.resample('W-SUN').sum()
        
        # For deaths (mu(t) * I)
        mu_values = np.array([mu_func(ti) for ti in t_eval])
        daily_deaths = mu_values * I
        pred_death_series = pd.Series(daily_deaths, index=sim_dates)
        pred_weekly_deaths = pred_death_series
        
        # Align with actual data and calculate errors
        cases_df = pd.DataFrame({'actual': actual_cases, 'predicted': pred_weekly_cases}).dropna()
        deaths_df = pd.DataFrame({'actual': actual_deaths, 'predicted': pred_weekly_deaths}).dropna()
        
        if len(cases_df) < 2 or len(deaths_df) < 2:
            return np.inf
        
        rmse_cases = np.sqrt(mean_squared_error(cases_df['actual'], cases_df['predicted']))
        rmse_deaths = np.sqrt(mean_squared_error(deaths_df['actual'], deaths_df['predicted']))
        
        # Normalize errors
        norm_rmse_cases = rmse_cases / norm_cases
        norm_rmse_deaths = rmse_deaths / norm_deaths
        
        # Combined error (you can adjust weights if needed)
        combined_error = norm_rmse_cases + norm_rmse_deaths
        
        return combined_error
        
    except Exception as e:
        # print(f"Error in objective: {e}")  # Uncomment for debugging
        return np.inf



# Define model parameters
N = 67000000  # UK population
gamma = 1/14  # Recovery rate (14 days)
sigma = 1/5   # Incubation rate (5 days)

# Ensure we have the required data
# Check if actual weekly cases and deaths exist in the correct format
if 'actual_weekly_cases' not in locals() or not isinstance(actual_weekly_cases, pd.Series):
    print("Warning: Resampling weekly cases from original data")
    actual_weekly_cases = cases.set_index('date')['weekly cases'].copy()

if 'actual_weekly_deaths' not in locals() or not isinstance(actual_weekly_deaths, pd.Series):
    print("Warning: Resampling weekly deaths from original data")
    actual_weekly_deaths = deaths.set_index('date')['weekly deaths'].copy()

# Get normalization factors
norm_factor_cases = np.mean(actual_weekly_cases) if np.mean(actual_weekly_cases) > 0 else 1.0
norm_factor_deaths = np.mean(actual_weekly_deaths) if np.mean(actual_weekly_deaths) > 0 else 1.0

# Set up time vector
t = np.linspace(0, 365, 365)  # One year simulation
start_date = actual_weekly_cases.index.min()  # Use first date from actual data

# Initial conditions
I0 = 4680  # Initial infected (adjust based on actual data)
E0 = I0 * 1.5  # Initial exposed typically higher than infected
S0 = N - I0 - E0  # Initial susceptible
R0 = 0  # Initial recovered
D0 = 0  # Initial deaths
V0 = 0  # Initial vaccinated
y0_seirdv = [S0, E0, I0, R0, D0, V0]

# Define nu function (vaccination rate)
nu_func = create_vaccination_func(
    vaccine_df_processed,
    N
)


# Parameter counts
n_terms_beta = 3  # Number of Fourier terms for beta (more terms = more flexibility)
n_beta_params = 2 + 2 * n_terms_beta  # beta0, T_beta, [A1...An, B1...Bn]

n_gaussians_mu = 3  # Number of Gaussians for mu (more peaks for waves of mortality)
n_mu_params = 1 + 3 * n_gaussians_mu  # offset, [amp1, mean1, std1, amp2, mean2, std2, ...]

print(f"Optimizing {n_beta_params} parameters for beta(t) and {n_mu_params} for mu(t)")
print(f"Total parameters: {n_beta_params + n_mu_params}")

# Initial guesses - carefully chosen to help convergence
beta0_guess = 0.15  # Initial guess for baseline transmission
T_beta_guess = 365  # Period of one year
fourier_coeff_guess = [0.1, 0.05, -0.05, -0.1, 0.0, 0.0]  # A1, A2, A3, B1, B2, B3
if len(fourier_coeff_guess) != 2 * n_terms_beta:
    fourier_coeff_guess = [0.0] * (2 * n_terms_beta)
    
initial_beta_params = [beta0_guess, T_beta_guess] + fourier_coeff_guess

# Death rate initial guesses - position Gaussians across the timeline
mu_offset_guess = 0.0005  
initial_mu_params = [mu_offset_guess]
peak_times = np.linspace(t[0] + 50, t[-1] - 50, n_gaussians_mu)  # Spread peaks across timeline
for i in range(n_gaussians_mu):
    amp = 0.003 if i == 0 else 0.002  # First wave typically stronger
    mean = peak_times[i]
    std = 30  # About one month standard deviation
    initial_mu_params.extend([amp, mean, std])

# Combine all initial guesses
initial_params_combined = initial_beta_params + initial_mu_params

# Define bounds
# Beta bounds
beta0_bounds = (0.05, 0.4)  
T_beta_bounds = (180, 500)  
coeff_bounds = [(-0.3, 0.3)] * (2 * n_terms_beta)  # Limit Fourier coefficients
beta_bounds = [beta0_bounds, T_beta_bounds] + coeff_bounds

# Mu bounds
mu_offset_bounds = (1e-5, 0.005)  # Very small baseline death rate
mu_amp_bounds = (1e-5, 0.015)     # Amplitude of death rate peaks
mu_mean_bounds = [(t[0] + 30, t[-1] - 30)] * n_gaussians_mu  # Keep peaks within timeline
mu_std_bounds = [(15, 90)] * n_gaussians_mu  # Width between 2 weeks and 3 months

mu_bounds = [mu_offset_bounds]
for i in range(n_gaussians_mu):
    mu_bounds.extend([mu_amp_bounds, mu_mean_bounds[i], mu_std_bounds[i]])

# Combine all bounds
bounds_combined = beta_bounds + mu_bounds

# Wrap the objective function with fixed arguments
objective_wrapped = lambda p: objective_combined(
    p, n_beta_params, n_mu_params, sigma, gamma, nu_func, N, y0_seirdv, t, actual_weekly_cases, actual_weekly_deaths, start_date, norm_factor_cases, norm_factor_deaths
)


print("\nStarting optimization with L-BFGS-B method...")
print("This may take several minutes...")

result = minimize(
    objective_wrapped, 
    initial_params_combined,
    method='L-BFGS-B',
    bounds=bounds_combined,
    options={
        'maxiter': 1000,
        'ftol': 1e-8, 
        'gtol': 1e-6,
        'disp': True
    }
)


if result.success or "CONVERGENCE" in result.message.upper():
    print("\nOptimization successful or converged!")
    optimal_params = result.x
    
    # Split parameters
    optimal_beta_params = optimal_params[:n_beta_params]
    optimal_mu_params = optimal_params[n_beta_params:]
    
    print(f"\nOptimal beta parameters (Fourier):")
    print(f"beta0 = {optimal_beta_params[0]:.4f}")
    print(f"T = {optimal_beta_params[1]:.1f}")
    print(f"Fourier coefficients: {optimal_beta_params[2:].round(4)}")
    
    print(f"\nOptimal mu parameters (Gaussian):")
    print(f"offset = {optimal_mu_params[0]:.6f}")
    for i in range(n_gaussians_mu):
        idx = 1 + i * 3
        print(f"Gaussian {i+1}: amp={optimal_mu_params[idx]:.6f}, mean={optimal_mu_params[idx+1]:.1f}, std={optimal_mu_params[idx+2]:.1f}")
    
    # Create optimal functions
    optimal_beta_func = lambda ti: beta_fourier(ti, optimal_beta_params)
    optimal_mu_func = lambda ti: mu_multi_gaussian(ti, optimal_mu_params)
    
    # Run final simulation
    solution_final = solve_ivp(
        seirdv_model_mu_opt, 
        (t[0], t[-1]), y0_seirdv,
        args=(optimal_beta_func, sigma, gamma, optimal_mu_func, nu_func, N),
        t_eval=t, method='RK45'
    )
    
    S_opt, E_opt, I_opt, R_opt, D_opt, V_opt = solution_final.y
    sim_dates = pd.date_range(start=start_date, periods=len(t), freq='D')
    
    # Calculate metrics for cases and deaths
    daily_inf_opt = sigma * E_opt
    pred_inf_series = pd.Series(daily_inf_opt, index=sim_dates)
    pred_cases_opt = pred_inf_series.resample('W-SUN').sum()
    
    mu_values_opt = np.array([optimal_mu_func(ti) for ti in t])
    daily_deaths_opt = mu_values_opt * I_opt
    pred_death_series = pd.Series(daily_deaths_opt, index=sim_dates)
    pred_deaths_opt = pred_death_series.resample('W-SUN').sum()
    
    # Calculate fit metrics
    cases_comp = pd.DataFrame({'actual': actual_weekly_cases, 'predicted': pred_cases_opt}).dropna()
    deaths_comp = pd.DataFrame({'actual': actual_weekly_deaths, 'predicted': pred_deaths_opt}).dropna()
    
    rmse_cases = np.sqrt(mean_squared_error(cases_comp['actual'], cases_comp['predicted']))
    r2_cases = r2_score(cases_comp['actual'], cases_comp['predicted'])
    
    rmse_deaths = np.sqrt(mean_squared_error(deaths_comp['actual'], deaths_comp['predicted']))
    r2_deaths = r2_score(deaths_comp['actual'], deaths_comp['predicted'])
    
    print(f"\nFit Metrics:")
    print(f"Cases: RMSE = {rmse_cases:,.2f}, R² = {r2_cases:.4f}")
    print(f"Deaths: RMSE = {rmse_deaths:,.2f}, R² = {r2_deaths:.4f}")
    
    # === Plot results ===
    
    # Plot 1: SEIRDV compartments
    plt.figure(figsize=(14, 7))
    plt.plot(sim_dates, S_opt, label='Susceptible (S)', color='blue', alpha=0.7)
    plt.plot(sim_dates, E_opt, label='Exposed (E)', color='orange', alpha=0.7)
    plt.plot(sim_dates, I_opt, label='Infected (I)', color='red', alpha=0.7)
    plt.plot(sim_dates, R_opt, label='Recovered (R)', color='green', alpha=0.7)
    plt.plot(sim_dates, D_opt, label='Deceased (D)', color='black', alpha=0.7)
    plt.plot(sim_dates, V_opt, label='Vaccinated (V)', color='purple', alpha=0.7)
    plt.title('SEIRDV Model Simulation with Optimized Parameters')
    plt.xlabel('Date')
    plt.ylabel('Number of Individuals')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    plt.savefig("seirdv_model_simulation_gaussian_death.png")  # Save the plot
    # Plot 2: Optimal parameter functions
    beta_values = [optimal_beta_func(ti) for ti in t]
    nu_values = [nu_func(ti) for ti in t]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Beta(t)
    axes[0].plot(sim_dates, beta_values, color='dodgerblue')
    axes[0].set_title('Optimal Time-Varying Parameters')
    axes[0].set_ylabel('Beta(t)\nTransmission Rate')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(bottom=0)
    
    # Mu(t)
    axes[1].plot(sim_dates, mu_values_opt, color='firebrick')
    axes[1].set_ylabel('Mu(t)\nDeath Rate')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim(bottom=0)
    
    # Nu(t)
    axes[2].plot(sim_dates, nu_values, color='darkorchid')
    axes[2].set_ylabel('Nu(t)\nVaccination Rate')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("seirdv_model_simulation_gaussian_death.png")
    
    # Plot 3: Model vs. actual data
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Cases
    axes[0].plot(cases_comp.index, cases_comp['actual'], 'o-', label='Actual Weekly Cases', 
                 color='black', markersize=4, alpha=0.7)
    axes[0].plot(cases_comp.index, cases_comp['predicted'], 'x--', 
                 label=f'Model Prediction (R² = {r2_cases:.3f})', 
                 color='red', markersize=4)
    axes[0].set_title('Model Fit: COVID-19 Cases and Deaths')
    axes[0].set_ylabel('Weekly Cases')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Deaths
    axes[1].plot(deaths_comp.index, deaths_comp['actual'], 'o-', label='Actual Weekly Deaths', 
                color='black', markersize=4, alpha=0.7)
    axes[1].plot(deaths_comp.index, deaths_comp['predicted'], 'x--', 
                label=f'Model Prediction (R² = {r2_deaths:.3f})', 
                color='blue', markersize=4)
    axes[1].set_ylabel('Weekly Deaths')
    axes[1].set_xlabel('Date')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
    plt.savefig("model_fit_cases_deaths_gaussian_death.png")  # Save the plot

else:
    print(f"\nOptimization failed: {result.message}")
    print(f"Final function value: {result.fun}")
    
    # curve fitting

if 'real_data' in locals() and 'real_cases_scaled' in locals() and real_cases_scaled is not None:
    # Extract infection and death data
    infection_data = real_cases_scaled
    
    # Get death data if available
    if 'daily confirmed deaths per 1m' in real_data.columns:
        death_data = real_data['daily confirmed deaths per 1m'].iloc[:365].values
        death_data = np.nan_to_num(death_data) * (N / 1_000_000)
    else:
        death_data = None
    
    # Define potential mathematical functions to fit the data
    
    # 1. Sum of Gaussian peaks - good for capturing multiple waves
    def multi_gaussian(x, *params):
        """Sum of multiple Gaussian functions to model multiple waves"""
        y = np.zeros_like(x, dtype=float)
        for i in range(0, len(params), 3):
            if i+2 < len(params):
                amp, mu, sigma = params[i], params[i+1], params[i+2]
                y += amp * np.exp(-((x - mu) / sigma)**2 / 2)
        return y
    
    # 2. Gompertz Peak - models asymmetric peaks 
    def gompertz_peak(x, a, b, c, d):
        """ Models a peak using a Gompertz-like function """
        # a: amplitude, b: peak time, c: growth rate, d: decay rate modifier
        # Growth component - Gompertz growth until peak
        growth = np.exp(-np.exp(-c * (x - b)))
        # Decay component - start at 1 at peak time, then decay
        # Use smooth exponential decay that equals 1 at x=b
        decay = np.exp(-d * np.maximum(0, x - b)**2)
        return a * growth * decay
    
    # 3. Multi-Weibull PDF - good for modeling asymmetric peaks
    def multi_weibull_pdf(x, *params):
        """ Sum of multiple Weibull PDF functions """
        y = np.zeros_like(x, dtype=float)
        for i in range(0, len(params), 3):
             if i+2 < len(params):
                amp, shape, scale = params[i], params[i+1], params[i+2]
                # loc parameter (shift) is often fixed near 0 or fitted too
                # Using weibull_min from scipy.stats for the PDF calculation
                # Note: scale corresponds to lambda, shape corresponds to k in standard notation
                # We scale by amplitude 'amp'
                y += amp * weibull_min.pdf(x, c=shape, scale=scale, loc=0) # Assuming loc=0
        return y
    
    # 4. Richards curve with decay 
    def richards_curve_with_decay(x, K, r, x0, nu, decay_rate):
        """ Richards' curve (Generalized Logistic) with exponential decay """
        # Safe division to avoid numerical issues
        sigmoid = np.clip(nu * np.exp(-r * (x - x0)), -700, 700) # Prevent overflow in exp
        denominator = np.maximum(1e-10, (1 + sigmoid)**(1 / nu)) # Avoid division by zero
        growth = K / denominator
        # For decay, use a proper post-peak decay that starts at the inflection point
        decay_mask = x > x0  # Only apply decay after x0
        decay = np.ones_like(x)
        decay[decay_mask] = np.exp(-decay_rate * (x[decay_mask] - x0))
        return growth * decay
    
    # 5. Multi-LogNormal PDF - good for data with long tails
    def multi_lognormal_pdf(x, *params):
        """ Sum of multiple Log-Normal PDF functions """
        # Log-normal is defined for x > 0. Add a small epsilon if x_data starts at 0.
        x_safe = np.maximum(x, 1e-9)
        y = np.zeros_like(x_safe, dtype=float)
        for i in range(0, len(params), 3):
             if i+2 < len(params):
                amp, s, scale = params[i], params[i+1], params[i+2] # s is the shape parameter (sigma)
                y += amp * lognorm.pdf(x_safe, s=s, scale=scale, loc=0) # Assuming loc=0
        return y
    
    # 6. Fourier series 
    def fourier_series(x, a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5):
        """Fourier series approximation with explicit parameters"""
        # Scale x to be in 0-2π range for better fitting - use the max value of the array, not just the current point
        max_x = 365  # Use a fixed period of 365 days
        scaled_x = 2 * np.pi * x / max_x
        
        # Calculate the Fourier series with proper frequency terms
        result = a0  # DC component (mean)
        result += a1 * np.cos(1 * scaled_x) + b1 * np.sin(1 * scaled_x)  # First harmonic
        result += a2 * np.cos(2 * scaled_x) + b2 * np.sin(2 * scaled_x)  # Second harmonic
        result += a3 * np.cos(3 * scaled_x) + b3 * np.sin(3 * scaled_x)  # Third harmonic
        result += a4 * np.cos(4 * scaled_x) + b4 * np.sin(4 * scaled_x)  # Fourth harmonic
        result += a5 * np.cos(5 * scaled_x) + b5 * np.sin(5 * scaled_x)  # Fifth harmonic
        return result
    
    # Prepare x data (days)
    x_data = np.arange(len(infection_data))
    
    # Try multiple functions and select the best fit for infections
    functions = [
        {"name": "Multi-Gaussian (3 waves)", 
         "func": multi_gaussian, 
         "p0": [10, 30, 10, 20, 100, 30, 30, 200, 40],  # Initial guesses for 3 waves
         "bounds": ([0, 0, 1, 0, 50, 1, 0, 150, 1], [100, 60, 50, 100, 150, 70, 100, 300, 100])},
        
        {"name": "Gompertz Peak", 
         "func": gompertz_peak, 
         "p0": [np.max(infection_data) * 0.8, 100, 0.1, 0.001],  # Guess peak height, time, growth, decay
         "bounds": ([0, 0, 0.001, 0.00001], [np.max(infection_data)*2, len(x_data), 1, 0.1])},
        
        {"name": "Multi-Weibull PDF (2 waves)", 
         "func": multi_weibull_pdf, 
         "p0": [np.max(infection_data)*50, 2, 50, np.max(infection_data)*80, 3, 200],  # Guess amp, shape, scale for each wave
         "bounds": ([0, 1, 10, 0, 1, 100], [np.max(infection_data)*200, 10, 150, np.max(infection_data)*200, 10, 300])},
        
        {"name": "Richards with Decay", 
         "func": richards_curve_with_decay, 
         "p0": [50, 0.1, 100, 1, 0.01],  # K, r, x0, nu, decay_rate - Updated initial values
         "bounds": ([1, 0.001, 0, 0.01, 0.0001], [10000, 1, 300, 10, 0.1])},  # Updated bounds
        
        {"name": "Multi-LogNormal PDF (2 waves)", 
         "func": multi_lognormal_pdf, 
         "p0": [np.max(infection_data)*50, 1, 50, np.max(infection_data)*80, 1, 200],  # Guess amp, shape(s), scale for each wave
         "bounds": ([0, 0.1, 10, 0, 0.1, 100], [np.max(infection_data)*200, 5, 150, np.max(infection_data)*200, 5, 300])},
        
        {"name": "Fourier (5 terms)", 
         "func": fourier_series, 
         "p0": [np.mean(infection_data), 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],  
         "bounds": ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], 
                   [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])}  
    ]
    
    # Prepare x data (days)
    x_data = np.arange(len(infection_data))
    
    # Fit each function to infection data
    infection_fits = []
    infection_fit_qualities = []
    
    print("Fitting mathematical functions to infection curve...")
    for func_info in functions:
        try:
            # Perform the curve fit with increased maximum iterations and tolerance
            params, pcov = curve_fit(
                func_info["func"], x_data, infection_data, 
                p0=func_info["p0"], 
                bounds=func_info["bounds"],
                maxfev=20000,  
                ftol=1e-8,     # More precise tolerance
                method='trf'   # Trust Region Reflective algorithm, good for bounded problems
            )
            
            # Generate fitted curve
            fitted_curve = func_info["func"](x_data, *params)
            
            # Calculate fit quality (R²)
            ss_tot = np.sum((infection_data - np.mean(infection_data))**2)
            ss_res = np.sum((infection_data - fitted_curve)**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((infection_data - fitted_curve)**2))
            
            # Store the results
            infection_fits.append({
                "name": func_info["name"],
                "fitted_curve": fitted_curve,
                "params": params,
                "func": func_info["func"],
                "r_squared": r_squared,
                "rmse": rmse
            })
            
            infection_fit_qualities.append((func_info["name"], r_squared, rmse))
            print(f"  {func_info['name']}: R² = {r_squared:.4f}, RMSE = {rmse:.4f}")
            
        except Exception as e:
            print(f"  Failed to fit {func_info['name']}: {e}")
    
    # Sort fits by R² (best fit first)
    infection_fit_qualities.sort(key=lambda x: x[1], reverse=True)
    best_infection_fit = next((fit for fit in infection_fits if fit["name"] == infection_fit_qualities[0][0]), None)
    
    # Do the same for death data if available
    death_fits = []
    death_fit_qualities = []
    best_death_fit = None
    
    if death_data is not None:
        print("\nFitting mathematical functions to death curve...")
        for func_info in functions:
            try:
                # Perform the curve fit
                params, pcov = curve_fit(
                    func_info["func"], x_data, death_data, 
                    p0=func_info["p0"], 
                    bounds=func_info["bounds"],
                    maxfev=20000,  
                    ftol=1e-8,     
                    method='trf'   # Trust Region Reflective algorithm
                )
                
                # Generate fitted curve
                fitted_curve = func_info["func"](x_data, *params)
                
                # Calculate fit quality (R²)
                ss_tot = np.sum((death_data - np.mean(death_data))**2)
                ss_res = np.sum((death_data - fitted_curve)**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((death_data - fitted_curve)**2))
                
                # Store the results
                death_fits.append({
                    "name": func_info["name"],
                    "fitted_curve": fitted_curve,
                    "params": params,
                    "func": func_info["func"],
                    "r_squared": r_squared,
                    "rmse": rmse
                })
                
                death_fit_qualities.append((func_info["name"], r_squared, rmse))
                print(f"  {func_info['name']}: R² = {r_squared:.4f}, RMSE = {rmse:.4f}")
                
            except Exception as e:
                print(f"  Failed to fit {func_info['name']}: {e}")
        
        # Sort fits by R² (best fit first)
        death_fit_qualities.sort(key=lambda x: x[1], reverse=True)
        best_death_fit = next((fit for fit in death_fits if fit["name"] == death_fit_qualities[0][0]), None)
    
    # Plot all successful fits against the real data (not just the best one)
    plt.figure(figsize=(15, 10))
    
    # Plot for infections
    plt.subplot(2, 1, 1)
    plt.plot(x_data, infection_data, 'k-', label=f'Real Infection Data ({entity_name})', alpha=0.7)
    
    # Plot all infection fits with different colors
    colors = ['r', 'g', 'b', 'm', 'c', 'y']
    for i, fit in enumerate(infection_fits):
        plt.plot(x_data, fit["fitted_curve"], '-', color=colors[i % len(colors)], 
                 label=f'{fit["name"]} (R² = {fit["r_squared"]:.4f})', alpha=0.8)
    
    plt.title('Mathematical Recreation of COVID-19 Infection Curve')
    plt.xlabel('Days')
    plt.ylabel('Daily New Cases')
    plt.legend()
    plt.grid(True)
    
    # Plot for deaths if available
    if death_data is not None:
        plt.subplot(2, 1, 2)
        plt.plot(x_data, death_data, 'k-', label=f'Real Death Data ({entity_name})', alpha=0.7)
        
        # Plot all death fits with different colors
        for i, fit in enumerate(death_fits):
            plt.plot(x_data, fit["fitted_curve"], '-', color=colors[i % len(colors)], 
                     label=f'{fit["name"]} (R² = {fit["r_squared"]:.4f})', alpha=0.8)
        
        plt.title('Mathematical Recreation of COVID-19 Death Curve')
        plt.xlabel('Days')
        plt.ylabel('Daily Deaths')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("mathematical_recreation_covid_curves.png")  # Save the plot
    
    # Function to generate code for different models
    def generate_model_code(fit_type, name, params):
        if fit_type == "Multi-Gaussian":
            code = []
            code.append(f"def multi_gaussian_{name}(x):")
            code.append("    # Sum of Gaussian peaks")
            gaussian_parts = []
            for i in range(0, len(params), 3):
                if i+2 < len(params):
                    amp, mu, sigma = params[i], params[i+1], params[i+2]
                    gaussian_parts.append(f"{amp:.6f} * np.exp(-((x - {mu:.6f}) / {sigma:.6f})**2 / 2)")
            code.append("    return " + " + ".join(gaussian_parts))
            return "\n".join(code)
            
        elif fit_type == "Gompertz Peak":
            a, b, c, d = params
            code = []
            code.append(f"def gompertz_peak_{name}(x):")
            code.append("    # Gompertz peak model")
            code.append(f"    growth = np.exp(-np.exp(-{c:.6f} * (x - {b:.6f})))")
            code.append(f"    decay = np.exp(-{d:.6f} * (x - {b:.6f})**2 * (x > {b:.6f}))")
            code.append(f"    return {a:.6f} * growth * decay")
            return "\n".join(code)
            
        elif fit_type == "Multi-Weibull PDF":
            code = []
            code.append(f"def multi_weibull_{name}(x):")
            code.append("    # Sum of Weibull PDFs")
            code.append("    from scipy.stats import weibull_min")
            code.append("    y = np.zeros_like(x, dtype=float)")
            for i in range(0, len(params), 3):
                if i+2 < len(params):
                    amp, shape, scale = params[i], params[i+1], params[i+2]
                    code.append(f"    y += {amp:.6f} * weibull_min.pdf(x, c={shape:.6f}, scale={scale:.6f}, loc=0)")
            code.append("    return y")
            return "\n".join(code)
            
        elif fit_type == "Richards with Decay":
            K, r, x0, nu, decay_rate = params
            code = []
            code.append(f"def richards_decay_{name}(x):")
            code.append("    # Richards curve with decay ")
            code.append(f"    sigmoid = np.clip({nu:.6f} * np.exp(-{r:.6f} * (x - {x0:.6f})), -700, 700) # Prevent overflow")
            code.append(f"    denominator = np.maximum(1e-10, (1 + sigmoid)**(1 / {nu:.6f})) # Avoid division by zero")
            code.append(f"    growth = {K:.6f} / denominator")
            code.append(f"    decay_mask = x > {x0:.6f}  # Only apply decay after x0")
            code.append("    decay = np.ones_like(x)")
            code.append(f"    decay[decay_mask] = np.exp(-{decay_rate:.6f} * (x[decay_mask] - {x0:.6f}))")
            code.append("    return growth * decay")
            return "\n".join(code)
            
        elif fit_type == "Multi-LogNormal PDF":
            code = []
            code.append(f"def multi_lognormal_{name}(x):")
            code.append("    # Sum of Log-Normal PDFs")
            code.append("    from scipy.stats import lognorm")
            code.append("    x_safe = np.maximum(x, 1e-9)")
            code.append("    y = np.zeros_like(x_safe, dtype=float)")
            for i in range(0, len(params), 3):
                if i+2 < len(params):
                    amp, s, scale = params[i], params[i+1], params[i+2]
                    code.append(f"    y += {amp:.6f} * lognorm.pdf(x_safe, s={s:.6f}, scale={scale:.6f}, loc=0)")
            code.append("    return y")
            return "\n".join(code)
            
        elif fit_type == "Fourier":
            a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = params
            code = []
            code.append(f"def fourier_{name}(x):")
            code.append("    # Fourier series with 5 terms")
            code.append("    # Scale x to be in 0-2π range using a fixed maximum value")
            code.append("    max_x = 365  # Use a fixed period of 365 days")
            code.append("    scaled_x = 2 * np.pi * x / max_x")
            code.append(f"    result = {a0:.6f}")
            code.append(f"    result += {a1:.6f} * np.cos(1 * scaled_x) + {b1:.6f} * np.sin(1 * scaled_x)")
            code.append(f"    result += {a2:.6f} * np.cos(2 * scaled_x) + {b2:.6f} * np.sin(2 * scaled_x)")
            code.append(f"    result += {a3:.6f} * np.cos(3 * scaled_x) + {b3:.6f} * np.sin(3 * scaled_x)")
            code.append(f"    result += {a4:.6f} * np.cos(4 * scaled_x) + {b4:.6f} * np.sin(4 * scaled_x)")
            code.append(f"    result += {a5:.6f} * np.cos(5 * scaled_x) + {b5:.6f} * np.sin(5 * scaled_x)")
            code.append("    return result")
            return "\n".join(code)
        
        return "# No code generation available for this model type"
    
    # Save the mathematical functions for future use
    if best_infection_fit:
        print("\nMathematical function for infection curve:")
        print(f"Function type: {best_infection_fit['name']}")
        
        # Generate code to recreate the function
        model_type = best_infection_fit["name"].split(' ')[0] if ' ' in best_infection_fit["name"] else best_infection_fit["name"]
        param_values = best_infection_fit["params"]
        
        print("\nPython code to recreate the infection curve:")
        print(generate_model_code(model_type, "infection", param_values))
    
    if best_death_fit:
        print("\nMathematical function for death curve:")
        print(f"Function type: {best_death_fit['name']}")
        
        # Generate code to recreate the function
        model_type = best_death_fit["name"].split(' ')[0] if ' ' in best_death_fit["name"] else best_death_fit["name"]
        param_values = best_death_fit["params"]
        
        print("\nPython code to recreate the death curve:")
        print(generate_model_code(model_type, "death", param_values))
    
    # If we successfully fit the Fourier series, show a separate plot comparing it
    fourier_infection_fit = next((fit for fit in infection_fits if fit["name"] == "Fourier (5 terms)"), None)
    fourier_death_fit = next((fit for fit in death_fits if fit["name"] == "Fourier (5 terms)"), None)
    
    if fourier_infection_fit or fourier_death_fit:
        plt.figure(figsize=(15, 10))
        
        if fourier_infection_fit:
            plt.subplot(2, 1, 1)
            plt.plot(x_data, infection_data, 'k-', label=f'Real Infection Data ({entity_name})', alpha=0.7)
            plt.plot(x_data, fourier_infection_fit["fitted_curve"], 'r-', 
                     label=f'Fourier Series (R² = {fourier_infection_fit["r_squared"]:.4f})')
            plt.title('Fourier Series Approximation of COVID-19 Infection Curve')
            plt.xlabel('Days')
            plt.ylabel('Daily New Cases')
            plt.legend()
            plt.grid(True)
        
        if fourier_death_fit:
            plt.subplot(2, 1, 2)
            plt.plot(x_data, death_data, 'k-', label=f'Real Death Data ({entity_name})', alpha=0.7)
            plt.plot(x_data, fourier_death_fit["fitted_curve"], 'r-', 
                     label=f'Fourier Series (R² = {fourier_death_fit["r_squared"]:.4f})')
            plt.title('Fourier Series Approximation of COVID-19 Death Curve')
            plt.xlabel('Days')
            plt.ylabel('Daily Deaths')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        plt.savefig("fourier_series_approximation_covid_curves.png")  
        
        if fourier_infection_fit:
            print("\nFor infection data:")
            print(f"  - R² value: {fourier_infection_fit['r_squared']:.4f}")
            print(f"  - RMSE: {fourier_infection_fit['rmse']:.4f}")
            
            # Analyze the coefficients
            a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = fourier_infection_fit["params"]
            amplitudes = [np.sqrt(a1**2 + b1**2), np.sqrt(a2**2 + b2**2), 
                        np.sqrt(a3**2 + b3**2), np.sqrt(a4**2 + b4**2), np.sqrt(a5**2 + b5**2)]
            dominant_term = np.argmax(amplitudes) + 1
            
            print(f"  - The mean (DC) component is {a0:.2f}")
            print(f"  - The dominant frequency is term {dominant_term}")
            print(f"  - This suggests cycles of approximately {365/dominant_term:.1f} days in the infection pattern")
        
        if fourier_death_fit:
            print("\nFor death data:")
            print(f"  - R² value: {fourier_death_fit['r_squared']:.4f}")
            print(f"  - RMSE: {fourier_death_fit['rmse']:.4f}")
            
            # Analyze the coefficients
            a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 = fourier_death_fit["params"]
            amplitudes = [np.sqrt(a1**2 + b1**2), np.sqrt(a2**2 + b2**2), 
                        np.sqrt(a3**2 + b3**2), np.sqrt(a4**2 + b4**2), np.sqrt(a5**2 + b5**2)]
            dominant_term = np.argmax(amplitudes) + 1
            
            print(f"  - The mean (DC) component is {a0:.2f}")
            print(f"  - The dominant frequency is term {dominant_term}")
            print(f"  - This suggests cycles of approximately {365/dominant_term:.1f} days in the death pattern")
else:
    print("Real data not available for mathematical function fitting.")