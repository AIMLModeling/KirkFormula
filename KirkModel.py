import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Calculate volatility parameters
r = 0.02
sigma1 = 0.3
T = 0.5
S0_1 = 100
sigma2 = 0.2
S0_2 = 100
def calculate_averages(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must be of equal length")
    averages = []
    for i in range(len(arr1)):
        avg = (arr1[i] + arr2[i]) / 2
        averages.append(avg)
    return averages

# Function to evaluate the formula for S3, given rho
def function_S3_K_rho(rho):
    # Simulating Brownian motions
    n = 5000000
    W = np.random.normal(0, 1, n)
    Z = np.random.normal(0, 1, n)
    # Generating values for K from 0 to 20
    K = np.arange(0, 21)
    # Initialize list to store S3 means for each K
    S3_means = []
    invS3_means = []
    for k in K:
        # Calculating B to have the same correlation as W and Z
        B = rho * W + np.sqrt(1 - rho**2) * Z
        # Simulating S1 and S2
        r, sigma1, sigma2, T, S0_1, S0_2 = 0, 0.3, 0.2, 0.5, 100, 100
        S1 = S0_1 * np.exp((r - (sigma1**2) / 2) * T + sigma1 * np.sqrt(T) * W)
        S2 = S0_2 * np.exp((r - (sigma2**2) / 2) * T + sigma2 * np.sqrt(T) * B)
        # Simulating inverted S1 and S2
        invW = -W
        invZ = -Z
        invB = rho * invW + np.sqrt(1 - rho**2) * invZ
        invS1 = S0_1 * np.exp((r - (sigma1**2) / 2) * T + sigma1 * np.sqrt(T) * invW)
        invS2 = S0_2 * np.exp((r - (sigma2**2) / 2) * T + sigma2 * np.sqrt(T) * invB)
        # Calculating S3
        S3 = np.maximum(S1 - S2 - k, 0)
        S3_mean = np.mean(S3)
        # Append mean to list
        S3_means.append(S3_mean)
        # Calculating invS3
        invS3 = np.maximum(invS1 - invS2 - k, 0)
        invS3_mean = np.mean(invS3)
        # Append mean to list
        invS3_means.append(invS3_mean)
        S3_total_mean = calculate_averages(S3_means, invS3_means)
    return np.array(S3_total_mean)

# Applying function_S3_K_rho to matrix_rho
matrix_rho = np.array([0.80, 0.85, 0.90, 0.95, 0.999]).reshape(-1, 1)
results_function_S3_K_rho = np.apply_along_axis(function_S3_K_rho, 1, matrix_rho)
# Create a meshgrid for k and rho
K = np.arange(0, 21)
rho_values = np.array([0.80, 0.85, 0.90, 0.95, 0.999])
K_mesh, rho_mesh = np.meshgrid(K, rho_values)
# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(rho_mesh, K_mesh, results_function_S3_K_rho, cmap='viridis', edgecolor='none')
ax.set_xlabel('rho')
ax.set_ylabel('K')
ax.set_zlabel('Spread Option Price')
ax.set_title('Average Price of Monte Carlo SImulation')
plt.show()

# Function to evaluate Kirk's approximation for spread option prices given rho
def function_Kirk_K_rho(rho):
    # Define the range of K values
    K = np.arange(0, 21)
    results = [function_Kirk_K(k, rho) for k in K]
    return np.array(results)

def function_Kirk_K(K, rho):
    # Calculate sigma
    sigma = np.sqrt(sigma1**2 - 2 * rho * sigma1 * sigma2 * (S0_2 / (S0_2 + K)) + (sigma2**2) * ((S0_2 / (S0_2 + K))**2))
    # Calculate d1 and d2
    S = (S0_1 / (S0_2 + K))
    d1 = (np.log(S) + 1/2 * (sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # Calculate N(d1) and N(d2)
    N_d1 = norm.cdf(d1, loc=0, scale=1)
    N_d2 = norm.cdf(d2, loc=0, scale=1)
    # Calculate the spread option price using Kirk's formula
    C_kirk = (np.exp(-r * T)) * ((S0_1 * N_d1) - ((S0_2 + K) * N_d2))
    return C_kirk

# Apply the function_Kirk_K_rho to each rho value
results_function_Kirk_K_rho = np.apply_along_axis(function_Kirk_K_rho, 1, matrix_rho)
results_function_Kirk_K_rho = np.squeeze(results_function_Kirk_K_rho)
# Function to calculate Kirk's formula for spread option prices
def function_Kirk_modif_K(K, rho):
    sigma = np.sqrt(sigma1**2 - 2*rho*sigma1*sigma2*(S0_2/(S0_2 + K)) + (sigma2**2) * ((S0_2/(S0_2 + K))**2))
    X_t = np.log(S0_1)
    Y_t = np.log(S0_2 + K)
    I_t = np.sqrt(sigma**2) + 1/2 * (((sigma2 * S0_2/(S0_2 + K)) - rho*sigma1)**2) * (1 / ((np.sqrt(sigma**2))**3)) * (sigma2**2) * ((S0_2 * K) / ((S0_2 + K)**2)) * (X_t - Y_t)
    S = (S0_1 / (S0_2 + K))
    d1_modif = (np.log(S) + 1/2 * (I_t**2) * T) / (I_t * (np.sqrt(T)))
    d2_modif = d1_modif - I_t * np.sqrt(T)
    N_d1_modif = norm.cdf(d1_modif, loc=0, scale=1)
    N_d2_modif = norm.cdf(d2_modif, loc=0, scale=1)
    C_kirk_modif = (np.exp(-r*T)) * ((S0_1*N_d1_modif) - ((S0_2 + K) * N_d2_modif))
    return C_kirk_modif

# Function to calculate Kirk's formula for spread option prices for each rho value
def function_Kirk_modif_K_rho(rho):
    K = np.arange(0, 21)
    results = [function_Kirk_modif_K(k, rho) for k in K]
    return np.array(results)



# Applying the matrix_rho values to the function_Kirk_modif_K_rho function
results_function_Kirk_modif_K_rho = np.apply_along_axis(function_Kirk_modif_K_rho, 1, matrix_rho)
results_function_Kirk_modif_K_rho = np.squeeze(results_function_Kirk_modif_K_rho)

# Calculating the error produced by Kirk's formula for each rho
error_Kirk = ((results_function_Kirk_K_rho * 100) / results_function_S3_K_rho) - 100
error_Kirk_modif = ((results_function_Kirk_modif_K_rho * 100) / results_function_S3_K_rho) - 100

for idx, rho in enumerate(rho_values):
    plt.plot(error_Kirk[idx,:], label=f'Original Kirk formula', color='blue')
    plt.plot(error_Kirk_modif[idx,:], label=f'Modified Kirk formula', color='red')
    plt.xlabel('K')
    plt.ylabel(f'Error of Kirk Formula ({rho})')
    plt.legend()
    plt.show()

