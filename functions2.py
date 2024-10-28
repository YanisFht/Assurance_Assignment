import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom, lognorm, gamma
from sklearn.metrics import mean_squared_error

def fit_and_plot_poisson(train_data):
    """Fit and plot Poisson distribution for the number of claims."""
    lambda_poisson = train_data['ClaimNb'].mean()
    poisson_dist = poisson(mu=lambda_poisson)

    plt.figure(figsize=(12, 6))
    plt.hist(train_data['ClaimNb'], bins=100, color='skyblue', edgecolor='black', density=True)
    plt.xlabel('Number of Claims')
    plt.ylabel('Frequency')
    plt.title('Distribution of the Number of Claims')
    x = np.arange(0, 10)
    plt.plot(x, poisson_dist.pmf(x), 'r-', lw=2)
    plt.show()

    return lambda_poisson, poisson_dist

def fit_and_plot_negbin(train_data):
    """Fit and plot Negative Binomial distribution for the number of claims."""
    mean_claims = train_data['ClaimNb'].mean()
    var_claims = train_data['ClaimNb'].var()

    r = mean_claims**2 / (var_claims - mean_claims)
    p = mean_claims / var_claims

    neg_bin_dist = nbinom(n=r, p=p)

    plt.figure(figsize=(12, 6))
    plt.hist(train_data['ClaimNb'], bins=100, color='skyblue', edgecolor='black', density=True, label="Observed Data")
    plt.xlabel('Number of Claims')
    plt.ylabel('Frequency')
    plt.title('Fitting Negative Binomial Distribution to Number of Claims')
    x = np.arange(0, 10)
    plt.plot(x, neg_bin_dist.pmf(x), 'r-', lw=2, label=f'NegBin Fit (r={r:.2f}, p={p:.2f})')
    plt.legend()
    plt.show()

    return r, p, neg_bin_dist

def calculate_log_likelihood(distribution, data, dist_type='pmf'):
    if dist_type == 'pmf':
        return distribution.logpmf(data).sum()
    elif dist_type == 'pdf':
        return distribution.logpdf(data).sum()
    
def calculate_aic_bic(distribution, data, dist_type='pmf'):

    if dist_type == 'pmf':
        log_likelihood = distribution.logpmf(data).sum()
    elif dist_type == 'pdf':
        log_likelihood = distribution.logpdf(data).sum()
    
    dist_name = distribution.dist.name
    
    if dist_name == 'poisson':
        k = 1 
    elif dist_name == 'nbinom':
        k = 2 
    elif dist_name in ['lognorm', 'gamma']:
        k = 2  
    
    n = len(data)
    
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return aic, bic



def simulate_frequency(valid_data, lambda_poisson, r, p):
    poisson_forecasts = poisson.rvs(mu=lambda_poisson, size=len(valid_data))
    negbin_forecasts = nbinom.rvs(n=r, p=p, size=len(valid_data))
    
    actual_claims = valid_data['ClaimNb'].values

    mse_poisson = mean_squared_error(actual_claims, poisson_forecasts)
    mse_negbin = mean_squared_error(actual_claims, negbin_forecasts)

    #print('Poisson MSE:', mse_poisson)
    #print('Negative Binomial MSE:', mse_negbin)

    return poisson_forecasts, negbin_forecasts

def fit_and_plot_lognormal(train_data):
    positive_claims = train_data['ClaimAmount'][train_data['ClaimAmount'] > 0]
    log_claims = np.log(positive_claims)
    mu, sigma = np.mean(log_claims), np.std(log_claims)

    shape = sigma
    loc = 0
    scale = np.exp(mu)

    lognorm_dist = lognorm(s=shape, loc=loc, scale=scale)

    plt.figure(figsize=(12, 6))
    plt.hist(positive_claims, bins=100, color='skyblue', edgecolor='black', density=True, alpha=0.6, label="Observed Data")
    x = np.linspace(positive_claims.min(), positive_claims.max(), 1000)
    plt.plot(x, lognorm_dist.pdf(x), 'r-', lw=2, label=f'Lognormal Fit (μ={mu:.2f}, σ={sigma:.2f})')
    plt.xlabel('Claim Amount')
    plt.ylabel('Density')
    plt.title('Fitting Lognormal Distribution to Claim Amount')
    plt.legend()
    plt.show()

    return lognorm_dist

def fit_and_plot_gamma(train_data):
    positive_claims = train_data['ClaimAmount'][train_data['ClaimAmount'] > 0]
    shape, loc, scale = gamma.fit(positive_claims, floc=0)
    gamma_dist = gamma(a=shape, loc=loc, scale=scale)

    plt.figure(figsize=(12, 6))
    plt.hist(positive_claims, bins=100, color='skyblue', edgecolor='black', density=True, alpha=0.6, label="Observed Data")
    x = np.linspace(positive_claims.min(), positive_claims.max(), 1000)
    plt.plot(x, gamma_dist.pdf(x), 'r-', lw=2, label=f'Gamma Fit (shape={shape:.2f}, scale={scale:.2f})')
    plt.xlabel('Claim Amount')
    plt.ylabel('Density')
    plt.title('Fitting Gamma Distribution to Claim Amount')
    plt.legend()
    plt.show()

    return gamma_dist

def simulate_severity(valid_data, lognorm_dist=None, gamma_dist=None):
    
    if lognorm_dist is not None:

        lognorm_forecasts = lognorm_dist.rvs(size=len(valid_data))
    else:
        lognorm_forecasts = None

    if gamma_dist is not None:

        gamma_forecasts = gamma_dist.rvs(size=len(valid_data))
    else:
        gamma_forecasts = None

    actual_claims = valid_data['ClaimAmount'].values

    if lognorm_forecasts is not None:
        mse_lognorm = mean_squared_error(actual_claims, lognorm_forecasts)
        #print('Lognormal MSE:', mse_lognorm)

    if gamma_forecasts is not None:
        mse_gamma = mean_squared_error(actual_claims, gamma_forecasts)
        #print('Gamma MSE:', mse_gamma)

    return lognorm_forecasts, gamma_forecasts


def monte_carlo_simulation(lambda_poisson, shape, scale, n_claims_dist='Poisson', p=None, n_simulations=100_000):
    total_losses_nextyear = np.zeros(n_simulations)

    for i in range(n_simulations):
        if n_claims_dist == 'Poisson':
            n_claims = poisson.rvs(mu=lambda_poisson)
        else:
            n_claims = nbinom.rvs(n=shape, p=p)

        if n_claims > 0:
            claim_amounts = gamma.rvs(a=shape, scale=scale, size=n_claims)
            total_losses_nextyear[i] = np.sum(claim_amounts)
        else:
            total_losses_nextyear[i] = 0

    mean_loss = np.mean(total_losses_nextyear)
    median_loss = np.median(total_losses_nextyear)
    percentile_95 = np.percentile(total_losses_nextyear, 95)
    percentile_99 = np.percentile(total_losses_nextyear, 99)

    print(f"Mean Total Loss: {mean_loss:.2f}")
    print(f"Median Total Loss: {median_loss:.2f}")
    print(f"95th Percentile of Total Loss: {percentile_95:.2f}")
    print(f"99th Percentile of Total Loss: {percentile_99:.2f}")

    plt.figure(figsize=(12, 6))
    plt.hist(total_losses_nextyear, bins=100, color='skyblue', edgecolor='black', density=True)
    plt.xlabel('Total Loss')
    plt.ylabel('Density')
    plt.title('Distribution of Total Losses Next Year')
    plt.show()

    return total_losses_nextyear


