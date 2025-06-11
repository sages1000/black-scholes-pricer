import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes Option Pricing Formula

    Parameters:
    - S: Current stock price
    - K: Strike price
    - T: Time to maturity (in years)
    - r: Risk-free interest rate (e.g., 0.05 for 5%)
    - sigma: Volatility of the stock (standard deviation)
    - option_type: "call" or "put"

    Returns:
    - Option price (float)
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option_type. Use 'call' or 'put'.")

    return price

def compute_prices_vs_volatility(S, K, T, r, option_type):
    volatilities = np.linspace(0.01, 1.0, 100)  # 1% to 100%
    prices = []

    for volatility in volatilities:
        price = black_scholes_price(S, K, T, r, volatility, option_type)
        prices.append(price)

    return volatilities, prices

def plot_prices_vs_volatility(volatilities, prices, option_type):
    plt.plot(volatilities, prices, label=f"{option_type.capitalize()} Option")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Option Price")
    plt.title(f"{option_type.capitalize()} Option Price vs. Volatility")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_both_options_vs_volatility(volatilities, call_prices, put_prices):

    plt.plot(volatilities, call_prices, label="Call Option", color='blue')
    plt.plot(volatilities, put_prices, label="Put Option", color='red')
    
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Option Price")
    plt.title("Call vs. Put Option Price vs. Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()



# Example usage:
if __name__ == "__main__":
    # Define inputs
    S = 100      # Current stock price
    K = 100      # Strike price
    T = 1        # Time to expiration (1 year)
    r = 0.05     # Risk-free interest rate (5%)
    sigma = 0.2  # Volatility (20%)

    # Calculate both call and put prices
    call_price = black_scholes_price(S, K, T, r, sigma, option_type="call")
    put_price = black_scholes_price(S, K, T, r, sigma, option_type="put")

    print(f"Call Option Price: ${call_price:.2f}")
    print(f"Put Option Price:  ${put_price:.2f}")

    vols, prices = compute_prices_vs_volatility(S, K, T, r, "call")
    call_vols, call_prices = compute_prices_vs_volatility(S, K, T, r, "call")
    put_vols, put_prices   = compute_prices_vs_volatility(S, K, T, r, "put")

    plot_both_options_vs_volatility(call_vols, call_prices, put_prices)                                   
    plot_prices_vs_volatility(vols, prices, "call")

