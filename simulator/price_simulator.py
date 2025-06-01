import numpy as np
import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt

class PriceSimulator:
    def __init__(self, initial_price: float, mu: float, sigma: float, dt: float):
        """
        Initialize the price simulator.
        
        Args:
            initial_price: Starting price
            mu: Drift parameter
            sigma: Volatility parameter
            dt: Time step size
        """
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def gbm_simulate(self, n_steps: int, n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Geometric Brownian Motion paths.
        
        Args:
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            
        Returns:
            Tuple of (time points, price paths)
        """
        # Generate time points
        t = np.linspace(0, n_steps * self.dt, n_steps + 1)
        
        # Generate random walks
        dW = np.random.normal(0, np.sqrt(self.dt), (n_paths, n_steps))
        
        # Calculate drift and diffusion terms
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * dW
        
        # Calculate cumulative returns
        returns = drift + diffusion
        cum_returns = np.cumsum(returns, axis=1)
        
        # Calculate price paths
        prices = self.initial_price * np.exp(cum_returns)
        prices = np.insert(prices, 0, self.initial_price, axis=1)
        
        return t, prices

    def ou_simulate(self, n_steps: int, n_paths: int, 
                   mean_reversion: float, long_term_mean: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Ornstein-Uhlenbeck process paths.
        
        Args:
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            mean_reversion: Mean reversion speed
            long_term_mean: Long-term mean level
            
        Returns:
            Tuple of (time points, price paths)
        """
        # Generate time points
        t = np.linspace(0, n_steps * self.dt, n_steps + 1)
        
        # Initialize price paths
        prices = np.zeros((n_paths, n_steps + 1))
        prices[:, 0] = self.initial_price
        
        # Generate random walks
        dW = np.random.normal(0, np.sqrt(self.dt), (n_paths, n_steps))
        
        # Simulate OU process
        for i in range(n_steps):
            prices[:, i+1] = prices[:, i] + mean_reversion * (long_term_mean - prices[:, i]) * self.dt + \
                            self.sigma * dW[:, i]
        
        return t, prices

    def save_to_csv(self, t: np.ndarray, prices: np.ndarray, filename: str):
        """
        Save simulation results to CSV file.
        
        Args:
            t: Time points
            prices: Price paths
            filename: Output filename
        """
        # Create DataFrame
        df = pd.DataFrame(prices.T, index=t)
        df.columns = [f'Path_{i+1}' for i in range(prices.shape[0])]
        
        
        df.to_csv(filename)
        print(f"Saved simulation results to {filename}")

    def plot_paths(self, t: np.ndarray, prices: np.ndarray, title: str):
        """
        Plot simulated price paths.
        
        Args:
            t: Time points
            prices: Price paths
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        for i in range(prices.shape[0]):
            plt.plot(t, prices[i], alpha=0.5)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Example usage
    simulator = PriceSimulator(
        initial_price=100.0,
        mu=0.05,
        sigma=0.2,
        dt=1/252  # Daily time step
    )
    
    # Simulate GBM
    t_gbm, prices_gbm = simulator.gbm_simulate(n_steps=252, n_paths=1000)
    simulator.save_to_csv(t_gbm, prices_gbm, "data/simulator_output_gbm.csv")
    simulator.plot_paths(t_gbm, prices_gbm, "GBM Price Paths")
    
    # Simulate OU
    t_ou, prices_ou = simulator.ou_simulate(
        n_steps=252,
        n_paths=1000,
        mean_reversion=0.1,
        long_term_mean=100.0
    )
    simulator.save_to_csv(t_ou, prices_ou, "data/simulator_output_ou.csv")
    simulator.plot_paths(t_ou, prices_ou, "OU Price Paths") 