import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class MLStrategy:
    def __init__(self):
        """Initialize the ML strategy."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'moneyness',
            'days_to_expiry',
            'implied_volatility',
            'volume',
            'open_interest',
            'delta',
            'gamma',
            'theta',
            'vega'
        ]

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the model.
        
        Args:
            data: Raw options data
            
        Returns:
            Processed features DataFrame
        """
        # Calculate moneyness
        data['moneyness'] = data['spot_price'] / data['strike_price']
        
        # Calculate days to expiry
        data['days_to_expiry'] = (data['expiry_date'] - data['date']).dt.days
        
        # Normalize features
        features = data[self.feature_columns].copy()
        features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns
        )
        
        return features

    def prepare_target(self, data: pd.DataFrame, 
                      target_type: str = 'return') -> pd.Series:
        """
        Prepare target variable for the model.
        
        Args:
            data: Raw options data
            target_type: Type of target ('return' or 'iv')
            
        Returns:
            Target Series
        """
        if target_type == 'return':
            # Calculate future returns
            data['future_return'] = data.groupby('option_id')['price'].shift(-1) / \
                                  data['price'] - 1
            return data['future_return']
        else:
            # Use implied volatility as target
            return data['implied_volatility']

    def train(self, data: pd.DataFrame, target_type: str = 'return'):
        """
        Train the XGBoost model.
        
        Args:
            data: Raw options data
            target_type: Type of target ('return' or 'iv')
        """
        # Prepare features and target
        X = self.prepare_features(data)
        y = self.prepare_target(data, target_type)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training R² score: {train_score:.4f}")
        print(f"Testing R² score: {test_score:.4f}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: Raw options data
            
        Returns:
            Array of predictions
        """
        X = self.prepare_features(data)
        return self.model.predict(X)

    def plot_feature_importance(self):
        """Plot feature importance."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

    def backtest(self, data: pd.DataFrame, threshold: float = 0.02) -> Dict:
        """
        Backtest the strategy.
        
        Args:
            data: Raw options data
            threshold: Return threshold for trading signals
            
        Returns:
            Dictionary containing backtest results
        """
        # Get predictions
        predictions = self.predict(data)
        
        # Generate trading signals
        signals = np.where(predictions > threshold, 1, 0)
        
        # Calculate returns
        data['strategy_return'] = data['future_return'] * signals
        
        # Calculate performance metrics
        total_return = (1 + data['strategy_return']).prod() - 1
        sharpe_ratio = data['strategy_return'].mean() / data['strategy_return'].std() * np.sqrt(252)
        max_drawdown = (data['strategy_return'].cumsum() - \
                       data['strategy_return'].cumsum().cummax()).min()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

if __name__ == "__main__":
    # Example usage
    # Load data (replace with actual data loading)
    data = pd.read_csv("data/options_data.csv")
    
    # Initialize and train strategy
    strategy = MLStrategy()
    strategy.train(data)
    
    # Plot feature importance
    strategy.plot_feature_importance()
    
    # Backtest strategy
    results = strategy.backtest(data)
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}") 