import pandas as pd
from skopt import gp_minimize
from backtesting import Backtest
from strategies import str  # Use str instead of IntegratedStrategy

# Load your data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = [
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Base Asset Volume", "Number of Trades",
        "Taker Buy Volume", "Taker Buy Base Asset Volume", "Ignore"
    ]
    data["Open Time"] = pd.to_datetime(data["Open Time"], unit="ms")
    data.set_index("Open Time", inplace=True)
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    return data

# Objective function for Bayesian Optimization
def objective(params):
    ema_period, rsi_period = params
    str.ema_period = ema_period
    str.rsi_period = rsi_period
    
    bt = Backtest(data, str, cash=1000000, commission=0.00075)
    stats = bt.run()
    return -stats['Equity Final [$]']  # Maximize Sharpe Ratio

# Load data
file_path = "BTCUSDT-1h-2024-11.csv"
data = load_data(file_path)

# Run Bayesian Optimization
result = gp_minimize(
    objective,
    [
        (50, 300),  # EMA period
        (5, 30)     # RSI period
    ],
    n_calls=50,
    random_state=42
)

# Save best parameters and score
best_params = result.x
best_score = -result.fun

# Assign the best parameters to the strategy
str.ema_period, str.rsi_period = best_params

# Final backtest with optimal parameters
bt = Backtest(data, str, cash=1000000, commission=0.00075)
final_stats = bt.run()

# Extract key results
results_text = f"""
Best Parameters:
- EMA Period: {best_params[0]}
- RSI Period: {best_params[1]}

Final Backtest Results:
- Final Equity: ${final_stats['Equity Final [$]']:.2f}
- Total Return: {final_stats['Return [%]']:.2f}%
- Total Trades: {final_stats['_trades']}

Detailed Statistics:
{final_stats}
"""

# Print results to console
print(results_text)

# Save results to a text file
with open("baysian_results.txt", "w") as file:
    file.write(results_text)

