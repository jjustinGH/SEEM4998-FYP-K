import pandas as pd
import numpy as np
import optuna

sentiment_df = pd.read_csv(r"C:\Users\user\Desktop\Yr4HW\CSCI2040\FYP\sentiment_history_monthly.csv")
price_df = pd.read_csv(r"C:\Users\user\Desktop\Yr4HW\CSCI2040\FYP\24Jan_to_Jun_price.csv")
price_df['Date'] = pd.to_datetime(price_df['Date'].str.split(' ').str[0]) 
stocks = sentiment_df['stock'].unique().tolist()

returns = {}
for stock in stocks:
    prices = price_df[['Date', stock]].sort_values('Date')
    prices['return'] = prices[stock].pct_change()  # (price_t - price_{t-1}) / price_{t-1}
    returns[stock] = prices[['Date', 'return']].dropna()

months = sentiment_df['month'].unique().tolist()
data_rows = []

for month in months:
    start_date = pd.to_datetime(month + '-01')
    end_date = start_date + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    next_start = start_date + pd.DateOffset(months=1)
    next_end = start_date + pd.DateOffset(months=4) - pd.Timedelta(days=1)
    
    for stock in stocks:
        # Get S and C_base from sentiment (though C_base not used in r_pred)
        sent_row = sentiment_df[(sentiment_df['month'] == month) & (sentiment_df['stock'] == stock)]
        if sent_row.empty:
            continue
        S = sent_row['S'].values[0]
        C_base = sent_row['C_base'].values[0]  # Included but not used yet
        
        # mu: mean daily return for the month
        stock_returns = returns[stock]
        month_returns = stock_returns[(stock_returns['Date'] >= start_date) & (stock_returns['Date'] <= end_date)]['return']
        mu = month_returns.mean() if not month_returns.empty else np.nan
        
        # actual_r_next: mean daily return for next 3 months (use available data up to Jun)
        next_returns = stock_returns[(stock_returns['Date'] >= next_start) & (stock_returns['Date'] <= next_end)]['return']
        actual_r_next = next_returns.mean() if not next_returns.empty else np.nan
        
        if not np.isnan(mu) and not np.isnan(actual_r_next):
            data_rows.append({
                'period': month,
                'stock': stock,
                'mu': mu,
                'S': S,
                'C_base': C_base,  # Optional
                'actual_r_next': actual_r_next
            })

df = pd.DataFrame(data_rows)
print(f"Loaded {len(df)} data points for BO.")
# -----------------------------
# Define your predictive model
# -----------------------------

def predict_r(mu, S, theta):
    a1, b1, a2, b2, c, a3, b3, d, bound1, bound2 = theta
    if mu > bound2:
        return mu * a1 + b1 * S
    elif bound1 < mu <= bound2:
        return (mu * a2 + b2 * S) * c
    else:
        return max(d, mu * a3 + b3 * S)

# -----------------------------
# Objective function for Optuna
# -----------------------------

def objective(trial):
    # 8 parameters, specify bounds
    a1 = trial.suggest_float('a1', 0.2, 0.9)
    b1 = trial.suggest_float('b1', 0.000, 0.01)
    a2 = trial.suggest_float('a2', 0.2, 0.9)
    b2 = trial.suggest_float('b2', 0.000, 0.01)
    c  = trial.suggest_float('c', 0.8, 1.3)
    a3 = trial.suggest_float('a3', 0.05, 0.4)
    b3 = trial.suggest_float('b3', 0.000, 0.005)
    d  = trial.suggest_float('d', 0.0000, 0.001)
    bound1 = trial.suggest_float('bound1', -0.01, 0.01)
    bound2 = trial.suggest_float('bound2', 0.001, 0.01)

    theta = [a1, b1, a2, b2, c, a3, b3, d, bound1, bound2]
    preds = df.apply(lambda row: predict_r(row['mu'], row['S'], theta), axis=1)
    mse = ((preds - df['actual_r_next']) ** 2).mean()
    return mse

# -----------------------------
# Bayesian Optimization with Optuna
# -----------------------------

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("Best parameters:", study.best_params)
    print("Lowest validation MSE:", study.best_value)
