import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def markov_jump_intensity(returns, macro_df, high_vol_threshold=0.02):
    # Align
    common_idx = returns.index.intersection(macro_df.index)
    if len(common_idx) < 5:
        return 0.01
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    # Target: 1 if |return| > threshold
    y = (np.abs(ret_aligned) > high_vol_threshold).astype(int)
    # Add pseudo‑events to avoid zero class
    if np.sum(y) == 0:
        # add one artificial event at the median macro
        y[0] = 1
    X = macro_aligned.values
    # Ridge regression (logistic is not stable, so we use linear regression on probability?)
    # Simpler: use Ridge to predict y directly
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_scaled, y)
    # Predict for today's macro
    last_macro = macro_df.iloc[-1].values.reshape(1, -1)
    last_scaled = scaler.transform(last_macro)
    intensity = model.predict(last_scaled)[0]
    # Clamp to [0,1] and ensure >0
    intensity = np.clip(intensity, 0.01, 0.99)
    return intensity

def continuous_time_markov_jump_score(returns, macro_df, high_vol_threshold=0.02):
    if len(returns) < 5 or macro_df is None:
        return 0.01
    return markov_jump_intensity(returns, macro_df, high_vol_threshold)
