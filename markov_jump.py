import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def markov_jump_intensity(returns, macro_df, high_vol_threshold=0.02):
    """
    Estimate jump intensity to high‑volatility state using macro covariates.
    """
    # Align returns and macro
    common_idx = returns.index.intersection(macro_df.index)
    if len(common_idx) < 10:
        return 0.0
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    # Binary target: 1 if absolute return > threshold
    y = (np.abs(ret_aligned) > high_vol_threshold).astype(int)
    # Check if both classes present
    if len(np.unique(y)) < 2:
        # No positive examples: return low probability
        return 0.0
    # Check for any NaN in macro (should have been handled, but drop)
    if macro_aligned.isna().any().any():
        # Drop rows with NaN in macro
        valid = ~macro_aligned.isna().any(axis=1)
        macro_aligned = macro_aligned[valid]
        y = y[valid]
        if len(macro_aligned) < 10 or len(np.unique(y)) < 2:
            return 0.0
    # Standardise macro
    scaler = StandardScaler()
    X = scaler.fit_transform(macro_aligned)
    # Fit logistic regression
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X, y)
    # Predict on last macro observation
    last_macro = macro_df.iloc[-1].values.reshape(1, -1)
    # Check last macro for NaN
    if np.any(np.isnan(last_macro)):
        return 0.0
    last_macro_scaled = scaler.transform(last_macro)
    prob_high = model.predict_proba(last_macro_scaled)[0, 1]
    return prob_high

def continuous_time_markov_jump_score(returns, macro_df, high_vol_threshold=0.02):
    """Score = probability of entering high‑volatility state today given macro."""
    if len(returns) < 5 or macro_df is None or len(macro_df) < 5:
        return 0.0
    return markov_jump_intensity(returns, macro_df, high_vol_threshold)
