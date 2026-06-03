import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def markov_jump_intensity(returns, macro_df, high_vol_threshold=0.02):
    """
    Estimate jump intensity to high‑volatility state using macro covariates.
    Uses logistic regression where target is 1 if |return| > threshold.
    Returns predicted probability for the last macro observation.
    """
    # Align returns and macro
    common_idx = returns.index.intersection(macro_df.index)
    if len(common_idx) < 10:
        return 0.0
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    # Drop any remaining NaN in macro
    macro_aligned = macro_aligned.dropna()
    ret_aligned = ret_aligned[macro_aligned.index]
    if len(ret_aligned) < 10:
        return 0.0
    # Binary target: 1 if absolute return > threshold
    y = (np.abs(ret_aligned) > high_vol_threshold).astype(int)
    # Standardise macro
    scaler = StandardScaler()
    X = scaler.fit_transform(macro_aligned)
    # Fit logistic regression
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X, y)
    # Predict on last macro observation (ensure no NaN)
    last_macro = macro_df.iloc[-1].values.reshape(1, -1)
    # Remove any NaN in last_macro by imputing with mean? For simplicity, if any NaN, return 0
    if np.any(np.isnan(last_macro)):
        return 0.0
    last_macro_scaled = scaler.transform(last_macro)
    prob_high = model.predict_proba(last_macro_scaled)[0, 1]
    return prob_high

def continuous_time_markov_jump_score(returns, macro_df, high_vol_threshold=0.02):
    """Score = probability of entering high‑volatility state today given macro."""
    if len(returns) < 5 or macro_df is None or macro_df.empty:
        return 0.0
    return markov_jump_intensity(returns, macro_df, high_vol_threshold)
