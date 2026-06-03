import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

def markov_jump_intensity(returns, macro_df, high_vol_threshold=0.02):
    """
    Estimate jump intensity to high‑volatility state using macro covariates.
    Model: λ(t) = exp(β₀ + β·macro(t))
    The "jump" is defined as a transition to a high‑volatility day (|return| > threshold).
    We estimate a Poisson regression where the count of high‑vol days in a rolling window?
    Alternatively, treat each day as a Bernoulli trial with probability of being high‑vol,
    and model log‑odds (logistic regression) with macro covariates.
    Here we use logistic regression to estimate P(high_vol | macro). Then the score = predicted probability.
    """
    # Align returns and macro
    common_idx = returns.index.intersection(macro_df.index)
    if len(common_idx) < 10:
        return 0.0
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    # Binary target: 1 if absolute return > threshold
    y = (np.abs(ret_aligned) > high_vol_threshold).astype(int)
    # Standardise macro
    scaler = StandardScaler()
    X = scaler.fit_transform(macro_aligned)
    # Fit logistic regression (or Poisson with offset = 1 day)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X, y)
    # Predict on last macro observation
    last_macro = macro_df.iloc[-1].values.reshape(1, -1)
    last_macro_scaled = scaler.transform(last_macro)
    prob_high = model.predict_proba(last_macro_scaled)[0, 1]
    return prob_high

def continuous_time_markov_jump_score(returns, macro_df, high_vol_threshold=0.02):
    """Score = probability of entering high‑volatility state today given macro."""
    if len(returns) < 5 or macro_df is None:
        return 0.0
    return markov_jump_intensity(returns, macro_df, high_vol_threshold)
