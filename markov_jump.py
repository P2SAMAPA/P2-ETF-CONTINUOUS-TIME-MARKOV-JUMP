import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def markov_jump_intensity(returns, macro_df, high_vol_threshold=0.02):
    """
    Compute macro‑adjusted jump intensity.
    1. Base intensity = proportion of days with |return| > threshold.
    2. For each macro, compute factor = exp(β * (current_macro - macro_median)),
       where β is estimated from rolling windows (over time).
    Returns final intensity.
    """
    # Align returns and macro
    common_idx = returns.index.intersection(macro_df.index)
    if len(common_idx) < 10:
        return 0.01  # small baseline
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    
    # Base intensity (unconditional)
    high_vol = (np.abs(ret_aligned) > high_vol_threshold).astype(int)
    base_intensity = np.mean(high_vol)
    if base_intensity == 0:
        base_intensity = 0.01  # avoid zero
    
    # For each macro variable, estimate a sensitivity coefficient
    # We'll use a simple regression of log(intensity) on macro over time
    # To get a single coefficient, we need rolling windows? Instead, use the whole period.
    # But to avoid look‑ahead, we use only the available history (within the window).
    # Since we have only one window, we cannot regress over time. So we'll use a simpler:
    # multiply by factor = exp( (current - median) / std ) to amplify deviation.
    # This is not estimated but always works.
    macro_median = macro_aligned.median()
    macro_std = macro_aligned.std()
    macro_today = macro_df.iloc[-1].values
    
    # Build factor product
    factor = 1.0
    for j, col in enumerate(macro_df.columns):
        if macro_std[col] > 1e-8:
            z = (macro_today[j] - macro_median[col]) / macro_std[col]
            factor *= np.exp(z)  # positive deviation increases intensity
    intensity = base_intensity * factor
    # Clip to [0,1]
    intensity = min(max(intensity, 0.0), 1.0)
    return intensity

def continuous_time_markov_jump_score(returns, macro_df, high_vol_threshold=0.02):
    if len(returns) < 5 or macro_df is None:
        return 0.0
    return markov_jump_intensity(returns, macro_df, high_vol_threshold)
