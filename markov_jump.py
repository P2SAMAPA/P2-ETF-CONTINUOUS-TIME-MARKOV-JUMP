import numpy as np

def markov_jump_intensity(returns, macro_df, high_vol_threshold=0.02):
    """
    Compute macro‑adjusted jump intensity by binning macro into high/low.
    Intensity = proportion of high‑vol days given macro regime.
    Uses the first macro column (e.g., VIX) as the driver.
    """
    # Align returns and macro
    common_idx = returns.index.intersection(macro_df.index)
    if len(common_idx) < 10:
        return 0.01
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    
    # Binary target: 1 if |return| > threshold
    high_vol = (np.abs(ret_aligned) > high_vol_threshold).astype(int)
    
    # Use first macro column (VIX or DXY) as regime driver
    first_macro = macro_aligned.iloc[:, 0].values
    median = np.median(first_macro)
    high_macro = (first_macro > median).astype(int)
    
    # Compute intensities for low and high macro regimes
    mask_low = (high_macro == 0)
    mask_high = (high_macro == 1)
    intensity_low = np.mean(high_vol[mask_low]) if np.any(mask_low) else 0.01
    intensity_high = np.mean(high_vol[mask_high]) if np.any(mask_high) else 0.01
    
    # Current macro value (last day)
    current_macro = macro_df.iloc[-1, 0]
    if np.isnan(current_macro):
        return 0.01
    
    # Select intensity based on current macro regime
    intensity = intensity_high if current_macro > median else intensity_low
    # Ensure strictly positive
    intensity = max(intensity, 0.01)
    return min(intensity, 1.0)  # cap at 1

def continuous_time_markov_jump_score(returns, macro_df, high_vol_threshold=0.02):
    if len(returns) < 5 or macro_df is None or macro_df.empty:
        return 0.01
    return markov_jump_intensity(returns, macro_df, high_vol_threshold)
