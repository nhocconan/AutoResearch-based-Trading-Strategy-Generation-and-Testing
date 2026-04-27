# 1d_KellyRatio_VolatilityAdjusted_V1
# Hypothesis: Kelly criterion with volatility scaling allocates optimal position size based on 20-day return/risk ratio, scaled by volatility regime.
# Uses 1w trend filter to align with higher timeframe momentum. Works in bull/bear by reducing size in high volatility and increasing in trending markets.
# Targets 10-20 trades/year to minimize fee drag on daily timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 20-day returns for Kelly numerator
    returns_20 = np.zeros(n)
    returns_20[20:] = (close[20:] - close[:-20]) / close[:-20]
    
    # Calculate 20-day volatility for Kelly denominator (annualized)
    returns_daily = np.diff(close) / close[:-1]
    returns_daily = np.concatenate([[0], returns_daily])
    vol_20 = pd.Series(returns_daily).rolling(window=20, min_periods=20).std() * np.sqrt(252)
    vol_20 = vol_20.values
    
    # Kelly ratio = (expected return) / (variance) - using 20-day return over 20-day variance
    kelly_ratio = np.zeros(n)
    for i in range(20, n):
        if vol_20[i] > 0 and not np.isnan(vol_20[i]):
            kelly_ratio[i] = returns_20[i] / (vol_20[i] ** 2)
        else:
            kelly_ratio[i] = 0
    
    # Scale Kelly by volatility regime (inverse volatility scaling)
    vol_median = np.nanmedian(vol_20[20:]) if np.any(~np.isnan(vol_20[20:])) else 0.5
    vol_scaling = np.where(vol_20 > 0, vol_median / vol_20, 1.0)
    vol_scaling = np.clip(vol_scaling, 0.5, 2.0)  # Limit scaling to prevent extreme positions
    
    # Final position size: Kelly * volatility scaling, capped at 0.30
    raw_position = kelly_ratio * vol_scaling
    position_size = np.clip(raw_position, -0.30, 0.30)
    
    # Apply trend filter: only take positions aligned with weekly trend
    signals = np.zeros(n)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(position_size[i]):
            signals[i] = 0.0
            continue
            
        # Only take long positions in uptrend, short in downtrend
        if position_size[i] > 0 and close[i] > ema50_1w_aligned[i]:
            signals[i] = position_size[i]
        elif position_size[i] < 0 and close[i] < ema50_1w_aligned[i]:
            signals[i] = position_size[i]
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KellyRatio_VolatilityAdjusted_V1"
timeframe = "1d"
leverage = 1.0