# 6h_Stochastic_Top_Bottom_Scalp_v1
# Hypothesis: Identify overbought/oversold conditions using Stochastic Oscillator with
# divergence confirmation and 1-day trend filter. Works in bull (fade tops) and bear
# (fade bottoms) by combining mean reversion with trend context to avoid fighting
# strong trends. Target: 50-150 trades over 4 years for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Stochastic Oscillator (14,3,3)
    lookback = 14
    lowest_low = np.zeros(n)
    highest_high = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - lookback + 1)
        lowest_low[i] = np.min(low[start_idx:i+1])
        highest_high[i] = np.max(high[start_idx:i+1])
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1, denominator)
    
    k_percent = 100 * ((close - lowest_low) / denominator)
    
    # Smooth %K to get %D (3-period SMA of %K)
    k_smooth = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    d_percent = pd.Series(k_smooth).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Stochastic and EMA50
    start_idx = max(50, 14 + 3 + 3)  # 14 for lookback, 3 for %K smoothing, 3 for %D smoothing
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(d_percent[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        k = k_percent[i]
        d = d_percent[i]
        
        if position == 0:
            # Long: Stochastic oversold (<20) with bullish crossover (%K > %D) and uptrend
            if k < 20 and d < 20 and k > d and k > k_percent[i-1] and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: Stochastic overbought (>80) with bearish crossover (%K < %D) and downtrend
            elif k > 80 and d > 80 and k < d and k < k_percent[i-1] and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Stochastic overbought (>80) or trend turns down
            if k > 80 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Stochastic oversold (<20) or trend turns up
            if k < 20 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Stochastic_Top_Bottom_Scalp_v1"
timeframe = "6h"
leverage = 1.0