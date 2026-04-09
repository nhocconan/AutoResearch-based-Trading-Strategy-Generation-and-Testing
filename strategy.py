#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter
# - Uses 6h Williams %R(14) for oversold/overbought signals (long < -80, short > -20)
# - Filters by 1d EMA(50) trend: only long when price > EMA50, short when price < EMA50
# - Exits when Williams %R reverts to opposite extreme (> -20 for longs, < -80 for shorts)
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag
# - Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# - Williams %R identifies exhaustion points; 1d EMA ensures we trade with higher timeframe trend

name = "6h_1d_williamsr_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or
            ema_50_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R reverts above -20 (overbought)
            if williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R reverts below -80 (oversold)
            if williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with 1d trend filter
            if (williams_r[i] < -80 and  # Oversold
                close_6h[i] > ema_50_1d_aligned[i]):  # Uptrend filter
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] > -20 and   # Overbought
                  close_6h[i] < ema_50_1d_aligned[i]):  # Downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals