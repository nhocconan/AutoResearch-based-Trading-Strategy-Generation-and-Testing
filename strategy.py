#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d Elder Ray regime filter
# - Uses 6h Williams %R(14) for oversold/overbought signals (long < -80, short > -20)
# - Filters by 1d Elder Ray regime: Bull Power > 0 for longs, Bear Power < 0 for shorts
# - Elder Ray confirms trend strength: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Only takes mean-reversion trades in the direction of the 1d trend (as measured by Elder Ray)
# - Exits when Williams %R reverts to midpoint (-50) or opposite extreme
# - Position size: 0.25 (25% of capital) to limit drawdown in volatile 6h markets
# - Target: 12-25 trades/year on 6h timeframe (48-100 total over 4 years) to minimize fee drag
# - Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# - Williams %R provides timely mean-reversion signals, Elder Ray filters for trend alignment

name = "6h_1d_williamsr_elderray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(13) for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denominator = highest_high - lowest_low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    williams_r = -100 * (highest_high - close) / denominator
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R reverts to midpoint (-50) or reaches overbought (-20)
            if williams_r[i] >= -50:  # Reverted to midpoint or above
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R reverts to midpoint (-50) or reaches oversold (-80)
            if williams_r[i] <= -50:  # Reverted to midpoint or below
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme with Elder Ray regime confirmation
            if (williams_r[i] <= -80 and      # Oversold
                bull_power_aligned[i] > 0):   # Bullish regime (Bull Power > 0)
                position = 1
                signals[i] = 0.25
            elif (williams_r[i] >= -20 and    # Overbought
                  bear_power_aligned[i] < 0): # Bearish regime (Bear Power < 0)
                position = -1
                signals[i] = -0.25
    
    return signals