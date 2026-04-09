#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d Elder Ray regime filter
# - Uses 6h Williams %R(14) for mean reversion signals: long when < -80, short when > -20
# - Filters with 1d Elder Ray: Bull Power > 0 for longs, Bear Power < 0 for shorts
# - Only trades when 1d trend is established (avoids choppy markets)
# - Exits when Williams %R reverts to -50 (mean reversion target)
# - Position size: 0.25 (25% of capital) to balance risk and minimize fee drag
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to minimize fee drag
# - Williams %R captures overextended moves; Elder Ray ensures trading with the 1d trend
# - Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)

name = "6h_1d_williamsr_elderray_v2"
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
    
    # 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align 1d Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denominator = highest_high_14 - lowest_low_14
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    williams_r = -100 * (highest_high_14 - close) / denominator
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R reverts to -50 (mean reversion target)
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R reverts to -50 (mean reversion target)
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with Elder Ray regime filter
            # Long: Williams %R oversold (< -80) AND Bull Power > 0 (uptrend)
            if williams_r[i] < -80 and bull_power_aligned[i] > 0:
                position = 1
                signals[i] = 0.25
            # Short: Williams %R overbought (> -20) AND Bear Power < 0 (downtrend)
            elif williams_r[i] > -20 and bear_power_aligned[i] < 0:
                position = -1
                signals[i] = -0.25
    
    return signals