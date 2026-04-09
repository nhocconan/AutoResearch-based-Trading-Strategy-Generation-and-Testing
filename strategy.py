#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w EMA200 trend filter + volume confirmation
# - Primary signal: 12h price breaks above/below 20-period Donchian channel
# - Trend filter: 1w EMA200 - price must be above EMA for longs, below for shorts
# - Volume confirmation: 12h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, EMA200 filter ensures alignment with
#   higher timeframe trend, reducing false signals in counter-trend moves

name = "12h_1w_donchian_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 210:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    
    # 1w EMA200 for trend direction
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 12h timeframe (completed 1w bar only)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower OR price crosses below EMA200
            if close[i] < lowest_low_20[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper OR price crosses above EMA200
            if close[i] > highest_high_20[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and EMA200 filter
            # Long: price > Donchian upper AND volume regime AND price above EMA200
            if close[i] > highest_high_20[i] and volume_regime[i] and close[i] > ema_200_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price < Donchian lower AND volume regime AND price below EMA200
            elif close[i] < lowest_low_20[i] and volume_regime[i] and close[i] < ema_200_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals