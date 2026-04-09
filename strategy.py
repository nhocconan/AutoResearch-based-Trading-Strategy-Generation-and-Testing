#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d EMA200 trend filter + volume spike confirmation
# - Primary signal: 6h price breaks above Donchian(20) high for long, below Donchian(20) low for short
# - Trend filter: 1d EMA200 - price must be above EMA200 for longs, below for shorts (higher timeframe alignment)
# - Volume confirmation: 6h volume > 1.5 * 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, EMA200 filter ensures alignment with higher timeframe trend, volume confirmation reduces false breakouts

name = "6h_1d_donchian_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Pre-compute 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute Donchian channels on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Donchian(20) upper and lower bands
    highest_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # 6h volume regime: volume > 1.5 * 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian(20) low OR price crosses below 1d EMA200
            if close_6h[i] < lowest_low[i] or close_6h[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian(20) high OR price crosses above 1d EMA200
            if close_6h[i] > highest_high[i] or close_6h[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and 1d EMA200 filter
            # Long: price breaks above Donchian(20) high AND volume spike AND price above 1d EMA200
            if (close_6h[i] > highest_high[i] and 
                volume_spike[i] and 
                close_6h[i] > ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian(20) low AND volume spike AND price below 1d EMA200
            elif (close_6h[i] < lowest_low[i] and 
                  volume_spike[i] and 
                  close_6h[i] < ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals