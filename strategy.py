#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA200 trend filter + volume confirmation
# - Primary signal: 4h price breaks above/below Donchian(20) channel
# - Trend filter: 1d EMA200 - price must be above EMA for longs, below for shorts
# - Volume confirmation: 4h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture momentum, EMA200 filter ensures
#   trades align with higher timeframe trend, reducing false signals in strong trends

name = "4h_1d_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend direction
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe (completed 1d bar only)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) channel
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume regime: volume > 20-period median volume
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
            # Exit: price crosses below Donchian lower band OR price crosses below EMA200
            if close[i] < lowest_low_20[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper band OR price crosses above EMA200
            if close[i] > highest_high_20[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and EMA200 filter
            # Long: price breaks above Donchian upper band AND volume regime AND price above EMA200
            if close[i] > highest_high_20[i] and volume_regime[i] and close[i] > ema_200_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band AND volume regime AND price below EMA200
            elif close[i] < lowest_low_20[i] and volume_regime[i] and close[i] < ema_200_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals