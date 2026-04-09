#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA200 trend filter + volume spike confirmation
# - Primary signal: 4h price breaks above Donchian(20) high for long, below Donchian(20) low for short
# - Trend filter: 1d EMA200 - price must be above EMA200 for longs, below for shorts (avoid counter-trend)
# - Volume confirmation: 4h volume > 1.5 * 20-period median volume (ensure participation)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Stoploss: implicit via Donchian breakout structure - exit when price retouches opposite Donchian level
# - Works in bull/bear: Donchian breakouts capture trends, EMA200 filter avoids false signals in strong opposite trends
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines

name = "4h_1d_donchian_ema_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:  # Need enough for EMA200
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
    
    # 4h Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume regime: volume > 1.5 * 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches or crosses below Donchian low (20-period)
            if close[i] <= lowest_low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or crosses above Donchian high (20-period)
            if close[i] >= highest_high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and EMA200 filter
            # Long: price breaks above Donchian high AND volume spike AND price above EMA200
            if close[i] > highest_high_20[i] and volume_spike[i] and close[i] > ema_200_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume spike AND price below EMA200
            elif close[i] < lowest_low_20[i] and volume_spike[i] and close[i] < ema_200_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals