#!/usr/bin/env python3
"""
#100801 - 4h_Donchian20_1dTrend_VolumeFilter_2025
Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume spike confirmation.
Works in bull: breakouts above upper band with strong trend and volume.
Works in bear: breakouts below lower band with strong trend and volume (shorts).
Volume filter prevents false breakouts. Trend filter ensures alignment with higher timeframe.
Target: 20-50 trades/year to minimize fee drag. Uses discrete position sizing (0.25) to reduce churn.
"""

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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above upper Donchian, above 1d EMA200, volume spike
        if (close[i] > high_roll[i] and 
            close[i] > ema200_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below lower Donchian, below 1d EMA200, volume spike
        elif (close[i] < low_roll[i] and 
              close[i] < ema200_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to middle of Donchian channel
        elif position == 1 and close[i] < (high_roll[i] + low_roll[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (high_roll[i] + low_roll[i]) / 2:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dTrend_VolumeFilter_2025"
timeframe = "4h"
leverage = 1.0