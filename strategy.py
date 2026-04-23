#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Donchian levels calculated from prior 1w bar's high-low
- Long: Close breaks above upper band (20-period high) + price > 1w EMA50 (uptrend) + volume > 1.5x 20-period avg
- Short: Close breaks below lower band (20-period low) + price < 1w EMA50 (downtrend) + volume > 1.5x 20-period avg
- Exit: Close reverts to mid-band (mean of upper/lower) OR opposite breakout
- 1w EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag on 1d timeframe
- Works in both bull (trend continuation via breakouts) and bear (mean reversion via mid-band returns)
"""

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
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data ONCE before loop for Donchian calculation and EMA50
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels for each 1w bar
    # Upper band = 20-period high, Lower band = 20-period low, Mid-band = (upper+lower)/2
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    upper_1w = high_series.rolling(window=20, min_periods=20).max().values
    lower_1w = low_series.rolling(window=20, min_periods=20).min().values
    mid_1w = (upper_1w + lower_1w) / 2.0
    
    # Align Donchian levels to 1d timeframe (available after 1w bar closes)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    mid_aligned = align_htf_to_ltf(prices, df_1w, mid_1w)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume MA, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(mid_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above upper band + price > 1w EMA50 (uptrend) + volume spike
            if volume_spike and close[i] > upper_aligned[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower band + price < 1w EMA50 (downtrend) + volume spike
            elif volume_spike and close[i] < lower_aligned[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close returns to mid-band (mean reversion) OR breaks below lower band (reversal)
            if close[i] <= mid_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close returns to mid-band (mean reversion) OR breaks above upper band (reversal)
            if close[i] >= mid_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0