#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with 12h EMA Trend and Volume Confirmation
# Hypothesis: Donchian channel breakouts aligned with 12h EMA trend capture momentum in both bull and bear markets.
# Volume filter ensures breakouts have conviction, reducing false signals.
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag.

name = "4h_donchian_breakout_12h_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    
    # Align 12h EMA to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Donchian channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower or trend changes
            if close[i] <= lowest_low[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper or trend changes
            if close[i] >= highest_high[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian upper with volume confirmation and uptrend
            if close[i] > highest_high[i] and volume[i] > vol_avg[i] and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower with volume confirmation and downtrend
            elif close[i] < lowest_low[i] and volume[i] > vol_avg[i] and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals