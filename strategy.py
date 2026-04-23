#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike confirmation.
Enter long when price breaks above Donchian upper band AND 12h EMA50 rising AND volume > 1.5x avg volume.
Enter short when price breaks below Donchian lower band AND 12h EMA50 falling AND volume > 1.5x avg volume.
Exit when price touches Donchian middle band (mean reversion) or trend reverses.
Designed for 4h timeframe to achieve 20-50 trades/year with discrete sizing (0.25) to minimize fee drag.
Works in both bull and bear markets by following the 12h trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    dc_middle = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        dc_upper[i] = np.max(high[i-lookback+1:i+1])
        dc_lower[i] = np.min(low[i-lookback+1:i+1])
        dc_middle[i] = (dc_upper[i] + dc_lower[i]) / 2.0
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 12h EMA50 slope (rising/falling)
    ema_50_slope = np.zeros_like(ema_50_aligned)
    ema_50_slope[1:] = ema_50_aligned[1:] - ema_50_aligned[:-1]
    
    # Calculate average volume (20-period) for volume spike confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(lookback-1, n):
        avg_volume[i] = np.mean(volume[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback-1, 50)  # need Donchian and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(dc_middle[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_50_slope[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 12h EMA50 rising AND volume spike
            if (close[i] > dc_upper[i] and 
                ema_50_slope[i] > 0 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND 12h EMA50 falling AND volume spike
            elif (close[i] < dc_lower[i] and 
                  ema_50_slope[i] < 0 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price touches Donchian middle band OR trend reverses
            exit_signal = False
            if position == 1:
                # Exit long when price <= Donchian middle OR trend breaks (EMA50 falling)
                if close[i] <= dc_middle[i] or ema_50_slope[i] < 0:
                    exit_signal = True
            elif position == -1:
                # Exit short when price >= Donchian middle OR trend breaks (EMA50 rising)
                if close[i] >= dc_middle[i] or ema_50_slope[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0