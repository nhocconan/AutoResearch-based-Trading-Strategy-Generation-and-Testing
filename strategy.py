#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
- Donchian(20) breakout captures medium-term momentum with controlled frequency
- 12h EMA50 defines the trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 1.5x 20-period MA) reduces false breakouts
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years)
- Uses discrete position sizing (0.25) to minimize fee churn
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need 12h EMA50, Donchian(20), vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND above 12h EMA50 AND volume spike
            if (close[i] > highest_20[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND below 12h EMA50 AND volume spike
            elif (close[i] < lowest_20[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band OR crosses 12h EMA50
            exit_signal = False
            if position == 1:
                # Exit long when price < Donchian lower band OR < 12h EMA50
                if close[i] < lowest_20[i] or close[i] < ema_50_12h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > Donchian upper band OR > 12h EMA50
                if close[i] > highest_20[i] or close[i] > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0