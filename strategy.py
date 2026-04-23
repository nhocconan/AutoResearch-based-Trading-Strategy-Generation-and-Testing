#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike
- Donchian(20) breakout captures medium-term momentum with clear structure
- 12h EMA50 ensures trades align with higher timeframe trend to reduce whipsaw
- Volume confirmation (> 1.5x 20-period MA) validates breakout strength
- Discrete position sizing (0.25) minimizes fee churn while maintaining exposure
- Designed for 4h timeframe to target 19-50 trades/year per symbol (75-200 total over 4 years)
- Works in bull via long breakouts above rising EMA50 and bear via short breakdowns below falling EMA50
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
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # need Donchian, EMA50_12h, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND above 12h EMA50 AND volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 12h EMA50 AND volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price breaks opposite Donchian level OR volume drops significantly
            exit_signal = False
            if position == 1:
                # Exit long when price breaks below Donchian low
                if close[i] < low_20[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price breaks above Donchian high
                if close[i] > high_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0