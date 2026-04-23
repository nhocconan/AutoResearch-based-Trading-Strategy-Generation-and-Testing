#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) Breakout with 1w EMA50 Trend Filter and Volume Confirmation
- Uses 12h Donchian channel breakout for entry signals
- 1w EMA50 defines higher timeframe trend filter: only trade in direction of weekly trend
- Volume confirmation (> 1.5x 20-period average) filters weak breakouts
- Exit on opposite Donchian(10) touch or trend reversal
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in bull markets via breakouts with trend, avoids bear markets via trend filter
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
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian(20) high AND above 1w EMA50 AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low AND below 1w EMA50 AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price touches opposite Donchian(10) OR trend reverses
            exit_signal = False
            
            # Calculate shorter Donchian(10) for exit
            highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values[i]
            lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values[i]
            
            if position == 1:
                # Exit long when price touches Donchian(10) low OR closes below 1w EMA50
                if (close[i] <= lowest_low_10 or close[i] < ema_50_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price touches Donchian(10) high OR closes above 1w EMA50
                if (close[i] >= highest_high_10 or close[i] > ema_50_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0