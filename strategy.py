#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Uses tight entry conditions: Donchian breakout + 1w EMA50 trend + volume > 2.0x 20-period MA
- Discrete position sizing (0.25) to minimize fee churn
- Exits when price closes back inside prior day's Donchian channel or loses 1w EMA50 trend
- Designed for 1d timeframe to minimize fee drag while capturing medium-term trends
- Target: 15-25 trades/year per symbol (<100 total over 4 years) to avoid fee drag
- Works in both bull and bear markets via trend filter (1w EMA50) and volume confirmation
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: 20-period high, Lower channel: 20-period low
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    upper_20_prev = np.roll(upper_20, 1)
    lower_20_prev = np.roll(lower_20, 1)
    upper_20_prev[0] = upper_20[0]  # first bar uses current
    lower_20_prev[0] = lower_20[0]
    
    # Align Donchian levels to 1d timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20_prev)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20_prev)
    
    # Volume confirmation: > 2.0x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50_1w, Donchian 20, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Upper20 (breakout) AND price > 1w EMA50 (uptrend) AND volume spike
            if (close[i] > upper_20_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < Lower20 (breakdown) AND price < 1w EMA50 (downtrend) AND volume spike
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside previous day's Donchian channel OR loss of trend
            exit_signal = False
            if position == 1:
                # Exit long when close < Lower20 (breakdown of support) OR price < 1w EMA50
                if close[i] < lower_20_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > Upper20 (breakout of resistance) OR price > 1w EMA50
                if close[i] > upper_20_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0